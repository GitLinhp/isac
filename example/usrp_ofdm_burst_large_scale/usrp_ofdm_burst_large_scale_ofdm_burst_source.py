"""
GNU Radio 嵌入式 Python 块：OFDM 突发源（System.transmit + TOML）

周期性输出 OFDM 时域突发，并在流上附加 UHD 所需的 stream tag：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

Style 1 约定：下游 USRP Sink 的 len_tag_name 必须留空，由 tag 而非包长 tag 界定突发边界。

配置来源：TOML（默认 implementaion/ofdm_burst_source_large_sacle.toml）经 load_config 加载；
GRC 变量 config_file / device / num_symbols / fft_size / subcarrier_spacing / cp_len
可配置，其中 OFDM 参数优先覆盖 TOML [ofdm]。

发射链：ZC 比特源 → 资源网格映射 → OFDM 调制 → 缓存为 _burst_buffer 后周期性重放。
"""

from __future__ import annotations

import time

import numpy as np
import pmt
import sionna
import torch
from gnuradio import gr

from isac.data_structures import SystemComponents, SystemParams
from isac.utils import load_config, set_random_seed
from isac_imp.constants import TAG_EOB, TAG_SOB, TAG_TIME

_DEFAULT_CONFIG = "implementaion/ofdm_burst_source_large_sacle.toml"


def _log_override(label: str, toml_val, grc_val) -> None:
    print(f"  [config] GRC 覆盖 TOML: {label} {toml_val}→{grc_val}")


class blk(gr.basic_block):
    """
    Style 1 OFDM 突发源：首次 general_work 时懒加载并缓存一帧 x_time，
    之后周期性重放并打 UHD 定时 tag。

    工作流程：突发期内从缓存输出样本并打 tag；突发结束后进入 idle 静默期，
    静默期不产出样本（general_work 返回 0），到期再启动下一突发。
    """

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        idle_ms=400.0,
        tx_amp=0.3,
        time_lead_s=0.3,
        startup_delay_s=0.2,
        num_symbols=32,
        fft_size=64,
        subcarrier_spacing=15000.0,
        cp_len=0,
        device="cpu",
        seed=42,
    ):
        """
        参数:
            config_file:         TOML 相对路径（经 load_config 解析）
            idle_ms:             两次突发之间的纯静默间隔 (毫秒)，不含突发本身时长
            tx_amp:              输出幅度缩放，建议 ≤ 1.0 以免 USRP 饱和
            time_lead_s:         tx_time 相对当前 wall-clock 的提前量 (秒)，
                                 给 USRP 调度器预留缓冲，避免“已过期”时刻
            startup_delay_s:     首突发启动前的初始等待 (秒)
            num_symbols:         OFDM 符号数，GRC 覆盖 TOML [ofdm]
            fft_size:            FFT 点数，GRC 覆盖 TOML [ofdm]
            subcarrier_spacing:  子载波间隔 (Hz)，GRC 覆盖 TOML [ofdm]
            cp_len:              循环前缀长度（采样点），GRC 覆盖 TOML [ofdm]
            device:              Sionna/Torch 计算设备（cpu / cuda）
            seed:                随机种子，变更后需重建突发缓冲
        """
        gr.basic_block.__init__(
            self,
            name="OFDM Burst Source",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._config_file = str(config_file)
        self._idle_ms = float(idle_ms)
        self._tx_amp = float(tx_amp)
        self._time_lead_s = float(time_lead_s)
        self._startup_delay_s = float(startup_delay_s)
        self._num_symbols = int(num_symbols)
        self._fft_size = int(fft_size)
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._cp_len = int(cp_len)
        self._device = str(device)
        self._seed = int(seed)

        # 突发状态机：是否正在输出、突发内样本索引、下次允许启动的 monotonic 时刻
        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = time.monotonic() + self._startup_delay_s
        # 懒加载缓存：OFDM 时域突发缓冲与已解析的 SystemParams
        self._burst_buffer: np.ndarray | None = None
        self._system_params = None

        # 不向上游传播 tag；初始输出缓冲占位，构建突发后按 burst_len 调整
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.set_min_output_buffer(4096)

    def _invalidate_burst(self) -> None:
        """GRC 参数变更时清空缓存，下次 general_work 触发重建。"""
        self._burst_buffer = None
        self._system_params = None

    def _ensure_burst(self) -> None:
        """懒加载入口：缓存未命中时构建突发缓冲。"""
        if self._burst_buffer is not None:
            return
        self._build_system_and_burst()

    def _build_system_and_burst(self) -> None:
        """加载 TOML、GRC 覆盖 [ofdm]、构建发射链并缓存突发缓冲。"""
        set_random_seed(self._seed)
        sionna.phy.config.device = self._device
        torch.set_num_threads(1)

        # 加载 TOML 并记录 GRC 对 [ofdm] 的覆盖
        raw = load_config(self._config_file)
        toml_ofdm = dict(raw.get("ofdm") or {})
        if toml_ofdm.get("num_symbols") != self._num_symbols:
            _log_override("num_symbols", toml_ofdm.get("num_symbols"), self._num_symbols)
        if toml_ofdm.get("fft_size") != self._fft_size:
            _log_override("fft_size", toml_ofdm.get("fft_size"), self._fft_size)
        if float(toml_ofdm.get("subcarrier_spacing", 0)) != self._subcarrier_spacing:
            _log_override(
                "subcarrier_spacing",
                toml_ofdm.get("subcarrier_spacing"),
                self._subcarrier_spacing,
            )
        if int(toml_ofdm.get("cyclic_prefix_length", 0)) != self._cp_len:
            _log_override(
                "cp",
                toml_ofdm.get("cyclic_prefix_length", 0),
                self._cp_len,
            )

        # 以 GRC 参数覆盖 TOML [ofdm] 后构建 SystemParams / SystemComponents
        ofdm = dict(toml_ofdm)
        ofdm["num_symbols"] = self._num_symbols
        ofdm["fft_size"] = self._fft_size
        ofdm["subcarrier_spacing"] = self._subcarrier_spacing
        ofdm["cyclic_prefix_length"] = self._cp_len
        raw["ofdm"] = ofdm

        params = SystemParams.from_dict(raw)
        self._system_params = params
        comps = SystemComponents.build_from_params(params, device=self._device)

        # 发射链：ZC 比特源 → 资源网格映射 → OFDM 调制
        x = comps.zc_source([1, 1, 1, comps.rg.num_data_symbols])
        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)

        # 转为 numpy 并施加幅度缩放，缓存供 general_work 重放
        burst = x_time.squeeze().detach().cpu().numpy().astype(np.complex64, copy=False)
        self._burst_buffer = (burst * self._tx_amp).astype(np.complex64, copy=False)

        # 派生采样率、突发长度与静默期，并调整输出缓冲
        self._samp_rate = float(params.ofdm.samp_rate)
        self._burst_len = int(self._burst_buffer.size)
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self.set_min_output_buffer(self._burst_len * 2)

    def _recompute_idle(self) -> None:
        """仅 idle_ms 变更时重算静默期，无需重建波形。"""
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        self._config_file = str(value)
        self._invalidate_burst()

    @property
    def num_symbols(self):
        return self._num_symbols

    @num_symbols.setter
    def num_symbols(self, value):
        self._num_symbols = int(value)
        self._invalidate_burst()

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value):
        self._fft_size = int(value)
        self._invalidate_burst()

    @property
    def subcarrier_spacing(self):
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, value):
        self._subcarrier_spacing = float(value)
        self._invalidate_burst()

    @property
    def cp_len(self):
        return self._cp_len

    @cp_len.setter
    def cp_len(self, value):
        self._cp_len = int(value)
        self._invalidate_burst()

    @property
    def samp_rate(self):
        return self._samp_rate

    @property
    def burst_len(self):
        return self._burst_len

    @property
    def idle_ms(self):
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        self._idle_ms = float(value)
        self._recompute_idle()

    @property
    def tx_amp(self):
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        self._tx_amp = float(value)
        self._invalidate_burst()

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

    @property
    def startup_delay_s(self):
        return self._startup_delay_s

    @startup_delay_s.setter
    def startup_delay_s(self, value):
        self._startup_delay_s = float(value)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = str(value)
        self._invalidate_burst()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)
        self._invalidate_burst()

    def forecast(self, noutput_items, ninput_items):
        """无输入端口，不向调度器声明输入需求。"""
        del noutput_items, ninput_items
        return []

    def _schedule_delay_s(self) -> float:
        """
        计算当前突发结束后，到下一突发开始前的等待时间（秒）。

        周期 = max(idle + burst, burst)；返回值 = 周期 - burst，即纯 idle 时长。
        """
        burst_s = self._burst_len / self._samp_rate
        period_s = max(self._idle_s + burst_s, burst_s)
        return max(0.0, period_s - burst_s)

    def _tx_time_pmt(self):
        """
        构造 tx_time tag 的 PMT 值：(整数秒, 小数秒)。

        使用 time.time() + time_lead_s，与 UHD timed TX 的绝对时刻格式一致。
        """
        t = time.time() + self._time_lead_s
        sec = int(t)
        frac = t - sec
        return pmt.make_tuple(pmt.from_uint64(sec), pmt.from_double(frac))

    def general_work(self, input_items, output_items):
        """
        主工作函数：按块从缓存重放 OFDM 突发，在突发首尾附加 UHD tag。

        静默期内返回 0（不消耗输出缓冲）；突发可分多次 general_work 完成。
        返回值 n 表示本次写入输出端口的有效样本数。
        """
        del input_items
        self._ensure_burst()
        out = output_items[0]

        # 非突发态：未到调度时刻则暂不输出
        if not self._burst_active:
            if time.monotonic() < self._next_burst_at:
                return 0
            self._burst_active = True
            self._burst_idx = 0

        # 本次最多输出的样本数（受输出缓冲与剩余突发长度限制）
        n_remain = self._burst_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        # 从缓存重放突发片段（非实时生成，保证波形一致）
        out[:n] = self._burst_buffer[self._burst_idx : self._burst_idx + n]

        abs_out = self.nitems_written(0)
        # 突发第一个样本：打 tx_sob 与 tx_time
        if self._burst_idx == 0:
            self.add_item_tag(0, abs_out, TAG_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, TAG_TIME, self._tx_time_pmt())

        self._burst_idx += n
        # 突发最后一个样本：打 tx_eob，进入 idle 并预约下一突发
        if self._burst_idx >= self._burst_len:
            self.add_item_tag(0, abs_out + n - 1, TAG_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
