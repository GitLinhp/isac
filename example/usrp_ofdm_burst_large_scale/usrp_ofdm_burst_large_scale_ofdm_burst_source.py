"""
GNU Radio 嵌入式 Python 块：OFDM 突发源（System.transmit + TOML）

周期性输出 OFDM 时域突发，并在流上附加 UHD 所需的 stream tag：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

Style 1 约定：下游 USRP Sink 的 len_tag_name 必须留空，由 tag 而非包长 tag 界定突发边界。

配置来源：TOML（默认 simulation/sensing/sensing_monostatic.toml）经 create_system 加载；
GRC 变量 config_file / device / num_symbols / fft_size / subcarrier_spacing / cp_len
可配置，其中 OFDM 参数优先覆盖 TOML [ofdm]。

发射链：System.transmit() → 缓存 x_time 周期性重放，x_rg 写入共享 session 供收端 sensing。

注意：__init__ 形参默认值须与 GRC 变量保持同步。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pmt
import torch
from gnuradio import gr

from isac.system import System
from isac_imp.constants import TAG_EOB, TAG_SOB, TAG_TIME, TPP_DONT, make_tx_time_pmt
from isac_imp.gr_setup import create_system, set_last_x_rg

_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"


class blk(gr.basic_block):
    """
    Style 1 OFDM 突发源：首次 general_work 时 create_system + transmit()，
    之后周期性重放并打 UHD 定时 tag。

    工作流程：突发期内从缓存输出样本并打 tag；突发结束后进入 idle 静默期，
    静默期不产出样本（general_work 返回 0），到期再启动下一突发。
    """

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        idle_ms=900.0,
        tx_amp=0.3,
        time_lead_s=0.3,
        startup_delay_s=0.2,
        num_symbols=32,
        fft_size=2048,
        subcarrier_spacing=15000.0,
        cp_len=0,
        device="cpu",
        seed=42,
    ):
        """
        参数:
            - config_file:         TOML 相对路径（经 load_config 解析）
            - idle_ms:             两次突发之间的纯静默间隔 (毫秒)，不含突发本身时长
            - tx_amp:              输出幅度缩放，建议 ≤ 1.0 以免 USRP 饱和
            - time_lead_s:         tx_time 相对当前 wall-clock 的提前量 (秒)
            - startup_delay_s:     首突发启动前的初始等待 (秒)
            - num_symbols:         OFDM 符号数，GRC 覆盖 TOML [ofdm]
            - fft_size:            FFT 点数，GRC 覆盖 TOML [ofdm]
            - subcarrier_spacing:  子载波间隔 (Hz)，GRC 覆盖 TOML [ofdm]
            - cp_len:              循环前缀长度（采样点），GRC 覆盖 TOML [ofdm]
            - device:              Sionna/Torch 计算设备（cpu / cuda）
            - seed:                随机种子，变更后需重建突发缓冲
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

        # 懒加载：共享 System、参考频域网格、时域突发缓冲
        self._system: System | None = None
        self._x_rg: Any = None
        self._burst_buffer: np.ndarray | None = None
        self._samp_rate = 0.0
        self._burst_len = 0
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

        # 不向上游传播 tag；初始输出缓冲占位，构建突发后按 burst_len 调整
        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(4096)

    def _ofdm_overrides(self) -> dict:
        return {
            "num_symbols": self._num_symbols,
            "fft_size": self._fft_size,
            "subcarrier_spacing": self._subcarrier_spacing,
            "cyclic_prefix_length": self._cp_len,
        }

    def _invalidate_burst(self) -> None:
        """GRC 参数变更时清空本地缓存，下次 general_work 触发重建。"""
        self._system = None
        self._x_rg = None
        self._burst_buffer = None

    def _ensure_burst(self) -> np.ndarray:
        """懒加载入口：缓存未命中时 create_system + transmit()，返回突发缓冲。"""
        if self._burst_buffer is None:
            self._build_system_and_burst()
        assert self._burst_buffer is not None
        return self._burst_buffer

    def _build_system_and_burst(self) -> None:
        """create_system（对齐 monostatic）+ transmit()，缓存时域突发与 x_rg。"""
        torch.set_num_threads(1)

        self._system = create_system(
            self._config_file,
            device=self._device,
            seed=self._seed,
            ofdm_overrides=self._ofdm_overrides(),
        )
        _, self._x_rg, x_time = self._system.transmit()
        set_last_x_rg(self._system, self._x_rg)

        burst = x_time.squeeze().detach().cpu().numpy().astype(np.complex64, copy=False)
        burst_buffer = (burst * self._tx_amp).astype(np.complex64, copy=False)
        self._burst_buffer = burst_buffer

        ofdm_params = self._system.params.ofdm
        if ofdm_params is None:
            raise RuntimeError("TOML 缺少 [ofdm] 配置")
        self._samp_rate = float(ofdm_params.samp_rate)
        self._burst_len = int(burst_buffer.size)
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

    def forecast(self, noutput_items, ninputs):
        """无输入端口，不向调度器声明输入需求。"""
        del noutput_items, ninputs
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
        return make_tx_time_pmt(time.time() + self._time_lead_s)

    def general_work(self, input_items, output_items):
        """
        主工作函数：按块从缓存重放 OFDM 突发，在突发首尾附加 UHD tag。

        静默期内返回 0（不消耗输出缓冲）；突发可分多次 general_work 完成。
        返回值 n 表示本次写入输出端口的有效样本数。
        """
        del input_items
        burst_buffer = self._ensure_burst()
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
        out[:n] = burst_buffer[self._burst_idx : self._burst_idx + n]

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
