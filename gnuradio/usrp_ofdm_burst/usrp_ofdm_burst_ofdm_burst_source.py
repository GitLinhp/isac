"""GNU Radio 嵌入式 Python 块：OFDM 突发源（运行时 ZC + OFDM，非 TransmitCache）。

周期性输出 OFDM 时域突发，并在流上附加 UHD Style 1 stream tag：
``tx_sob`` + ``tx_time`` + ``tx_eob``（USRP Sink 的 ``len_tag_name`` 须留空）。

职责：用 TOML + GRC 覆盖的 ``[ofdm]`` 经 ``SystemComponents`` 现场生成一帧
（ZC → 资源网格 → 调制），乘 ``tx_amp`` 后缓存并周期重放。**不**读取
``TransmitCache`` / ``x_time.npy``。

调度：``startup_delay_s`` 为首启延迟；突发间隙由 ``idle_ms`` 与突发时长共同决定。
``tx_time`` 为 ``time.time() + time_lead_s`` 的绝对 Unix 时刻。

GRC 变量 ``config_file`` / ``device`` / ``num_symbols`` / ``fft_size`` /
``subcarrier_spacing`` / ``cp_len`` 可配置；其中 OFDM 参数优先覆盖 TOML ``[ofdm]``。
``__init__`` 形参默认值须与 GRC 变量保持同步。
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

from isac_imp.burst_pack import TAG_TX_EOB, TAG_TX_SOB, TAG_TX_TIME

_DEFAULT_CONFIG = "implementaion/ofdm_burst_source.toml"
"""默认 TOML（相对 ``config/``）；须与 GRC 默认值一致。"""


def _log_override(label: str, toml_val, grc_val) -> None:
    """GRC 参数覆盖 TOML 对应项时打印一行差异。"""
    print(f"  [config] GRC 覆盖 TOML: {label} {toml_val}→{grc_val}")


class blk(gr.basic_block):
    """Style 1 OFDM 突发源：懒生成一帧 ``x_time``，周期性重放并打 UHD 定时 tag。

    生命周期：``__init__`` 仅存参与调度状态 → 首次写出时
    ``_build_system_and_burst`` → ``general_work`` 分段拷贝缓冲并打
    SOB/TIME（首样）与 EOB（末样）；idle 期返回 0。
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
        """构造块并初始化调度状态（不立即生成波形）。

        参数:
        -------
        - config_file : str
            TOML 路径（相对 ``config/`` 或仓库根）
        - idle_ms : float
            两次突发之间的纯静默间隔（毫秒），不含突发本身时长
        - tx_amp : float
            输出幅度缩放（生成缓冲时乘入）
        - time_lead_s : float
            ``tx_time`` 相对 ``time.time()`` 的提前量（秒）
        - startup_delay_s : float
            首突发启动前的初始等待（秒）
        - num_symbols / fft_size / subcarrier_spacing / cp_len
            GRC 侧 OFDM 覆盖项，优先写入 ``[ofdm]``
        - device : str
            Sionna / Torch 设备
        - seed : int
            随机种子（ZC / 映射可复现）

        内部状态（调度）:
        - ``_burst_active``：当前是否在突发期内
        - ``_burst_idx``：缓冲内已写出偏移
        - ``_next_burst_at``：下一突发允许开始的 ``monotonic`` 时刻
        - ``_burst_buffer``：缩放后的一维 ``complex64`` 突发样点；``None`` 表示待重建
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

        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = time.monotonic() + self._startup_delay_s
        self._burst_buffer: np.ndarray | None = None
        self._system_params = None

        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.set_min_output_buffer(4096)

    def _invalidate_burst(self) -> None:
        """清空突发缓冲与参数缓存；下次写出时 ``_ensure_burst`` 重建。"""
        self._burst_buffer = None
        self._system_params = None

    def _ensure_burst(self) -> None:
        """懒加载入口：缓冲未命中时构建系统与突发。"""
        if self._burst_buffer is not None:
            return
        self._build_system_and_burst()

    def _build_system_and_burst(self) -> None:
        """加载 TOML、GRC 覆盖 ``[ofdm]``、构建组件并缓存突发缓冲。

        流程：覆盖 OFDM 几何 → ``SystemParams`` / ``SystemComponents`` →
        ZC → ``rg_mapper`` → ``modulator`` → squeeze → ``* tx_amp`` →
        记录 ``samp_rate`` / ``burst_len`` / ``idle_s``。
        """
        set_random_seed(self._seed)
        sionna.phy.config.device = self._device
        torch.set_num_threads(1)

        raw = load_config(self._config_file)
        toml_ofdm = dict(raw.get("ofdm") or {})
        if toml_ofdm.get("num_symbols") != self._num_symbols:
            _log_override(
                "num_symbols", toml_ofdm.get("num_symbols"), self._num_symbols
            )
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

        ofdm = dict(toml_ofdm)
        ofdm["num_symbols"] = self._num_symbols
        ofdm["fft_size"] = self._fft_size
        ofdm["subcarrier_spacing"] = self._subcarrier_spacing
        ofdm["cyclic_prefix_length"] = self._cp_len
        raw["ofdm"] = ofdm

        params = SystemParams.from_dict(raw)
        self._system_params = params
        comps = SystemComponents.build_from_params(params, device=self._device)

        x = comps.zc_source([1, 1, 1, comps.rg.num_data_symbols])
        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)

        burst = x_time.squeeze().detach().cpu().numpy().astype(np.complex64, copy=False)
        self._burst_buffer = (burst * self._tx_amp).astype(np.complex64, copy=False)

        self._samp_rate = float(params.ofdm.samp_rate)
        self._burst_len = int(self._burst_buffer.size)
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self.set_min_output_buffer(self._burst_len * 2)

    def _recompute_idle(self) -> None:
        """由 ``idle_ms`` 更新秒级 idle，不重建波形。"""
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        """更换 TOML；清空缓冲以便按新配置重建。"""
        self._config_file = str(value)
        self._invalidate_burst()

    @property
    def num_symbols(self):
        return self._num_symbols

    @num_symbols.setter
    def num_symbols(self, value):
        """覆盖 OFDM 符号数；几何变更需重建突发。"""
        self._num_symbols = int(value)
        self._invalidate_burst()

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value):
        """覆盖 FFT 点数；几何变更需重建突发。"""
        self._fft_size = int(value)
        self._invalidate_burst()

    @property
    def subcarrier_spacing(self):
        return self._subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, value):
        """覆盖子载波间隔；几何变更需重建突发。"""
        self._subcarrier_spacing = float(value)
        self._invalidate_burst()

    @property
    def cp_len(self):
        return self._cp_len

    @cp_len.setter
    def cp_len(self, value):
        """覆盖循环前缀长度；几何变更需重建突发。"""
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
        """仅更新调度间隔，不重建波形。"""
        self._idle_ms = float(value)
        self._recompute_idle()

    @property
    def tx_amp(self):
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        """幅度写入缓冲，变更需重建突发。"""
        self._tx_amp = float(value)
        self._invalidate_burst()

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        """仅影响后续 ``tx_time``，不重建波形。"""
        self._time_lead_s = float(value)

    @property
    def startup_delay_s(self):
        return self._startup_delay_s

    @startup_delay_s.setter
    def startup_delay_s(self, value):
        """仅存参；已启动后的 ``_next_burst_at`` 不自动回写。"""
        self._startup_delay_s = float(value)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        """计算设备变更需重建突发。"""
        self._device = str(value)
        self._invalidate_burst()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        """随机种子变更需重建突发。"""
        self._seed = int(value)
        self._invalidate_burst()

    def forecast(self, noutput_items, ninput_items):
        """源块无输入流，返回空消耗列表。"""
        del noutput_items, ninput_items
        return []

    def _schedule_delay_s(self) -> float:
        """突发结束后到下一突发的纯 idle 秒数（思路同 ``schedule_idle_delay_s``）。"""
        burst_s = self._burst_len / self._samp_rate
        period_s = max(self._idle_s + burst_s, burst_s)
        return max(0.0, period_s - burst_s)

    def _tx_time_pmt(self):
        """构造 ``tx_time`` PMT：``time.time() + time_lead_s`` → ``(uint64, double)``。"""
        t = time.time() + self._time_lead_s
        sec = int(t)
        frac = t - sec
        return pmt.make_tuple(pmt.from_uint64(sec), pmt.from_double(frac))

    def general_work(self, input_items, output_items):
        """写出突发样点；idle 返回 0；首样打 SOB/TIME，末样打 EOB。"""
        del input_items
        self._ensure_burst()
        out = output_items[0]

        if not self._burst_active:
            if time.monotonic() < self._next_burst_at:
                return 0
            self._burst_active = True
            self._burst_idx = 0

        n_remain = self._burst_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        out[:n] = self._burst_buffer[self._burst_idx : self._burst_idx + n]

        abs_out = self.nitems_written(0)
        if self._burst_idx == 0:
            self.add_item_tag(0, abs_out, TAG_TX_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, TAG_TX_TIME, self._tx_time_pmt())

        self._burst_idx += n
        if self._burst_idx >= self._burst_len:
            self.add_item_tag(0, abs_out + n - 1, TAG_TX_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
