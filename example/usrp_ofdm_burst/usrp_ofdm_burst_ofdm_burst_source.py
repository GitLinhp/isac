"""
GNU Radio 嵌入式 Python 块：OFDM 突发源（System.transmit + TOML）

周期性输出 OFDM 时域突发，并在流上附加 UHD stream tag：
  tx_sob, tx_time, tx_eob  (Style 1; USRP Sink len_tag_name 须留空)

GRC 变量 config_file / device / num_symbols / fft_size / subcarrier_spacing / cp_len
可配置；其中 OFDM 参数优先覆盖 TOML [ofdm]。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pmt
import sionna
import torch
from gnuradio import gr

try:
    _repo_root = Path(__file__).resolve().parents[2]
except NameError:
    _repo_root = Path.cwd().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from isac.data_structures import SystemComponents, SystemParams
from isac.utils import load_config, set_random_seed
from isac_imp.constants import TAG_EOB, TAG_SOB, TAG_TIME

_DEFAULT_CONFIG = "implementaion/ofdm_burst_source.toml"


def _log_override(label: str, toml_val, grc_val) -> None:
    print(f"  [config] GRC 覆盖 TOML: {label} {toml_val}→{grc_val}")


class blk(gr.basic_block):
    """
    Style 1 OFDM 突发源：初始化时生成并缓存一帧 x_time，
    周期性重放并打 UHD 定时 tag。
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
        self._burst_buffer = None
        self._system_params = None

    def _ensure_burst(self) -> None:
        if self._burst_buffer is not None:
            return
        self._build_system_and_burst()

    def _build_system_and_burst(self) -> None:
        """加载 TOML、GRC 覆盖 [ofdm]、构建组件并缓存突发缓冲。"""
        set_random_seed(self._seed)
        sionna.phy.config.device = self._device
        torch.set_num_threads(1)

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
        del noutput_items, ninput_items
        return []

    def _schedule_delay_s(self) -> float:
        burst_s = self._burst_len / self._samp_rate
        period_s = max(self._idle_s + burst_s, burst_s)
        return max(0.0, period_s - burst_s)

    def _tx_time_pmt(self):
        t = time.time() + self._time_lead_s
        sec = int(t)
        frac = t - sec
        return pmt.make_tuple(pmt.from_uint64(sec), pmt.from_double(frac))

    def general_work(self, input_items, output_items):
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
            self.add_item_tag(0, abs_out, TAG_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, TAG_TIME, self._tx_time_pmt())

        self._burst_idx += n
        if self._burst_idx >= self._burst_len:
            self.add_item_tag(0, abs_out + n - 1, TAG_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
