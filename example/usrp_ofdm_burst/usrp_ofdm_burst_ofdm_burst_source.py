"""
GNU Radio 嵌入式 Python 块：OFDM 突发源（System.transmit + TOML）

周期性输出 OFDM 时域突发，并在流上附加 UHD stream tag：
  tx_sob, tx_time, tx_eob  (Style 1; USRP Sink len_tag_name 须留空)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pmt
from gnuradio import gr

_TAG_SOB = pmt.intern("tx_sob")
_TAG_EOB = pmt.intern("tx_eob")
_TAG_TIME = pmt.intern("tx_time")

_DEFAULT_CONFIG = "implementaion/ofdm_burst_source.toml"


class blk(gr.basic_block):
    """
    Style 1 OFDM 突发源：初始化时调用 System.transmit() 缓存一帧 x_time，
    周期性重放并打 UHD 定时 tag。
    """

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        idle_ms=400.0,
        tx_amp=0.3,
        time_lead_s=0.05,
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
        self._device = str(device)
        self._seed = int(seed)

        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = 0.0
        self._burst_buffer: np.ndarray | None = None

        self.set_tag_propagation_policy(gr.TPP_DONT)
        # 延迟到首次 general_work：GRC 编译期不实例化 Sionna
        self.set_min_output_buffer(4096)

    def _ensure_burst(self) -> None:
        if self._burst_buffer is not None:
            return
        self._build_system_and_burst()

    def _build_system_and_burst(self) -> None:
        """加载 TOML、构建 System、生成并缓存 OFDM 突发缓冲。"""
        import sionna
        import torch

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from isac.system import System
        from isac.utils import set_random_seed

        set_random_seed(self._seed)
        sionna.phy.config.device = self._device
        torch.set_num_threads(1)

        args = argparse.Namespace(
            config_file=self._config_file,
            device=self._device,
            batch_size=1,
        )
        self._system = System(args)

        _, _, x_time = self._system.transmit()
        burst = x_time.squeeze().detach().cpu().numpy().astype(np.complex64, copy=False)
        self._burst_buffer = (burst * self._tx_amp).astype(np.complex64, copy=False)

        self._samp_rate = float(self._system.params.ofdm.samp_rate)
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
        self._burst_buffer = None

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
        self._burst_buffer = None

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = str(value)
        self._burst_buffer = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)
        self._burst_buffer = None

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
            self.add_item_tag(0, abs_out, _TAG_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, _TAG_TIME, self._tx_time_pmt())

        self._burst_idx += n
        if self._burst_idx >= self._burst_len:
            self.add_item_tag(0, abs_out + n - 1, _TAG_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
