"""
Embedded Python Block: Style 1 USRP burst source (sine wave)

Generates periodic complex sinusoid bursts with UHD stream tags:
  tx_sob, tx_time, tx_eob  (Style 1; USRP Sink len_tag_name must be empty)
"""

from __future__ import annotations

import time

import numpy as np
import pmt
from gnuradio import gr

_TAG_SOB = pmt.intern("tx_sob")
_TAG_EOB = pmt.intern("tx_eob")
_TAG_TIME = pmt.intern("tx_time")


class blk(gr.basic_block):
    """Style 1 timed burst source outputting a complex sinusoid."""

    def __init__(
        self,
        samp_rate=1e6,
        tone_freq=100e3,
        burst_ms=100.0,
        idle_ms=400.0,
        tx_amp=0.3,
        time_lead_s=0.05,
    ):
        gr.basic_block.__init__(
            self,
            name="Sine Burst Source",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._samp_rate = float(samp_rate)
        self._tone_freq = float(tone_freq)
        self._burst_ms = float(burst_ms)
        self._idle_ms = float(idle_ms)
        self._tx_amp = float(tx_amp)
        self._time_lead_s = float(time_lead_s)
        self._phase_inc = 2.0 * np.pi * self._tone_freq / self._samp_rate
        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = 0.0

        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.set_min_output_buffer(self._burst_len * 2)

    def _recompute_timing(self) -> None:
        self._phase_inc = 2.0 * np.pi * self._tone_freq / self._samp_rate
        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self.set_min_output_buffer(self._burst_len * 2)

    @property
    def samp_rate(self):
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value):
        self._samp_rate = float(value)
        self._recompute_timing()

    @property
    def tone_freq(self):
        return self._tone_freq

    @tone_freq.setter
    def tone_freq(self, value):
        self._tone_freq = float(value)
        self._recompute_timing()

    @property
    def burst_ms(self):
        return self._burst_ms

    @burst_ms.setter
    def burst_ms(self, value):
        self._burst_ms = float(value)
        self._recompute_timing()

    @property
    def idle_ms(self):
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        self._idle_ms = float(value)
        self._recompute_timing()

    @property
    def tx_amp(self):
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        self._tx_amp = float(value)

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

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

        idx = np.arange(self._burst_idx, self._burst_idx + n, dtype=np.float64)
        phase = self._phase_inc * idx
        out[:n] = (self._tx_amp * (np.cos(phase) + 1j * np.sin(phase))).astype(
            np.complex64, copy=False
        )

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
