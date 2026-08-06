"""统计 ``packet_len`` tag 产出率，用于诊断 overflow / 对齐问题。"""

from __future__ import annotations

import time

import numpy as np
import pmt
from gnuradio import gr


class CpiRateProbeBlock(gr.sync_block):
    """透传 IQ 并每秒打印 CPI（``packet_len`` tag）计数。"""

    def __init__(
        self,
        length_tag_key: str = "packet_len",
        log_interval_s: float = 1.0,
        label: str = "cpi_probe",
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="CPI Rate Probe",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._log_interval_s = float(log_interval_s)
        self._label = str(label)

        self._cpi_count = 0
        self._last_log_mono = 0.0
        self._last_log_cpi = 0

    @property
    def cpi_count(self) -> int:
        return self._cpi_count

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        out[:n] = inp

        in_base = self.nitems_read(0)
        for tag in self.get_tags_in_range(0, in_base, in_base + n):
            if pmt.eq(tag.key, self._length_tag_key):
                self._cpi_count += 1

        now = time.monotonic()
        if now - self._last_log_mono >= self._log_interval_s:
            cpi_delta = self._cpi_count - self._last_log_cpi
            print(
                f"[{self._label}] CPI/s={cpi_delta / self._log_interval_s:.1f} "
                f"total_cpi={self._cpi_count}",
                flush=True,
            )
            self._last_log_mono = now
            self._last_log_cpi = self._cpi_count
        return n


blk = CpiRateProbeBlock
