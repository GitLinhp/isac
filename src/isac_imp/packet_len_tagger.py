"""在 CPI 边界为标量 IQ 流添加 ``packet_len`` tag。

scheduled 模式：在 UHD ``rx_time`` tag 处打 tag（与 ``NUM_SAMPS_AND_DONE`` 突发边界对齐）。
continuous 模式：按 ``burst_len_samples`` 样点计数相位打 tag（回退路径）。
"""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import TAG_RX_TIME, TPP_DONT


class PacketLenTaggerBlock(gr.sync_block):
    """scheduled RX：``rx_time`` 边界 → ``packet_len``；否则样点计数相位。"""

    def __init__(
        self,
        burst_len_samples: int = 10240,
        length_tag_key: str = "packet_len",
        use_rx_time: bool = True,
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="Packet Len Tagger",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._burst_len = int(burst_len_samples)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._use_rx_time = bool(use_rx_time)
        self._last_tag_abs: int | None = None
        self.set_tag_propagation_policy(TPP_DONT)

    @property
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len = int(value)

    @property
    def use_rx_time(self) -> bool:
        return self._use_rx_time

    @use_rx_time.setter
    def use_rx_time(self, value: bool) -> None:
        self._use_rx_time = bool(value)

    def start(self) -> bool:
        self._last_tag_abs = None
        return True

    def _tag_modulo(self, in_base: int, out_base: int, n: int) -> None:
        if self._burst_len <= 0:
            return
        pos = 0
        start_phase = in_base % self._burst_len
        if start_phase != 0:
            pos = self._burst_len - start_phase
        while pos < n:
            abs_out = out_base + pos
            self.add_item_tag(
                0,
                abs_out,
                self._length_tag_key,
                pmt.from_long(self._burst_len),
            )
            self._last_tag_abs = in_base + pos
            pos += self._burst_len

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        out[:n] = inp

        in_base = self.nitems_read(0)
        out_base = self.nitems_written(0)

        if self._use_rx_time:
            tagged_any = False
            for i in range(n):
                abs_in = in_base + i
                for tag in self.get_tags_in_range(0, abs_in, abs_in + 1):
                    if not pmt.eq(tag.key, TAG_RX_TIME):
                        continue
                    if (
                        self._last_tag_abs is not None
                        and abs_in - self._last_tag_abs < self._burst_len
                    ):
                        break
                    self.add_item_tag(
                        0,
                        out_base + i,
                        self._length_tag_key,
                        pmt.from_long(self._burst_len),
                    )
                    self._last_tag_abs = abs_in
                    tagged_any = True
                    break
            if not tagged_any:
                self._tag_modulo(in_base, out_base, n)
        else:
            self._tag_modulo(in_base, out_base, n)

        return n


blk = PacketLenTaggerBlock
