"""CPI 频域符号流 tag 归一化 epy 块。

剥离上游 ``packet_len`` tag 并按 ``transpose_len`` 符号重打，供 GR
``radar_ofdm_divide_vcvc`` / ``transpose_matrix_vcvc`` 等 tagged_stream 链使用。
"""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import TPP_DONT


class OfdmCpiTagNormalizerBlock(gr.sync_block):
    """1:1 透传频域向量；每 CPI 首符号打 ``packet_len=transpose_len`` tag。"""

    def __init__(
        self,
        fft_len: int = 2048,
        transpose_len: int = 4,
        length_tag_key: str = "packet_len",
    ) -> None:
        self._fft_len = int(fft_len)
        self._transpose_len = int(transpose_len)
        gr.sync_block.__init__(
            self,
            name="OFDM CPI Tag Normalizer",
            in_sig=[(np.complex64, self._fft_len)],
            out_sig=[(np.complex64, self._fft_len)],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._sym_idx = 0
        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._transpose_len * 2)

    def start(self) -> bool:
        self._sym_idx = 0
        return True

    def work(self, input_items, output_items) -> int:
        inp = input_items[0]
        out = output_items[0]
        n = min(len(inp), len(out))
        abs_base = self.nitems_written(0)

        for i in range(n):
            if self._sym_idx == 0:
                self.add_item_tag(
                    0,
                    abs_base + i,
                    self._length_tag_key,
                    pmt.from_long(self._transpose_len),
                )
            out[i][:] = inp[i]
            self._sym_idx += 1
            if self._sym_idx >= self._transpose_len:
                self._sym_idx = 0

        return n


blk = OfdmCpiTagNormalizerBlock
