"""Sionna BinarySource + Mapper QPSK 符号源，替代 GR random/repack/chunks 链。"""

from __future__ import annotations

import sys

import numpy as np
import pmt
import torch
from gnuradio import gr
from sionna.phy.mapping import BinarySource, Mapper

from isac.utils.misc import set_random_seed
from isac_imp.burst_pack import TPP_DONT

_LOG_PREFIX = "[SionnaQpskSymbolSource]"


def _build_qpsk_symbols(
    n_symbols: int,
    *,
    num_bits_per_symbol: int,
    device: str,
    seed: int,
) -> np.ndarray:
    set_random_seed(seed)
    binary_source = BinarySource(device=device)
    mapper = Mapper("qam", num_bits_per_symbol, device=device)
    bits = binary_source([1, 1, 1, n_symbols * num_bits_per_symbol])
    with torch.inference_mode():
        syms = mapper(bits).reshape(-1)
        if device != "cpu":
            syms = syms.cpu()
        return syms.detach().numpy().astype(np.complex64, copy=False)


class SionnaQpskSymbolSourceBlock(gr.basic_block):
    """固定 seed 生成 QPSK 符号并循环重放；CPI 首符号打 ``length_tag_key`` tag。"""

    def __init__(
        self,
        qpsk_symbols_per_packet: int = 8184,
        length_tag_key: str = "packet_len",
        num_bits_per_symbol: int = 2,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna QPSK Symbol Source",
            in_sig=None,
            out_sig=[np.complex64],
        )
        self._qpsk_symbols_per_packet = int(qpsk_symbols_per_packet)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._device = str(device)
        self._seed = int(seed)
        self._sym_buf: np.ndarray | None = None
        self._sym_idx = 0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._qpsk_symbols_per_packet * 2)

    def _log(self, msg: str) -> None:
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    def start(self) -> bool:
        torch.set_num_threads(1)
        self._sym_buf = _build_qpsk_symbols(
            self._qpsk_symbols_per_packet,
            num_bits_per_symbol=self._num_bits_per_symbol,
            device=self._device,
            seed=self._seed,
        )
        self._sym_idx = 0
        self._log(
            f"loaded {self._sym_buf.size} QPSK symbols "
            f"(seed={self._seed}, device={self._device})"
        )
        return True

    def forecast(self, noutput_items: int, ninputs) -> list:
        del noutput_items, ninputs
        return []

    def general_work(self, input_items, output_items) -> int:
        del input_items
        if self._sym_buf is None:
            return 0

        out = output_items[0]
        max_out = len(out)
        n_out = 0
        abs_base = self.nitems_written(0)

        while n_out < max_out:
            if self._sym_idx == 0:
                self.add_item_tag(
                    0,
                    abs_base + n_out,
                    self._length_tag_key,
                    pmt.from_long(self._qpsk_symbols_per_packet),
                )
            out[n_out] = self._sym_buf[self._sym_idx]
            n_out += 1
            self._sym_idx += 1
            if self._sym_idx >= self._qpsk_symbols_per_packet:
                self._sym_idx = 0

        return n_out
