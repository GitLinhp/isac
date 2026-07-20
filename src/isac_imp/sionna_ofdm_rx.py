"""Sionna 接收链：OFDMDemodulator，替代 GR CP remover + FFT。"""

from __future__ import annotations

import sys

import numpy as np
import pmt
import torch
from gnuradio import gr
from sionna.phy.ofdm import OFDMDemodulator

from isac_imp.burst_pack import TPP_DONT

_LOG_PREFIX = "[SionnaOfdmRx]"


class SionnaOfdmRxBlock(gr.basic_block):
    """按 CPI 缓冲时域 burst，OFDMDemodulator 解调至资源网格向量流。

    - 输入：标量 complex64（``burst_len_samples`` 为一帧 CPI）
    - 输出：vlen=fft_len，CPI 首符号打 ``length_tag_key=transpose_len``
    - 输出格式与 ``ResourceGridMapper`` / ``SionnaOfdmTxBlock`` out1 一致（自然序）
    """

    def __init__(
        self,
        fft_len: int = 2048,
        transpose_len: int = 4,
        cp_len: int = 512,
        l_min: int = 0,
        length_tag_key: str = "packet_len",
        device: str = "cpu",
    ) -> None:
        self._fft_len = int(fft_len)
        self._transpose_len = int(transpose_len)
        self._cp_len = int(cp_len)
        self._l_min = int(l_min)
        self._burst_len_samples = self._transpose_len * (
            self._fft_len + self._cp_len
        )
        gr.basic_block.__init__(
            self,
            name="Sionna OFDM RX",
            in_sig=[np.complex64],
            out_sig=[(np.complex64, self._fft_len)],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._device = str(device)
        self._demodulator: OFDMDemodulator | None = None
        self._in_buf = np.zeros(0, dtype=np.complex64)
        self._pending_symbols: np.ndarray | None = None
        self._sym_idx = 0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(max(self._transpose_len * 2, 4))

    def _log(self, msg: str) -> None:
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    def start(self) -> bool:
        torch.set_num_threads(1)
        self._demodulator = OFDMDemodulator(
            fft_size=self._fft_len,
            l_min=self._l_min,
            cyclic_prefix_length=self._cp_len,
            device=self._device,
        )
        self._in_buf = np.zeros(0, dtype=np.complex64)
        self._pending_symbols = None
        self._sym_idx = 0
        self._log(
            f"ready burst_len={self._burst_len_samples} "
            f"symbols={self._transpose_len} fft_len={self._fft_len} "
            f"cp_len={self._cp_len} l_min={self._l_min}"
        )
        return True

    def _demodulate_burst(self, burst: np.ndarray) -> np.ndarray:
        assert self._demodulator is not None
        with torch.inference_mode():
            x_time = torch.from_numpy(burst).unsqueeze(0).to(
                device=self._device, dtype=torch.complex64
            )
            x_rg = self._demodulator(x_time)
            grid = x_rg.squeeze(0).cpu().numpy().astype(np.complex64, copy=False)
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        if grid.shape != (self._transpose_len, self._fft_len):
            grid = grid.reshape(self._transpose_len, self._fft_len)
        return grid

    def forecast(self, noutput_items: int, ninputs: int) -> list:
        if noutput_items <= 0:
            return [0] * ninputs
        bursts = (noutput_items + self._transpose_len - 1) // self._transpose_len
        return [bursts * self._burst_len_samples] * ninputs

    def general_work(self, input_items, output_items) -> int:
        if self._demodulator is None:
            return 0

        in_stream = input_items[0]
        out_stream = output_items[0]
        n_read = len(in_stream)
        if n_read > 0:
            self._in_buf = np.concatenate([self._in_buf, in_stream[:n_read]])

        n_out = 0
        max_out = len(out_stream)
        abs_out = self.nitems_written(0)

        while n_out < max_out:
            if self._pending_symbols is not None:
                if self._sym_idx == 0:
                    self.add_item_tag(
                        0,
                        abs_out + n_out,
                        self._length_tag_key,
                        pmt.from_long(self._transpose_len),
                    )
                out_stream[n_out][:] = self._pending_symbols[self._sym_idx]
                self._sym_idx += 1
                n_out += 1
                if self._sym_idx >= self._transpose_len:
                    self._pending_symbols = None
                    self._sym_idx = 0
                continue

            if self._in_buf.size < self._burst_len_samples:
                break

            burst = self._in_buf[: self._burst_len_samples]
            self._in_buf = self._in_buf[self._burst_len_samples :]
            self._pending_symbols = self._demodulate_burst(burst)
            self._sym_idx = 0

        if n_read > 0:
            self.consume(0, n_read)
        return n_out
