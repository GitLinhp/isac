"""
Embedded Python Block: Torch batch FFT (GR fft_vxx replacement)

Vector in/out (vlen = fft_len). Set the GRC variable fft_len, then run
sync_ofdm_loopback_sionna_epy.py and grcc before starting the flowgraph.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from gnuradio import gr

_UHD_TEST_DIR = "/home/caict/Desktop/gunradio_test/uhd_test"
if _UHD_TEST_DIR not in sys.path:
    sys.path.insert(0, _UHD_TEST_DIR)


class blk(gr.sync_block):
    """Batch parallel FFT for HPD header/payload vector streams."""

    def __init__(self, fft_len=2048, device="cpu"):
        self._fft_len = int(fft_len)
        self._device = str(device)
        gr.sync_block.__init__(
            self,
            name="Torch Batch FFT",
            in_sig=[(np.complex64, self._fft_len)],
            out_sig=[(np.complex64, self._fft_len)],
        )
        torch.set_num_threads(1)

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = str(value)

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = min(len(inp), len(out))
        if n <= 0:
            return 0

        batch = np.stack(
            [np.asarray(inp[i], dtype=np.complex64) for i in range(n)], axis=0
        )
        x = torch.from_numpy(batch)
        dev = self._device
        if dev != "cpu":
            x = x.to(dev)
        with torch.inference_mode():
            freq = torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1)
        if dev != "cpu":
            freq = freq.cpu()
        freq_np = freq.numpy().astype(np.complex64, copy=False)
        for i in range(n):
            out[i][:] = freq_np[i]
        return n
