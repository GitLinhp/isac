"""Sionna 发射链：ResourceGridMapper + OFDMModulator，替代 GR carrier_allocator / IFFT / CP。"""

from __future__ import annotations

import sys

import numpy as np
import pmt
import torch
from gnuradio import gr
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.ofdm import OFDMModulator, ResourceGrid, ResourceGridMapper

from isac.utils.misc import set_random_seed
from isac_imp.burst_pack import TPP_DONT

_LOG_PREFIX = "[SionnaOfdmTx]"


def _resolve_guard_carriers(fft_len: int, transpose_len: int) -> tuple[int, int]:
    """Return ``num_guard_carriers`` so ``rg.num_data_symbols == transpose_len * (fft_len - 2)``."""
    target = int(transpose_len) * (int(fft_len) - 2)
    for guards in ((1, 0), (0, 1)):
        rg = ResourceGrid(
            num_ofdm_symbols=int(transpose_len),
            fft_size=int(fft_len),
            subcarrier_spacing=60e3,
            num_guard_carriers=guards,
            dc_null=True,
            pilot_pattern=None,
            device="cpu",
        )
        if rg.num_data_symbols == target:
            return guards
    raise ValueError(
        f"无法对齐 GR 数据符号数: target={target}, fft_len={fft_len}, "
        f"transpose_len={transpose_len}"
    )


def _build_waveforms(
    *,
    fft_len: int,
    transpose_len: int,
    subcarrier_spacing: float,
    cp_len: int,
    num_bits_per_symbol: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    set_random_seed(seed)
    guards = _resolve_guard_carriers(fft_len, transpose_len)
    rg = ResourceGrid(
        num_ofdm_symbols=int(transpose_len),
        fft_size=int(fft_len),
        subcarrier_spacing=float(subcarrier_spacing),
        cyclic_prefix_length=int(cp_len),
        num_guard_carriers=guards,
        dc_null=True,
        pilot_pattern=None,
        device=device,
    )
    expected_data = int(transpose_len) * (int(fft_len) - 2)
    if rg.num_data_symbols != expected_data:
        raise ValueError(
            f"ResourceGrid num_data_symbols={rg.num_data_symbols} != {expected_data}"
        )

    binary_source = BinarySource(device=device)
    mapper = Mapper("qam", int(num_bits_per_symbol), device=device)
    rg_mapper = ResourceGridMapper(rg, device=device)
    modulator = OFDMModulator(cyclic_prefix_length=int(cp_len), device=device)

    with torch.inference_mode():
        bits = binary_source([1, 1, 1, rg.num_data_symbols * int(num_bits_per_symbol)])
        x = mapper(bits)
        x_rg = rg_mapper(x)
        x_time = modulator(x_rg)
        x_rg_np = x_rg.squeeze().cpu().numpy().astype(np.complex64, copy=False)
        x_time_np = x_time.reshape(-1).cpu().numpy().astype(np.complex64, copy=False)

    if x_rg_np.ndim == 1:
        x_rg_np = x_rg_np.reshape(1, -1)
    if x_rg_np.shape != (int(transpose_len), int(fft_len)):
        x_rg_np = x_rg_np.reshape(int(transpose_len), int(fft_len))

    burst_len = int(transpose_len) * (int(fft_len) + int(cp_len))
    if x_time_np.size != burst_len:
        raise ValueError(f"x_time 样点数 {x_time_np.size} != 期望 {burst_len}")

    # 自然序资源网格（与 OFDMDemodulator 输出一致，供 divide port0）
    freq = x_rg_np.astype(np.complex64, copy=False)
    return x_time_np, freq, guards, int(rg.num_data_symbols)


class SionnaOfdmTxBlock(gr.basic_block):
    """BinarySource + Mapper + ResourceGridMapper + OFDMModulator 双输出发射源。"""

    def __init__(
        self,
        fft_len: int = 2048,
        transpose_len: int = 4,
        subcarrier_spacing: float = 60e3,
        cp_len: int = 512,
        length_tag_key: str = "packet_len",
        num_bits_per_symbol: int = 2,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self._fft_len = int(fft_len)
        self._transpose_len = int(transpose_len)
        self._burst_len_samples = self._transpose_len * (self._fft_len + int(cp_len))
        gr.basic_block.__init__(
            self,
            name="Sionna OFDM TX",
            in_sig=None,
            out_sig=[np.complex64, (np.complex64, self._fft_len)],
        )
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._cp_len = int(cp_len)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._device = str(device)
        self._seed = int(seed)
        self._time_buf: np.ndarray | None = None
        self._freq_buf: np.ndarray | None = None
        self._time_idx = 0
        self._sym_idx = 0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(max(self._burst_len_samples * 2, self._transpose_len * 2))

    def _log(self, msg: str) -> None:
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    def start(self) -> bool:
        torch.set_num_threads(1)
        time_buf, freq, guards, n_data = _build_waveforms(
            fft_len=self._fft_len,
            transpose_len=self._transpose_len,
            subcarrier_spacing=self._subcarrier_spacing,
            cp_len=self._cp_len,
            num_bits_per_symbol=self._num_bits_per_symbol,
            device=self._device,
            seed=self._seed,
        )
        self._time_buf = time_buf
        self._freq_buf = freq
        self._time_idx = 0
        self._sym_idx = 0
        self._log(
            f"loaded burst_len={self._burst_len_samples} freq={freq.shape} "
            f"num_data_symbols={n_data} guards={guards} seed={self._seed}"
        )
        return True

    def forecast(self, noutput_items: int, ninputs: int) -> list:
        del noutput_items, ninputs
        return []

    def general_work(self, input_items, output_items) -> int:
        del input_items
        if self._time_buf is None or self._freq_buf is None:
            return 0

        out_time = output_items[0]
        out_freq = output_items[1]
        max_time = len(out_time)
        max_freq = len(out_freq)

        n_time = 0
        n_freq = 0
        abs_time_base = self.nitems_written(0)
        abs_freq_base = self.nitems_written(1)

        while n_time < max_time:
            if self._time_idx == 0:
                self.add_item_tag(
                    0,
                    abs_time_base + n_time,
                    self._length_tag_key,
                    pmt.from_long(self._burst_len_samples),
                )
            out_time[n_time] = self._time_buf[self._time_idx]
            n_time += 1
            self._time_idx += 1
            if self._time_idx >= self._burst_len_samples:
                self._time_idx = 0

        while n_freq < max_freq:
            if self._sym_idx == 0:
                self.add_item_tag(
                    1,
                    abs_freq_base + n_freq,
                    self._length_tag_key,
                    pmt.from_long(self._transpose_len),
                )
            out_freq[n_freq][:] = self._freq_buf[self._sym_idx]
            n_freq += 1
            self._sym_idx += 1
            if self._sym_idx >= self._transpose_len:
                self._sym_idx = 0

        if n_freq > 0:
            self.produce(1, n_freq)
        if n_time > 0:
            return n_time
        if n_freq > 0:
            return gr.WORK_CALLED_PRODUCE
        return 0


# 兼容旧 GRC / import
SionnaResourceGridTxBlock = SionnaOfdmTxBlock
