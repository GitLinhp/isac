"""Sionna ResourceGrid 发射频域 epy 块。

在 ``usrp_ofdm_echotimer_dd`` 流图中替代 GR ``digital_ofdm_carrier_allocator``，
输出 fftshift 后的频域 OFDM 符号向量流（vlen=fft_len）::

    SionnaResourceGridTx out0
      ├→ GR IFFT + CP → echotimer（时域发射）
      └→ OfdmRangeProfileBlock in0（TX 频域参考）

``start()`` 按 ``seed`` 一次性生成固定 CPI 频域栅格；``work()`` 周期性重放，
并在每个 CPI 首符号打 ``packet_len`` tag（值为 ``transpose_len`` 符号数）。
"""

from __future__ import annotations

import sys

import numpy as np
import pmt
import torch
from gnuradio import gr
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper

from isac.utils.misc import set_random_seed
from isac_imp.burst_pack import TPP_DONT

_LOG_PREFIX = "[SionnaResourceGridTx]"


def _resolve_guard_carriers(fft_len: int, transpose_len: int) -> tuple[int, int]:
    """选取 guard 配置，使 ``rg.num_data_symbols == transpose_len * (fft_len - 2)``。

    与流图变量 ``n_carriers = fft_len - 2``、QPSK（2 bit/符号）下的
    ``packet_len = transpose_len * n_carriers // 4`` 保持一致。
    """
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


def _build_freq_grid(
    *,
    fft_len: int,
    transpose_len: int,
    subcarrier_spacing: float,
    cp_len: int,
    num_bits_per_symbol: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, tuple[int, int], int]:
    """BinarySource → QAM → ResourceGridMapper，返回 fftshift 频域 CPI。

    Returns:
        freq: 形状 ``(transpose_len, fft_len)``，与 GR carrier_allocator
            ``output_is_shifted=True`` 一致。
        guards: 选用的 ``num_guard_carriers`` 元组。
        num_data_symbols: ResourceGrid 数据 RE 总数。
    """
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

    with torch.inference_mode():
        bits = binary_source([1, 1, 1, rg.num_data_symbols * int(num_bits_per_symbol)])
        x = mapper(bits)
        x_rg = rg_mapper(x)
        x_rg_np = x_rg.squeeze().cpu().numpy().astype(np.complex64, copy=False)

    if x_rg_np.ndim == 1:
        x_rg_np = x_rg_np.reshape(1, -1)
    if x_rg_np.shape != (int(transpose_len), int(fft_len)):
        x_rg_np = x_rg_np.reshape(int(transpose_len), int(fft_len))

    freq = np.fft.fftshift(x_rg_np, axes=-1).astype(np.complex64, copy=False)
    return freq, guards, int(rg.num_data_symbols)


class SionnaResourceGridTxBlock(gr.sync_block):
    """无输入；输出 fftshift 频域 OFDM 符号向量流（vlen=fft_len）。"""

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
        gr.sync_block.__init__(
            self,
            name="Sionna ResourceGrid TX",
            in_sig=None,
            out_sig=[(np.complex64, self._fft_len)],
        )
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._cp_len = int(cp_len)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._device = str(device)
        self._seed = int(seed)
        self._freq_buf: np.ndarray | None = None
        self._sym_idx = 0

        # 本块在 CPI 首符号自行打 packet_len tag，不传播上游 tag。
        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._transpose_len * 4)

    def start(self) -> bool:
        # 限制 Torch 线程，避免与 GR scheduler 争抢 CPU。
        torch.set_num_threads(1)
        freq, guards, n_data = _build_freq_grid(
            fft_len=self._fft_len,
            transpose_len=self._transpose_len,
            subcarrier_spacing=self._subcarrier_spacing,
            cp_len=self._cp_len,
            num_bits_per_symbol=self._num_bits_per_symbol,
            device=self._device,
            seed=self._seed,
        )
        self._freq_buf = freq
        self._sym_idx = 0
        return True

    def work(self, input_items, output_items) -> int:
        del input_items
        if self._freq_buf is None:
            return 0

        out = output_items[0]
        n_out = len(out)
        abs_base = self.nitems_written(0)

        for i in range(n_out):
            if self._sym_idx == 0:
                # tag 值 = CPI 符号数（非 packet_len 字节数），供 CP prefixer 对齐。
                self.add_item_tag(
                    0,
                    abs_base + i,
                    self._length_tag_key,
                    pmt.from_long(self._transpose_len),
                )
            out[i][:] = self._freq_buf[self._sym_idx]
            self._sym_idx += 1
            if self._sym_idx >= self._transpose_len:
                self._sym_idx = 0

        return n_out
