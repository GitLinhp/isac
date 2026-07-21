"""OFDM 距离谱 epy 块。

替代 GR 链 ``radar_ofdm_divide_vcvc`` + range FFT + mag² + integrate + nlog10，
在 ``usrp_ofdm_echotimer_dd`` 中流图连接为::

    in0 ← SionnaResourceGridTx（TX 频域参考）
    in1 ← fft_vxx_0_0（RX 经 CP remover + FFT）
    out0 → qtgui_vector_sink_f（dB 距离谱，vlen=fft_len*zeropadding_fac）

双输入按样点序号配对（非 tag），依赖上游 CPI 符号流已对齐。
"""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np
import pmt
from gnuradio import gr
from gnuradio.fft import window

from isac_imp.burst_pack import TPP_DONT

_LOG_PREFIX = "[OfdmRangeProfile]"


def _build_discarded_mask(
    discarded_carriers: Sequence[int],
    fft_len: int,
    vlen_in: int,
) -> np.ndarray | None:
    """将相对 DC 的子载波索引映射为 fftshift 后数组上的布尔掩码。"""
    if not discarded_carriers:
        return None
    idx = np.asarray(discarded_carriers, dtype=np.int64) + (vlen_in // 2)
    mask = np.zeros(vlen_in, dtype=bool)
    mask[idx] = True
    return mask


def compute_symbol_range_power(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
    discarded_mask: np.ndarray | None = None,
    apply_discarded: bool = True,
) -> np.ndarray:
    """单符号距离维功率谱：频域除法 → 零填充 → BH 窗 → FFT → |·|²。

    等价于 GR ``ofdm_divide_vcvc`` 单符号输出后再做 range FFT 与取模平方；
    零填充仅占用 ``h_pad[0:fft_len]``，高索引为距离分辨率扩展。
    """
    if apply_discarded and discarded_mask is not None:
        h = np.zeros(fft_len, dtype=np.complex64)
        active = ~discarded_mask
        h[active] = (tx[active] / rx[active]).astype(np.complex64, copy=False)
    else:
        h = (tx / rx).astype(np.complex64, copy=False)

    h_pad = np.zeros(vlen_out, dtype=np.complex64)
    h_pad[:fft_len] = h
    h_win = h_pad * bh_window
    rd = np.fft.fft(h_win)
    return (np.abs(rd) ** 2).astype(np.float32, copy=False)


def compute_cpi_range_profile_db(
    tx_batch: np.ndarray,
    rx_batch: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
    discarded_mask: np.ndarray | None = None,
    num_sync_words: int = 0,
    n_db: float = 10.0,
) -> np.ndarray:
    """CPI 非相干积累后转 dB，形状 ``(vlen_out,)`` float32。

    前 ``num_sync_words`` 个符号不做 discarded 掩码，与 ``radar_ofdm_divide``
    的 ``num_sync_words`` 行为一致。
    """
    power_sum = np.zeros(vlen_out, dtype=np.float64)
    n_sym = tx_batch.shape[0]
    for k in range(n_sym):
        p = compute_symbol_range_power(
            tx_batch[k],
            rx_batch[k],
            bh_window=bh_window,
            fft_len=fft_len,
            vlen_out=vlen_out,
            discarded_mask=discarded_mask,
            apply_discarded=k >= num_sync_words,
        )
        power_sum += p
    return (n_db * np.log10(power_sum)).astype(np.float32, copy=False)


class OfdmRangeProfileBlock(gr.basic_block):
    """双输入 TX/RX 频域符号 → 单输出 CPI 距离谱（dB）。"""

    def __init__(
        self,
        fft_len: int = 2048,
        zeropadding_fac: int = 2,
        transpose_len: int = 4,
        discarded_carriers: Sequence[int] = (),
        num_sync_words: int = 0,
        length_tag_key: str = "packet_len",
        n_db: float = 10.0,
    ) -> None:
        del length_tag_key  # 保留 GRC 参数；配对不依赖 tag
        self._fft_len = int(fft_len)
        self._vlen_out = self._fft_len * int(zeropadding_fac)
        self._transpose_len = int(transpose_len)
        self._num_sync_words = int(num_sync_words)
        self._n_db = float(n_db)
        self._discarded_mask = _build_discarded_mask(
            discarded_carriers, self._fft_len, self._fft_len
        )

        gr.basic_block.__init__(
            self,
            name="OFDM Range Profile",
            in_sig=[
                (np.complex64, self._fft_len),
                (np.complex64, self._fft_len),
            ],
            out_sig=[(np.float32, self._vlen_out)],
        )
        # 每 transpose_len 个输入符号产出 1 条距离谱。
        self.set_relative_rate(1, self._transpose_len)
        self.set_tag_propagation_policy(TPP_DONT)

        self._bh_window = np.asarray(
            window.blackmanharris(self._vlen_out), dtype=np.float32
        )
        self._sym_idx = 0
        self._cpi_symbol_idx = 0
        self._power_acc = np.zeros(self._vlen_out, dtype=np.float64)

    def start(self) -> bool:
        self._sym_idx = 0
        self._cpi_symbol_idx = 0
        self._power_acc.fill(0.0)
        return True

    def forecast(self, noutput_items: int, ninputs) -> list:
        del ninputs
        # 两路输入需等量符号，各 noutput * transpose_len。
        need = noutput_items * self._transpose_len
        return [need, need]

    def general_work(self, input_items, output_items) -> int:
        in_tx = input_items[0]
        in_rx = input_items[1]
        out = output_items[0]

        n_avail = min(len(in_tx), len(in_rx))
        if n_avail <= 0:
            self.consume(0, 0)
            self.consume(1, 0)
            return 0

        n_produced = 0
        n_consumed = 0

        while n_consumed < n_avail and n_produced < len(out):
            tx = in_tx[n_consumed]
            rx = in_rx[n_consumed]
            apply_discarded = self._cpi_symbol_idx >= self._num_sync_words
            self._power_acc += compute_symbol_range_power(
                tx,
                rx,
                bh_window=self._bh_window,
                fft_len=self._fft_len,
                vlen_out=self._vlen_out,
                discarded_mask=self._discarded_mask,
                apply_discarded=apply_discarded,
            )
            self._sym_idx += 1
            self._cpi_symbol_idx += 1
            n_consumed += 1

            if self._sym_idx >= self._transpose_len:
                # CPI 结束：非相干积累转 dB，重置状态机。
                out[n_produced][:] = (
                    self._n_db * np.log10(self._power_acc)
                ).astype(np.float32, copy=False)
                n_produced += 1
                self._sym_idx = 0
                self._cpi_symbol_idx = 0
                self._power_acc.fill(0.0)

        self.consume(0, n_consumed)
        self.consume(1, n_consumed)
        return n_produced
