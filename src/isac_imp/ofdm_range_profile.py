"""OFDM 距离谱 epy 块。

替代 GR 链 ``radar_ofdm_divide_vcvc`` + range FFT + mag² + integrate + nlog10，
在 ``usrp_ofdm_echotimer_dd`` 中流图连接为::

    in0 ← SionnaResourceGridTx（TX 频域参考）
    in1 ← fft_vxx_0_0（RX 经 CP remover + FFT）
    out0 → qtgui_vector_sink_f（dB 距离谱，vlen=fft_len*zeropadding_fac）
    out1 → RangeMusicBlock（CPI 复数距离谱，可选，供 1D MUSIC）

双输入按样点序号配对（非 tag），依赖上游 CPI 符号流已对齐。
全谱频域除法（无 discarded 掩码），out0 固定 10·log10(功率和)。
"""

from __future__ import annotations

import numpy as np
from gnuradio import gr
from gnuradio.fft import window

from isac_imp.burst_pack import TPP_DONT

_N_DB = 10.0


def _symbol_divide_pad_window(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """频域除法 → 零填充 → BH 窗，返回窗后序列（range FFT 输入）。"""
    h = (tx / rx).astype(np.complex64, copy=False)
    h_pad = np.zeros(vlen_out, dtype=np.complex64)
    h_pad[:fft_len] = h
    return (h_pad * bh_window).astype(np.complex64, copy=False)


def compute_symbol_range_spectrum(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """单符号复数距离谱：等价于 GR divide + range FFT（不做 |·|²）。"""
    h_win = _symbol_divide_pad_window(
        tx,
        rx,
        bh_window=bh_window,
        fft_len=fft_len,
        vlen_out=vlen_out,
    )
    return np.fft.fft(h_win).astype(np.complex64, copy=False)


def compute_symbol_range_power(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """单符号距离维功率谱：频域除法 → 零填充 → BH 窗 → FFT → |·|²。

    等价于 GR ``ofdm_divide_vcvc`` 单符号输出后再做 range FFT 与取模平方；
    零填充仅占用 ``h_pad[0:fft_len]``，高索引为距离分辨率扩展。
    """
    rd = compute_symbol_range_spectrum(
        tx,
        rx,
        bh_window=bh_window,
        fft_len=fft_len,
        vlen_out=vlen_out,
    )
    return (np.abs(rd) ** 2).astype(np.float32, copy=False)


def compute_cpi_range_profile_db(
    tx_batch: np.ndarray,
    rx_batch: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """CPI 非相干积累后转 dB，形状 ``(vlen_out,)`` float32。"""
    power_sum = np.zeros(vlen_out, dtype=np.float64)
    for k in range(tx_batch.shape[0]):
        power_sum += compute_symbol_range_power(
            tx_batch[k],
            rx_batch[k],
            bh_window=bh_window,
            fft_len=fft_len,
            vlen_out=vlen_out,
        )
    return (_N_DB * np.log10(power_sum)).astype(np.float32, copy=False)


class OfdmRangeProfileBlock(gr.basic_block):
    """双输入 TX/RX 频域符号 → CPI dB 距离谱 + 可选 CPI 复数距离谱（MUSIC）。"""

    def __init__(
        self,
        fft_len: int = 2048,
        zeropadding_fac: int = 2,
        transpose_len: int = 4,
    ) -> None:
        self._fft_len = int(fft_len)
        self._vlen_out = self._fft_len * int(zeropadding_fac)
        self._transpose_len = int(transpose_len)

        gr.basic_block.__init__(
            self,
            name="OFDM Range Profile",
            in_sig=[
                (np.complex64, self._fft_len),
                (np.complex64, self._fft_len),
            ],
            out_sig=[
                (np.float32, self._vlen_out),
                (np.complex64, self._vlen_out),
            ],
        )
        # 每 transpose_len 个输入符号产出 1 条距离谱。
        self.set_relative_rate(1, self._transpose_len)
        self.set_tag_propagation_policy(TPP_DONT)

        self._bh_window = np.asarray(
            window.blackmanharris(self._vlen_out), dtype=np.float32
        )
        self._sym_idx = 0
        self._power_acc = np.zeros(self._vlen_out, dtype=np.float64)
        self._complex_acc = np.zeros(self._vlen_out, dtype=np.complex128)

    def start(self) -> bool:
        self._sym_idx = 0
        self._power_acc.fill(0.0)
        self._complex_acc.fill(0.0)
        return True

    def forecast(self, noutput_items: int, ninputs) -> list:
        del ninputs
        # 两路输入需等量符号，各 noutput * transpose_len。
        need = noutput_items * self._transpose_len
        return [need, need]

    def general_work(self, input_items, output_items) -> int:
        in_tx = input_items[0]
        in_rx = input_items[1]
        out_db = output_items[0]
        out_cx = output_items[1]

        n_avail = min(len(in_tx), len(in_rx))
        if n_avail <= 0:
            self.consume(0, 0)
            self.consume(1, 0)
            return 0

        n_produced = 0
        n_consumed = 0

        while n_consumed < n_avail and n_produced < len(out_db):
            tx = in_tx[n_consumed]
            rx = in_rx[n_consumed]
            spec = compute_symbol_range_spectrum(
                tx,
                rx,
                bh_window=self._bh_window,
                fft_len=self._fft_len,
                vlen_out=self._vlen_out,
            )
            self._complex_acc += spec.astype(np.complex128, copy=False)
            self._power_acc += (np.abs(spec) ** 2).astype(np.float64, copy=False)
            self._sym_idx += 1
            n_consumed += 1

            if self._sym_idx >= self._transpose_len:
                out_db[n_produced][:] = (
                    _N_DB * np.log10(self._power_acc)
                ).astype(np.float32, copy=False)
                out_cx[n_produced][:] = self._complex_acc.astype(
                    np.complex64, copy=False
                )
                n_produced += 1
                self._sym_idx = 0
                self._power_acc.fill(0.0)
                self._complex_acc.fill(0.0)

        self.consume(0, n_consumed)
        self.consume(1, n_consumed)
        return n_produced
