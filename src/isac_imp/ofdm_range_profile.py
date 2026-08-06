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
import pmt
from gnuradio import gr
from gnuradio.fft import window

from isac_imp.burst_pack import PORT_TX_SCHEDULE, TPP_DONT, parse_uhd_time_pmt

_N_DB = 10.0


def compute_symbol_divide_padded(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """单符号 OFDM Divide H(f)：tx/rx 后零填充至 ``vlen_out``（等价 GR divide 输出）。"""
    h = (tx / rx).astype(np.complex64, copy=False)
    h_pad = np.zeros(vlen_out, dtype=np.complex64)
    h_pad[:fft_len] = h
    return h_pad


def _symbol_divide_pad_window(
    tx: np.ndarray,
    rx: np.ndarray,
    *,
    bh_window: np.ndarray,
    fft_len: int,
    vlen_out: int,
) -> np.ndarray:
    """频域除法 → 零填充 → BH 窗，返回窗后序列（range FFT 输入）。"""
    h_pad = compute_symbol_divide_padded(
        tx, rx, fft_len=fft_len, vlen_out=vlen_out
    )
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
        self._tx_skip_until_tag = False
        self._rx_skip_until_tag = False
        self._align_pending = False
        self._align_search = max(int(transpose_len) * 8, 32)

        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_in(self._schedule_port)
        self.set_msg_handler(self._schedule_port, self._on_tx_schedule)

    def start(self) -> bool:
        self._sym_idx = 0
        self._power_acc.fill(0.0)
        self._complex_acc.fill(0.0)
        self._request_stream_resync()
        return True

    def _request_stream_resync(self) -> None:
        self._tx_skip_until_tag = True
        self._rx_skip_until_tag = True
        self._align_pending = True
        self._sym_idx = 0
        self._power_acc.fill(0.0)
        self._complex_acc.fill(0.0)

    def _on_tx_schedule(self, msg: pmt.pmt) -> None:
        if pmt.is_pair(msg):
            msg = pmt.cdr(msg)
        if pmt.is_tuple(msg):
            parse_uhd_time_pmt(msg)
        self._request_stream_resync()

    def _consume_to_cpi_boundary(self, port: int, n_avail: int) -> int:
        """将输入流对齐到最近的 ``packet_len`` tag（CPI 首符号）。"""
        skip_flag = self._tx_skip_until_tag if port == 0 else self._rx_skip_until_tag
        if not skip_flag or n_avail <= 0:
            return 0
        base = self.nitems_read(port)
        tag_key = pmt.intern("packet_len")
        best: int | None = None
        for tag in self.get_tags_in_range(port, base, base + n_avail):
            if not pmt.eq(tag.key, tag_key):
                continue
            off = int(tag.offset - base)
            if 0 <= off < n_avail and (best is None or off < best):
                best = off
        if best is None:
            return 0
        if port == 0:
            self._tx_skip_until_tag = False
        else:
            self._rx_skip_until_tag = False
        return best

    def _find_rx_align_offset(self, in_tx: np.ndarray, in_rx: np.ndarray) -> tuple[int, float]:
        """搜索使 TX/RX 频域符号相关性最大的 RX 偏移。"""
        tlen = self._transpose_len
        max_off = min(self._align_search, len(in_rx) - tlen + 1)
        if max_off <= 0 or len(in_tx) < tlen:
            return 0, 0.0
        best_off = 0
        best_score = -1.0
        tx_batch = in_tx[:tlen]
        for off in range(max_off):
            score = 0.0
            valid = True
            for k in range(tlen):
                rx_sym = in_rx[off + k]
                rx_norm = float(np.linalg.norm(rx_sym))
                if rx_norm < 1e-9:
                    valid = False
                    break
                score += float(np.abs(np.vdot(tx_batch[k], rx_sym)))
            if valid and score > best_score:
                best_score = score
                best_off = off
        return best_off, best_score

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

        tx_off = 0
        rx_tag_off = 0
        align_off = 0
        align_score = 0.0
        if self._tx_skip_until_tag and len(in_tx) > 0:
            tx_off = self._consume_to_cpi_boundary(0, len(in_tx))
            if tx_off > 0:
                self.consume(0, tx_off)
        if self._rx_skip_until_tag and len(in_rx) > 0:
            rx_tag_off = self._consume_to_cpi_boundary(1, len(in_rx))
            if rx_tag_off > 0:
                self.consume(1, rx_tag_off)

        in_tx = in_tx[tx_off:]
        in_rx = in_rx[rx_tag_off:]

        if (
            self._align_pending
            and len(in_tx) >= self._transpose_len
            and len(in_rx) >= self._transpose_len
        ):
            align_off, align_score = self._find_rx_align_offset(in_tx, in_rx)
            if align_off > 0:
                self.consume(1, align_off)
                in_rx = in_rx[align_off:]
            self._align_pending = False
            if not hasattr(self, "_dbg_alignments"):
                self._dbg_alignments = 0
            self._dbg_alignments += 1
            if self._dbg_alignments <= 5:
                from isac_imp.agent_debug_log import agent_log

                agent_log(
                    "ofdm_range_profile.py:general_work",
                    "rx align search",
                    {
                        "align_off": align_off,
                        "align_score": align_score,
                        "tx_off": tx_off,
                        "rx_tag_off": rx_tag_off,
                    },
                    hypothesis_id="H4",
                    run_id="post-fix-v3",
                )

        n_avail = min(len(in_tx), len(in_rx))
        if n_avail <= 0:
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
        if n_produced > 0:
            if not hasattr(self, "_dbg_profiles"):
                self._dbg_profiles = 0
            self._dbg_profiles += n_produced
            if self._dbg_profiles <= 5 or self._dbg_profiles % 20 == 0:
                from isac_imp.agent_debug_log import agent_log

                vec0 = out_db[0]
                peak_db = float(np.nanmax(vec0))
                peak_bin = int(np.nanargmax(vec0))
                step_hint = 3e8 / (2.0 * self._fft_len * 60e3 * (self._vlen_out // self._fft_len))
                agent_log(
                    "ofdm_range_profile.py:general_work",
                    "range profile produced",
                    {
                        "count": self._dbg_profiles,
                        "peak_db": peak_db,
                        "peak_bin": peak_bin,
                        "peak_range_m": peak_bin * step_hint,
                        "tx_off": tx_off,
                        "rx_tag_off": rx_tag_off,
                        "align_off": align_off,
                    },
                    hypothesis_id="H4",
                    run_id="post-fix-v3",
                )
        return n_produced


class OfdmDivideCpiBlock(gr.basic_block):
    """双输入 TX/RX 频域符号 → 每 CPI 输出 flatten 的 Divide H(f) 向量。"""

    def __init__(
        self,
        fft_len: int = 2048,
        zeropadding_fac: int = 4,
        transpose_len: int = 4,
    ) -> None:
        self._fft_len = int(fft_len)
        self._vlen_out = self._fft_len * int(zeropadding_fac)
        self._transpose_len = int(transpose_len)
        self._vlen_cpi = self._vlen_out * self._transpose_len

        gr.basic_block.__init__(
            self,
            name="OFDM Divide CPI",
            in_sig=[
                (np.complex64, self._fft_len),
                (np.complex64, self._fft_len),
            ],
            out_sig=[(np.complex64, self._vlen_cpi)],
        )
        self.set_relative_rate(1, self._transpose_len)
        self.set_tag_propagation_policy(TPP_DONT)

        self._sym_idx = 0
        self._divide_buf = np.zeros(
            (self._transpose_len, self._vlen_out), dtype=np.complex64
        )

    def start(self) -> bool:
        self._sym_idx = 0
        self._divide_buf.fill(0)
        return True

    def forecast(self, noutput_items: int, ninputs) -> list:
        del ninputs
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
            self._divide_buf[self._sym_idx][:] = compute_symbol_divide_padded(
                tx,
                rx,
                fft_len=self._fft_len,
                vlen_out=self._vlen_out,
            )
            self._sym_idx += 1
            n_consumed += 1

            if self._sym_idx >= self._transpose_len:
                out[n_produced][:] = self._divide_buf.ravel()
                n_produced += 1
                self._sym_idx = 0

        self.consume(0, n_consumed)
        self.consume(1, n_consumed)
        return n_produced
