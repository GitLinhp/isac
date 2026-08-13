#!/usr/bin/env python3
"""统一 OFDM 接收处理：tag / 延迟补偿 + Torch CUDA 去 CP / FFT / 距离谱 + 显示/MUSIC。

无中间 GR 块；流图连接::

    in0  ← USRP Source（complex64，含 rx_time）
    tx_schedule / tx_freq_cpi ← TX 计划 epoch + 频域 CPI 消息（后者粘性复用）

内部直接刷新 Range Profile 窗口，并异步跑 1D MUSIC（无流输出）。
"""

from __future__ import annotations

import queue
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Optional

import numpy as np
import pmt
import torch
from gnuradio import gr
from gnuradio.fft import window

from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac_imp.burst_pack import (
    PORT_TX_FREQ_CPI,
    PORT_TX_SCHEDULE,
    TAG_RX_TIME,
    TPP_DONT,
    parse_tx_freq_cpi_msg,
    parse_uhd_time_pmt,
)
from isac_imp.ofdm_burst_tx_source import _normalize_torch_device
from isac_imp.range_profile_plot import _RangeProfileDisplay, publish_range_estimates
from isac_imp.range_profile_roi_slice import compute_range_roi

_N_DB = 10.0
_FORECAST_MAX = 16384
_DSP_Q_MAX = 2
_LOG_MUSIC = "[RangeMusic]"


def _remove_cp_fft_torch(
    td_burst: torch.Tensor,
    *,
    fft_len: int,
    cp_len: int,
    n_sym: int,
) -> torch.Tensor:
    """时域 CPI → fftshift 频域 ``(n_sym, fft_len)``，对齐 GR CP remover + fft_vcc(fwd, shift)."""
    sym_len = fft_len + cp_len
    td = td_burst.reshape(n_sym, sym_len)
    td_no_cp = td[:, cp_len:]
    return torch.fft.fftshift(torch.fft.fft(td_no_cp, dim=-1), dim=-1)


def _range_profile_torch(
    tx_freq: torch.Tensor,
    rx_freq: torch.Tensor,
    bh_window: torch.Tensor,
    fft_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 ``(db_profile, complex_acc)``，形状均为 ``(vlen_out,)``。"""
    vlen = int(bh_window.numel())
    n_sym = tx_freq.shape[0]
    h = tx_freq / rx_freq
    h_pad = torch.zeros((n_sym, vlen), dtype=torch.complex64, device=tx_freq.device)
    h_pad[:, :fft_len] = h
    h_win = h_pad * bh_window
    spec = torch.fft.fft(h_win, dim=-1)
    power = (spec.abs() ** 2).sum(dim=0)
    db = (_N_DB * torch.log10(torch.clamp(power, min=1e-30))).to(torch.float32)
    complex_acc = spec.sum(dim=0).to(torch.complex64)
    return db, complex_acc


def process_rx_cpi_torch(
    td_burst: np.ndarray,
    tx_freq: np.ndarray,
    *,
    fft_len: int,
    cp_len: int,
    n_sym: int,
    bh_window: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """主机 CPI → CUDA DSP → 主机 dB / 复数谱。"""
    device = _normalize_torch_device(device)
    with torch.inference_mode():
        td_t = torch.as_tensor(td_burst, device=device, dtype=torch.complex64)
        tx_t = torch.as_tensor(tx_freq, device=device, dtype=torch.complex64)
        bh_t = torch.as_tensor(bh_window, device=device, dtype=torch.float32)
        rx_f = _remove_cp_fft_torch(td_t, fft_len=fft_len, cp_len=cp_len, n_sym=n_sym)
        db_t, cx_t = _range_profile_torch(tx_t, rx_f, bh_t, fft_len)
        if str(td_t.device).startswith("cuda"):
            torch.cuda.synchronize()
        db = db_t.detach().cpu().numpy().astype(np.float32, copy=False)
        cx = cx_t.detach().cpu().numpy().astype(np.complex64, copy=False)
    return db, cx


def process_rx_cpi_numpy(
    td_burst: np.ndarray,
    tx_freq: np.ndarray,
    *,
    fft_len: int,
    cp_len: int,
    n_sym: int,
    bh_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy 参考路径（校验用）。"""
    sym_len = fft_len + cp_len
    td = td_burst.reshape(n_sym, sym_len)
    td_no_cp = td[:, cp_len:]
    rx_f = np.fft.fftshift(np.fft.fft(td_no_cp, axis=-1), axes=-1).astype(np.complex64)
    vlen = int(bh_window.size)
    power = np.zeros(vlen, dtype=np.float64)
    cacc = np.zeros(vlen, dtype=np.complex128)
    for k in range(n_sym):
        h = (tx_freq[k] / rx_f[k]).astype(np.complex64)
        h_pad = np.zeros(vlen, dtype=np.complex64)
        h_pad[:fft_len] = h
        spec = np.fft.fft(h_pad * bh_window).astype(np.complex64)
        power += np.abs(spec) ** 2
        cacc += spec
    db = (_N_DB * np.log10(np.maximum(power, 1e-30))).astype(np.float32)
    return db, cacc.astype(np.complex64)


class OfdmBurstRxBlock(gr.basic_block):
    """USRP IQ + TX 频域参考 → Torch CUDA 距离谱 + 内置 plot/MUSIC。"""

    def __init__(
        self,
        fft_len: int = 2048,
        cp_len: int = 512,
        num_symbols: int = 4,
        zeropadding_fac: int = 4,
        num_delay_samp: int = 0,
        device: str = "cuda",
        length_tag_key: str = "packet_len",
        log_interval_s: float = 1.0,
        range_roi: tuple[float, float] = (0.0, 30.0),
        range_bin_step: float = 0.305,
        music_enable: bool = True,
        num_sources: int = 1,
        subarray_size: int = 16,
        threshold: float = 0.1,
        **_ignored,
    ) -> None:
        del _ignored
        self._fft_len = int(fft_len)
        self._cp_len = int(cp_len)
        self._num_symbols = int(num_symbols)
        self._sym_len = self._fft_len + self._cp_len
        self._burst_len = self._num_symbols * self._sym_len
        self._vlen_out = self._fft_len * int(zeropadding_fac)
        self._num_delay = max(0, int(num_delay_samp))
        self._device = str(device)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._log_interval_s = float(log_interval_s)
        self._srcid = pmt.intern("ofdm_burst_rx")

        self._range_roi = (float(range_roi[0]), float(range_roi[1]))
        self._range_bin_step = float(range_bin_step)
        self._music_enable = bool(music_enable)
        self._num_sources = int(num_sources)
        self._subarray_size = int(subarray_size)
        self._threshold = float(threshold)
        self._plot_period_s = 0.10
        self._last_plot_t = 0.0
        self._start_bin = 0
        self._num_bins = 1
        self._x_start_m = 0.0
        self._recompute_roi()

        gr.basic_block.__init__(
            self,
            name="OFDM Burst RX",
            in_sig=[np.complex64],
            out_sig=None,
        )
        self.set_tag_propagation_policy(TPP_DONT)

        self._bh_window = np.asarray(
            window.blackmanharris(self._vlen_out), dtype=np.float32
        )

        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        self._display = _RangeProfileDisplay(
            title="Range Profile",
            xlabel="Range (m)",
            ylabel="Power (dB)",
            axis_x=(self._x_start_m, x_max),
        )
        self._music_estimator = RangeMusicEstimator()
        self._music_executor: Optional[ThreadPoolExecutor] = None
        self._music_busy = False
        self._music_queue: queue.Queue[list[float]] = queue.Queue()
        self._music_frame = 0

        self._dsp_executor: Optional[ThreadPoolExecutor] = None
        self._dsp_busy = False
        self._dsp_input_q: Deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._dsp_result_q: queue.Queue[tuple[np.ndarray, np.ndarray, float]] = (
            queue.Queue()
        )
        self._last_dsp_ms = 0.0
        self._dsp_drop = 0

        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_in(self._schedule_port)
        self.set_msg_handler(self._schedule_port, self._on_tx_schedule)
        self._freq_port = pmt.intern(PORT_TX_FREQ_CPI)
        self.message_port_register_in(self._freq_port)
        self.set_msg_handler(self._freq_port, self._on_tx_freq_cpi)

        self._iq_buf: np.ndarray | None = None
        self._iq_pos = 0
        self._collecting = False
        self._last_rx_tag_abs: int | None = None

        self._tx_cpi_queue: Deque[np.ndarray] = deque()
        self._tx_cpi_latest: np.ndarray | None = None

        self._cpi_count = 0
        self._cpi_window_t0 = 0.0
        self._cpi_window_n = 0
        self._rx_time_seen = 0

    def _recompute_roi(self) -> None:
        start_bin, num_bins, x_start_m = compute_range_roi(
            range_roi=self._range_roi,
            range_bin_step=self._range_bin_step,
            vlen_in=self._vlen_out,
        )
        self._start_bin = start_bin
        self._num_bins = num_bins
        self._x_start_m = x_start_m

    # ----- properties (flowgraph callbacks) -----

    @property
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        del value

    @property
    def num_delay_samp(self) -> int:
        return self._num_delay

    @num_delay_samp.setter
    def num_delay_samp(self, value: int) -> None:
        self._num_delay = max(0, int(value))

    @property
    def num_delay_samps(self) -> int:
        return self._num_delay

    @num_delay_samps.setter
    def num_delay_samps(self, value: int) -> None:
        self._num_delay = max(0, int(value))

    @property
    def range_roi(self) -> tuple[float, float]:
        return self._range_roi

    @range_roi.setter
    def range_roi(self, value: tuple[float, float]) -> None:
        self._range_roi = (float(value[0]), float(value[1]))
        self._recompute_roi()
        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        self._display.set_axis_x(self._x_start_m, x_max)

    @property
    def range_bin_step(self) -> float:
        return self._range_bin_step

    @range_bin_step.setter
    def range_bin_step(self, value: float) -> None:
        self._range_bin_step = float(value)
        self._recompute_roi()
        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        self._display.set_axis_x(self._x_start_m, x_max)

    @property
    def music_enable(self) -> bool:
        return self._music_enable

    @music_enable.setter
    def music_enable(self, value: bool) -> None:
        self._music_enable = bool(value)
        if not self._music_enable:
            publish_range_estimates([], method_name="MUSIC")

    def start(self) -> bool:
        self._iq_buf = None
        self._iq_pos = 0
        self._collecting = False
        self._last_rx_tag_abs = None
        self._tx_cpi_queue.clear()
        # Keep sticky freq ref across start() only if already set; clear on fresh run.
        self._tx_cpi_latest = None
        self._cpi_count = 0
        self._cpi_window_t0 = time.monotonic()
        self._cpi_window_n = 0
        self._rx_time_seen = 0
        self._last_plot_t = 0.0
        self._music_busy = False
        self._music_frame = 0
        self._dsp_busy = False
        self._last_dsp_ms = 0.0
        self._dsp_drop = 0
        self._dsp_input_q.clear()
        while not self._music_queue.empty():
            try:
                self._music_queue.get_nowait()
            except queue.Empty:
                break
        while not self._dsp_result_q.empty():
            try:
                self._dsp_result_q.get_nowait()
            except queue.Empty:
                break
        if self._music_executor is not None:
            self._music_executor.shutdown(wait=False, cancel_futures=True)
        self._music_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ofdm_burst_rx_music"
        )
        if self._dsp_executor is not None:
            self._dsp_executor.shutdown(wait=False, cancel_futures=True)
        self._dsp_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ofdm_burst_rx_dsp"
        )
        self._display.show()
        return True

    def stop(self) -> bool:
        if self._music_executor is not None:
            self._music_executor.shutdown(wait=False, cancel_futures=True)
            self._music_executor = None
        if self._dsp_executor is not None:
            self._dsp_executor.shutdown(wait=False, cancel_futures=True)
            self._dsp_executor = None
        self._music_busy = False
        self._dsp_busy = False
        publish_range_estimates([], method_name="MUSIC")
        self._display.close()
        return True

    def _on_tx_schedule(self, msg: pmt.pmt) -> None:
        if pmt.is_pair(msg):
            msg = pmt.cdr(msg)
        if pmt.is_tuple(msg):
            parse_uhd_time_pmt(msg)

    def _enqueue_tx_cpi(self, cpi: np.ndarray) -> None:
        arr = np.asarray(cpi, dtype=np.complex64)
        self._tx_cpi_latest = arr
        self._tx_cpi_queue.append(arr)
        while len(self._tx_cpi_queue) > 2:
            self._tx_cpi_queue.popleft()

    def _on_tx_freq_cpi(self, msg: pmt.pmt) -> None:
        try:
            cpi = parse_tx_freq_cpi_msg(
                msg, n_sym=self._num_symbols, fft_len=self._fft_len
            )
        except Exception:
            traceback.print_exc()
            return
        self._enqueue_tx_cpi(cpi)

    def _apply_delay(self, buf: np.ndarray) -> None:
        delay = min(self._num_delay, len(buf))
        if delay <= 0:
            return
        tail = len(buf) - delay
        if tail > 0:
            buf[:tail] = buf[delay:]
        buf[tail:] = 0

    def _maybe_log_status(self, *, cpi_just_done: bool = False) -> None:
        if cpi_just_done:
            self._cpi_count += 1
            self._cpi_window_n += 1
        if self._log_interval_s <= 0:
            return
        now = time.monotonic()
        dt = now - self._cpi_window_t0
        if dt < self._log_interval_s:
            return
        rate = self._cpi_window_n / dt if dt > 0 else 0.0
        dsp_q = len(self._dsp_input_q) + int(self._dsp_busy)
        print(
            f"[ofdm_burst_rx] CPI/s≈{rate:.1f} total={self._cpi_count} "
            f"tx_q={len(self._tx_cpi_queue)} "
            f"tx_ref={int(self._tx_cpi_latest is not None)} "
            f"iq_pos={self._iq_pos}/{self._burst_len} "
            f"collecting={int(self._collecting)} rx_time_seen={self._rx_time_seen} "
            f"dsp_ms={self._last_dsp_ms:.1f} dsp_q={dsp_q} dsp_drop={self._dsp_drop}",
            flush=True,
        )
        self._cpi_window_t0 = now
        self._cpi_window_n = 0

    def _drain_music(self) -> None:
        while True:
            try:
                ranges = self._music_queue.get_nowait()
            except queue.Empty:
                break
            publish_range_estimates(ranges, method_name="MUSIC")

    def _drain_dsp(self) -> None:
        got = False
        while True:
            try:
                db, cx, dsp_ms = self._dsp_result_q.get_nowait()
            except queue.Empty:
                break
            got = True
            self._last_dsp_ms = float(dsp_ms)
            self._emit_plot_and_music(db, cx)
            self._maybe_log_status(cpi_just_done=True)
        if got:
            self._kick_dsp()

    def _run_dsp(self, td: np.ndarray, tx: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            db, cx = process_rx_cpi_torch(
                td,
                tx,
                fft_len=self._fft_len,
                cp_len=self._cp_len,
                n_sym=self._num_symbols,
                bh_window=self._bh_window,
                device=self._device,
            )
            dsp_ms = (time.perf_counter() - t0) * 1000.0
            self._dsp_result_q.put((db, cx, dsp_ms))
        except Exception:
            traceback.print_exc()
            self._dsp_result_q.put(
                (
                    np.full(self._vlen_out, np.nan, dtype=np.float32),
                    np.zeros(self._vlen_out, dtype=np.complex64),
                    (time.perf_counter() - t0) * 1000.0,
                )
            )
        finally:
            self._dsp_busy = False

    def _kick_dsp(self) -> None:
        if self._dsp_busy or not self._dsp_input_q or self._dsp_executor is None:
            return
        td, tx = self._dsp_input_q.popleft()
        self._dsp_busy = True
        self._dsp_executor.submit(self._run_dsp, td, tx)

    def _run_music(self, profile: np.ndarray, frame_idx: int) -> None:
        try:
            peaks = self._music_estimator(
                profile,
                range_bin_step=self._range_bin_step,
                range_roi=self._range_roi,
                num_sources=self._num_sources,
                subarray_size=self._subarray_size,
                threshold=self._threshold,
            )
            ranges = peaks.peak_ranges_m.tolist()
            self._music_queue.put(ranges)
            if ranges:
                print(f"{_LOG_MUSIC} frame #{frame_idx} — 1D MUSIC 距离估计 (m):")
                for i, r in enumerate(ranges):
                    print(f"  峰 {i + 1}: {r:.3f} m")
            else:
                print(f"{_LOG_MUSIC} frame #{frame_idx} — 未检测到谱峰")
        except Exception:
            traceback.print_exc()
        finally:
            self._music_busy = False

    def _emit_plot_and_music(self, db: np.ndarray, cx: np.ndarray) -> None:
        self._drain_music()
        now = time.monotonic()
        if now - self._last_plot_t >= self._plot_period_s:
            self._last_plot_t = now
            s, n = self._start_bin, self._num_bins
            y = db[s : s + n]
            x = self._x_start_m + np.arange(n, dtype=np.float64) * self._range_bin_step
            self._display.post_profile(x, y)

        if (
            self._music_enable
            and not self._music_busy
            and self._music_executor is not None
        ):
            self._music_frame += 1
            self._music_busy = True
            self._music_executor.submit(
                self._run_music, np.asarray(cx, dtype=np.complex64).copy(), self._music_frame
            )

    def _try_process_pending(self) -> bool:
        if self._iq_buf is None or self._iq_pos < self._burst_len:
            return False
        if self._tx_cpi_queue:
            tx = self._tx_cpi_queue.popleft()
        elif self._tx_cpi_latest is not None:
            tx = self._tx_cpi_latest
        else:
            return False
        td = self._iq_buf
        self._apply_delay(td)
        self._iq_buf = None
        self._iq_pos = 0
        self._collecting = False
        if len(self._dsp_input_q) >= _DSP_Q_MAX and self._dsp_busy:
            self._dsp_drop += 1
            return True
        if len(self._dsp_input_q) >= _DSP_Q_MAX:
            self._dsp_input_q.popleft()
            self._dsp_drop += 1
        self._dsp_input_q.append((td, tx))
        self._kick_dsp()
        return True

    def _ingest_iq(self, in_iq: np.ndarray, read_base: int) -> int:
        """Bulk-consume IQ: find rx_time, then memcpy CPI (avoid per-sample Python)."""
        n = len(in_iq)
        if n <= 0:
            return 0
        consumed = 0
        while consumed < n:
            if self._iq_buf is not None and self._iq_pos >= self._burst_len:
                if not self._try_process_pending():
                    break

            if not self._collecting:
                tags = self.get_tags_in_range(0, read_base + consumed, read_base + n)
                start_off: int | None = None
                for tag in tags:
                    if not pmt.eq(tag.key, TAG_RX_TIME):
                        continue
                    off = int(tag.offset - read_base)
                    if off < consumed:
                        continue
                    if (
                        self._last_rx_tag_abs is not None
                        and tag.offset - self._last_rx_tag_abs < self._burst_len
                    ):
                        continue
                    start_off = off
                    self._last_rx_tag_abs = int(tag.offset)
                    break
                if start_off is None:
                    return n
                consumed = start_off
                self._iq_buf = np.zeros(self._burst_len, dtype=np.complex64)
                self._iq_pos = 0
                self._collecting = True
                self._rx_time_seen += 1
                if consumed >= n:
                    break

            assert self._iq_buf is not None
            need = self._burst_len - self._iq_pos
            take = min(need, n - consumed)
            if take <= 0:
                break
            self._iq_buf[self._iq_pos : self._iq_pos + take] = in_iq[
                consumed : consumed + take
            ]
            self._iq_pos += take
            consumed += take
            if self._iq_pos >= self._burst_len:
                self._try_process_pending()
        return consumed

    def forecast(self, noutput_items: int, ninputs) -> list:
        del noutput_items, ninputs
        partial_iq = (
            self._collecting
            and self._iq_buf is not None
            and 0 < self._iq_pos < self._burst_len
        )
        if partial_iq:
            need_iq = max(1, min(self._burst_len - self._iq_pos, _FORECAST_MAX))
        elif self._tx_cpi_latest is not None or self._tx_cpi_queue or self._collecting:
            need_iq = max(1, min(self._burst_len, _FORECAST_MAX))
        else:
            need_iq = 1
        return [need_iq]

    def general_work(self, input_items, output_items) -> int:
        del output_items
        in_iq = input_items[0]

        self._drain_dsp()

        iq_consumed = self._ingest_iq(in_iq, self.nitems_read(0))
        if iq_consumed > 0:
            self.consume(0, iq_consumed)

        if self._iq_buf is not None and self._iq_pos >= self._burst_len:
            self._try_process_pending()

        self._drain_dsp()
        self._drain_music()
        self._maybe_log_status()

        if iq_consumed > 0:
            return gr.WORK_CALLED_PRODUCE
        return 0


blk = OfdmBurstRxBlock
