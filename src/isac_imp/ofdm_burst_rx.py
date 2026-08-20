#!/usr/bin/env python3
"""统一 OFDM 接收处理：tag / 延迟补偿 + Torch CUDA OFDMDemodulator / 距离谱 + 显示/MUSIC。

无中间 GR 块；流图连接::

    in0..inN-1  ← USRP Source 各通道（complex64，含 rx_time）
    tx_freq_cpi / tx_schedule ← TX 频域 CPI 消息（粘性复用）+ 计划 epoch

``num_channels>1`` 时分路拼 CPI，凑齐后一次批处理 CUDA 距离谱；每通道独立 Range Profile 窗。
"""

from __future__ import annotations

import queue
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Deque, Optional

import numpy as np
import pmt
import torch
from gnuradio import gr
from gnuradio.fft import window

from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac.sensing.detection.ula_aoa_estimator import UlaAoaEstimator
from isac_imp.burst_pack import (
    PORT_TX_FREQ_CPI,
    PORT_TX_SCHEDULE,
    TAG_RX_TIME,
    TPP_DONT,
    parse_tx_freq_cpi_msg,
    parse_uhd_time_pmt,
)
from isac_imp.ofdm_burst_tx_source import _normalize_torch_device
from isac_imp.range_profile_plot import (
    _RangeProfileDisplay,
    publish_aoa_deg,
    publish_range_estimates,
)
from isac_imp.range_profile_record_limiter import notify_record_limit_reached
from isac_imp.range_profile_roi_slice import compute_range_roi
from isac_imp.record_paths import repo_data_dir
from isac_imp.rx_phase_calibration import (
    DEFAULT_CAL_FRAMES,
    RxPhaseCalibrator,
    apply_phase_weights,
    load_phase_cal,
    relative_phases_deg,
    residual_phase_deg,
    save_phase_cal,
)
from sionna.phy.ofdm import OFDMDemodulator

_N_DB = 10.0
_FORECAST_MAX = 16384
_DSP_Q_MAX = 2
_LOG_MUSIC = "[RangeMusic]"
_LOG_AOA = "[UlaAoa]"
_LOG_PHASE_CAL = "[PhaseCal]"

_phase_cal_done_handlers: list = []
_phase_display_handlers: list = []


def bind_phase_cal_done_handler(handler) -> None:
    """流图可绑定：校准结束后取消 Phase Cal 勾选。"""
    if handler not in _phase_cal_done_handlers:
        _phase_cal_done_handlers.append(handler)


def notify_phase_cal_done() -> None:
    for handler in list(_phase_cal_done_handlers):
        try:
            handler()
        except Exception:
            traceback.print_exc()


def bind_phase_display_handler(handler) -> None:
    """流图可绑定：刷新四通道相对相位标签。``handler(phases_deg: list[float])``。"""
    if handler not in _phase_display_handlers:
        _phase_display_handlers.append(handler)


def notify_channel_phases(phases_deg: list[float]) -> None:
    for handler in list(_phase_display_handlers):
        try:
            handler(list(phases_deg))
        except Exception:
            traceback.print_exc()


def default_phase_cal_path() -> str:
    return str(
        Path(repo_data_dir("data", "usrp_ofdm_single_bs_range"))
        / "rx_phase_cal.npz"
    )


def _log_tag_from_plot_title(plot_title: str) -> str:
    title = str(plot_title).strip() or "Range Profile"
    for part in reversed(title.split()):
        low = part.lower()
        if low.startswith("ch") and low[2:].isdigit():
            return low
    return title


def _range_profile_torch(
    tx_freq: torch.Tensor,
    rx_freq: torch.Tensor,
    bh_window: torch.Tensor,
    fft_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单通道：``tx/rx`` 为 ``(n_sym, fft)`` → ``(db, cx)`` 均为 ``(vlen,)``。"""
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


def _range_profile_batch_torch(
    tx_freq: torch.Tensor,
    rx_freq: torch.Tensor,
    bh_window: torch.Tensor,
    fft_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """批通道：``rx`` 为 ``(n_ch, n_sym, fft)``，``tx`` 为 ``(n_sym, fft)``。"""
    vlen = int(bh_window.numel())
    n_ch, n_sym, _ = rx_freq.shape
    h = tx_freq.unsqueeze(0) / rx_freq
    h_pad = torch.zeros(
        (n_ch, n_sym, vlen), dtype=torch.complex64, device=rx_freq.device
    )
    h_pad[:, :, :fft_len] = h
    h_win = h_pad * bh_window.view(1, 1, -1)
    spec = torch.fft.fft(h_win, dim=-1)
    power = (spec.abs() ** 2).sum(dim=1)
    db = (_N_DB * torch.log10(torch.clamp(power, min=1e-30))).to(torch.float32)
    complex_acc = spec.sum(dim=1).to(torch.complex64)
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
    demod: Optional[OFDMDemodulator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """主机 CPI → CUDA DSP → 主机 dB / 复数谱。

    ``td_burst`` 可为 ``(burst,)`` 或 ``(n_ch, burst)``。
    RX 频域经 ``OFDMDemodulator``（ortho，``l_min=0``）再 ×√N。
    """
    device = _normalize_torch_device(device)
    td_np = np.asarray(td_burst, dtype=np.complex64)
    batched = td_np.ndim == 2
    with torch.inference_mode():
        td_t = torch.as_tensor(td_np, device=device, dtype=torch.complex64)
        tx_t = torch.as_tensor(tx_freq, device=device, dtype=torch.complex64)
        bh_t = torch.as_tensor(bh_window, device=device, dtype=torch.float32)
        if demod is None:
            demod = OFDMDemodulator(
                fft_size=fft_len,
                l_min=0,
                cyclic_prefix_length=cp_len,
                device=device,
            )
        rx_f = demod(td_t) * (float(fft_len) ** 0.5)
        if int(rx_f.shape[-2]) != int(n_sym):
            raise RuntimeError(
                f"demod n_sym {rx_f.shape[-2]} != expected {n_sym}"
            )
        if batched:
            db_t, cx_t = _range_profile_batch_torch(tx_t, rx_f, bh_t, fft_len)
        else:
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
    """NumPy 参考路径（校验用，单通道）。"""
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
    """USRP IQ + TX 频域参考 → Torch CUDA 距离谱 + 内置 plot/MUSIC。

    ``num_channels=1``：单流口（兼容旧流图）。
    ``num_channels>1``：多流口分路采集，凑齐后一次批 CUDA。
    """

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
        plot_title: str = "Range Profile",
        num_channels: int = 1,
        plot_title_prefix: str = "Range Profile",
        record_enable: bool = False,
        record_file_path: str = "",
        record_max_frames: int = 55,
        array_spacing_m: float = 0.025,
        carrier_freq_hz: float = 6.0e9,
        aoa_enable: bool = False,
        phase_cal_capture: bool = False,
        phase_cal_frames: int = DEFAULT_CAL_FRAMES,
        phase_cal_path: str = "",
        phase_bias_deg: tuple[float, ...] | list[float] | None = None,
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
        self._num_channels = max(1, int(num_channels))

        self._range_roi = (float(range_roi[0]), float(range_roi[1]))
        self._range_bin_step = float(range_bin_step)
        self._music_enable = bool(music_enable)
        self._num_sources = int(num_sources)
        self._subarray_size = int(subarray_size)
        self._threshold = float(threshold)
        self._array_spacing_m = max(1e-6, float(array_spacing_m))
        self._carrier_freq_hz = max(1.0, float(carrier_freq_hz))
        self._aoa_enable = bool(aoa_enable)
        self._phase_cal_path = str(phase_cal_path or "").strip() or default_phase_cal_path()
        self._phase_cal_frames = max(1, int(phase_cal_frames))
        self._phase_bias_deg = np.zeros(self._num_channels, dtype=np.float64)
        if phase_bias_deg is not None:
            self._set_phase_bias_deg(phase_bias_deg)
        self._plot_period_s = 0.10
        self._last_plot_t = 0.0
        self._start_bin = 0
        self._num_bins = 1
        self._x_start_m = 0.0
        self._recompute_roi()

        self._record_enable = False
        self._record_file_path = str(record_file_path or "")
        self._record_fp: BinaryIO | None = None
        self._record_frames = 0
        self._record_max_frames = max(1, int(record_max_frames))
        self._record_limit_notified = False
        _record_enable_init = bool(record_enable)

        gr.basic_block.__init__(
            self,
            name="OFDM Burst RX",
            in_sig=[np.complex64] * self._num_channels,
            out_sig=None,
        )
        self.set_tag_propagation_policy(TPP_DONT)

        self._bh_window = np.asarray(
            window.blackmanharris(self._vlen_out), dtype=np.float32
        )

        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        prefix = str(plot_title_prefix).strip() or "Range Profile"
        if self._num_channels == 1:
            titles = [
                str(plot_title).strip() if str(plot_title).strip() else prefix
            ]
        else:
            titles = [f"{prefix} ch{i}" for i in range(self._num_channels)]
        self._log_tag = (
            f"{self._num_channels}ch"
            if self._num_channels > 1
            else _log_tag_from_plot_title(titles[0])
        )
        self._displays = [
            _RangeProfileDisplay(
                title=t,
                xlabel="Range (m)",
                ylabel="Power (dB)",
                axis_x=(self._x_start_m, x_max),
            )
            for t in titles
        ]
        # Back-compat alias for single-channel callers
        self._display = self._displays[0]

        if _record_enable_init and self._record_file_path:
            self._open_record_file()
            self._record_enable = self._record_fp is not None

        self._music_estimator = RangeMusicEstimator()
        self._music_executor: Optional[ThreadPoolExecutor] = None
        self._music_busy = False
        self._music_queue: queue.Queue[list[float]] = queue.Queue()
        self._music_frame = 0

        self._aoa_estimator = UlaAoaEstimator()
        self._aoa_executor: Optional[ThreadPoolExecutor] = None
        self._aoa_busy = False
        self._aoa_queue: queue.Queue[tuple[list[float], float]] = queue.Queue()
        self._aoa_frame = 0

        self._phase_cal = RxPhaseCalibrator(
            self._num_channels, target_frames=self._phase_cal_frames
        )
        self._try_load_phase_cal()
        if bool(phase_cal_capture):
            self._phase_cal.start_capture(target_frames=self._phase_cal_frames)

        self._dsp_executor: Optional[ThreadPoolExecutor] = None
        self._dsp_busy = False
        self._dsp_input_q: Deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._dsp_result_q: queue.Queue[
            tuple[np.ndarray, np.ndarray, float]
        ] = queue.Queue()
        self._last_dsp_ms = 0.0
        self._dsp_drop = 0
        self._demod: Optional[OFDMDemodulator] = None

        self._freq_port = pmt.intern(PORT_TX_FREQ_CPI)
        self.message_port_register_in(self._freq_port)
        self.set_msg_handler(self._freq_port, self._on_tx_freq_cpi)
        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_in(self._schedule_port)
        self.set_msg_handler(self._schedule_port, self._on_tx_schedule)

        n = self._num_channels
        self._iq_buf: list[np.ndarray | None] = [None] * n
        self._iq_pos = [0] * n
        self._collecting = [False] * n
        self._last_rx_tag_abs: list[int | None] = [None] * n
        self._ready: list[np.ndarray | None] = [None] * n

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

    def _ensure_demod(self) -> OFDMDemodulator:
        if self._demod is None:
            device = _normalize_torch_device(self._device)
            self._demod = OFDMDemodulator(
                fft_size=self._fft_len,
                l_min=0,
                cyclic_prefix_length=self._cp_len,
                device=device,
            )
        return self._demod

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
        for disp in self._displays:
            disp.set_axis_x(self._x_start_m, x_max)

    @property
    def range_bin_step(self) -> float:
        return self._range_bin_step

    @range_bin_step.setter
    def range_bin_step(self, value: float) -> None:
        self._range_bin_step = float(value)
        self._recompute_roi()
        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        for disp in self._displays:
            disp.set_axis_x(self._x_start_m, x_max)

    @property
    def music_enable(self) -> bool:
        return self._music_enable

    @music_enable.setter
    def music_enable(self, value: bool) -> None:
        self._music_enable = bool(value)
        if not self._music_enable:
            publish_range_estimates([], method_name="MUSIC")

    @property
    def aoa_enable(self) -> bool:
        return self._aoa_enable

    @aoa_enable.setter
    def aoa_enable(self, value: bool) -> None:
        self._aoa_enable = bool(value)
        if not self._aoa_enable:
            publish_aoa_deg(None)

    @property
    def array_spacing_m(self) -> float:
        return self._array_spacing_m

    @array_spacing_m.setter
    def array_spacing_m(self, value: float) -> None:
        self._array_spacing_m = max(1e-6, float(value))

    @property
    def carrier_freq_hz(self) -> float:
        return self._carrier_freq_hz

    @carrier_freq_hz.setter
    def carrier_freq_hz(self, value: float) -> None:
        self._carrier_freq_hz = max(1.0, float(value))

    @property
    def phase_cal_capture(self) -> bool:
        return self._phase_cal.capturing

    @phase_cal_capture.setter
    def phase_cal_capture(self, value: bool) -> None:
        if bool(value):
            self._phase_cal.start_capture(target_frames=self._phase_cal_frames)
            print(
                f"{_LOG_PHASE_CAL} capture started "
                f"(target={self._phase_cal_frames} frames)",
                flush=True,
            )
        else:
            self._phase_cal.stop_capture()

    @property
    def phase_cal_frames(self) -> int:
        return self._phase_cal_frames

    @phase_cal_frames.setter
    def phase_cal_frames(self, value: int) -> None:
        self._phase_cal_frames = max(1, int(value))

    @property
    def phase_cal_path(self) -> str:
        return self._phase_cal_path

    @phase_cal_path.setter
    def phase_cal_path(self, value: str) -> None:
        path = str(value or "").strip() or default_phase_cal_path()
        self._phase_cal_path = path

    def _set_phase_bias_deg(self, value: tuple[float, ...] | list[float] | np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != self._num_channels:
            raise ValueError(
                f"phase_bias_deg length {arr.size} != num_channels {self._num_channels}"
            )
        self._phase_bias_deg = np.clip(arr, -180.0, 180.0)

    @property
    def phase_bias_deg(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self._phase_bias_deg.tolist())

    @phase_bias_deg.setter
    def phase_bias_deg(self, value: tuple[float, ...] | list[float] | np.ndarray) -> None:
        self._set_phase_bias_deg(value)

    def _effective_weights(self) -> np.ndarray:
        """``w_eff = w_auto * exp(-j * deg2rad(bias))``。"""
        w_auto = self._phase_cal.weights
        bias = np.deg2rad(self._phase_bias_deg)
        return (w_auto * np.exp(-1j * bias).astype(np.complex64)).astype(np.complex64)

    def _try_load_phase_cal(self) -> None:
        path = Path(self._phase_cal_path)
        if not path.is_file():
            return
        try:
            weights, freq, n_frames = load_phase_cal(path)
            if weights.size != self._num_channels:
                print(
                    f"{_LOG_PHASE_CAL} skip load {path}: "
                    f"channels {weights.size} != {self._num_channels}",
                    flush=True,
                )
                return
            self._phase_cal.set_weights(weights)
            print(
                f"{_LOG_PHASE_CAL} loaded {path} "
                f"(n_frames={n_frames}, freq={freq:.3e} Hz)",
                flush=True,
            )
        except Exception:
            traceback.print_exc()

    def _finish_phase_cal(self, result, cx: np.ndarray | None = None) -> None:
        try:
            save_phase_cal(
                self._phase_cal_path,
                result.weights,
                carrier_freq_hz=self._carrier_freq_hz,
                n_frames=result.n_frames,
            )
            # Auto-cal replaces channel phases; clear manual bias so w_eff == w_auto
            self._phase_bias_deg = np.zeros(self._num_channels, dtype=np.float64)
            phases = ", ".join(f"{p:.1f}" for p in result.mean_phase_deg.tolist())
            print(
                f"{_LOG_PHASE_CAL} done: saved {self._phase_cal_path} "
                f"n={result.n_frames} peak≈{result.peak_range_m:.3f} m "
                f"mean_phase_deg=[{phases}]; manual bias reset to 0",
                flush=True,
            )
            if cx is not None:
                try:
                    resid, _pb, _rm = relative_phases_deg(
                        cx,
                        self._effective_weights(),
                        range_roi=self._range_roi,
                        range_bin_step=self._range_bin_step,
                    )
                    resid_s = ", ".join(f"{p:.1f}" for p in resid.tolist())
                    print(
                        f"{_LOG_PHASE_CAL} residual_phase_deg after cal "
                        f"(bias=0): [{resid_s}]",
                        flush=True,
                    )
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        notify_phase_cal_done()

    @property
    def record_enable(self) -> bool:
        return self._record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        enabled = bool(value)
        if enabled and not self._record_enable:
            self._record_frames = 0
            self._record_limit_notified = False
            self._open_record_file()
            self._record_enable = self._record_fp is not None
        elif not enabled and self._record_enable:
            self._close_record_file()
            self._record_enable = False

    @property
    def record_file_path(self) -> str:
        return self._record_file_path

    @record_file_path.setter
    def record_file_path(self, value: str) -> None:
        path = str(value or "")
        if path == self._record_file_path:
            return
        self._record_file_path = path
        if self._record_enable:
            self._open_record_file()

    @property
    def record_max_frames(self) -> int:
        return self._record_max_frames

    @record_max_frames.setter
    def record_max_frames(self, value: int) -> None:
        self._record_max_frames = max(1, int(value))

    def _open_record_file(self) -> None:
        self._close_record_file()
        path = str(self._record_file_path or "").strip()
        if not path:
            print(
                f"[ofdm_burst_rx {self._log_tag}] record_file_path empty; skip open",
                flush=True,
            )
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._record_fp = open(p, "wb")
        self._record_frames = 0
        self._record_limit_notified = False
        print(
            f"[ofdm_burst_rx {self._log_tag}] record open {p} "
            f"(frame=({self._num_channels},{self._vlen_out}) c64, "
            f"max={self._record_max_frames})",
            flush=True,
        )

    def _close_record_file(self) -> None:
        fp = self._record_fp
        if fp is None:
            return
        try:
            fp.flush()
            fp.close()
        except OSError:
            traceback.print_exc()
        self._record_fp = None
        print(
            f"[ofdm_burst_rx {self._log_tag}] record close "
            f"frames={self._record_frames}",
            flush=True,
        )

    def _write_record_frame(self, cx_arr: np.ndarray) -> None:
        fp = self._record_fp
        if fp is None or not self._record_enable:
            return
        frame = np.asarray(cx_arr, dtype=np.complex64)
        if frame.ndim == 1:
            frame = frame[np.newaxis, :]
        if frame.shape != (self._num_channels, self._vlen_out):
            raise ValueError(
                f"record frame shape {frame.shape} != "
                f"({self._num_channels}, {self._vlen_out})"
            )
        frame.ravel(order="C").tofile(fp)
        self._record_frames += 1
        if (
            not self._record_limit_notified
            and self._record_frames >= self._record_max_frames
        ):
            self._record_limit_notified = True
            n = self._record_frames
            mx = self._record_max_frames
            self._close_record_file()
            self._record_enable = False
            notify_record_limit_reached(n, mx)

    def start(self) -> bool:
        n = self._num_channels
        self._iq_buf = [None] * n
        self._iq_pos = [0] * n
        self._collecting = [False] * n
        self._last_rx_tag_abs = [None] * n
        self._ready = [None] * n
        self._tx_cpi_queue.clear()
        self._tx_cpi_latest = None
        self._cpi_count = 0
        self._cpi_window_t0 = time.monotonic()
        self._cpi_window_n = 0
        self._rx_time_seen = 0
        self._last_plot_t = 0.0
        self._music_busy = False
        self._music_frame = 0
        self._aoa_busy = False
        self._aoa_frame = 0
        self._dsp_busy = False
        self._last_dsp_ms = 0.0
        self._dsp_drop = 0
        self._dsp_input_q.clear()
        self._demod = None
        while not self._music_queue.empty():
            try:
                self._music_queue.get_nowait()
            except queue.Empty:
                break
        while not self._aoa_queue.empty():
            try:
                self._aoa_queue.get_nowait()
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
        if self._aoa_executor is not None:
            self._aoa_executor.shutdown(wait=False, cancel_futures=True)
        self._aoa_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ofdm_burst_rx_aoa"
        )
        if self._dsp_executor is not None:
            self._dsp_executor.shutdown(wait=False, cancel_futures=True)
        self._dsp_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ofdm_burst_rx_dsp"
        )
        for disp in self._displays:
            disp.show()
        return True

    def stop(self) -> bool:
        if self._music_executor is not None:
            self._music_executor.shutdown(wait=False, cancel_futures=True)
            self._music_executor = None
        if self._aoa_executor is not None:
            self._aoa_executor.shutdown(wait=False, cancel_futures=True)
            self._aoa_executor = None
        if self._dsp_executor is not None:
            self._dsp_executor.shutdown(wait=False, cancel_futures=True)
            self._dsp_executor = None
        self._music_busy = False
        self._aoa_busy = False
        self._dsp_busy = False
        self._demod = None
        if self._record_enable:
            self._close_record_file()
            self._record_enable = False
        publish_range_estimates([], method_name="MUSIC")
        publish_aoa_deg(None)
        for disp in self._displays:
            disp.close()
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
        print(
            f"[ofdm_burst_rx {self._log_tag}] CPI/s≈{rate:.1f} "
            f"total={self._cpi_count} dsp_drop={self._dsp_drop}",
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

    def _drain_aoa(self) -> None:
        while True:
            try:
                angles, range_m = self._aoa_queue.get_nowait()
            except queue.Empty:
                break
            if angles:
                publish_aoa_deg(float(angles[0]))
            else:
                publish_aoa_deg(None)

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
                demod=self._ensure_demod(),
            )
            dsp_ms = (time.perf_counter() - t0) * 1000.0
            self._dsp_result_q.put((db, cx, dsp_ms))
        except Exception:
            traceback.print_exc()
            if self._num_channels > 1:
                nan_db = np.full(
                    (self._num_channels, self._vlen_out), np.nan, dtype=np.float32
                )
                z_cx = np.zeros(
                    (self._num_channels, self._vlen_out), dtype=np.complex64
                )
            else:
                nan_db = np.full(self._vlen_out, np.nan, dtype=np.float32)
                z_cx = np.zeros(self._vlen_out, dtype=np.complex64)
            self._dsp_result_q.put(
                (nan_db, z_cx, (time.perf_counter() - t0) * 1000.0)
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

    def _run_aoa(self, cx_cal: np.ndarray, frame_idx: int) -> None:
        try:
            peaks = self._aoa_estimator(
                cx_cal,
                spacing_m=self._array_spacing_m,
                carrier_freq_hz=self._carrier_freq_hz,
                range_bin_step=self._range_bin_step,
                range_roi=self._range_roi,
                num_sources=self._num_sources,
                threshold=self._threshold,
            )
            angles = peaks.peak_angles_deg.tolist()
            self._aoa_queue.put((angles, float(peaks.peak_range_m)))
            if angles:
                print(
                    f"{_LOG_AOA} frame #{frame_idx} — "
                    f"range≈{peaks.peak_range_m:.3f} m, aoa≈{angles[0]:.2f} deg"
                )
                for i, a in enumerate(angles):
                    print(f"  峰 {i + 1}: {a:.2f} deg")
                try:
                    resid = residual_phase_deg(
                        cx_cal,
                        np.ones(cx_cal.shape[0], dtype=np.complex64),
                        peaks.peak_bin_global,
                    )
                    # cx_cal already weighted; residual vs ch0 should be near 0 if cal ok
                    print(
                        f"  residual_phase_deg={np.array2string(resid, precision=1)}",
                        flush=True,
                    )
                except Exception:
                    pass
            else:
                print(f"{_LOG_AOA} frame #{frame_idx} — 未检测到方位峰", flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            self._aoa_busy = False

    def _emit_plot_and_music(self, db: np.ndarray, cx: np.ndarray) -> None:
        self._drain_music()
        self._drain_aoa()
        now = time.monotonic()
        db_arr = np.asarray(db)
        cx_arr = np.asarray(cx)
        if db_arr.ndim == 1:
            db_arr = db_arr[np.newaxis, :]
            cx_arr = cx_arr[np.newaxis, :]
        try:
            self._write_record_frame(cx_arr)
        except Exception:
            traceback.print_exc()

        # Phase cal uses raw cx; recording already wrote raw
        if self._phase_cal.capturing and self._num_channels > 1:
            try:
                result = self._phase_cal.ingest_frame(
                    cx_arr,
                    range_roi=self._range_roi,
                    range_bin_step=self._range_bin_step,
                )
                if result is not None:
                    self._finish_phase_cal(result, cx=cx_arr)
                else:
                    print(
                        f"{_LOG_PHASE_CAL} "
                        f"{self._phase_cal.frames_collected}/"
                        f"{self._phase_cal.target_frames}",
                        flush=True,
                    )
            except Exception:
                traceback.print_exc()

        if now - self._last_plot_t >= self._plot_period_s:
            self._last_plot_t = now
            s, n = self._start_bin, self._num_bins
            x = self._x_start_m + np.arange(n, dtype=np.float64) * self._range_bin_step
            for ch, disp in enumerate(self._displays):
                y = db_arr[ch, s : s + n]
                disp.post_profile(x, y)
            if self._num_channels > 1:
                try:
                    w_eff = self._effective_weights()
                    phases, _peak_bin, _rm = relative_phases_deg(
                        cx_arr,
                        w_eff,
                        range_roi=self._range_roi,
                        range_bin_step=self._range_bin_step,
                    )
                    notify_channel_phases(phases.tolist())
                except Exception:
                    traceback.print_exc()

        if (
            self._music_enable
            and not self._music_busy
            and self._music_executor is not None
        ):
            self._music_frame += 1
            self._music_busy = True
            # Serial MUSIC: use ch0 complex profile (same as single-ch behavior)
            self._music_executor.submit(
                self._run_music,
                np.asarray(cx_arr[0], dtype=np.complex64).copy(),
                self._music_frame,
            )

        if (
            self._aoa_enable
            and self._num_channels > 1
            and not self._aoa_busy
            and self._aoa_executor is not None
        ):
            self._aoa_frame += 1
            self._aoa_busy = True
            cx_cal = apply_phase_weights(cx_arr, self._effective_weights())
            self._aoa_executor.submit(
                self._run_aoa,
                np.asarray(cx_cal, dtype=np.complex64).copy(),
                self._aoa_frame,
            )

    def _enqueue_dsp_batch(self, td: np.ndarray, tx: np.ndarray) -> None:
        if len(self._dsp_input_q) >= _DSP_Q_MAX and self._dsp_busy:
            self._dsp_drop += 1
            return
        if len(self._dsp_input_q) >= _DSP_Q_MAX:
            self._dsp_input_q.popleft()
            self._dsp_drop += 1
        self._dsp_input_q.append((td, tx))
        self._kick_dsp()

    def _try_flush_ready(self) -> bool:
        """If all channels have a ready CPI and TX ref exists, enqueue one batch."""
        if any(r is None for r in self._ready):
            return False
        if self._tx_cpi_queue:
            tx = self._tx_cpi_queue.popleft()
        elif self._tx_cpi_latest is not None:
            tx = self._tx_cpi_latest
        else:
            return False
        if self._num_channels == 1:
            td = self._ready[0]
            assert td is not None
            self._ready[0] = None
            self._enqueue_dsp_batch(td, tx)
        else:
            stacked = np.stack([r for r in self._ready], axis=0)  # type: ignore[misc]
            self._ready = [None] * self._num_channels
            self._enqueue_dsp_batch(stacked, tx)
        return True

    def _finish_channel_cpi(self, ch: int) -> bool:
        """Move completed channel buffer into ready slot; try batch flush."""
        buf = self._iq_buf[ch]
        if buf is None or self._iq_pos[ch] < self._burst_len:
            return False
        if self._ready[ch] is not None:
            # Previous ready not consumed yet — hold collecting state
            return False
        self._apply_delay(buf)
        self._ready[ch] = buf
        self._iq_buf[ch] = None
        self._iq_pos[ch] = 0
        self._collecting[ch] = False
        self._try_flush_ready()
        return True

    def _ingest_iq_ch(self, ch: int, in_iq: np.ndarray, read_base: int) -> int:
        """Bulk-consume one channel: find rx_time, then memcpy CPI."""
        n = len(in_iq)
        if n <= 0:
            return 0
        consumed = 0
        while consumed < n:
            if (
                self._iq_buf[ch] is not None
                and self._iq_pos[ch] >= self._burst_len
            ):
                if not self._finish_channel_cpi(ch):
                    break

            if not self._collecting[ch]:
                tags = self.get_tags_in_range(
                    ch, read_base + consumed, read_base + n
                )
                start_off: int | None = None
                for tag in tags:
                    if not pmt.eq(tag.key, TAG_RX_TIME):
                        continue
                    off = int(tag.offset - read_base)
                    if off < consumed:
                        continue
                    last = self._last_rx_tag_abs[ch]
                    if last is not None and tag.offset - last < self._burst_len:
                        continue
                    start_off = off
                    self._last_rx_tag_abs[ch] = int(tag.offset)
                    break
                if start_off is None:
                    return n
                consumed = start_off
                self._iq_buf[ch] = np.zeros(self._burst_len, dtype=np.complex64)
                self._iq_pos[ch] = 0
                self._collecting[ch] = True
                self._rx_time_seen += 1
                if consumed >= n:
                    break

            assert self._iq_buf[ch] is not None
            need = self._burst_len - self._iq_pos[ch]
            take = min(need, n - consumed)
            if take <= 0:
                break
            buf = self._iq_buf[ch]
            assert buf is not None
            pos = self._iq_pos[ch]
            buf[pos : pos + take] = in_iq[consumed : consumed + take]
            self._iq_pos[ch] = pos + take
            consumed += take
            if self._iq_pos[ch] >= self._burst_len:
                self._finish_channel_cpi(ch)
        return consumed

    def forecast(self, noutput_items: int, ninputs) -> list:
        del noutput_items, ninputs
        needs: list[int] = []
        for ch in range(self._num_channels):
            partial = (
                self._collecting[ch]
                and self._iq_buf[ch] is not None
                and 0 < self._iq_pos[ch] < self._burst_len
            )
            if partial:
                need = max(
                    1, min(self._burst_len - self._iq_pos[ch], _FORECAST_MAX)
                )
            elif (
                self._tx_cpi_latest is not None
                or self._tx_cpi_queue
                or self._collecting[ch]
                or self._ready[ch] is not None
            ):
                need = max(1, min(self._burst_len, _FORECAST_MAX))
            else:
                need = 1
            needs.append(need)
        return needs

    def general_work(self, input_items, output_items) -> int:
        del output_items
        self._drain_dsp()

        any_consumed = False
        for ch in range(self._num_channels):
            iq_consumed = self._ingest_iq_ch(
                ch, input_items[ch], self.nitems_read(ch)
            )
            if iq_consumed > 0:
                self.consume(ch, iq_consumed)
                any_consumed = True
            if (
                self._iq_buf[ch] is not None
                and self._iq_pos[ch] >= self._burst_len
            ):
                self._finish_channel_cpi(ch)

        self._try_flush_ready()
        self._drain_dsp()
        self._drain_music()
        self._maybe_log_status()

        if any_consumed:
            return gr.WORK_CALLED_PRODUCE
        return 0


blk = OfdmBurstRxBlock
