"""方式 A RX 通道相位校准：等长功分/环回相对 ch0 校到 0°。

累计多帧在 ROI 非相干峰 bin 上的相对相位，复平面平均后生成权值
``w_k = exp(-j φ̄_k)``，持久化为 npz。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from isac.sensing.detection.ula_aoa_estimator import incoherent_peak_bin

DEFAULT_CAL_FRAMES = 20


@dataclass
class PhaseCalResult:
    """一次校准完成结果。"""

    weights: np.ndarray  # (n_ch,) complex64
    n_frames: int
    mean_phase_deg: np.ndarray  # (n_ch,) relative to ch0 before correction
    peak_range_m: float


class RxPhaseCalibrator:
    """方式 A：累计帧 → 相对 ch0 相位权值。"""

    def __init__(
        self,
        num_channels: int,
        *,
        target_frames: int = DEFAULT_CAL_FRAMES,
        normalize_amplitude: bool = False,
    ) -> None:
        self._n_ch = max(1, int(num_channels))
        self._target_frames = max(1, int(target_frames))
        self._normalize_amplitude = bool(normalize_amplitude)
        self._acc = np.zeros(self._n_ch, dtype=np.complex128)
        self._n = 0
        self._last_range_m = float("nan")
        self._weights = np.ones(self._n_ch, dtype=np.complex64)
        self._capturing = False

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def capturing(self) -> bool:
        return self._capturing

    @property
    def frames_collected(self) -> int:
        return self._n

    @property
    def target_frames(self) -> int:
        return self._target_frames

    def set_weights(self, weights: Sequence[complex] | np.ndarray) -> None:
        w = np.asarray(weights, dtype=np.complex64).reshape(-1)
        if w.size != self._n_ch:
            raise ValueError(f"weights length {w.size} != num_channels {self._n_ch}")
        self._weights = w.copy()
        self._weights[0] = np.complex64(1.0 + 0.0j)

    def start_capture(self, *, target_frames: Optional[int] = None) -> None:
        if target_frames is not None:
            self._target_frames = max(1, int(target_frames))
        self._acc[:] = 0.0
        self._n = 0
        self._last_range_m = float("nan")
        self._capturing = True

    def stop_capture(self) -> None:
        self._capturing = False

    def ingest_frame(
        self,
        cx: np.ndarray,
        *,
        range_roi: tuple[float, float],
        range_bin_step: float,
    ) -> Optional[PhaseCalResult]:
        """若正在采集则累计一帧；满帧后锁定权值并返回结果。"""
        if not self._capturing:
            return None

        arr = np.asarray(cx, dtype=np.complex64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[0] != self._n_ch:
            raise ValueError(
                f"cx channels {arr.shape[0]} != calibrator channels {self._n_ch}"
            )

        peak_bin, range_m, _, _ = incoherent_peak_bin(
            arr, range_roi=range_roi, range_bin_step=range_bin_step
        )
        snap = arr[:, peak_bin]
        ref = snap[0]
        if abs(ref) < 1e-20:
            return None

        # Relative phasors vs ch0 (unit magnitude for phase average)
        rel = snap * np.conj(ref)
        mag = np.abs(rel)
        mag = np.maximum(mag, 1e-20)
        unit = rel / mag
        self._acc += unit.astype(np.complex128)
        self._n += 1
        self._last_range_m = float(range_m)

        if self._n < self._target_frames:
            return None

        mean = self._acc / float(self._n)
        phases = np.angle(mean)
        phases[0] = 0.0
        weights = np.exp(-1j * phases).astype(np.complex64)
        weights[0] = np.complex64(1.0 + 0.0j)

        if self._normalize_amplitude:
            # Use last snap magnitudes (optional; plan default is phase-only)
            amps = np.abs(snap)
            amps = np.maximum(amps, 1e-20)
            weights = (weights * (amps[0] / amps)).astype(np.complex64)
            weights[0] = np.complex64(1.0 + 0.0j)

        self._weights = weights
        self._capturing = False
        return PhaseCalResult(
            weights=weights.copy(),
            n_frames=int(self._n),
            mean_phase_deg=np.rad2deg(phases).astype(np.float64),
            peak_range_m=float(self._last_range_m),
        )


def apply_phase_weights(cx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``cx_cal[k,:] = w_k * cx[k,:]``；支持 ``(vlen,)`` 或 ``(n_ch, vlen)``。"""
    arr = np.asarray(cx, dtype=np.complex64)
    w = np.asarray(weights, dtype=np.complex64).reshape(-1)
    if arr.ndim == 1:
        if w.size != 1:
            raise ValueError("1-D cx requires scalar/1-element weights")
        return (arr * w[0]).astype(np.complex64, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"cx must be 1-D or 2-D, got {arr.shape}")
    if w.size != arr.shape[0]:
        raise ValueError(f"weights {w.size} != channels {arr.shape[0]}")
    return (arr * w[:, np.newaxis]).astype(np.complex64, copy=False)


def save_phase_cal(
    path: str | Path,
    weights: np.ndarray,
    *,
    carrier_freq_hz: float,
    n_frames: int,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weights=np.asarray(weights, dtype=np.complex64),
        freq=np.float64(carrier_freq_hz),
        n_frames=np.int32(n_frames),
    )
    return out


def load_phase_cal(path: str | Path) -> tuple[np.ndarray, float, int]:
    """返回 ``(weights, freq_hz, n_frames)``。"""
    data = np.load(Path(path))
    weights = np.asarray(data["weights"], dtype=np.complex64).reshape(-1)
    freq = float(np.asarray(data["freq"]).reshape(-1)[0]) if "freq" in data else 0.0
    n_frames = int(np.asarray(data["n_frames"]).reshape(-1)[0]) if "n_frames" in data else 0
    return weights, freq, n_frames


def residual_phase_deg(cx: np.ndarray, weights: np.ndarray, peak_bin: int) -> np.ndarray:
    """校准后峰 bin 相对 ch0 残差相位 (deg)。"""
    arr = apply_phase_weights(cx, weights)
    if arr.ndim == 1:
        return np.zeros(1, dtype=np.float64)
    snap = arr[:, int(peak_bin)]
    ref = snap[0]
    if abs(ref) < 1e-20:
        return np.full(arr.shape[0], np.nan, dtype=np.float64)
    phases = np.rad2deg(np.angle(snap * np.conj(ref))).astype(np.float64)
    phases[0] = 0.0
    return phases


def relative_phases_deg(
    cx: np.ndarray,
    weights: np.ndarray,
    *,
    range_roi: tuple[float, float],
    range_bin_step: float,
) -> tuple[np.ndarray, int, float]:
    """ROI 非相干峰上相对 ch0 相位 (deg)。

    返回 ``(phases_deg, peak_bin, peak_range_m)``。
    """
    arr = np.asarray(cx, dtype=np.complex64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    peak_bin, range_m, _, _ = incoherent_peak_bin(
        arr, range_roi=range_roi, range_bin_step=range_bin_step
    )
    return residual_phase_deg(arr, weights, peak_bin), int(peak_bin), float(range_m)
