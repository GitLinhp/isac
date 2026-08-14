"""4-元（及一般）ULA 空间 MUSIC 方位角估计。

输入为校准后的多通道复数距离谱 ``cx``，形状 ``(n_ch, vlen)``。
在 ROI 内非相干求峰，取峰邻域 bin 作快拍，扫描 ULA 空间 MUSIC 伪谱。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from isac_imp.range_profile_roi_slice import compute_range_roi

MUSIC_EPS = 1e-12
COV_DIAG_LOAD_REL = 1e-8
COV_DIAG_LOAD_MIN = 1e-20
C_LIGHT = 299_792_458.0
DEFAULT_ANGLE_STEP_DEG = 0.5
DEFAULT_PEAK_HALF_WIDTH_BINS = 2
MAX_PEAKS = 10


@dataclass(frozen=True)
class UlaAoaPeaks:
    """ULA 方位角估计结果。"""

    peak_angles_deg: np.ndarray
    scores: np.ndarray
    peak_range_m: float
    peak_bin_global: int

    @staticmethod
    def empty() -> UlaAoaPeaks:
        return UlaAoaPeaks(
            peak_angles_deg=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
            peak_range_m=float("nan"),
            peak_bin_global=-1,
        )


def ula_steering_vectors(
    angles_deg: np.ndarray,
    *,
    num_elements: int,
    spacing_m: float,
    wavelength_m: float,
) -> np.ndarray:
    """ULA 导向矢量，形状 ``(num_elements, num_angles)``。

    \(a_n(\\theta)=\\exp\\bigl(j 2\\pi (d/\\lambda) n \\sin\\theta\\bigr)\)，
    \(n=0\\ldots N-1\)，\(+\sin\\theta\) 指向通道序号增大方向。
    """
    n = np.arange(int(num_elements), dtype=np.float64)[:, np.newaxis]
    theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64).reshape(1, -1))
    phase = 2.0 * np.pi * (float(spacing_m) / float(wavelength_m)) * n * np.sin(theta)
    return np.exp(1j * phase).astype(np.complex64)


def incoherent_peak_bin(
    cx: np.ndarray,
    *,
    range_roi: tuple[float, float],
    range_bin_step: float,
) -> tuple[int, float, int, int]:
    """ROI 内非相干功率峰 → ``(global_bin, range_m, start_bin, num_bins)``。"""
    arr = np.asarray(cx)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"cx must be (n_ch, vlen), got {arr.shape}")

    vlen = int(arr.shape[1])
    start_bin, num_bins, x_start_m = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=range_bin_step,
        vlen_in=vlen,
    )
    if num_bins < 1:
        raise ValueError("empty range ROI")

    power = np.sum(np.abs(arr[:, start_bin : start_bin + num_bins]) ** 2, axis=0)
    local = int(np.argmax(power))
    global_bin = start_bin + local
    range_m = float(x_start_m + local * float(range_bin_step))
    return global_bin, range_m, start_bin, num_bins


def _noise_subspace(
    cov: np.ndarray,
    *,
    num_sources: int,
    threshold: float,
) -> Optional[np.ndarray]:
    del threshold  # reserved for auto rank; fixed num_sources for now
    m = cov.shape[0]
    cov64 = np.asarray(cov, dtype=np.complex128)
    identity = np.eye(m, dtype=np.complex128)
    trace_real = max(float(np.real(np.trace(cov64))), MUSIC_EPS)
    base_load = max(trace_real / m * COV_DIAG_LOAD_REL, COV_DIAG_LOAD_MIN)
    n_sig = max(1, min(int(num_sources), m - 1))

    for scale in (1.0, 1e2, 1e4, 1e6, 1e8):
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                cov64 + base_load * scale * identity
            )
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigenvalues.real)[::-1]
        eigenvectors = eigenvectors[:, order]
        return eigenvectors[:, n_sig:].astype(np.complex64)
    return None


def _greedy_select_angles(
    scores: np.ndarray,
    angles_deg: np.ndarray,
    num_peaks: int,
    *,
    min_separation_deg: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    sel_a: list[float] = []
    sel_s: list[float] = []
    for idx in order:
        a = float(angles_deg[idx])
        if any(abs(a - sa) < min_separation_deg for sa in sel_a):
            continue
        sel_a.append(a)
        sel_s.append(float(scores[idx]))
        if len(sel_a) >= num_peaks:
            break
    if not sel_a:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        np.asarray(sel_a, dtype=np.float64),
        np.asarray(sel_s, dtype=np.float64),
    )


class UlaAoaEstimator:
    """多通道复数距离谱 → ULA 空间 MUSIC 方位角。"""

    def __call__(
        self,
        cx: Sequence[Sequence[complex]] | np.ndarray,
        *,
        spacing_m: float,
        carrier_freq_hz: float,
        range_bin_step: float,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: int = 1,
        threshold: float = 0.1,
        angle_min_deg: float = -90.0,
        angle_max_deg: float = 90.0,
        angle_step_deg: float = DEFAULT_ANGLE_STEP_DEG,
        peak_half_width_bins: int = DEFAULT_PEAK_HALF_WIDTH_BINS,
    ) -> UlaAoaPeaks:
        arr = np.asarray(cx, dtype=np.complex64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        n_ch, vlen = arr.shape
        if n_ch < 2 or vlen < 1:
            return UlaAoaPeaks.empty()

        fc = float(carrier_freq_hz)
        d = float(spacing_m)
        if fc <= 0.0 or d <= 0.0:
            return UlaAoaPeaks.empty()
        wavelength = C_LIGHT / fc

        try:
            peak_bin, range_m, _s, _n = incoherent_peak_bin(
                arr,
                range_roi=range_roi,
                range_bin_step=range_bin_step,
            )
        except ValueError:
            return UlaAoaPeaks.empty()

        half = max(0, int(peak_half_width_bins))
        lo = max(0, peak_bin - half)
        hi = min(vlen, peak_bin + half + 1)
        snaps = arr[:, lo:hi]  # (n_ch, T)
        t = snaps.shape[1]
        if t < 1:
            return UlaAoaPeaks.empty()

        cov = (snaps @ snaps.conj().T) / float(t)
        cov = 0.5 * (cov + cov.conj().T)

        noise = _noise_subspace(
            cov, num_sources=int(num_sources), threshold=float(threshold)
        )
        if noise is None:
            return UlaAoaPeaks.empty()

        angles = np.arange(
            float(angle_min_deg),
            float(angle_max_deg) + 0.5 * float(angle_step_deg),
            float(angle_step_deg),
            dtype=np.float64,
        )
        if angles.size == 0:
            return UlaAoaPeaks.empty()

        steering = ula_steering_vectors(
            angles,
            num_elements=n_ch,
            spacing_m=d,
            wavelength_m=wavelength,
        )
        proj = noise.conj().T @ steering
        denom = np.sum(np.abs(proj) ** 2, axis=0)
        spectrum = 1.0 / (denom + MUSIC_EPS)
        scores = spectrum.real.astype(np.float64)

        n_out = max(1, min(int(num_sources), MAX_PEAKS, n_ch - 1))
        peak_angles, peak_scores = _greedy_select_angles(scores, angles, n_out)
        if peak_angles.size == 0:
            return UlaAoaPeaks.empty()

        return UlaAoaPeaks(
            peak_angles_deg=peak_angles,
            scores=peak_scores,
            peak_range_m=range_m,
            peak_bin_global=int(peak_bin),
        )
