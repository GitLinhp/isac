"""1D 距离维 ESPRIT：零多普勒 CPI 复数距离谱闭式超分辨检峰。

输入为单帧 CPI 相干积分后的复数距离向量（长度 ``fft_len * zeropadding_fac``）。
在 ROI 内对幅度谱局部峰开窗，Hankel SVD + 旋转不变 ESPRIT  refine 距离 bin。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .range_music_estimator import (
    DEFAULT_SUBARRAY_SIZE,
    MAX_CANDIDATES,
    MAX_PEAKS,
    MIN_PEAK_THRESHOLD,
    _compute_roi_slice,
)

DEFAULT_WINDOW_SIZE = 32
MAX_REFINE_BINS = 3.0


@dataclass(frozen=True)
class RangeEspritPeaks:
    """1D 距离 ESPRIT 检峰结果。"""

    peak_bins: np.ndarray
    peak_ranges_m: np.ndarray
    scores: np.ndarray

    @staticmethod
    def empty() -> RangeEspritPeaks:
        return RangeEspritPeaks(
            peak_bins=np.empty(0, dtype=np.float64),
            peak_ranges_m=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
        )


def _resolve_num_output_peaks(num_sources: Optional[int]) -> int:
    n = int(num_sources if num_sources is not None else 1)
    return max(1, min(n, MAX_PEAKS))


def _wrap_bin(bin_val: float, num_bins: int) -> float:
    if num_bins <= 0:
        return 0.0
    b = float(bin_val) % float(num_bins)
    if b < 0:
        b += float(num_bins)
    return b


def _local_maxima_candidates_1d(
    magnitude: np.ndarray,
    *,
    max_candidates: int = MAX_CANDIDATES,
    min_peak_ratio: float = MIN_PEAK_THRESHOLD,
) -> np.ndarray:
    """1D 局部极大值候选 bin（含边界）。"""
    n = magnitude.size
    if n < 3:
        top = min(max_candidates, n)
        return np.argsort(magnitude)[-top:].astype(np.float64)

    peaks: list[int] = []
    gate = magnitude.max() * min_peak_ratio
    if magnitude[0] >= gate and magnitude[0] >= magnitude[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        if (
            magnitude[i] >= gate
            and magnitude[i] >= magnitude[i - 1]
            and magnitude[i] >= magnitude[i + 1]
        ):
            peaks.append(i)
    if magnitude[n - 1] >= gate and magnitude[n - 1] >= magnitude[n - 2]:
        peaks.append(n - 1)

    if not peaks:
        top = min(max_candidates, n)
        return np.argsort(magnitude)[-top:].astype(np.float64)

    peak_arr = np.asarray(peaks, dtype=np.int64)
    mags = magnitude[peak_arr]
    order = np.argsort(mags)[::-1]
    selected = peak_arr[order[: min(max_candidates, peak_arr.size)]]
    return selected.astype(np.float64)


def _hankel_matrix(spectrum: np.ndarray, subarray_size: int) -> Optional[np.ndarray]:
    n = spectrum.size
    cols = n - subarray_size + 1
    if cols < 1:
        return None
    h = np.lib.stride_tricks.as_strided(
        spectrum,
        shape=(subarray_size, cols),
        strides=(spectrum.strides[0], spectrum.strides[0]),
        writeable=False,
    )
    return np.ascontiguousarray(h)


def _esprit_bin_in_window(
    window: np.ndarray,
    *,
    subarray_size: int,
) -> Optional[float]:
    """单源 ESPRIT：在窗口内估计 fractional bin（相对窗口起点）。"""
    win_len = window.size
    l_dim = min(int(subarray_size), win_len)
    if l_dim < 2:
        return None

    hankel = _hankel_matrix(window, l_dim)
    if hankel is None:
        return None

    try:
        u, _, _ = np.linalg.svd(hankel, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    es = u[:, :1]
    es1, es2 = es[:-1, :], es[1:, :]
    phi = np.linalg.pinv(es1) @ es2
    z = np.linalg.eigvals(phi)[0]
    return _wrap_bin(float(np.angle(z)) / (2.0 * np.pi) * win_len, win_len)


def _refine_candidates_esprit(
    spectrum: np.ndarray,
    candidates: np.ndarray,
    *,
    subarray_size: int,
    window_size: int,
) -> list[tuple[float, float]]:
    """对每个幅度候选开窗做 1 源 ESPRIT，返回 ``(score, global_bin)``。"""
    n = spectrum.size
    magnitude = np.abs(spectrum)
    half = max(int(window_size) // 2, subarray_size)
    refined: list[tuple[float, float]] = []

    for center in candidates:
        c = int(np.clip(round(center), 0, n - 1))
        left = max(0, c - half)
        right = min(n, c + half + 1)
        window = spectrum[left:right]
        if window.size < subarray_size + 1:
            continue

        local_bin = _esprit_bin_in_window(window, subarray_size=subarray_size)
        if local_bin is None:
            global_bin = float(c)
        else:
            global_bin = float(left) + local_bin
            if abs(global_bin - float(c)) > MAX_REFINE_BINS:
                global_bin = float(c)

        idx = int(np.clip(round(global_bin), 0, n - 1))
        score = float(magnitude[idx])
        refined.append((score, global_bin))

    return refined


def _select_top_peaks(
    scored: list[tuple[float, float]],
    *,
    num_output: int,
    min_separation_bins: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    scored.sort(key=lambda t: t[0], reverse=True)
    sel_bins: list[float] = []
    sel_scores: list[float] = []

    for score, b in scored:
        if any(abs(b - sb) < min_separation_bins for sb in sel_bins):
            continue
        sel_bins.append(b)
        sel_scores.append(score)
        if len(sel_bins) >= num_output:
            break

    if not sel_bins:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(sel_scores, dtype=np.float64),
        np.asarray(sel_bins, dtype=np.float64),
    )


class RangeEspritEstimator:
    """1D 距离维 ESPRIT 估计器（峰开窗 refine）。"""

    def __call__(
        self,
        profile_complex: Sequence[complex] | np.ndarray,
        *,
        range_bin_step: float,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: Optional[int] = 1,
        subarray_size: int = DEFAULT_SUBARRAY_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> RangeEspritPeaks:
        profile = np.asarray(profile_complex, dtype=np.complex64).reshape(-1)
        vlen = profile.size
        if vlen < 1:
            return RangeEspritPeaks.empty()

        num_output = _resolve_num_output_peaks(num_sources)
        subarray_size = int(subarray_size)
        window_size = max(int(window_size), subarray_size + 1)
        if subarray_size < 2:
            return RangeEspritPeaks.empty()

        start_bin, num_bins, x_start_m = _compute_roi_slice(
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            vlen=vlen,
        )
        spectrum = profile[start_bin : start_bin + num_bins]
        if num_bins < subarray_size + 1:
            return RangeEspritPeaks.empty()

        magnitude = np.abs(spectrum)
        candidates = _local_maxima_candidates_1d(magnitude)
        refined = _refine_candidates_esprit(
            spectrum,
            candidates,
            subarray_size=subarray_size,
            window_size=window_size,
        )
        sel_scores, sel_bins = _select_top_peaks(refined, num_output=num_output)
        if sel_bins.size == 0:
            return RangeEspritPeaks.empty()

        peak_ranges_m = x_start_m + sel_bins * float(range_bin_step)
        return RangeEspritPeaks(
            peak_bins=sel_bins,
            peak_ranges_m=peak_ranges_m,
            scores=sel_scores,
        )
