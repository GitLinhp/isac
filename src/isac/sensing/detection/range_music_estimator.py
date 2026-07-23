"""1D 距离维 MUSIC：零多普勒 CPI 复数距离谱超分辨检峰。

输入为单帧 CPI 相干积分后的复数距离向量（长度 ``fft_len * zeropadding_fac``）。
在 ROI 内做空间平滑 + 候选点伪谱扫描，输出 bin 与物理距离 (m)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MUSIC_EPS = 1e-12
COV_DIAG_LOAD_REL = 1e-8
COV_DIAG_LOAD_MIN = 1e-20
NUM_SNAPSHOTS = 2048
MAX_CANDIDATES = 200
MIN_PEAK_THRESHOLD = 0.05
MAX_PEAKS = 10
DEFAULT_SUBARRAY_SIZE = 16


@dataclass(frozen=True)
class RangeMusicPeaks:
    """1D 距离 MUSIC 检峰结果。"""

    peak_bins: np.ndarray
    """ROI 局部 bin 索引（float，可为亚 bin）。"""
    peak_ranges_m: np.ndarray
    """物理距离 (m)。"""
    scores: np.ndarray
    """MUSIC 综合分数。"""

    @staticmethod
    def empty() -> RangeMusicPeaks:
        return RangeMusicPeaks(
            peak_bins=np.empty(0, dtype=np.float64),
            peak_ranges_m=np.empty(0, dtype=np.float64),
            scores=np.empty(0, dtype=np.float64),
        )


def _compute_roi_slice(
    *,
    range_roi: tuple[float, float],
    range_bin_step: float,
    vlen: int,
) -> tuple[int, int, float]:
    """与 ``isac_imp.range_profile_roi_slice.compute_range_roi`` 一致的 ROI 切片。"""
    from isac_imp.range_profile_roi_slice import compute_range_roi

    start_bin, num_bins, x_start_m = compute_range_roi(
        range_roi=range_roi,
        range_bin_step=range_bin_step,
        vlen_in=vlen,
    )
    return start_bin, num_bins, x_start_m


def _local_maxima_candidates_1d(
    magnitude: np.ndarray,
    *,
    max_candidates: int = MAX_CANDIDATES,
    min_peak_ratio: float = MIN_PEAK_THRESHOLD,
) -> np.ndarray:
    """1D 局部极大值候选 bin 索引。"""
    n = magnitude.size
    if n < 3:
        top = min(max_candidates, n)
        return np.argsort(magnitude)[-top:].astype(np.float64)

    peaks: list[int] = []
    gate = magnitude.max() * min_peak_ratio
    if magnitude[0] >= gate and magnitude[0] >= magnitude[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        if magnitude[i] >= gate and magnitude[i] >= magnitude[i - 1] and magnitude[i] >= magnitude[i + 1]:
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


def _noise_subspace_from_covariance(
    cov: np.ndarray,
    *,
    num_sources: Optional[int],
    threshold: float,
    subarray_size: int,
) -> Optional[np.ndarray]:
    """样本协方差特征分解 → 噪声子空间列向量 ``(M, K)``。"""
    cov64 = np.asarray(cov, dtype=np.complex128)
    identity = np.eye(subarray_size, dtype=np.complex128)
    trace_real = max(float(np.real(np.trace(cov64))), MUSIC_EPS)
    base_load_f = max(trace_real / subarray_size * COV_DIAG_LOAD_REL, COV_DIAG_LOAD_MIN)

    last_exc: Optional[BaseException] = None
    for scale in (1.0, 1e2, 1e4, 1e6, 1e8):
        load = base_load_f * scale
        r_mat = cov64 + load * identity
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(r_mat)
        except np.linalg.LinAlgError as exc:
            last_exc = exc
            continue

        order = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues.real[order]
        eigenvectors = eigenvectors[:, order]

        if num_sources is None:
            norm_eig = eigenvalues / (eigenvalues[0] + MUSIC_EPS)
            num_signal = int(np.sum(norm_eig > threshold))
            num_signal = max(1, min(num_signal, subarray_size - 1))
        else:
            num_signal = max(1, min(int(num_sources), subarray_size - 1))

        return eigenvectors[:, num_signal:].astype(np.complex64)

    del last_exc
    return None


def _build_snapshots(
    spectrum: np.ndarray,
    subarray_size: int,
    num_snapshots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """滑动子阵快拍，形状 ``(subarray_size, num_snapshots)``。"""
    n = spectrum.size
    max_offset = n - subarray_size
    if max_offset < 0:
        raise ValueError(
            f"ROI 长度 {n} 小于子阵尺寸 {subarray_size}，无法构建空间平滑快拍"
        )
    offsets = rng.integers(0, max_offset + 1, size=num_snapshots)
    snapshots = np.empty((subarray_size, num_snapshots), dtype=np.complex64)
    for t, off in enumerate(offsets):
        snapshots[:, t] = spectrum[off : off + subarray_size]
    return snapshots


def _steering_vectors(
    candidates: np.ndarray,
    num_bins: int,
    subarray_size: int,
) -> np.ndarray:
    """导向向量，形状 ``(subarray_size, num_candidates)``。"""
    row_idx = np.arange(subarray_size, dtype=np.float64)
    norm_pos = candidates / num_bins
    phase = 2.0 * np.pi * norm_pos[np.newaxis, :] * row_idx[:, np.newaxis]
    return np.exp(1j * phase).astype(np.complex64)


def _batch_music_scores(
    candidates: np.ndarray,
    magnitude: np.ndarray,
    noise_subspace: np.ndarray,
    num_bins: int,
    subarray_size: int,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float64)

    steering = _steering_vectors(candidates, num_bins, subarray_size)
    projection = noise_subspace.conj().T @ steering
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    pseudospectrum = 1.0 / (denominator + MUSIC_EPS)
    local_amp = magnitude[np.clip(candidates.astype(np.int64), 0, magnitude.size - 1)]
    return (pseudospectrum.real * local_amp).astype(np.float64)


def _greedy_select_peaks_1d(
    scores: np.ndarray,
    bin_idx: np.ndarray,
    num_output_peaks: int,
    min_separation_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    sel_scores: list[float] = []
    sel_bins: list[float] = []

    for idx in order:
        b = float(bin_idx[idx])
        if any(abs(b - sb) < min_separation_bins for sb in sel_bins):
            continue
        sel_scores.append(float(scores[idx]))
        sel_bins.append(b)
        if len(sel_scores) >= num_output_peaks:
            break

    if not sel_scores:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(sel_scores, dtype=np.float64),
        np.asarray(sel_bins, dtype=np.float64),
    )


def _resolve_num_output_peaks(num_sources: Optional[int]) -> int:
    n = int(num_sources if num_sources is not None else 1)
    return max(1, min(n, MAX_PEAKS))


class RangeMusicEstimator:
    """1D 距离维 MUSIC 估计器。"""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        profile_complex: Sequence[complex] | np.ndarray,
        *,
        range_bin_step: float,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: Optional[int] = 1,
        subarray_size: int = DEFAULT_SUBARRAY_SIZE,
        threshold: float = 0.1,
    ) -> RangeMusicPeaks:
        """对 CPI 复数距离谱做 1D MUSIC 检峰。

        参数
        ----
        profile_complex :
            全谱复数距离向量，长度 ``vlen``。
        range_bin_step :
            距离 bin 步进 (m)。
        range_roi :
            ``(min_m, max_m)`` 搜索 ROI。
        num_sources :
            期望信号源数；``None`` 时由特征值阈值自动估计维数。
        subarray_size :
            空间平滑子阵长度。
        threshold :
            自动估计信号维数时的归一化特征值阈值。
        """
        profile = np.asarray(profile_complex, dtype=np.complex64).reshape(-1)
        vlen = profile.size
        if vlen < 1:
            return RangeMusicPeaks.empty()

        start_bin, num_bins, x_start_m = _compute_roi_slice(
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            vlen=vlen,
        )
        spectrum = profile[start_bin : start_bin + num_bins]
        subarray_size = min(int(subarray_size), num_bins)
        if subarray_size < 2:
            return RangeMusicPeaks.empty()

        try:
            snapshots = _build_snapshots(
                spectrum, subarray_size, NUM_SNAPSHOTS, self._rng
            )
        except ValueError:
            return RangeMusicPeaks.empty()

        cov = snapshots @ snapshots.conj().T / NUM_SNAPSHOTS
        cov = 0.5 * (cov + cov.conj().T)

        noise_subspace = _noise_subspace_from_covariance(
            cov,
            num_sources=num_sources,
            threshold=threshold,
            subarray_size=subarray_size,
        )
        if noise_subspace is None:
            return RangeMusicPeaks.empty()

        magnitude = np.abs(spectrum)
        candidates = _local_maxima_candidates_1d(magnitude)
        scores = _batch_music_scores(
            candidates,
            magnitude,
            noise_subspace,
            num_bins,
            subarray_size,
        )

        num_output = _resolve_num_output_peaks(num_sources)
        sel_scores, sel_bins = _greedy_select_peaks_1d(scores, candidates, num_output)
        if sel_bins.size == 0:
            return RangeMusicPeaks.empty()

        peak_ranges_m = x_start_m + sel_bins * float(range_bin_step)
        return RangeMusicPeaks(
            peak_bins=sel_bins,
            peak_ranges_m=peak_ranges_m,
            scores=sel_scores,
        )
