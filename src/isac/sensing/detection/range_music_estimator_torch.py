"""1D 距离维 MUSIC（Torch）：与 ``range_music_estimator`` NumPy 版对齐，可在 CUDA 上运行。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from isac.sensing.detection.range_music_estimator import (
    COV_DIAG_LOAD_MIN,
    COV_DIAG_LOAD_REL,
    DEFAULT_SUBARRAY_SIZE,
    MAX_PEAKS,
    MUSIC_EPS,
    NUM_SNAPSHOTS,
    RangeMusicPeaks,
    _compute_roi_slice,
    _greedy_select_peaks_1d,
    _local_maxima_candidates_1d,
    _resolve_num_output_peaks,
)


def _noise_subspace_from_covariance_torch(
    cov: torch.Tensor,
    *,
    num_sources: Optional[int],
    threshold: float,
    subarray_size: int,
) -> torch.Tensor | None:
    identity = torch.eye(subarray_size, dtype=cov.dtype, device=cov.device)
    trace_real = max(float(torch.real(torch.trace(cov)).item()), MUSIC_EPS)
    base_load_f = max(trace_real / subarray_size * COV_DIAG_LOAD_REL, COV_DIAG_LOAD_MIN)

    for scale in (1.0, 1e2, 1e4, 1e6, 1e8):
        load = base_load_f * scale
        r_mat = cov + load * identity
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(r_mat)
        except RuntimeError:
            continue

        order = torch.argsort(eigenvalues.real, descending=True)
        eigenvalues = eigenvalues.real[order]
        eigenvectors = eigenvectors[:, order]

        if num_sources is None:
            norm_eig = eigenvalues / (eigenvalues[0] + MUSIC_EPS)
            num_signal = int((norm_eig > threshold).sum().item())
            num_signal = max(1, min(num_signal, subarray_size - 1))
        else:
            num_signal = max(1, min(int(num_sources), subarray_size - 1))

        return eigenvectors[:, num_signal:].to(torch.complex64)

    return None


def _build_snapshots_torch(
    spectrum: torch.Tensor,
    subarray_size: int,
    num_snapshots: int,
    generator: torch.Generator,
) -> torch.Tensor:
    n = int(spectrum.shape[0])
    max_offset = n - subarray_size
    if max_offset < 0:
        raise ValueError(
            f"ROI 长度 {n} 小于子阵尺寸 {subarray_size}，无法构建空间平滑快拍"
        )
    # Generator 固定在 CPU，offsets 再搬到 spectrum.device（兼容无 CUDA Generator 的环境）
    offsets = torch.randint(
        0,
        max_offset + 1,
        (num_snapshots,),
        generator=generator,
    ).to(device=spectrum.device)

    row = torch.arange(subarray_size, device=spectrum.device)
    idx = offsets.unsqueeze(1) + row.unsqueeze(0)  # (T, L)
    return spectrum[idx].T.contiguous()  # (L, T)


def _batch_music_scores_torch(
    candidates: np.ndarray,
    magnitude: np.ndarray,
    noise_subspace: torch.Tensor,
    num_bins: int,
    subarray_size: int,
) -> np.ndarray:
    if candidates.size == 0:
        return np.empty(0, dtype=np.float64)

    device = noise_subspace.device
    cand_t = torch.as_tensor(candidates, dtype=torch.float64, device=device)
    row_idx = torch.arange(subarray_size, dtype=torch.float64, device=device)
    norm_pos = cand_t / float(num_bins)
    phase = 2.0 * np.pi * norm_pos.unsqueeze(0) * row_idx.unsqueeze(1)
    steering = torch.exp(1j * phase).to(torch.complex64)
    projection = noise_subspace.conj().T @ steering
    denominator = torch.sum(torch.abs(projection) ** 2, dim=0)
    pseudospectrum = 1.0 / (denominator + MUSIC_EPS)
    local_idx = np.clip(candidates.astype(np.int64), 0, magnitude.size - 1)
    local_amp = torch.as_tensor(magnitude[local_idx], dtype=torch.float64, device=device)
    scores = (pseudospectrum.real * local_amp).detach().cpu().numpy().astype(np.float64)
    return scores


class RangeMusicEstimatorTorch:
    """1D 距离维 MUSIC（Torch），默认 seed=42 与 NumPy 版一致意图。"""

    def __init__(self, seed: int = 42, device: torch.device | str = "cpu") -> None:
        self._seed = int(seed)
        self.device = torch.device(device)

    def __call__(
        self,
        profile_complex: Sequence[complex] | np.ndarray | torch.Tensor,
        *,
        range_bin_step: float,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: Optional[int] = 1,
        subarray_size: int = DEFAULT_SUBARRAY_SIZE,
        threshold: float = 0.1,
        cfar: np.ndarray | None = None,
    ) -> RangeMusicPeaks:
        if isinstance(profile_complex, torch.Tensor):
            profile = profile_complex.to(
                device=self.device, dtype=torch.complex64
            ).reshape(-1)
        else:
            profile = torch.as_tensor(
                np.asarray(profile_complex, dtype=np.complex64).reshape(-1),
                dtype=torch.complex64,
                device=self.device,
            )
        vlen = int(profile.numel())
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

        gen = torch.Generator()
        gen.manual_seed(self._seed)
        try:
            snapshots = _build_snapshots_torch(
                spectrum, subarray_size, NUM_SNAPSHOTS, gen
            )
        except ValueError:
            return RangeMusicPeaks.empty()

        cov = snapshots @ snapshots.conj().T / NUM_SNAPSHOTS
        cov = 0.5 * (cov + cov.conj().T)

        noise_subspace = _noise_subspace_from_covariance_torch(
            cov,
            num_sources=num_sources,
            threshold=threshold,
            subarray_size=subarray_size,
        )
        if noise_subspace is None:
            return RangeMusicPeaks.empty()

        magnitude = torch.abs(spectrum).detach().cpu().numpy().astype(np.float64)
        candidates = _local_maxima_candidates_1d(magnitude, cfar=cfar)
        scores = _batch_music_scores_torch(
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
