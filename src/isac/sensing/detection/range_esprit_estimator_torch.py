"""1D 距离维 ESPRIT（Torch）：与 ``range_esprit_estimator`` NumPy 版对齐，可在 CUDA 上运行。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from isac.sensing.detection.range_esprit_estimator import (
    DEFAULT_WINDOW_SIZE,
    MAX_REFINE_BINS,
    RangeEspritPeaks,
    _resolve_num_output_peaks,
    _select_top_peaks,
    _wrap_bin,
)
from isac.sensing.detection.range_music_estimator import (
    DEFAULT_SUBARRAY_SIZE,
    _compute_roi_slice,
    _local_maxima_candidates_1d,
)


def _esprit_bin_in_window_torch(
    window: torch.Tensor,
    *,
    subarray_size: int,
) -> float | None:
    win_len = int(window.numel())
    l_dim = min(int(subarray_size), win_len)
    if l_dim < 2:
        return None
    cols = win_len - l_dim + 1
    if cols < 1:
        return None

    # Hankel via unfold
    hankel = window.unfold(0, l_dim, 1).T.contiguous()  # (L, cols)
    try:
        u, _, _ = torch.linalg.svd(hankel, full_matrices=False)
    except RuntimeError:
        return None

    es = u[:, :1]
    es1, es2 = es[:-1, :], es[1:, :]
    try:
        phi = torch.linalg.pinv(es1) @ es2
        z = torch.linalg.eigvals(phi)[0]
    except RuntimeError:
        return None
    angle = float(torch.angle(z).real.item())
    return _wrap_bin(angle / (2.0 * np.pi) * win_len, win_len)


def _refine_candidates_esprit_torch(
    spectrum: torch.Tensor,
    candidates: np.ndarray,
    *,
    subarray_size: int,
    window_size: int,
) -> list[tuple[float, float]]:
    n = int(spectrum.numel())
    magnitude = torch.abs(spectrum)
    mag_np = magnitude.detach().cpu().numpy()
    half = max(int(window_size) // 2, subarray_size)
    refined: list[tuple[float, float]] = []

    for center in candidates:
        c = int(np.clip(round(center), 0, n - 1))
        left = max(0, c - half)
        right = min(n, c + half + 1)
        window = spectrum[left:right]
        if int(window.numel()) < subarray_size + 1:
            continue

        local_bin = _esprit_bin_in_window_torch(window, subarray_size=subarray_size)
        if local_bin is None:
            global_bin = float(c)
        else:
            global_bin = float(left) + local_bin
            if abs(global_bin - float(c)) > MAX_REFINE_BINS:
                global_bin = float(c)

        idx = int(np.clip(round(global_bin), 0, n - 1))
        score = float(mag_np[idx])
        refined.append((score, global_bin))

    return refined


class RangeEspritEstimatorTorch:
    """1D 距离维 ESPRIT（Torch）。"""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def __call__(
        self,
        profile_complex: Sequence[complex] | np.ndarray | torch.Tensor,
        *,
        range_bin_step: float,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: Optional[int] = 1,
        subarray_size: int = DEFAULT_SUBARRAY_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        cfar: np.ndarray | None = None,
    ) -> RangeEspritPeaks:
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

        magnitude = torch.abs(spectrum).detach().cpu().numpy().astype(np.float64)
        candidates = _local_maxima_candidates_1d(magnitude, cfar=cfar)
        refined = _refine_candidates_esprit_torch(
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
