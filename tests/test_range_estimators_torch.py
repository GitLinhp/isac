"""Torch 版 1D 距离 MUSIC / ESPRIT 与 NumPy 对齐测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isac.sensing.detection.range_esprit_estimator_torch import RangeEspritEstimatorTorch
from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac.sensing.detection.range_music_estimator_torch import RangeMusicEstimatorTorch
from isac_imp.cooperative_monostatic_pipeline import (
    divide_cpi_to_roi_range_profile,
    divide_cpi_to_roi_range_profile_torch,
    grc_cooperative_processing_params,
)


def _synthetic_profile(
    peak_bins: list[int],
    *,
    vlen: int = 512,
    amplitude: float = 8.0,
    width: float = 1.5,
    noise_std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(vlen, dtype=np.float64)
    profile = np.zeros(vlen, dtype=np.complex64)
    for b in peak_bins:
        profile += amplitude * np.exp(-0.5 * ((x - b) / width) ** 2)
    profile += (
        rng.normal(0, noise_std, vlen) + 1j * rng.normal(0, noise_std, vlen)
    ).astype(np.complex64)
    return profile


def test_music_torch_cpu_matches_numpy_strongest_peak() -> None:
    step = 0.1
    profile = _synthetic_profile([80, 200], amplitude=8.0)
    np_peaks = RangeMusicEstimator(seed=123)(
        profile, range_bin_step=step, range_roi=(0.0, 50.0), num_sources=1
    )
    torch_peaks = RangeMusicEstimatorTorch(seed=123, device="cpu")(
        profile, range_bin_step=step, range_roi=(0.0, 50.0), num_sources=1
    )
    assert np_peaks.peak_ranges_m.size == 1
    assert torch_peaks.peak_ranges_m.size == 1
    assert abs(float(np_peaks.peak_ranges_m[0]) - float(torch_peaks.peak_ranges_m[0])) < step


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_music_torch_cuda_matches_torch_cpu() -> None:
    step = 0.1
    profile = _synthetic_profile([60], amplitude=9.0, seed=1)
    cpu = RangeMusicEstimatorTorch(seed=42, device="cpu")(
        profile, range_bin_step=step, range_roi=(0.0, 50.0), num_sources=1
    )
    cuda = RangeMusicEstimatorTorch(seed=42, device="cuda:0")(
        profile, range_bin_step=step, range_roi=(0.0, 50.0), num_sources=1
    )
    assert cpu.peak_ranges_m.size == 1
    assert cuda.peak_ranges_m.size == 1
    assert abs(float(cpu.peak_ranges_m[0]) - float(cuda.peak_ranges_m[0])) < step


def test_esprit_torch_cpu_returns_peak_near_truth() -> None:
    step = 0.1
    profile = _synthetic_profile([100], amplitude=10.0, width=1.2, noise_std=0.02)
    peaks = RangeEspritEstimatorTorch(device="cpu")(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 50.0),
        num_sources=1,
    )
    assert peaks.peak_ranges_m.size >= 1
    assert abs(float(peaks.peak_ranges_m[0]) - 10.0) < 2 * step


def test_divide_cpi_roi_torch_close_to_numpy() -> None:
    proc = grc_cooperative_processing_params()
    fft_len = int(proc["fft_len"])
    zp = int(proc["zeropadding_fac"])
    tr = int(proc["transpose_len"])
    rng = np.random.default_rng(7)
    cpi = (
        rng.normal(size=fft_len * zp * tr) + 1j * rng.normal(size=fft_len * zp * tr)
    ).astype(np.complex64)
    roi_np = divide_cpi_to_roi_range_profile(
        cpi,
        range_bin_step=float(proc["range_bin_step"]),
        range_roi=(0.0, 4.0),
        fft_len=fft_len,
        zeropadding_fac=zp,
        transpose_len=tr,
    )
    roi_t = divide_cpi_to_roi_range_profile_torch(
        cpi,
        range_bin_step=float(proc["range_bin_step"]),
        range_roi=(0.0, 4.0),
        fft_len=fft_len,
        zeropadding_fac=zp,
        transpose_len=tr,
        device="cpu",
    ).numpy()
    assert roi_np.shape == roi_t.shape
    assert np.max(np.abs(roi_np - roi_t)) < 1e-3
