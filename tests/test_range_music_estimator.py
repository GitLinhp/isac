"""1D 距离 MUSIC 估计器单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from isac.sensing.detection.range_music_estimator import RangeMusicEstimator


def _synthetic_profile(
    peak_bins: list[int],
    *,
    vlen: int = 512,
    amplitude: float = 5.0,
    width: float = 2.0,
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


@pytest.fixture
def estimator() -> RangeMusicEstimator:
    return RangeMusicEstimator(seed=123)


def test_two_peaks_within_one_bin(estimator: RangeMusicEstimator) -> None:
    vlen = 512
    step = 0.1
    peak_bins = [80, 200]
    profile = _synthetic_profile(peak_bins, vlen=vlen, amplitude=8.0, width=1.5)

    peaks = estimator(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 50.0),
        num_sources=2,
        subarray_size=16,
    )

    assert peaks.peak_ranges_m.size == 2
    expected_m = [b * step for b in peak_bins]
    for est, exp in zip(peaks.peak_ranges_m, expected_m):
        assert abs(est - exp) < step


def test_single_source_returns_strongest_peak(estimator: RangeMusicEstimator) -> None:
    vlen = 512
    step = 0.1
    profile = _synthetic_profile([60, 180], vlen=vlen, amplitude=6.0, width=1.5)

    peaks = estimator(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 50.0),
        num_sources=1,
    )

    assert peaks.peak_ranges_m.size == 1
    assert abs(peaks.peak_ranges_m[0] - 18.0) < step


def test_roi_excludes_outside_peak(estimator: RangeMusicEstimator) -> None:
    vlen = 512
    step = 0.1
    profile = _synthetic_profile([40, 400], vlen=vlen, amplitude=7.0, width=1.5)

    peaks = estimator(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 10.0),
        num_sources=2,
    )

    assert peaks.peak_ranges_m.size >= 1
    assert np.all(peaks.peak_ranges_m <= 10.0 + step)
    assert abs(peaks.peak_ranges_m[0] - 4.0) < step


def test_boundary_bin_zero_is_candidate(estimator: RangeMusicEstimator) -> None:
    vlen = 512
    step = 0.1
    profile = _synthetic_profile([0], vlen=vlen, amplitude=10.0, width=1.2, noise_std=0.02)

    peaks = estimator(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 30.0),
        num_sources=1,
    )

    assert peaks.peak_ranges_m.size == 1
    assert abs(peaks.peak_ranges_m[0]) < step


def test_boundary_bin_last_in_roi(estimator: RangeMusicEstimator) -> None:
    vlen = 512
    step = 0.1
    # ROI (0, 30) -> about 301 bins; place peak at last ROI bin index 300
    roi_last_bin = 300
    profile = _synthetic_profile(
        [roi_last_bin],
        vlen=vlen,
        amplitude=10.0,
        width=1.2,
        noise_std=0.02,
    )

    peaks = estimator(
        profile,
        range_bin_step=step,
        range_roi=(0.0, 30.0),
        num_sources=1,
    )

    assert peaks.peak_ranges_m.size == 1
    expected_m = roi_last_bin * step
    assert abs(peaks.peak_ranges_m[0] - expected_m) < step
