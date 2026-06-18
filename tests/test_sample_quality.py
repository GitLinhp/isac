"""样本质量门控单元测试。"""

from dataclasses import dataclass

import numpy as np
import torch
from sionna.phy.ofdm import ResourceGrid

from isac.sensing.sample_quality import (
    SampleQualityConfig,
    _dd_peak_result_from_magnitude,
    check_los_path,
    evaluate_sample_quality,
)
from isac.sensing.sensing_performance import SensingPerformance


@dataclass
class _MockPaths:
    tau: np.ndarray
    valid: np.ndarray
    a_cpx: np.ndarray

    def cir(self, out_type: str = "numpy"):
        return self.a_cpx, self.tau


@dataclass
class _MockScene:
    paths: _MockPaths


def _sensing_performance(fc: float = 4.9e9) -> SensingPerformance:
    rg = ResourceGrid(
        num_ofdm_symbols=64,
        fft_size=128,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=32,
        dc_null=False,
        device="cpu",
    )
    return SensingPerformance(rg, carrier_frequency=fc)


def _synthetic_dd_magnitude(
    sp: SensingPerformance,
    range_m: float,
    velocity_mps: float,
    *,
    peak_amp: float = 10.0,
    background: float = 0.01,
) -> np.ndarray:
    """构造在几何 bin 处有尖峰的 DD 幅度谱 ``(多普勒, 时延)``。"""
    s, f = sp.rg.num_ofdm_symbols, sp.rg.fft_size
    delay_bin = int(round(range_m / sp.range_resolution))
    fd_hz = -velocity_mps * 2.0 * sp.carrier_frequency / 3e8
    doppler_bin = int(np.argmin(np.abs(sp.doppler_bins - fd_hz)))

    mag = np.full((s, f), background, dtype=np.float64)
    mag[doppler_bin, delay_bin] = peak_amp
    return mag


def test_check_los_path_pass():
    tau_geo = 2.0 * 50.0 / 3e8
    scene = _MockScene(
        paths=_MockPaths(
            tau=np.array([[[tau_geo, tau_geo * 2]]]),
            valid=np.array([[[True, True]]]),
            a_cpx=np.array([[[1.0 + 0j, 0.2 + 0j]]]),
        )
    )
    result = check_los_path(scene, 50.0, cfg=SampleQualityConfig(min_los_ratio=0.3))
    assert result.passed
    assert result.los_ratio == 1.0


def test_check_los_path_weak():
    tau_geo = 2.0 * 50.0 / 3e8
    scene = _MockScene(
        paths=_MockPaths(
            tau=np.array([[[tau_geo * 3, tau_geo]]]),
            valid=np.array([[[True, True]]]),
            a_cpx=np.array([[[1.0 + 0j, 0.05 + 0j]]]),
        )
    )
    result = check_los_path(scene, 50.0, cfg=SampleQualityConfig(min_los_ratio=0.3))
    assert not result.passed
    assert result.reason == "weak_los"


def test_check_dd_peak_pass():
    sp = _sensing_performance()
    range_m = 40.0
    vel = 2.0
    mag = _synthetic_dd_magnitude(sp, range_m, vel, peak_amp=50.0, background=0.001)
    result = _dd_peak_result_from_magnitude(
        mag,
        range_m,
        vel,
        sp,
        cfg=SampleQualityConfig(min_peak_prominence_db=6.0, max_bin_offset=2),
    )
    assert result.passed
    assert result.peak_prominence_db is not None
    assert result.peak_prominence_db > 6.0


def test_check_dd_peak_low_prominence():
    sp = _sensing_performance()
    mag = np.full((sp.rg.num_ofdm_symbols, sp.rg.fft_size), 0.5, dtype=np.float64)
    result = _dd_peak_result_from_magnitude(
        mag,
        40.0,
        2.0,
        sp,
        cfg=SampleQualityConfig(min_peak_prominence_db=6.0),
    )
    assert not result.passed
    assert result.reason == "low_peak_prominence"


def test_evaluate_sample_quality_combined():
    sp = _sensing_performance()
    tau_geo = 2.0 * 40.0 / 3e8
    scene = _MockScene(
        paths=_MockPaths(
            tau=np.array([[[tau_geo]]]),
            valid=np.array([[[True]]]),
            a_cpx=np.array([[[1.0 + 0j]]]),
        )
    )
    mag = _synthetic_dd_magnitude(sp, 40.0, 1.0, peak_amp=80.0, background=0.001)
    dd_only = _dd_peak_result_from_magnitude(
        mag, 40.0, 1.0, sp, cfg=SampleQualityConfig(min_peak_prominence_db=6.0)
    )
    assert dd_only.passed
    los = check_los_path(scene, 40.0, cfg=SampleQualityConfig())
    assert los.passed
