"""sensing_attrs_from_system ROI 与全谱属性测试。"""

from __future__ import annotations

import pytest
import sionna.phy.config
import torch

from isac import PROJECT_ROOT
from isac.collection import sensing_attrs_from_system
from isac.data_structures.types import MusicPeaks
from isac.sensing.evaluation import SensingEstimator
from isac.sensing.geometry import monostatic_range_velocity
from isac.sensing.metric import SpectrumMetric
from isac.system import System
from isac.utils import set_random_seed

_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@pytest.fixture
def collection_system() -> System:
    sionna.phy.config.device = "cpu"
    set_random_seed(42)
    return System(_CONFIG, device="cpu")


def test_sensing_attrs_use_roi_limits_when_configured(
    collection_system: System,
) -> None:
    system = collection_system
    sp = system.components.sensing_performance
    dd_roi = system.components.dd_spectrum_roi
    assert sp is not None
    assert dd_roi is not None

    sensing = sensing_attrs_from_system(system)

    assert sensing["num_doppler_bins"] != sp.rg.num_ofdm_symbols
    assert sensing["num_doppler_bins"] == dd_roi.num_doppler_bins
    assert sensing["max_range_m"] < sp.max_range_monostatic
    assert sensing["max_velocity_mps"] < sp.max_velocity_monostatic
    assert sensing["max_range_m"] == pytest.approx(29.28, rel=0.02)
    # round(5.0 / dv) * dv；dv≈1.171 → 有效约 4.684（配置 5.0 的 bin 对齐近似上界）
    assert sensing["max_velocity_mps"] == pytest.approx(4.684, rel=0.02)
    assert sensing["num_doppler_bins"] == 2 * dd_roi.doppler_half_bins() + 1


def test_sensing_attrs_roi_velocity_roundtrip(collection_system: System) -> None:
    system = collection_system
    sp = system.components.sensing_performance
    dd_roi = system.components.dd_spectrum_roi
    assert sp is not None
    assert dd_roi is not None

    sensing = sensing_attrs_from_system(system)
    num_doppler_bins = int(sensing["num_doppler_bins"])

    range_m, vel_mps = monostatic_range_velocity(
        [12.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    metric = SpectrumMetric(sp)
    delay_bin, doppler_bin = metric.physical_to_local_bins(
        range_m,
        vel_mps,
        num_doppler_bins=num_doppler_bins,
        sens_mode="monostatic",
    )
    peaks = MusicPeaks.from_local_bins(delay_bin, doppler_bin, device="cpu")
    estimator = SensingEstimator(sp, "cpu", dd_spectrum_roi=dd_roi)
    estimate = estimator(peaks, sens_mode="monostatic", log_peaks=False)

    assert estimate.est_ranges[0].item() == pytest.approx(range_m, rel=0.02)
    assert estimate.est_velocities[0].item() == pytest.approx(vel_mps, rel=0.05)
