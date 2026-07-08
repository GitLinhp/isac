"""单基地 DD-CNN 前向与标签工具测试。"""

import pytest
import torch

from isac.data_structures.types import MusicPeaks
from isac.models import MonostaticDelayDopplerCNN, dd_spectrum_to_features
from isac.sensing.evaluation import SensingEstimator
from isac.sensing.geometry import monostatic_range_velocity
from isac.sensing.metric import SpectrumMetric
from isac.sensing.spectrum import DelayDopplerRoi, SensingPerformance


def _sensing_performance() -> SensingPerformance:
    from sionna.phy.ofdm import ResourceGrid

    rg = ResourceGrid(
        num_ofdm_symbols=512,
        fft_size=2048,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=32,
        dc_null=False,
        device="cpu",
    )
    return SensingPerformance(rg, carrier_frequency=6e9)


def test_monostatic_cnn_forward_shape():
    model = MonostaticDelayDopplerCNN(
        in_channels=2,
        range_resolution=2.5,
        velocity_resolution=0.5,
        max_range_m=317.5,
        max_velocity_mps=64.0,
        num_doppler_bins=256,
    )
    model.eval()
    x = torch.randn(1, 2, 256, 128)
    with torch.no_grad():
        bins = model.forward_bins(x)
        peaks = model(x)
    assert bins.shape == (1, 2)
    assert isinstance(peaks, MusicPeaks)
    assert peaks.peaks_delay.shape == (1,)
    assert peaks.peaks_doppler.shape == (1,)


def test_dd_features_from_cropped_spectrum():
    h_dd = torch.randn(128, 64, dtype=torch.complex64)
    feat = dd_spectrum_to_features(h_dd)
    assert feat.shape == (2, 128, 64)


def test_monostatic_range_velocity():
    bs = [0.0, 0.0, 0.0]
    tgt = [30.0, 0.0, 0.0]
    vel = [5.0, 0.0, 0.0]
    r, v = monostatic_range_velocity(tgt, vel, bs)
    assert abs(r - 30.0) < 1e-6
    assert abs(v - 5.0) < 1e-6


def test_kinematics_to_music_peaks_via_sensing_estimator():
    sp = _sensing_performance()
    roi = DelayDopplerRoi(
        max_range_m=310.0,
        max_velocity_mps=150.0,
        sensing_performance=sp,
    )
    h_full = torch.zeros(sp.rg.num_ofdm_symbols, sp.rg.fft_size)
    roi.crop(h_full, sens_mode="monostatic")
    num_doppler_bins = roi.num_doppler_bins

    range_m, vel_mps = monostatic_range_velocity(
        [30.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
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
    estimator = SensingEstimator(sp, "cpu", dd_spectrum_roi=roi)
    estimate = estimator(peaks, sens_mode="monostatic", log_peaks=False)
    assert estimate.est_ranges[0].item() == pytest.approx(30.0, rel=0.02)
    assert estimate.est_velocities[0].item() == pytest.approx(5.0, rel=0.05)


def test_physical_local_bins_consistency():
    sp = _sensing_performance()
    m = SpectrumMetric(sp)
    num_doppler_bins = 256
    range_m = torch.tensor([50.0])
    vel_mps = torch.tensor([-1.0])
    delay_bin, doppler_bin = m.physical_to_local_bins(
        range_m,
        vel_mps,
        num_doppler_bins=num_doppler_bins,
    )
    _, _, r_back, v_back = m.local_bins_to_range_velocity(
        delay_bin,
        doppler_bin,
        num_doppler_bins=num_doppler_bins,
        sens_mode="monostatic",
    )
    assert r_back.item() == pytest.approx(50.0, rel=1e-6)
    assert v_back.item() == pytest.approx(-1.0, rel=1e-6)
