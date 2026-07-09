"""单基地 DD-CNN 前向与标签工具测试。"""

import pytest
import torch

from isac.data_structures.types import MusicPeaks
from isac.models import (
    SensingCNN,
    dd_spectrum_to_features,
    kinematics_to_target_bins,
    spectrum_tensor_to_features,
)
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
    model = SensingCNN(in_channels=2)
    model.eval()
    h_dd = torch.randn(256, 128, dtype=torch.complex64)
    with torch.no_grad():
        bins = model(h_dd)
    assert bins.shape == (1, 2)


def test_monostatic_cnn_batch_forward():
    model = SensingCNN(in_channels=2)
    model.eval()
    batch = torch.randn(4, 256, 128, dtype=torch.complex64)
    with torch.no_grad():
        bins = model(batch)
    assert bins.shape == (4, 2)


@pytest.mark.parametrize("num_layers", [1, 2, 4])
def test_monostatic_cnn_num_layers_forward(num_layers: int):
    model = SensingCNN(in_channels=2, num_layers=num_layers)
    model.eval()
    batch = torch.randn(2, 256, 128, dtype=torch.complex64)
    with torch.no_grad():
        bins = model(batch)
    assert bins.shape == (2, 2)


def test_monostatic_cnn_num_layers_invalid():
    with pytest.raises(ValueError, match="num_layers"):
        SensingCNN(in_channels=2, num_layers=0)


def test_monostatic_cnn_base_channels_dropout_roi_input():
    """ROI 典型尺寸 8×13 与小宽度/高 dropout 配置仍可前向。"""
    model = SensingCNN(
        in_channels=2,
        base_channels=16,
        num_layers=2,
        dropout=0.4,
    )
    model.eval()
    h_dd = torch.randn(8, 13, dtype=torch.complex64)
    with torch.no_grad():
        bins = model(h_dd)
    assert bins.shape == (1, 2)


def test_dd_features_from_cropped_spectrum():
    h_dd = torch.randn(128, 64, dtype=torch.complex64)
    feat = dd_spectrum_to_features(h_dd)
    assert feat.shape == (2, 128, 64)


def test_spectrum_tensor_to_features_batch():
    batch = torch.randn(3, 64, 32, dtype=torch.complex64)
    feats = spectrum_tensor_to_features(batch)
    assert feats.shape == (3, 2, 64, 32)
    assert feats.dtype == torch.float32


def test_monostatic_range_velocity():
    bs = [0.0, 0.0, 0.0]
    tgt = [30.0, 0.0, 0.0]
    vel = [5.0, 0.0, 0.0]
    r, v = monostatic_range_velocity(tgt, vel, bs)
    assert abs(r - 30.0) < 1e-6
    assert abs(v - 5.0) < 1e-6


def test_kinematics_to_target_bins_shape():
    sp = _sensing_performance()
    pos = torch.tensor([[30.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    vel = torch.tensor([[5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    bs = torch.tensor([0.0, 0.0, 0.0])
    bins = kinematics_to_target_bins(
        pos,
        vel,
        bs,
        sensing_performance=sp,
        num_doppler_bins=256,
    )
    assert bins.shape == (2, 2)


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
