"""单基地 DD-CNN 前向与标签工具测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.models import MonostaticDelayDopplerCNN


def test_monostatic_cnn_forward_shape():
    model = MonostaticDelayDopplerCNN(
        in_channels=2,
        range_resolution=2.5,
        velocity_resolution=0.5,
        max_range_m=317.5,
        max_velocity_mps=64.0,
    )
    model.eval()
    x = torch.randn(4, 2, 256, 128)
    with torch.no_grad():
        bins = model.forward_bins(x)
        y = model.bins_to_physical(bins)
    assert y.shape == (4, 2)
    assert bins.shape == (4, 2)
    assert torch.allclose(y, model(x))


def test_dd_features_from_cropped_spectrum():
    from isac.models import dd_spectrum_to_features

    h_dd = torch.randn(128, 64, dtype=torch.complex64)
    feat = dd_spectrum_to_features(h_dd)
    assert feat.shape == (2, 128, 64)


def test_monostatic_labels():
    from isac.models import monostatic_labels_from_kinematics

    bs = [0.0, 0.0, 0.0]
    tgt = [30.0, 0.0, 0.0]
    vel = [5.0, 0.0, 0.0]
    r, v = monostatic_labels_from_kinematics(tgt, vel, bs)
    assert abs(r - 30.0) < 1e-6
    assert abs(v - 5.0) < 1e-6


def test_roi_sensing_limits():
    from isac.models import roi_sensing_limits

    sp = SimpleNamespace(range_resolution=2.5, velocity_resolution=0.4)
    max_range_m, max_velocity_mps = roi_sensing_limits(
        sp,
        max_range_m=157.5,
        max_velocity_mps=25.6,
    )
    assert max_range_m == pytest.approx(63 * 2.5)
    assert max_velocity_mps == pytest.approx(64 * 0.4)


def test_physical_bins_roundtrip():
    from isac.models import bins_to_physical, physical_to_bins

    range_m = torch.tensor([25.0, 100.0])
    velocity_mps = torch.tensor([1.0, -2.5])
    bins = physical_to_bins(
        range_m,
        velocity_mps,
        range_resolution=2.5,
        velocity_resolution=0.5,
    )
    assert bins[0, 0].item() == pytest.approx(10.0)
    assert bins[1, 0].item() == pytest.approx(40.0)
    assert bins[0, 1].item() == pytest.approx(2.0)
    assert bins[1, 1].item() == pytest.approx(-5.0)

    restored = bins_to_physical(
        bins,
        range_resolution=2.5,
        velocity_resolution=0.5,
    )
    assert torch.allclose(restored[:, 0], range_m)
    assert torch.allclose(restored[:, 1], velocity_mps)


def test_model_physical_to_bins():
    range_m = torch.tensor([50.0])
    velocity_mps = torch.tensor([-1.0])
    bins = MonostaticDelayDopplerCNN.physical_to_bins(
        range_m,
        velocity_mps,
        range_resolution=10.0,
        velocity_resolution=5.0,
    )
    assert bins[0, 0].item() == pytest.approx(5.0)
    assert bins[0, 1].item() == pytest.approx(-0.2)
