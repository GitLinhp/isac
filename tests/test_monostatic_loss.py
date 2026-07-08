"""单基地复合感知损失测试。"""

import pytest
import torch
from types import SimpleNamespace

from isac.models import MonostaticSensingLoss, MonostaticSensingLossConfig


def test_perfect_prediction_zero_loss():
    criterion = MonostaticSensingLoss()
    y = torch.tensor([[10.0, 64.0], [5.0, 128.0]])
    assert criterion(y, y).item() == pytest.approx(0.0)


def test_known_error_values():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[5.0, 128.0]])
    target = torch.tensor([[0.0, 130.0]])
    assert criterion(pred, target).item() == pytest.approx(29.0)


def test_equal_weight_sum():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[5.0, 128.0], [5.0, 128.0]])
    target = torch.tensor([[0.0, 130.0], [0.0, 130.0]])
    assert criterion(pred, target).item() == pytest.approx(29.0)


def test_velocity_weight_scaling():
    criterion = MonostaticSensingLoss(
        MonostaticSensingLossConfig(velocity_weight=2.0),
    )
    pred = torch.tensor([[5.0, 128.0]])
    target = torch.tensor([[0.0, 130.0]])
    expected = 25.0 + 2.0 * 4.0
    assert criterion(pred, target).item() == pytest.approx(expected)


def test_invalid_shape_raises():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[0.5]])
    target = torch.tensor([[0.0]])
    with pytest.raises(ValueError, match="\\(B, 2\\)"):
        criterion(pred, target)


def test_target_local_bins_from_peaks():
    target_bins = MonostaticSensingLoss.target_local_bins_from_peaks(
        torch.tensor([10.0, 5.0]),
        torch.tensor([64.0, 128.0]),
    )
    assert target_bins.shape == (2, 2)
    assert target_bins[0, 0].item() == pytest.approx(10.0)
    assert target_bins[1, 0].item() == pytest.approx(5.0)
    assert target_bins[0, 1].item() == pytest.approx(64.0)
    assert target_bins[1, 1].item() == pytest.approx(128.0)


def test_target_local_bins_from_physical():
    sp = SimpleNamespace(
        range_resolution_monostatic=10.0,
        velocity_resolution_monostatic=5.0,
    )
    target_bins = MonostaticSensingLoss.target_local_bins_from_physical(
        torch.tensor([100.0, 200.0]),
        torch.tensor([10.0, -10.0]),
        num_doppler_bins=256,
        sensing_performance=sp,  # type: ignore[arg-type]
    )
    assert target_bins[0, 0].item() == pytest.approx(10.0)
    assert target_bins[1, 0].item() == pytest.approx(20.0)
    assert target_bins[0, 1].item() == pytest.approx(126.0)
    assert target_bins[1, 1].item() == pytest.approx(130.0)
