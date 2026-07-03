"""单基地复合感知损失测试。"""

import pytest
import torch

from isac.models import MonostaticSensingLoss, MonostaticSensingLossConfig


def test_perfect_prediction_zero_loss():
    criterion = MonostaticSensingLoss()
    y = torch.tensor([[10.0, 2.0], [5.0, -1.0]])
    assert criterion(y, y).item() == pytest.approx(0.0)


def test_known_error_values():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[5.0, 0.0]])
    target = torch.tensor([[0.0, 2.0]])
    assert criterion(pred, target).item() == pytest.approx(29.0)


def test_equal_weight_sum():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    target = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    assert criterion(pred, target).item() == pytest.approx(29.0)


def test_velocity_weight_scaling():
    criterion = MonostaticSensingLoss(
        MonostaticSensingLossConfig(velocity_weight=2.0),
    )
    pred = torch.tensor([[5.0, 0.0]])
    target = torch.tensor([[0.0, 2.0]])
    expected = 25.0 + 2.0 * 4.0
    assert criterion(pred, target).item() == pytest.approx(expected)


def test_invalid_shape_raises():
    criterion = MonostaticSensingLoss()
    pred = torch.tensor([[0.5]])
    target = torch.tensor([[0.0]])
    with pytest.raises(ValueError, match="\\(B, 2\\)"):
        criterion(pred, target)


def test_target_bins_from_physical_labels():
    target_bins = MonostaticSensingLoss.target_bins_from_physical_labels(
        torch.tensor([100.0, 200.0]),
        torch.tensor([10.0, -10.0]),
        range_resolution=10.0,
        velocity_resolution=5.0,
    )
    assert target_bins.shape == (2, 2)
    assert target_bins[0, 0].item() == pytest.approx(10.0)
    assert target_bins[1, 0].item() == pytest.approx(20.0)
    assert target_bins[0, 1].item() == pytest.approx(2.0)
    assert target_bins[1, 1].item() == pytest.approx(-2.0)
