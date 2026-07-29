"""邻域软标签 CE 与 Region 分类损失 / mixup 测试。"""

from __future__ import annotations

import pytest
import torch

from isac.models import TargetSubregionCrossEntropyLoss, apply_feature_mixup
from isac_imp.record_target_metadata import SUBREGION_COUNT, SUBREGION_GRID_N


def test_neighbor_soft_targets_center() -> None:
    crit = TargetSubregionCrossEntropyLoss(
        num_classes=SUBREGION_COUNT,
        neighbor_smooth=0.2,
        grid_n=SUBREGION_GRID_N,
    )
    # id=5 → (x=1,y=1)，四邻 1,4,6,9
    soft = crit.soft_targets_from_ids(
        torch.tensor([5]), dtype=torch.float32, device=torch.device("cpu")
    )
    assert soft.shape == (1, SUBREGION_COUNT)
    assert soft[0, 5].item() == pytest.approx(0.8)
    for nid in (1, 4, 6, 9):
        assert soft[0, nid].item() == pytest.approx(0.05)
    assert soft[0].sum().item() == pytest.approx(1.0)


def test_neighbor_soft_targets_corner() -> None:
    crit = TargetSubregionCrossEntropyLoss(
        num_classes=SUBREGION_COUNT,
        neighbor_smooth=0.2,
        grid_n=SUBREGION_GRID_N,
    )
    # id=0 仅右、上 → 1 与 4
    soft = crit.soft_targets_from_ids(
        torch.tensor([0]), dtype=torch.float32, device=torch.device("cpu")
    )
    assert soft[0, 0].item() == pytest.approx(0.8)
    assert soft[0, 1].item() == pytest.approx(0.1)
    assert soft[0, 4].item() == pytest.approx(0.1)
    assert soft[0].sum().item() == pytest.approx(1.0)


def test_neighbor_ce_forward_finite() -> None:
    crit = TargetSubregionCrossEntropyLoss(
        num_classes=SUBREGION_COUNT,
        neighbor_smooth=0.2,
        grid_n=SUBREGION_GRID_N,
    )
    logits = torch.randn(8, SUBREGION_COUNT, requires_grad=True)
    targets = torch.randint(0, SUBREGION_COUNT, (8,))
    loss = crit(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_hard_ce_unchanged_when_no_neighbor() -> None:
    crit = TargetSubregionCrossEntropyLoss(
        num_classes=SUBREGION_COUNT,
        neighbor_smooth=0.0,
        label_smoothing=0.0,
    )
    logits = torch.randn(4, SUBREGION_COUNT)
    targets = torch.tensor([0, 5, 15, 3])
    loss = crit(logits, targets)
    ref = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(loss, ref)


def test_feature_mixup_shapes_and_grad() -> None:
    crit = TargetSubregionCrossEntropyLoss(
        num_classes=SUBREGION_COUNT,
        neighbor_smooth=0.2,
        grid_n=SUBREGION_GRID_N,
    )
    dual = torch.randn(4, 2, 8, 16, requires_grad=True)
    dual_b = torch.randn(4, 2, 8, 16)
    targets = torch.tensor([0, 5, 15, 3])
    soft_a = crit.soft_targets_from_ids(
        targets, dtype=dual.dtype, device=dual.device
    )
    soft_b = crit.soft_targets_from_ids(
        targets.flip(0), dtype=dual.dtype, device=dual.device
    )
    dual_m, soft_m = apply_feature_mixup(dual, dual_b, soft_a, soft_b, lam=0.7)
    assert dual_m.shape == dual.shape
    assert soft_m.shape == soft_a.shape
    assert torch.allclose(soft_m.sum(dim=-1), torch.ones(4), atol=1e-5)
    logits = dual_m.flatten(1)[:, :SUBREGION_COUNT]
    loss = crit(logits, targets, soft_targets=soft_m)
    loss.backward()
    assert dual.grad is not None
    assert torch.isfinite(dual.grad).all()
