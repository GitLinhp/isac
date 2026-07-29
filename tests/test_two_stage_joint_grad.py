"""两阶段串联全局 RMSE 的端到端梯度 smoke 测试。"""

from __future__ import annotations

import torch

from isac.models import (
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    CooperativeMonostaticTwoStageCNN,
    TargetPositionRmseLoss,
    decode_xy_topk_region_probs,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT


def test_serial_rmse_grads_reach_region_and_fine() -> None:
    region = CooperativeMonostaticRegionCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    fine = CooperativeMonostaticFineCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    two_stage = CooperativeMonostaticTwoStageCNN(region, fine)
    two_stage.train()
    dual = torch.randn(4, 4, 40, requires_grad=False)
    target_xy = torch.randn(4, 2)
    criterion = TargetPositionRmseLoss()

    pred_xy, topk_ids, topk_probs = decode_xy_topk_region_probs(
        two_stage, dual, topk=3
    )
    assert pred_xy.requires_grad
    # topk_probs 来自 Region softmax，应可反传
    assert topk_probs.requires_grad
    loss = criterion(pred_xy, target_xy)
    loss.backward()

    region_grads = [p.grad for p in region.parameters() if p.requires_grad]
    fine_grads = [p.grad for p in fine.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in region_grads
    )
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in fine_grads
    )
    assert topk_ids.shape == (4, 3)


def test_direct_joint_rmse_plus_ce_grads() -> None:
    """直接联合：全局 RMSE + Region CE 均可反传到两塔。"""
    from isac.models import TargetSubregionCrossEntropyLoss

    region = CooperativeMonostaticRegionCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    fine = CooperativeMonostaticFineCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    two_stage = CooperativeMonostaticTwoStageCNN(region, fine)
    two_stage.train()
    dual = torch.randn(4, 4, 40)
    target_xy = torch.randn(4, 2)
    true_sid = torch.randint(0, SUBREGION_COUNT, (4,))
    rmse_crit = TargetPositionRmseLoss()
    ce_crit = TargetSubregionCrossEntropyLoss(num_classes=SUBREGION_COUNT)

    pred_xy, logits = two_stage(dual)
    loss = rmse_crit(pred_xy, target_xy) + 1.0 * ce_crit(logits, true_sid)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in region.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in fine.parameters()
    )
