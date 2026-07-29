"""串联 TwoStage Region→Fine 解码与指标测试。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isac.models import (
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    CooperativeMonostaticTwoStageCNN,
    decode_xy_topk_region_probs,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT


def _make_two_stage() -> CooperativeMonostaticTwoStageCNN:
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
    return CooperativeMonostaticTwoStageCNN(region, fine)


def test_topk1_matches_serial_forward() -> None:
    two_stage = _make_two_stage()
    two_stage.eval()
    batch = torch.randn(4, 4, 40)
    with torch.no_grad():
        xy_k1, ids_k1, probs_k1 = decode_xy_topk_region_probs(
            two_stage, batch, topk=1
        )
        xy_direct, logits = two_stage(batch)
        pred_top1 = logits.argmax(dim=-1)
    assert ids_k1.shape == (4, 1)
    assert torch.equal(ids_k1[:, 0], pred_top1)
    assert torch.allclose(probs_k1, torch.ones(4, 1), atol=1e-5)
    assert torch.allclose(xy_k1, xy_direct, atol=1e-5)


def test_topk3_region_metrics_only() -> None:
    two_stage = _make_two_stage()
    two_stage.eval()
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        # 强制 Region 输出固定 logits：通过直接兼容路径验证 topk 排序
        logits = torch.full((2, SUBREGION_COUNT), -8.0)
        logits[:, 3] = 5.0
        logits[:, 7] = 4.0
        logits[:, 12] = 3.0
        fine = two_stage.fine_model
        xy, ids, probs = decode_xy_topk_region_probs(
            fine, batch, logits, topk=3
        )
        expected_xy = fine(batch, F.softmax(logits, dim=-1))
    assert ids.shape == (2, 3)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert list(ids[0].tolist()) == [3, 7, 12]
    assert torch.allclose(xy, expected_xy, atol=1e-5)
    assert 7 in ids[0].tolist()


def test_topk3_probs_renormalize() -> None:
    two_stage = _make_two_stage()
    two_stage.eval()
    batch = torch.randn(1, 4, 32)
    logits = torch.zeros(1, SUBREGION_COUNT)
    logits[0, 0] = 2.0
    logits[0, 1] = 1.0
    logits[0, 2] = 0.5
    with torch.no_grad():
        _, _, probs = decode_xy_topk_region_probs(
            two_stage.fine_model, batch, logits, topk=3
        )
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert probs[0, 0] >= probs[0, 1] >= probs[0, 2]
