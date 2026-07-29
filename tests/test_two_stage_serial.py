"""CooperativeMonostaticTwoStageCNN 串联模块测试。"""

from __future__ import annotations

import torch

from isac.models import (
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    CooperativeMonostaticTwoStageCNN,
    load_cooperative_monostatic_two_stage_checkpoints,
    subregion_id_to_one_hot,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT


def _make_models():
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
    return region, fine


def test_two_stage_forward_shape() -> None:
    region, fine = _make_models()
    two_stage = CooperativeMonostaticTwoStageCNN(region, fine)
    two_stage.eval()
    batch = torch.randn(3, 4, 40)
    with torch.no_grad():
        xy, logits = two_stage(batch)
    assert xy.shape == (3, 2)
    assert logits.shape == (3, SUBREGION_COUNT)
    assert torch.isfinite(xy).all()


def test_two_stage_oracle_override() -> None:
    region, fine = _make_models()
    two_stage = CooperativeMonostaticTwoStageCNN(region, fine)
    two_stage.eval()
    batch = torch.randn(2, 4, 40)
    sid = torch.tensor([1, 7], dtype=torch.int64)
    override = subregion_id_to_one_hot(sid, SUBREGION_COUNT)
    with torch.no_grad():
        xy_oracle, logits = two_stage(batch, region_probs_override=override)
        xy_direct = fine(batch, override)
    assert torch.allclose(xy_oracle, xy_direct, atol=1e-5)
    # override 不改变 logits（仍来自 Region）
    assert logits.shape == (2, SUBREGION_COUNT)


def test_two_stage_grad_through_softmax() -> None:
    region, fine = _make_models()
    two_stage = CooperativeMonostaticTwoStageCNN(region, fine)
    two_stage.train()
    batch = torch.randn(2, 4, 40)
    xy, _logits = two_stage(batch)
    loss = xy.pow(2).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in region.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in fine.parameters()
    )


def test_load_two_stage_checkpoints(tmp_path) -> None:
    region, fine = _make_models()
    region_path = tmp_path / "region.pth"
    fine_path = tmp_path / "fine.pth"
    torch.save(
        {
            "model_state_dict": region.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.3,
            "model_kind": "region",
            "pool_mode": "attention",
            "num_classes": SUBREGION_COUNT,
        },
        region_path,
    )
    torch.save(
        {
            "model_state_dict": fine.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.3,
            "model_kind": "fine",
            "pool_mode": "attention",
            "num_classes": SUBREGION_COUNT,
        },
        fine_path,
    )
    loaded = load_cooperative_monostatic_two_stage_checkpoints(
        region_path, fine_path, "cpu"
    )
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        xy, logits = loaded(batch)
    assert xy.shape == (2, 2)
    assert logits.shape == (2, SUBREGION_COUNT)
