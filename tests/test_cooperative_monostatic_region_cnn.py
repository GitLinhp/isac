"""Cooperative Monostatic Region / Fine CNN smoke tests。"""

from __future__ import annotations

import pytest
import torch

from isac.models import (
    CooperativeMonostaticFineCNN,
    CooperativeMonostaticRegionCNN,
    TargetSubregionCrossEntropyLoss,
    load_cooperative_monostatic_fine_cnn_checkpoint,
    load_cooperative_monostatic_region_cnn_checkpoint,
    subregion_id_to_one_hot,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT


def test_region_cnn_forward_shape() -> None:
    model = CooperativeMonostaticRegionCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    model.eval()
    batch = torch.randn(3, 4, 40)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (3, SUBREGION_COUNT)


def test_region_cnn_complex_input() -> None:
    model = CooperativeMonostaticRegionCNN(
        in_channels=4, base_channels=16, num_layers=2
    )
    model.eval()
    dual = torch.randn(2, 2, 34, dtype=torch.complex64)
    with torch.no_grad():
        logits = model(dual)
    assert logits.shape == (2, SUBREGION_COUNT)


def test_region_ce_loss() -> None:
    criterion = TargetSubregionCrossEntropyLoss(num_classes=SUBREGION_COUNT)
    logits = torch.randn(4, SUBREGION_COUNT)
    targets = torch.tensor([0, 5, 15, 3], dtype=torch.int64)
    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_region_cnn_checkpoint_roundtrip(tmp_path) -> None:
    model = CooperativeMonostaticRegionCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        dropout=0.1,
        pool_mode="attention",
        num_classes=SUBREGION_COUNT,
    )
    path = tmp_path / "region.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "model_kind": "region",
            "pool_mode": "attention",
            "num_classes": SUBREGION_COUNT,
        },
        path,
    )
    loaded = load_cooperative_monostatic_region_cnn_checkpoint(path, "cpu")
    batch = torch.randn(2, 4, 40)
    with torch.no_grad():
        assert loaded(batch).shape == (2, SUBREGION_COUNT)


def test_fine_cnn_forward_shape() -> None:
    model = CooperativeMonostaticFineCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        num_classes=SUBREGION_COUNT,
    )
    model.eval()
    batch = torch.randn(3, 4, 40)
    probs = torch.softmax(torch.randn(3, SUBREGION_COUNT), dim=-1)
    with torch.no_grad():
        xy = model(batch, probs)
    assert xy.shape == (3, 2)
    assert torch.isfinite(xy).all()


def test_subregion_id_to_one_hot() -> None:
    sid = torch.tensor([0, 5, 15], dtype=torch.int64)
    oh = subregion_id_to_one_hot(sid, SUBREGION_COUNT)
    assert oh.shape == (3, SUBREGION_COUNT)
    assert torch.allclose(oh.sum(dim=-1), torch.ones(3))
    assert oh[1, 5].item() == 1.0


def test_fine_cnn_checkpoint_roundtrip(tmp_path) -> None:
    model = CooperativeMonostaticFineCNN(
        in_channels=4,
        base_channels=16,
        num_layers=2,
        dropout=0.1,
        pool_mode="gap",
        num_classes=SUBREGION_COUNT,
    )
    path = tmp_path / "fine.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "model_kind": "fine",
            "pool_mode": "gap",
            "num_classes": SUBREGION_COUNT,
        },
        path,
    )
    loaded = load_cooperative_monostatic_fine_cnn_checkpoint(path, "cpu")
    batch = torch.randn(2, 4, 40)
    probs = torch.softmax(torch.randn(2, SUBREGION_COUNT), dim=-1)
    with torch.no_grad():
        assert loaded(batch, probs).shape == (2, 2)


def test_fine_ckpt_rejects_old_embed(tmp_path) -> None:
    path = tmp_path / "old_fine.pth"
    torch.save(
        {
            "model_state_dict": {"region_embed.weight": torch.zeros(16, 32)},
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "model_kind": "fine",
        },
        path,
    )
    with pytest.raises(ValueError, match="region_embed"):
        load_cooperative_monostatic_fine_cnn_checkpoint(path, "cpu")


def test_region_ckpt_rejects_wrong_kind(tmp_path) -> None:
    path = tmp_path / "bad.pth"
    torch.save(
        {
            "model_state_dict": {},
            "in_channels": 4,
            "base_channels": 16,
            "num_layers": 2,
            "dropout": 0.1,
            "model_kind": "fine",
        },
        path,
    )
    with pytest.raises(ValueError, match="model_kind"):
        load_cooperative_monostatic_region_cnn_checkpoint(path, "cpu")
