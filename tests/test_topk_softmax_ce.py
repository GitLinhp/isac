"""Forced top-k softmax CE 损失测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from isac.models import (
    TargetSubregionTopKSoftmaxCELoss,
    forced_topk_set_indices,
)
from isac_imp.record_target_metadata import SUBREGION_COUNT


def test_forced_topk_includes_target_when_missing() -> None:
    # logits: class 0 highest, then 1, 2; true=5 not in top-3
    logits = torch.zeros(1, SUBREGION_COUNT)
    logits[0, 0] = 3.0
    logits[0, 1] = 2.0
    logits[0, 2] = 1.0
    logits[0, 5] = 0.0
    s = forced_topk_set_indices(logits, torch.tensor([5]), topk=3)
    assert s.shape == (1, 3)
    ids = set(int(x) for x in s[0].tolist())
    assert 5 in ids
    assert ids == {0, 1, 5}


def test_forced_topk_keeps_natural_topk_when_target_inside() -> None:
    logits = torch.zeros(1, SUBREGION_COUNT)
    logits[0, 5] = 3.0
    logits[0, 1] = 2.0
    logits[0, 2] = 1.0
    s = forced_topk_set_indices(logits, torch.tensor([5]), topk=3)
    ids = set(int(x) for x in s[0].tolist())
    assert ids == {5, 1, 2}


def test_topk1_matches_standard_nll() -> None:
    crit = TargetSubregionTopKSoftmaxCELoss(num_classes=SUBREGION_COUNT, topk=1)
    logits = torch.randn(8, SUBREGION_COUNT)
    targets = torch.randint(0, SUBREGION_COUNT, (8,))
    loss = crit(logits, targets)
    ref = F.nll_loss(F.log_softmax(logits, dim=-1), targets)
    assert torch.allclose(loss, ref, atol=1e-6)


def test_topk_ce_forward_finite_grad() -> None:
    crit = TargetSubregionTopKSoftmaxCELoss(num_classes=SUBREGION_COUNT, topk=3)
    logits = torch.randn(8, SUBREGION_COUNT, requires_grad=True)
    targets = torch.randint(0, SUBREGION_COUNT, (8,))
    loss = crit(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_topk_ce_invalid_k() -> None:
    with pytest.raises(ValueError):
        TargetSubregionTopKSoftmaxCELoss(num_classes=16, topk=0)
    with pytest.raises(ValueError):
        TargetSubregionTopKSoftmaxCELoss(num_classes=16, topk=17)
