"""前 keep_frac 裁剪 RMSE 损失单测。"""

from __future__ import annotations

import pytest
import torch

from isac.models import (
    TargetPositionRmseLoss,
    session_aggregated_trimmed_best_rmse_loss,
    trimmed_best_rmse_loss,
)
from isac.models.loss import _validate_keep_frac


def test_trimmed_best_rmse_frac_one_matches_full_rmse():
    pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    criterion = TargetPositionRmseLoss()
    full = criterion(pred, target)
    trimmed = trimmed_best_rmse_loss(pred, target, keep_frac=1.0)
    assert torch.allclose(full, trimmed, atol=1e-6, rtol=1e-6)


def test_trimmed_best_rmse_drops_outlier():
    pred = torch.tensor(
        [
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [10.0, 10.0],  # outlier
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    full = TargetPositionRmseLoss()(pred, target)
    trimmed = trimmed_best_rmse_loss(pred, target, keep_frac=0.8)
    assert float(trimmed) < float(full)
    # 5 * 0.8 -> keep 4 best; should match RMSE of first 4 only
    without_out = TargetPositionRmseLoss()(pred[:4], target[:4])
    assert torch.allclose(trimmed, without_out, atol=1e-5, rtol=1e-5)


def test_trimmed_best_rmse_invalid_frac():
    with pytest.raises(ValueError, match="keep_frac"):
        _validate_keep_frac(0.0)
    with pytest.raises(ValueError, match="keep_frac"):
        _validate_keep_frac(1.5)
    pred = torch.randn(4, 2)
    target = torch.randn(4, 2)
    with pytest.raises(ValueError, match="keep_frac"):
        trimmed_best_rmse_loss(pred, target, keep_frac=0.0)


def test_session_aggregated_trimmed_best_rmse_frac_one():
    pred = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    # sessions: 0,0,1,1 -> session means (1,0) and (0,2)
    session_index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    from isac.models.loss import session_aggregated_target_rmse_loss

    full = session_aggregated_target_rmse_loss(pred, target, session_index)
    trimmed = session_aggregated_trimmed_best_rmse_loss(
        pred, target, session_index, keep_frac=1.0
    )
    assert torch.allclose(full, trimmed, atol=1e-6, rtol=1e-6)
