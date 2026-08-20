"""小米单站测距模型：损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetRangeRmseLoss(nn.Module):
    """标量距离 batch RMSE 损失（米，可微）。"""

    def __init__(self, *, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _as_1d(x: torch.Tensor) -> torch.Tensor:
        t = x.reshape(-1)
        return t

    def forward(
        self,
        pred_range: torch.Tensor,
        target_range: torch.Tensor,
    ) -> torch.Tensor:
        pred = self._as_1d(pred_range)
        target = self._as_1d(target_range)
        if pred.shape != target.shape:
            raise ValueError(
                "pred_range 与 target_range 形状须一致，"
                f"收到 {tuple(pred_range.shape)} 与 {tuple(target_range.shape)}"
            )
        sq = (pred - target) ** 2
        return torch.sqrt(sq.mean() + self.eps)

    @staticmethod
    def mean_abs_error_m(
        pred_range: torch.Tensor,
        target_range: torch.Tensor,
    ) -> torch.Tensor:
        """batch 平均绝对误差 (m)。"""
        pred = pred_range.reshape(-1)
        target = target_range.reshape(-1)
        if pred.shape != target.shape:
            raise ValueError(
                "pred_range 与 target_range 形状须一致，"
                f"收到 {tuple(pred_range.shape)} 与 {tuple(target_range.shape)}"
            )
        return torch.abs(pred - target).mean()
