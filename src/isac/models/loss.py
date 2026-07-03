"""单基地感知复合损失：分辨率 bin 空间分维度 MSE 加权。"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MonostaticSensingLossConfig:
    """复合感知损失超参数。"""

    range_weight: float = 1.0
    velocity_weight: float = 1.0
    reduction: Literal["mean"] = "mean"


class MonostaticSensingLoss(nn.Module):
    """单基地距离/速度复合损失（bin 空间）。

    对距离 bin、多普勒 bin 预测与标签分别计算 MSE，
    ``forward`` 返回加权复合 MSE 标量。
    """

    def __init__(self, cfg: MonostaticSensingLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MonostaticSensingLossConfig()

    @staticmethod
    def _validate_inputs(
        y_pred_bins: torch.Tensor, y_target_bins: torch.Tensor
    ) -> None:
        if y_pred_bins.shape != y_target_bins.shape:
            raise ValueError(
                "y_pred_bins 与 y_target_bins 形状须一致，"
                f"收到 {tuple(y_pred_bins.shape)} 与 {tuple(y_target_bins.shape)}",
            )
        if y_pred_bins.ndim != 2 or y_pred_bins.shape[-1] != 2:
            raise ValueError(
                "y_pred_bins 与 y_target_bins 形状须为 (B, 2)，"
                f"收到 {tuple(y_pred_bins.shape)}",
            )

    @staticmethod
    def target_bins_from_physical_labels(
        range_m: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        range_resolution: float,
        velocity_resolution: float,
    ) -> torch.Tensor:
        """将物理标签转为 ``(B, 2)`` bin 监督目标。"""
        from .model_design import MonostaticDelayDopplerCNN

        return MonostaticDelayDopplerCNN.physical_to_bins(
            range_m,
            velocity_mps,
            range_resolution=range_resolution,
            velocity_resolution=velocity_resolution,
        )

    def forward(
        self,
        y_pred_bins: torch.Tensor,
        y_target_bins: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(y_pred_bins, y_target_bins)
        err = y_pred_bins - y_target_bins
        mse_range = err[:, 0].pow(2).mean()
        mse_velocity = err[:, 1].pow(2).mean()
        return (
            self.cfg.range_weight * mse_range
            + self.cfg.velocity_weight * mse_velocity
        )
