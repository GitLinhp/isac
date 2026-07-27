"""单基地感知复合损失：ROI 局部 bin 空间分维度 MSE 加权。"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from ..sensing.metric import SpectrumMetric
from ..sensing.spectrum.sensing_performance import SensingPerformance


@dataclass(frozen=True)
class MonostaticSensingLossConfig:
    """复合感知损失超参数。"""

    range_weight: float = 1.0
    velocity_weight: float = 1.0
    reduction: Literal["mean"] = "mean"


class MonostaticSensingLoss(nn.Module):
    """单基地距离/速度复合损失（ROI 局部 bin 空间）。

    对 ``peaks_delay``、``peaks_doppler`` 预测与标签分别计算 MSE。
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
    def target_local_bins_from_physical(
        range_m: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        num_doppler_bins: int,
        sensing_performance: SensingPerformance,
    ) -> torch.Tensor:
        """将物理标签转为 ``(B, 2)`` ROI 局部 bin 监督目标。"""
        metric = SpectrumMetric(sensing_performance)
        delay_bin, doppler_bin = metric.physical_to_local_bins(
            range_m,
            velocity_mps,
            num_doppler_bins=num_doppler_bins,
            sens_mode="monostatic",
        )
        return torch.stack([delay_bin, doppler_bin], dim=-1).to(
            dtype=range_m.dtype,
            device=range_m.device,
        )

    @staticmethod
    def target_local_bins_from_peaks(
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
    ) -> torch.Tensor:
        """由 ``peaks_delay`` / ``peaks_doppler`` 构造 ``(B, 2)`` 监督目标。"""
        return torch.stack([peaks_delay, peaks_doppler], dim=-1)

    def forward(
        self,
        y_pred_bins: torch.Tensor,
        y_target_bins: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(y_pred_bins, y_target_bins)
        err = y_pred_bins - y_target_bins
        mse_delay = err[:, 0].pow(2).mean()
        mse_doppler = err[:, 1].pow(2).mean()
        return (
            self.cfg.range_weight * mse_delay
            + self.cfg.velocity_weight * mse_doppler
        )
