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


class TargetPositionRmseLoss(nn.Module):
    """目标平面位置 batch RMSE 损失（可微）。"""

    def __init__(self, *, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    @staticmethod
    def _validate_inputs(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> None:
        if pred_xy.shape != target_xy.shape:
            raise ValueError(
                "pred_xy 与 target_xy 形状须一致，"
                f"收到 {tuple(pred_xy.shape)} 与 {tuple(target_xy.shape)}",
            )
        if pred_xy.ndim != 2 or pred_xy.shape[-1] != 2:
            raise ValueError(
                "pred_xy 与 target_xy 形状须为 (B, 2)，"
                f"收到 {tuple(pred_xy.shape)}",
            )

    def forward(
        self,
        pred_xy: torch.Tensor,
        target_xy: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(pred_xy, target_xy)
        sq_dist = ((pred_xy - target_xy) ** 2).sum(dim=-1)
        if sample_weight is not None:
            if sample_weight.shape != sq_dist.shape:
                raise ValueError(
                    "sample_weight 形状须为 (B,)，"
                    f"收到 {tuple(sample_weight.shape)}"
                )
            weight_sum = sample_weight.sum().clamp_min(self.eps)
            return torch.sqrt((sq_dist * sample_weight).sum() / weight_sum + self.eps)
        return torch.sqrt(sq_dist.mean() + self.eps)

    @staticmethod
    def mean_euclidean_error_m(
        pred_xy: torch.Tensor,
        target_xy: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """batch 内（可选加权）平均欧氏距离 (m)。"""
        dist = torch.linalg.vector_norm(pred_xy - target_xy, dim=-1)
        if sample_weight is None:
            return dist.mean()
        weight_sum = sample_weight.sum().clamp_min(1e-12)
        return (dist * sample_weight).sum() / weight_sum


def monostatic_ranges_from_xy(
    target_xy: torch.Tensor,
    *,
    dev0_xy: tuple[float, float] | torch.Tensor = (0.0, -2.0),
    dev1_xy: tuple[float, float] | torch.Tensor = (-2.0, 0.0),
) -> torch.Tensor:
    """由目标 ``(B, 2)`` xy 与双站坐标计算几何单站距离 ``(B, 2)`` = ``[r0, r1]``。"""
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy 须为 (B, 2)，收到 {tuple(target_xy.shape)}")
    d0 = torch.as_tensor(dev0_xy, dtype=target_xy.dtype, device=target_xy.device).reshape(2)
    d1 = torch.as_tensor(dev1_xy, dtype=target_xy.dtype, device=target_xy.device).reshape(2)
    r0 = torch.linalg.vector_norm(target_xy - d0, dim=-1)
    r1 = torch.linalg.vector_norm(target_xy - d1, dim=-1)
    return torch.stack([r0, r1], dim=-1)


def aux_range_rmse_loss(
    pred_ranges: torch.Tensor,
    target_ranges: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """双站距离辅助头 RMSE（可微）。"""
    if pred_ranges.shape != target_ranges.shape:
        raise ValueError(
            "pred_ranges 与 target_ranges 形状须一致，"
            f"收到 {tuple(pred_ranges.shape)} 与 {tuple(target_ranges.shape)}"
        )
    if pred_ranges.ndim != 2 or pred_ranges.shape[-1] != 2:
        raise ValueError(
            f"pred_ranges 须为 (B, 2)，收到 {tuple(pred_ranges.shape)}"
        )
    sq = ((pred_ranges - target_ranges) ** 2).sum(dim=-1)
    if sample_weight is not None:
        if sample_weight.shape != sq.shape:
            raise ValueError(
                "sample_weight 形状须为 (B,)，"
                f"收到 {tuple(sample_weight.shape)}"
            )
        weight_sum = sample_weight.sum().clamp_min(eps)
        return torch.sqrt((sq * sample_weight).sum() / weight_sum + eps)
    return torch.sqrt(sq.mean() + eps)


def session_aggregated_target_rmse_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    session_index: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """按 session 聚合预测与标签后计算 batch 平均 session RMSE（可微）。"""
    if pred_xy.shape != target_xy.shape:
        raise ValueError(
            "pred_xy 与 target_xy 形状须一致，"
            f"收到 {tuple(pred_xy.shape)} 与 {tuple(target_xy.shape)}"
        )
    if pred_xy.ndim != 2 or pred_xy.shape[-1] != 2:
        raise ValueError(f"pred_xy 须为 (B, 2)，收到 {tuple(pred_xy.shape)}")
    if session_index.ndim != 1 or session_index.shape[0] != pred_xy.shape[0]:
        raise ValueError(
            "session_index 须为 (B,) 且与 batch 对齐，"
            f"收到 {tuple(session_index.shape)} vs batch {pred_xy.shape[0]}"
        )

    unique_sessions = torch.unique(session_index)
    session_sq: list[torch.Tensor] = []
    session_weights: list[torch.Tensor] = []
    for sess in unique_sessions:
        mask = session_index == sess
        pred_mean = pred_xy[mask].mean(dim=0)
        target_mean = target_xy[mask].mean(dim=0)
        sq_dist = ((pred_mean - target_mean) ** 2).sum()
        session_sq.append(sq_dist)
        if sample_weight is not None:
            session_weights.append(sample_weight[mask].mean())
        else:
            session_weights.append(torch.tensor(1.0, device=pred_xy.device, dtype=pred_xy.dtype))

    sq_tensor = torch.stack(session_sq)
    w_tensor = torch.stack(session_weights)
    weight_sum = w_tensor.sum().clamp_min(eps)
    return torch.sqrt((sq_tensor * w_tensor).sum() / weight_sum + eps)
