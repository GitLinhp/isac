from __future__ import annotations

import math
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


def _validate_keep_frac(keep_frac: float) -> None:
    if not (0.0 < float(keep_frac) <= 1.0):
        raise ValueError(f"keep_frac 须满足 0 < keep_frac <= 1，收到 {keep_frac}")


def _trimmed_rmse_from_sq(
    sq_dist: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    keep_frac: float = 0.8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """对一维平方误差向量取最小的前 ``keep_frac`` 再算 RMSE。"""
    _validate_keep_frac(keep_frac)
    if sq_dist.ndim != 1:
        raise ValueError(f"sq_dist 须为 (N,)，收到 {tuple(sq_dist.shape)}")
    n = int(sq_dist.shape[0])
    if n == 0:
        raise ValueError("sq_dist 不能为空")
    if sample_weight is not None and sample_weight.shape != sq_dist.shape:
        raise ValueError(
            "sample_weight 形状须与 sq_dist 一致，"
            f"收到 {tuple(sample_weight.shape)} vs {tuple(sq_dist.shape)}"
        )
    k = max(1, int(math.ceil(float(keep_frac) * n)))
    k = min(k, n)
    keep_sq, idx = torch.topk(sq_dist, k, largest=False)
    if sample_weight is not None:
        w = sample_weight[idx]
        weight_sum = w.sum().clamp_min(eps)
        return torch.sqrt((keep_sq * w).sum() / weight_sum + eps)
    return torch.sqrt(keep_sq.mean() + eps)


def trimmed_best_rmse_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    keep_frac: float = 0.8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """全 batch 误差最小的前 ``keep_frac`` 样本 RMSE（可微）。

    取 ``e_i = ||pred-target||^2`` 最小的 ``ceil(keep_frac * B)`` 个，再
    ``sqrt(mean(e_kept))``；有 ``sample_weight`` 时在 kept 子集上加权。
    """
    TargetPositionRmseLoss._validate_inputs(pred_xy, target_xy)
    sq_dist = ((pred_xy - target_xy) ** 2).sum(dim=-1)
    return _trimmed_rmse_from_sq(
        sq_dist, sample_weight=sample_weight, keep_frac=keep_frac, eps=eps
    )


def session_aggregated_trimmed_best_rmse_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    session_index: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    keep_frac: float = 0.8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """session 聚合后再对 session 级误差取前 ``keep_frac`` RMSE（可微）。"""
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
    _validate_keep_frac(keep_frac)

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
            session_weights.append(
                torch.tensor(1.0, device=pred_xy.device, dtype=pred_xy.dtype)
            )

    sq_tensor = torch.stack(session_sq)
    w_tensor = torch.stack(session_weights)
    return _trimmed_rmse_from_sq(
        sq_tensor, sample_weight=w_tensor, keep_frac=keep_frac, eps=eps
    )


def apply_feature_mixup(
    dual: torch.Tensor,
    dual_b: torch.Tensor,
    soft_a: torch.Tensor,
    soft_b: torch.Tensor,
    *,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同 batch 特征与软标签线性插值（``dual_b`` / ``soft_b`` 已按 perm 对齐）。"""
    if dual.shape != dual_b.shape:
        raise ValueError(
            f"dual 与 dual_b 须同形，收到 {tuple(dual.shape)} vs {tuple(dual_b.shape)}"
        )
    if soft_a.shape != soft_b.shape or soft_a.shape[0] != dual.shape[0]:
        raise ValueError(
            "soft_a/soft_b 须同形且 batch 与 dual 对齐，"
            f"收到 soft_a={tuple(soft_a.shape)} soft_b={tuple(soft_b.shape)} "
            f"dual_batch={dual.shape[0]}"
        )
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam 须在 [0, 1]，收到 {lam}")
    dual_mixed = lam * dual + (1.0 - lam) * dual_b
    soft = lam * soft_a + (1.0 - lam) * soft_b
    return dual_mixed, soft


def forced_topk_set_indices(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """构造强制包含真值的 top-k 索引集合 ``(B, topk)``。

    - 若真值已在 ``TopK(logits, k)`` 中：``S = TopK``
    - 否则：``S = TopK(logits excluding y, k-1) ∪ {y}``
    """
    if logits.ndim != 2:
        raise ValueError(f"logits 须为 (B, C)，收到 {tuple(logits.shape)}")
    if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            "target_ids 须为 (B,) 且与 batch 对齐，"
            f"收到 {tuple(target_ids.shape)} vs batch {logits.shape[0]}"
        )
    batch, num_classes = logits.shape
    k = int(topk)
    if k < 1 or k > num_classes:
        raise ValueError(f"topk 须在 [1, {num_classes}]，收到 {topk}")

    target = target_ids.long()
    if k == 1:
        return target.unsqueeze(-1)

    _topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
    in_topk = (topk_idx == target.unsqueeze(-1)).any(dim=-1)

    logits_excl = logits.clone()
    neg_inf = torch.finfo(logits.dtype).min
    logits_excl.scatter_(1, target.unsqueeze(-1), neg_inf)
    _excl_vals, topk_excl_idx = torch.topk(logits_excl, k=k - 1, dim=-1)
    s_forced = torch.cat([topk_excl_idx, target.unsqueeze(-1)], dim=-1)

    not_in = (~in_topk).unsqueeze(-1)
    return torch.where(not_in.expand_as(topk_idx), s_forced, topk_idx)


class TargetSubregionTopKSoftmaxCELoss(nn.Module):
    """Forced top-k softmax CE：``L = -log(sum_{j in S} p_j)``。

    ``S`` 为强制包含真值的 k 元候选集，与推理 top-k 融合对齐。
    ``topk=1`` 时退化为标准 ``-log p_y``（无 class weight / label smoothing）。
    """

    def __init__(
        self,
        *,
        num_classes: int = 16,
        topk: int = 3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes 须 >= 2，收到 {num_classes}")
        if topk < 1 or topk > num_classes:
            raise ValueError(
                f"topk 须在 [1, {num_classes}]，收到 {topk}"
            )
        self.num_classes = int(num_classes)
        self.topk = int(topk)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"logits 须为 (B, {self.num_classes})，收到 {tuple(logits.shape)}"
            )
        if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
            raise ValueError(
                "target_ids 须为 (B,) 且与 batch 对齐，"
                f"收到 {tuple(target_ids.shape)} vs batch {logits.shape[0]}"
            )
        probs = nn.functional.softmax(logits, dim=-1)
        s_idx = forced_topk_set_indices(
            logits, target_ids, topk=self.topk
        )
        mass = probs.gather(1, s_idx).sum(dim=-1).clamp_min(1e-12)
        return (-torch.log(mass)).mean()


class TargetSubregionCrossEntropyLoss(nn.Module):
    """子区域分类交叉熵（可选类别权重、label smoothing、邻域软标签）。

    ``neighbor_smooth`` > 0 时：真值格权重 ``1-α``，四邻接格（上下左右）
    均分 ``α``；再对该软目标做 ``-sum t log_softmax(logits)``。
    """

    def __init__(
        self,
        *,
        num_classes: int = 16,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        neighbor_smooth: float = 0.0,
        grid_n: int = 4,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes 须 >= 2，收到 {num_classes}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing 须在 [0, 1)，收到 {label_smoothing}"
            )
        if not (0.0 <= neighbor_smooth < 1.0):
            raise ValueError(
                f"neighbor_smooth 须在 [0, 1)，收到 {neighbor_smooth}"
            )
        if grid_n < 1 or grid_n * grid_n != num_classes:
            raise ValueError(
                f"grid_n^2 须等于 num_classes，收到 grid_n={grid_n} "
                f"num_classes={num_classes}"
            )
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)
        self.neighbor_smooth = float(neighbor_smooth)
        self.grid_n = int(grid_n)
        if class_weight is not None:
            weight = torch.as_tensor(class_weight, dtype=torch.float32)
            if weight.ndim != 1 or weight.numel() != self.num_classes:
                raise ValueError(
                    f"class_weight 须为 ({self.num_classes},)，"
                    f"收到 {tuple(weight.shape)}"
                )
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None  # type: ignore[assignment]

    def _neighbor_ids(self, sid: int) -> list[int]:
        x = sid % self.grid_n
        y = sid // self.grid_n
        out: list[int] = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_n and 0 <= ny < self.grid_n:
                out.append(ny * self.grid_n + nx)
        return out

    def soft_targets_from_ids(
        self,
        target_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """``(B,)`` hard id → ``(B, C)`` 软目标（邻域平滑 + 可选均匀 smoothing）。"""
        batch = int(target_ids.shape[0])
        soft = torch.zeros(
            batch, self.num_classes, dtype=dtype, device=device
        )
        alpha = self.neighbor_smooth
        ids = target_ids.long().tolist()
        for i, sid in enumerate(ids):
            sid_i = int(sid)
            neighbors = self._neighbor_ids(sid_i)
            if alpha <= 0.0 or not neighbors:
                soft[i, sid_i] = 1.0
            else:
                soft[i, sid_i] = 1.0 - alpha
                share = alpha / float(len(neighbors))
                for nid in neighbors:
                    soft[i, nid] += share
            if self.label_smoothing > 0.0:
                eps = self.label_smoothing
                soft[i] = soft[i] * (1.0 - eps) + eps / float(self.num_classes)
        return soft

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        soft_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"logits 须为 (B, {self.num_classes})，收到 {tuple(logits.shape)}"
            )
        if target_ids.ndim != 1 or target_ids.shape[0] != logits.shape[0]:
            raise ValueError(
                "target_ids 须为 (B,) 且与 batch 对齐，"
                f"收到 {tuple(target_ids.shape)} vs batch {logits.shape[0]}"
            )
        weight = self.class_weight
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)

        use_soft = soft_targets is not None or self.neighbor_smooth > 0.0
        if use_soft:
            if soft_targets is None:
                soft_targets = self.soft_targets_from_ids(
                    target_ids, dtype=logits.dtype, device=logits.device
                )
            if soft_targets.shape != logits.shape:
                raise ValueError(
                    "soft_targets 须与 logits 同形，"
                    f"收到 {tuple(soft_targets.shape)} vs {tuple(logits.shape)}"
                )
            log_prob = nn.functional.log_softmax(logits, dim=-1)
            # 可选按类权重缩放 soft 目标
            if weight is not None:
                soft_w = soft_targets * weight.unsqueeze(0)
                soft_w = soft_w / soft_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                soft_w = soft_targets
            return -(soft_w * log_prob).sum(dim=-1).mean()

        return nn.functional.cross_entropy(
            logits,
            target_ids.long(),
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
