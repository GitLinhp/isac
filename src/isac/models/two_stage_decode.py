"""两阶段定位解码：RegionCNN → FineCNN 串联 → 全局 (x, y)。

top-k 仅用于 Region 分类指标，不参与 xy 融合。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isac_imp.record_target_metadata import SUBREGION_COUNT


def _region_topk_from_logits(
    region_logits: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 Region logits 提取 top-k ids 与再归一化概率（仅指标用）。"""
    if region_logits.ndim != 2:
        raise ValueError(
            f"region_logits 须为 (B, C)，收到 {tuple(region_logits.shape)}"
        )
    _batch, num_classes = region_logits.shape
    k = int(topk)
    if k < 1:
        raise ValueError(f"topk 须 >= 1，收到 {topk}")
    if k > num_classes:
        raise ValueError(f"topk={k} 超过 num_classes={num_classes}")

    probs = F.softmax(region_logits, dim=-1)
    topk_probs_raw, topk_ids = torch.topk(probs, k=k, dim=-1)
    weight_sum = topk_probs_raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    topk_probs = topk_probs_raw / weight_sum
    return topk_ids, topk_probs


def decode_xy_topk_region_probs(
    two_stage: torch.nn.Module,
    dual_profiles: torch.Tensor,
    region_logits: torch.Tensor | None = None,
    *,
    topk: int = 3,
    region_probs_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """串联 Region→Fine 解码全局坐标，并返回 Region top-k 指标。

    1. ``xy, logits = two_stage(dual_profiles)``（单次串联前向）
    2. 从 ``logits`` 取 top-k ids/probs（仅 Region 指标，不参与 xy）

    兼容旧签名：若传入 ``region_logits`` 且 ``two_stage`` 实际为 Fine 模型，
    则视为 ``fine_model(dual, softmax(logits))``（无 Region 再前向）。
    推荐直接传入 :class:`~isac.models.CooperativeMonostaticTwoStageCNN`。

    Parameters
    ----------
    two_stage
        ``CooperativeMonostaticTwoStageCNN``，或（兼容）Fine 模型
    dual_profiles
        模型输入特征 ``(B, C, L)``
    region_logits
        可选；若 ``two_stage`` 为 Fine 且给出 logits，则用
        ``softmax(logits)`` 条件化 Fine（兼容旧调用）
    topk
        Region 指标取前 k 个区域
    region_probs_override
        仅 TwoStage 路径：oracle ablation 概率覆盖

    Returns
    -------
    xy : Tensor
        ``(B, 2)`` 全局坐标 (m)
    topk_ids : Tensor
        ``(B, topk)`` int64 区域 id
    topk_probs : Tensor
        ``(B, topk)`` 再归一化权重
    """
    # 推荐路径：TwoStage 串联模块
    if hasattr(two_stage, "region_model") and hasattr(two_stage, "fine_model"):
        xy, logits = two_stage(
            dual_profiles,
            region_probs_override=region_probs_override,
        )
        topk_ids, topk_probs = _region_topk_from_logits(logits, topk=topk)
        return xy, topk_ids, topk_probs

    # 兼容：fine_model + 外部 region_logits
    if region_logits is None:
        raise TypeError(
            "非 TwoStage 调用须提供 region_logits；"
            "推荐传入 CooperativeMonostaticTwoStageCNN"
        )
    probs = F.softmax(region_logits, dim=-1)
    if region_probs_override is not None:
        probs = region_probs_override
    xy = two_stage(dual_profiles, probs)
    topk_ids, topk_probs = _region_topk_from_logits(region_logits, topk=topk)
    return xy, topk_ids, topk_probs


def decode_xy_hard_top1(
    two_stage: torch.nn.Module,
    dual_profiles: torch.Tensor,
    region_logits: torch.Tensor | None = None,
    *,
    region_probs_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard top-1 解码（``topk=1`` 的便捷别名）。"""
    return decode_xy_topk_region_probs(
        two_stage,
        dual_profiles,
        region_logits,
        topk=1,
        region_probs_override=region_probs_override,
    )


__all__ = [
    "SUBREGION_COUNT",
    "decode_xy_hard_top1",
    "decode_xy_topk_region_probs",
]
