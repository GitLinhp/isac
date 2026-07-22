"""空场景背景对消：有目标 CFR 减去无目标背景 CFR，抑制静态杂波与直射。"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


def subtract_background_cfr(h_freq: ArrayLike, h_bg: ArrayLike) -> ArrayLike:
    """``H_clean = H - H_bg``，形状与 dtype 与 ``h_freq`` 对齐。"""
    is_torch = isinstance(h_freq, torch.Tensor)
    if is_torch:
        h = h_freq
        bg = (
            h_bg
            if isinstance(h_bg, torch.Tensor)
            else torch.as_tensor(h_bg, device=h.device, dtype=h.dtype)
        )
        if bg.shape != h.shape:
            raise ValueError(
                f"h_bg 形状须与 h_freq 一致，收到 {tuple(bg.shape)} vs {tuple(h.shape)}"
            )
        return h - bg.to(device=h.device, dtype=h.dtype)

    h_np = np.asarray(h_freq)
    bg_np = np.asarray(h_bg)
    if bg_np.shape != h_np.shape:
        raise ValueError(
            f"h_bg 形状须与 h_freq 一致，收到 {bg_np.shape} vs {h_np.shape}"
        )
    return (h_np.astype(np.complex128) - bg_np.astype(np.complex128)).astype(
        h_np.dtype, copy=False
    )


def snapshot_target_pose(target: Any) -> dict[str, Any]:
    """缓存目标位姿，供移出场景后恢复。"""
    pose: dict[str, Any] = {
        "position": list(np.asarray(target.position, dtype=np.float64).reshape(-1)),
        "velocity": list(np.asarray(target.velocity, dtype=np.float64).reshape(-1)),
    }
    scaling = getattr(target, "scaling", None)
    if scaling is not None:
        pose["scaling"] = scaling
    orientation = getattr(target, "orientation", None)
    if orientation is not None:
        pose["orientation"] = list(
            np.asarray(orientation, dtype=np.float64).reshape(-1)
        )
    return pose


def remove_targets_from_scene(rt_simulator: Any) -> list[tuple[str, Any, dict[str, Any]]]:
    """从场景移除全部 ``rt_targets``，返回 ``[(name, target, pose), ...]`` 快照。

    调用后须 ``rt_simulator.paths(update=True)``。
    """
    snapshots: list[tuple[str, Any, dict[str, Any]]] = []
    for name, target in list(rt_simulator.rt_targets.items()):
        pose = snapshot_target_pose(target)
        rt_simulator.scene.edit(remove=target)
        del rt_simulator.rt_targets[name]
        snapshots.append((name, target, pose))
    return snapshots


def restore_targets_to_scene(
    rt_simulator: Any,
    snapshots: list[tuple[str, Any, dict[str, Any]]],
) -> None:
    """按快照顺序将目标加回场景并恢复位姿。

    调用后须 ``rt_simulator.paths(update=True)``。
    """
    for name, target, pose in snapshots:
        rt_simulator.scene.edit(add=target)
        target(**pose)
        rt_simulator.rt_targets[name] = target
