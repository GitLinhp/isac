"""采集辅助工具：路径交互检查、几何真值与输出文件名 slug。"""

from __future__ import annotations

import numpy as np
import torch
from sionna.rt import Paths

from ..channel.rt.rx_target_tx_geometric import RxTargetTxGeometric
from ..channel.rt.rt_simulator import RTSimulator
from ..channel.rt.rt_target import RTTarget


def los_truth_from_kinematics(
    pos: np.ndarray,
    vel: np.ndarray,
    rt_simulator: RTSimulator,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由目标运动学计算默认三元组 ``(range, radial_velocity)``，形状均为标量张量。"""
    target_name = next(iter(rt_simulator.rt_targets.keys()))
    target_states = {
        target_name: [
            np.asarray(pos, dtype=np.float64),
            np.asarray(vel, dtype=np.float64),
        ],
    }
    geom = RxTargetTxGeometric.from_states(
        target_states,
        rt_simulator.rx_states,
        rt_simulator.tx_states,
        device=device,
    )
    return geom.range_tensor[0, 0, 0], geom.vel_tensor[0, 0, 0]


def scene_slug_from_rt_simulator(rt_simulator: RTSimulator) -> str:
    """输出文件名用：取 ``rt_simulator_params.filename``；未配置或为空时用 ``\"None\"``。"""
    raw = getattr(rt_simulator.rt_simulator_params, "filename", None)
    if raw is None:
        return "None"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "None"
    return s


def accept_episode_kinematics(
    rt_simulator: RTSimulator,
    position: np.ndarray,
) -> bool:
    """场景障碍物过滤：位置不在 mesh AABB（含 safe_margin）内则采纳。"""
    return bool(rt_simulator.scene_filter(position))


def paths_intersect_object(paths: Paths, object_id: int) -> bool:
    """任一路径在任一 bounce 深度与 ``object_id`` 相交则返回 True。"""
    return bool(np.any(np.asarray(paths.objects) == object_id))


def paths_intersect_target(
    rt_simulator: RTSimulator,
    target: RTTarget,
    paths: Paths | None = None,
) -> bool:
    """目标位姿更新后，判断是否存在与该目标 mesh 相交的路径。"""
    resolved = paths if paths is not None else rt_simulator.paths(update=True)
    return paths_intersect_object(resolved, int(target.object_id))
