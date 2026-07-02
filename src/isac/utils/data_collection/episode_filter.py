"""Episode 采样点采纳条件（预留扩展接口）。"""

from __future__ import annotations

import numpy as np

from isac.channel.rt.rt_scene_filter import RTSceneFilter


def accept_episode_kinematics(
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    ori: np.ndarray,
    scene_filter: RTSceneFilter,
) -> bool:
    """判定是否采纳该 ROI 采样点。

    以 ``scene_filter(pos)`` 判别目标位置是否落在障碍物 AABB 之外；
    ``vel`` / ``ori`` 保留供后续扩展（如速度约束）。
    """
    _ = (vel, ori)
    return scene_filter(pos)
