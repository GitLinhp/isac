"""场景障碍物 AABB 过滤：收发机校验与蒙特卡洛采样点有效性判定。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .rt_scene import RTScene


def _point_inside_aabb(
    position: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> bool:
    """点是否落在轴对齐包围盒内（含边界）。"""
    x, y, z = np.asarray(position, dtype=np.float64).reshape(3)
    return bool(
        box_min[0] <= x <= box_max[0]
        and box_min[1] <= y <= box_max[1]
        and box_min[2] <= z <= box_max[2]
    )


def _collect_mesh_obstacle_boxes(
    scene: RTScene,
    *,
    safe_margin: float,
) -> list[dict[str, np.ndarray | str]]:
    """收集场景 mesh 障碍物 AABB（排除 ground/terrain/floor）。"""
    obstacles: list[dict[str, np.ndarray | str]] = []
    margin = float(safe_margin)
    for name, obj in scene.objects.items():
        name_lower = name.lower()
        if "ground" in name_lower or "terrain" in name_lower or "floor" in name_lower:
            continue
        try:
            if hasattr(obj, "mi_mesh") and obj.mi_mesh is not None:
                bbox = obj.mi_mesh.bbox()
                obstacles.append(
                    {
                        "name": name,
                        "min": np.array(bbox.min, dtype=np.float64) - margin,
                        "max": np.array(bbox.max, dtype=np.float64) + margin,
                    }
                )
        except Exception:
            continue
    return obstacles


def validate_transceivers_not_in_obstacles(
    scene: RTScene,
    *,
    safe_margin: float = 0.0,
) -> None:
    """场景初始化时校验收发机 ``position`` 未落入任何障碍物包围盒。"""
    if not scene.transceivers:
        return
    boxes = _collect_mesh_obstacle_boxes(scene, safe_margin=safe_margin)
    for tc_name, tc in scene.transceivers.items():
        pos = np.asarray(tc.position, dtype=np.float64).reshape(3)
        for obs in boxes:
            box_min = np.asarray(obs["min"], dtype=np.float64)
            box_max = np.asarray(obs["max"], dtype=np.float64)
            if _point_inside_aabb(pos, box_min, box_max):
                raise ValueError(
                    f"收发机 {tc_name!r} 位置 {pos.tolist()} 落入障碍物 "
                    f"{obs['name']!r} 的包围盒 "
                    f"(min={box_min.tolist()}, max={box_max.tolist()})。"
                    "请调整 [rt_scene.transceivers.*.position] 或场景布局。"
                )


class SceneFilter:
    """场景障碍物过滤器：基于场景对象 AABB（轴对齐包围盒）进行点有效性判定。"""

    def __init__(self, scene: RTScene, safe_margin: float = 1.0):
        """
        初始化障碍物过滤器。

        参数:
        -------
        scene: RTScene
            射线追踪场景对象，其中包含所有三维实体对象。
        safe_margin: float
            包围盒外扩的安全距离，用于判定障碍物时的冗余缓冲，防止穿透边界取样。
        """
        self.safe_margin = safe_margin
        self.obstacles = _collect_mesh_obstacle_boxes(scene, safe_margin=safe_margin)

    def is_valid(self, position: np.ndarray) -> bool:
        """
        判断指定三维点位置是否在所有障碍物包围盒之外。

        参数:
        -------
        position: np.ndarray
            要判断的三维坐标，形如 [x, y, z]

        返回:
        -------
        bool
            True: 点有效（未落入任意障碍物内），False: 点无效（落入某障碍物内）
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        for obs in self.obstacles:
            if _point_inside_aabb(
                pos,
                np.asarray(obs["min"], dtype=np.float64),
                np.asarray(obs["max"], dtype=np.float64),
            ):
                return False
        return True
