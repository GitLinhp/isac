"""场景障碍物 AABB 过滤：蒙特卡洛采样点有效性判定。

从 ``RTSimulator.scene`` 的 mesh 对象收集轴对齐包围盒（AABB），判断三维点是否落在
建筑物等实体障碍物内部。实例通常挂在 ``RTSimulator.scene_filter``（与 ``scene`` 同级）。
典型调用方：

- 目标蒙特卡洛采样：``sim.scene_filter(position)``
- ``RTSimulator.validate_transceivers_not_in_obstacles``：初始化时校验收发机位置
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sionna.rt.scene import Scene

# 收集 mesh 障碍物 AABB 时，按对象名子串排除的场景构件（匹配时不区分大小写）。
# 这些对象通常表示地面、顶棚或装饰性几何，不应阻挡目标/采样点放置。
EXCLUDED_MESH_OBSTACLE_NAME_SUBSTRINGS: frozenset[str] = frozenset(
    ("ground", "terrain", "floor", "ceiling", "baseboard")
)


class SceneFilter:
    """场景障碍物过滤器：基于 mesh AABB 判定点是否可放置。

    实例化时一次性收集并缓存 ``self.obstacles``；之后通过 ``filter(position)``
    或 ``filter.__call__(position)`` 做 O(n) 包围盒检测，n 为障碍物数量。
    """

    def __init__(self, scene: Scene, safe_margin: float = 1.0):
        """
        初始化障碍物过滤器。

        参数:
        -------
        scene: Scene
            射线追踪场景，从中读取 ``scene.objects`` 的 mesh 包围盒。
        safe_margin: float
            各轴方向对 AABB 外扩的距离（米）。采样场景常用正值，避免点落在
            墙体表面或数值误差导致的边界穿透；收发机初始化校验通常传 ``0.0``。
        """
        self.safe_margin = safe_margin
        self.obstacles = self._collect_mesh_obstacle_boxes(
            scene, safe_margin=safe_margin
        )

    @staticmethod
    def _collect_mesh_obstacle_boxes(
        scene: Scene,
        *,
        safe_margin: float,
    ) -> list[dict[str, np.ndarray | str]]:
        """收集场景 mesh 障碍物 AABB。

        遍历 ``scene.objects``，跳过名称命中
        ``EXCLUDED_MESH_OBSTACLE_NAME_SUBSTRINGS`` 的对象，对其余含
        ``mi_mesh`` 的实体取 bbox 并按 ``safe_margin`` 外扩。

        参数:
        -------
        - scene: Scene
            射线追踪场景。
        - safe_margin: float
            包围盒 min/max 各轴外扩量（米）。

        返回:
        -------
        - list[dict[str, np.ndarray | str]]
            每项含 ``"name"``（对象名）、``"min"`` / ``"max"``（float64 三维向量）。
        """
        # 收集场景 mesh 障碍物 AABB
        obstacles: list[dict[str, np.ndarray | str]] = []
        margin = float(safe_margin)  # 包围盒 min/max 各轴外扩量（米）

        for name, obj in scene.objects.items():
            name_lower = name.lower()  # 对象名小写
            # 跳过名称命中 EXCLUDED_MESH_OBSTACLE_NAME_SUBSTRINGS 的对象
            if any(
                excluded in name_lower
                for excluded in EXCLUDED_MESH_OBSTACLE_NAME_SUBSTRINGS
            ):
                continue
            try:
                # 获取对象的 mesh 包围盒
                if hasattr(obj, "mi_mesh") and obj.mi_mesh is not None:
                    bbox = obj.mi_mesh.bbox()
                    # 外扩 min/max，使判定区域略大于原始 mesh
                    obstacles.append(
                        {
                            "name": name,
                            # 外扩 min/max，使判定区域略大于原始 mesh
                            "min": np.array(bbox.min, dtype=np.float64)
                            - margin,  # 外扩 min
                            "max": np.array(bbox.max, dtype=np.float64)
                            + margin,  # 外扩 max
                        }
                    )
            except Exception:
                # 个别对象 bbox 不可用时中断整场景收集
                raise RuntimeError(f"对象 {name!r} 的 mesh 包围盒不可用。")
        return obstacles

    def __call__(self, position: np.ndarray) -> bool:
        """
        判断三维点是否在所有障碍物 AABB 之外。

        边界判定为闭区间（含墙面）：坐标落在 ``[min, max]`` 上视为在障碍物内。

        参数:
        -------
        - position: np.ndarray
            三维坐标，形如 ``[x, y, z]``，可为一维数组或嵌套序列。

        返回:
        -------
        - bool
            - ``True``：点有效，未落入任一障碍物；
            - ``False``：点无效，至少落入一个障碍物包围盒。
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        x, y, z = pos
        for obs in self.obstacles:
            box_min = np.asarray(obs["min"], dtype=np.float64)
            box_max = np.asarray(obs["max"], dtype=np.float64)
            if (
                box_min[0] <= x <= box_max[0]
                and box_min[1] <= y <= box_max[1]
                and box_min[2] <= z <= box_max[2]
            ):
                return False
        return True
