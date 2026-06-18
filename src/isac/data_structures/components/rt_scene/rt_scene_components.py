"""
射线追踪运行时场景构建
"""

from dataclasses import dataclass
from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ...params.rt_scene import RtSceneParams
from ....channel.rt.rt_scene import RTScene


@dataclass
class RTSceneComponents:
    """射线追踪场景组件"""

    rt_scene: Optional[RTScene] = None

    @classmethod
    def build_from_params(
        cls,
        rt_scene_params: Optional[RtSceneParams],
        carrier_frequency: float,
        resource_grid: Optional[ResourceGrid] = None,
    ) -> "RTSceneComponents":
        if rt_scene_params is None:
            return cls(rt_scene=None)
        scene = RTScene(scene_params=rt_scene_params)
        scene.frequency = float(carrier_frequency)
        if resource_grid is not None:
            scene.bandwidth = float(resource_grid.bandwidth)
        return cls(rt_scene=scene)
