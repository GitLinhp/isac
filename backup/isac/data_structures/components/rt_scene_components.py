"""
射线追踪运行时场景构建（与 ``rt_scene_params`` 对应）
"""

from dataclasses import dataclass
from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ..params import SystemParams
from ...channel.rt.rt_scene import RTScene


@dataclass
class RTSceneComponents:
    """射线追踪场景组件"""

    rt_scene: Optional[RTScene] = None

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        resource_grid: Optional[ResourceGrid] = None,
    ) -> "RTSceneComponents":
        """由 ``SystemParams.channel.rt_scene`` 构造 ``RTScene``；未配置时 ``rt_scene`` 为 ``None``。

        Sionna RT 的路径多普勒使用 ``Scene.frequency``（及 ``λ=c/f``）；必须与 TOML 中的
        ``carrier_frequency`` 一致，否则仅改配置载频时仿真多普勒频移不会随之变化。
        若提供 ``resource_grid``，同时将 ``scene.bandwidth`` 与 OFDM 带宽对齐。
        """
        p = system_params.channel.rt_scene
        if p is None:
            return cls(rt_scene=None)
        scene = RTScene(scene_params=p)
        scene.frequency = float(system_params.carrier_frequency)
        if resource_grid is not None:
            scene.bandwidth = float(resource_grid.bandwidth)
        return cls(rt_scene=scene)
