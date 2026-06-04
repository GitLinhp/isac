"""
信道运行时组件构建（与 ``channel_params`` 对应）
"""

from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ...channel.channel import Channel
from ...channel.rt.rt_scene import RTScene


def build_channel(
    rg: ResourceGrid,
    rt_scene: Optional[RTScene],
) -> Channel:
    """构建 ``Channel``；``paths`` 与现有逻辑一致（``lambda: rt_scene.paths``）。"""
    return Channel(
        rg=rg,
        paths=lambda: rt_scene.paths,
    )
