"""
信道运行时组件构建（与 ``channel_params`` 对应）
"""

from dataclasses import dataclass
from typing import Optional

from sionna.phy.ofdm import ResourceGrid

from ...channel.channel import Channel
from ...channel.rt.rt_scene import RTScene


@dataclass
class ChannelComponents:
    """信道组件"""

    channel: Channel

    @classmethod
    def build_from_params(
        cls,
        rg: ResourceGrid,
        rt_scene: Optional[RTScene],
    ) -> "ChannelComponents":
        """构建 ``Channel``；``paths`` 与现有逻辑一致（``lambda: rt_scene.paths``）。"""
        return cls(
            channel=Channel(
                rg=rg,
                paths=lambda: rt_scene.paths,
            )
        )
