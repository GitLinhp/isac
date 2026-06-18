"""
系统参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .channel import ChannelParams
from .ofdm import OFDMParams
from .rt_scene import RtSceneParams
from .sensing import SensingParams
from .static_target import StaticTargetConfig


@dataclass
class SystemParams:
    """系统配置（五表聚合）。"""

    ofdm: OFDMParams = field(default_factory=OFDMParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    sensing: SensingParams = field(default_factory=SensingParams)
    rt_scene: Optional[RtSceneParams] = None
    static_target: Optional[StaticTargetConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        """从配置字典创建系统配置对象。"""
        rt_scene_cfg = config_dict.get("rt_scene")
        static_target_cfg = config_dict.get("static_target")
        ofdm = OFDMParams.from_dict(config_dict.get("ofdm", {}))
        channel = ChannelParams.from_dict(config_dict.get("channel", {}))

        rt_scene: Optional[RtSceneParams] = None
        if isinstance(rt_scene_cfg, dict) and rt_scene_cfg:
            rt_scene = RtSceneParams.from_dict(rt_scene_cfg)

        static_target: Optional[StaticTargetConfig] = None
        if isinstance(static_target_cfg, dict) and static_target_cfg:
            static_target = StaticTargetConfig.from_dict(static_target_cfg)

        params = cls(
            ofdm=ofdm,
            channel=channel,
            sensing=SensingParams.from_dict(config_dict.get("sensing") or {}),
            rt_scene=rt_scene,
            static_target=static_target,
        )
        params._validate_channel_dependencies()
        return params

    def _validate_channel_dependencies(self) -> None:
        if self.channel.type == "rt" and self.rt_scene is None:
            raise ValueError("channel.type='rt' 要求配置 [rt_scene]")
        if self.channel.type == "rcs" and self.static_target is None:
            raise ValueError("channel.type='rcs' 要求配置 [static_target]")
