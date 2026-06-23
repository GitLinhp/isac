"""
系统参数数据结构和配置类
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .basic_params import OFDMParams, SourceParams, StreamManagementParams
from .channel_params import ChannelParams, RtSceneParams, RCSSceneParams
from .sensing_params import (
    CFARParams,
    MTDParams,
    MTIParams,
    MusicParams,
    WindowParams,
)


@dataclass
class SystemParams:
    """系统配置（嵌套 Params，顺序对齐 system_components）。"""

    carrier_frequency: Optional[float] = None
    """载波频率"""
    source: Optional[SourceParams] = None
    """信源"""
    stream_management: Optional[StreamManagementParams] = None
    """流管理"""
    ofdm: Optional[OFDMParams] = None
    """OFDM"""

    channel: Optional[ChannelParams] = None
    """信道"""
    rt_scene: Optional[RtSceneParams] = None
    """射线追踪场景"""
    rcs_scene: Optional[RCSSceneParams] = None
    """RCS 点目标场景"""

    mti: Optional[MTIParams] = None
    """动目标显示"""
    mtd: Optional[MTDParams] = None
    """动目标检测"""
    windows: Optional[WindowParams] = None
    """时延 / 多普勒窗"""
    cfar: Optional[CFARParams] = None
    """CFAR 检测"""
    music: Optional[MusicParams] = None
    """MUSIC 谱估计"""

    @staticmethod
    def _parse_section(
        config_dict: Dict[str, Any],
        key: str,
        parser: Callable[[Dict[str, Any]], Any],
    ) -> Optional[Any]:
        if key not in config_dict:
            return None
        raw = config_dict[key]
        if not isinstance(raw, dict):
            raise TypeError(f"{key} 须为表(dict)，收到 {type(raw)!r}")
        if not raw:
            return None
        return parser(raw)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        carrier_frequency = (
            float(config_dict["carrier_frequency"])
            if "carrier_frequency" in config_dict
            else None
        )
        source = cls._parse_section(config_dict, "source", SourceParams.from_dict)
        stream_management = cls._parse_section(
            config_dict, "stream_management", StreamManagementParams.from_dict
        )
        ofdm = cls._parse_section(config_dict, "ofdm", OFDMParams.from_dict)

        channel = cls._parse_section(config_dict, "channel", ChannelParams.from_dict)
        rt_scene = cls._parse_section(config_dict, "rt_scene", RtSceneParams.from_dict)
        rcs_scene = cls._parse_section(
            config_dict, "rcs_scene", RCSSceneParams.from_dict
        )

        mti = cls._parse_section(config_dict, "mti", MTIParams.from_dict)
        mtd = cls._parse_section(config_dict, "mtd", MTDParams.from_dict)
        windows = cls._parse_section(config_dict, "windows", WindowParams.from_dict)
        cfar = cls._parse_section(config_dict, "cfar", CFARParams.from_dict)
        music = cls._parse_section(config_dict, "music", MusicParams.from_dict)

        params = cls(
            carrier_frequency=carrier_frequency,
            source=source,
            stream_management=stream_management,
            ofdm=ofdm,
            channel=channel,
            rt_scene=rt_scene,
            rcs_scene=rcs_scene,
            mti=mti,
            mtd=mtd,
            windows=windows,
            cfar=cfar,
            music=music,
        )
        params._validate_channel_dependencies()
        return params

    def _validate_channel_dependencies(self) -> None:
        if self.channel is None:
            return
        if self.channel.type == "rt" and self.rt_scene is None:
            raise ValueError("channel.type='rt' 要求配置 [rt_scene]")
        if self.channel.type == "rcs" and self.rcs_scene is None:
            raise ValueError("channel.type='rcs' 要求配置 [rcs_scene]")
