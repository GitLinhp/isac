"""
系统参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .channel_params import ChannelParams
from .ofdm_params import OFDMParams
from .qam_params import QAMParams
from .sensing_params import SensingParams


@dataclass
class SystemParams:
    """系统配置"""

    carrier_frequency: float = 2.6e9
    """载波频率(Hz)"""
    qam: QAMParams = field(default_factory=QAMParams)
    """QAM 调制配置"""
    ofdm: OFDMParams = field(default_factory=OFDMParams)
    """OFDM配置"""
    channel: ChannelParams = field(default_factory=ChannelParams)
    """信道配置（SNR、射线追踪场景等）"""
    sensing: SensingParams = field(default_factory=SensingParams)
    """感知配置（含时延/多普勒窗等）"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        """从配置字典创建系统配置对象"""
        return cls(
            carrier_frequency=config_dict.get("carrier_frequency", 2.6e9),
            qam=QAMParams.from_dict(config_dict),
            ofdm=OFDMParams.from_dict(config_dict.get("ofdm", {})),
            channel=ChannelParams.from_dict(config_dict),
            sensing=SensingParams.from_dict(config_dict.get("sensing") or {}),
        )
