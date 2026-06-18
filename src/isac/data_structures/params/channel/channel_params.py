"""
信道施加相关参数
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal


@dataclass
class ChannelParams:
    """信道配置：类型、信噪比与码率。"""

    type: Literal["rt", "rcs"] = "rt"
    """``rt``：Sionna 射线追踪；``rcs``：静态点目标散射。"""
    snr_db: float = 10.0
    """接收端信噪比 (dB)"""
    coderate: float = 1.0
    """码率（无 LDPC 时取 1）"""

    def __post_init__(self) -> None:
        if self.type not in ("rt", "rcs"):
            raise ValueError("channel.type must be 'rt' or 'rcs'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChannelParams":
        if not isinstance(config_dict, dict):
            config_dict = {}
        raw_type = config_dict.get("type", "rt")
        if not isinstance(raw_type, str):
            raise ValueError(f"channel.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            snr_db=float(config_dict.get("snr_db", 10.0)),
            coderate=float(config_dict.get("coderate", 1.0)),
        )
