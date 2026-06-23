"""信道类型与 SNR 配置。"""

from dataclasses import dataclass
from typing import Any, Dict, Literal


@dataclass
class ChannelParams:
    """信道配置"""

    type: Literal["rt", "rcs"] = "rt"
    snr_db: float = 10.0

    def __post_init__(self) -> None:
        if self.type not in ("rt", "rcs"):
            raise ValueError("channel.type must be 'rt' or 'rcs'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChannelParams":
        raw_type = config_dict.get("type", "rt")
        if not isinstance(raw_type, str):
            raise ValueError(f"channel.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            snr_db=float(config_dict.get("snr_db", 10.0)),
        )
