"""
感知参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class SensingWindowsParams:
    """感知流程中的窗配置。"""

    delay_window: Optional[Union[str, Dict[str, Any]]] = None
    doppler_window: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class SensingCFARParams:
    """感知中的 CFAR 检测配置。"""

    cfar_type: str = "ca"
    k: Optional[int] = None
    guard: Union[int, list[int]] = 2
    trailing: Union[int, list[int]] = 20
    pfa: float = 1e-4
    detector: str = "linear"
    offset: Optional[float] = None

    def __post_init__(self) -> None:
        t = self.cfar_type.strip().lower()
        if t not in ("ca", "os"):
            raise ValueError("sensing.cfar.type must be 'ca' or 'os'")
        object.__setattr__(self, "cfar_type", t)
        if t == "os" and self.k is None:
            raise ValueError("sensing.cfar: type 'os' requires integer 'k' in config")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SensingCFARParams":
        raw_type = config_dict.get("type", "ca")
        if not isinstance(raw_type, str):
            raise ValueError(f"sensing.cfar.type must be a string, got {type(raw_type)!r}")
        cfar_type = raw_type.strip().lower()
        k_raw = config_dict.get("k", None)
        k: Optional[int]
        if k_raw is None:
            k = None
        else:
            k = int(k_raw)
        return cls(
            cfar_type=cfar_type,
            k=k,
            guard=config_dict.get("guard", 2),
            trailing=config_dict.get("trailing", 20),
            pfa=config_dict.get("pfa", 1e-4),
            detector=config_dict.get("detector", "linear"),
            offset=config_dict.get("offset", None),
        )


@dataclass
class SensingParams:
    """感知相关顶层配置。"""

    windows: SensingWindowsParams = field(default_factory=SensingWindowsParams)
    cfar: SensingCFARParams = field(default_factory=SensingCFARParams)

    @classmethod
    def from_dict(cls, sensing_dict: Dict[str, Any]) -> "SensingParams":
        if not isinstance(sensing_dict, dict):
            sensing_dict = {}
        w = sensing_dict.get("windows")
        if not isinstance(w, dict):
            w = {}
        delay = w.get("delay_window")
        doppler = w.get("doppler_window")
        cfar_dict = sensing_dict.get("cfar")
        return cls(
            windows=SensingWindowsParams(delay_window=delay, doppler_window=doppler),
            cfar=SensingCFARParams.from_dict(cfar_dict if isinstance(cfar_dict, dict) else {}),
        )
