"""感知处理参数：MTI/MTD、窗函数、CFAR 与 MUSIC。"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class DelayDopplerRoiParams:
    """时延–多普勒谱 ROI（物理量语义）。"""

    max_range_m: float
    max_velocity_mps: float

    def __post_init__(self) -> None:
        if self.max_range_m <= 0:
            raise ValueError(
                f"dd_spectrum_roi.max_range_m 须为正，收到 {self.max_range_m}"
            )
        if self.max_velocity_mps <= 0:
            raise ValueError(
                "dd_spectrum_roi.max_velocity_mps 须为正，"
                f"收到 {self.max_velocity_mps}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DelayDopplerRoiParams":
        return cls(
            max_range_m=float(config_dict.get("max_range_m", 50.0)),
            max_velocity_mps=float(config_dict.get("max_velocity_mps", 10.0)),
        )


@dataclass
class MTIParams:
    """动目标显示（MTI）配置"""

    filter_order: int = 1
    prf: Optional[float] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MTIParams":
        prf_raw = config_dict.get("prf", None)
        return cls(
            filter_order=int(config_dict.get("filter_order", 1)),
            prf=float(prf_raw) if prf_raw is not None else None,
        )


@dataclass
class MTDParams:
    """动目标检测（MTD）配置"""

    num_filters: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MTDParams":
        nf = config_dict.get("num_filters", None)
        return cls(
            num_filters=int(nf) if nf is not None else None,
        )


@dataclass
class WindowParams:
    """时延 / 多普勒窗配置"""

    delay_window: Optional[Union[str, Dict[str, Any]]] = None
    doppler_window: Optional[Union[str, Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WindowParams":
        return cls(
            delay_window=config_dict.get("delay_window"),
            doppler_window=config_dict.get("doppler_window"),
        )


@dataclass
class CFARParams:
    """CFAR 检测配置"""

    type: str = "ca"
    k: Optional[int] = None
    guard: Union[int, list[int]] = 2
    trailing: Union[int, list[int]] = 20
    pfa: float = 1e-4
    detector: str = "linear"
    offset: Optional[float] = None

    def __post_init__(self) -> None:
        t = self.type.strip().lower()
        if t not in ("ca", "os"):
            raise ValueError("cfar.type must be 'ca' or 'os'")
        self.type = t
        if t == "os" and self.k is None:
            raise ValueError("cfar: type 'os' requires integer 'k' in config")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CFARParams":
        raw_type = config_dict.get("type", "ca")
        if not isinstance(raw_type, str):
            raise ValueError(f"cfar.type must be a string, got {type(raw_type)!r}")
        k_raw = config_dict.get("k", None)
        cfar_k: Optional[int] = int(k_raw) if k_raw is not None else None
        return cls(
            type=raw_type.strip().lower(),
            k=cfar_k,
            guard=config_dict.get("guard", 2),
            trailing=config_dict.get("trailing", 20),
            pfa=config_dict.get("pfa", 1e-4),
            detector=config_dict.get("detector", "linear"),
            offset=config_dict.get("offset", None),
        )


@dataclass
class MusicParams:
    """MUSIC 估计器默认调用参数"""

    threshold: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicParams":
        return cls(
            threshold=float(config_dict.get("threshold", 0.1)),
        )
