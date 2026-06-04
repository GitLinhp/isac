"""
感知参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union


@dataclass
class SensingWindowsParams:
    """感知流程中的窗配置（原始 TOML 结构，在 DelayDopplerSpectrum 内解析为 WindowSpec）。

    ``delay_window`` / ``doppler_window`` 为 ``None`` 时不加窗（TOML 未配置或整段注释时）。
    """

    delay_window: Optional[Union[str, Dict[str, Any]]] = None
    doppler_window: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class SensingCFARParams:
    """感知中的 CFAR 检测配置。"""

    cfar_type: str = "ca"
    """``ca``：CA-CFAR（``cfar_ca_2d``）；``os``：OS-CFAR（``cfar_os_2d``），需配置 ``k``。"""
    k: Optional[int] = None
    """OS-CFAR 有序统计索引；仅 ``cfar_type=os`` 时必填。"""
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
class SensingSourceParams:
    """发射参考信号类型（与 ``[sensing.source]`` 对应；TOML 键 ``type`` 与字段同名）。"""

    type: Literal["binary", "zc"] = "binary"
    """``binary``：比特流 + QAM；``zc``：Zadoff-Chu（不经 Mapper）。"""
    root_index: int = 1
    """ZC 根索引 ``u``（须满足 ``gcd(u, N)=1``，``N`` 为 ``ResourceGrid.num_data_symbols``）。"""
    normalize: bool = True
    """ZC 是否按 ``sqrt(N)`` 归一化。"""

    def __post_init__(self) -> None:
        k = self.type
        if k not in ("binary", "zc"):
            raise ValueError("sensing.source.type must be 'binary' or 'zc'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SensingSourceParams":
        if not isinstance(config_dict, dict):
            config_dict = {}
        raw_type = config_dict.get("type", "binary")
        if not isinstance(raw_type, str):
            raise ValueError(f"sensing.source.type must be a string, got {type(raw_type)!r}")
        src_type = raw_type.strip().lower()
        ri = config_dict.get("root_index", 1)
        norm = config_dict.get("normalize", True)
        return cls(
            type=src_type,
            root_index=int(ri),
            normalize=bool(norm),
        )


@dataclass
class SensingParams:
    """感知相关顶层配置。"""

    windows: SensingWindowsParams = field(default_factory=SensingWindowsParams)
    cfar: "SensingCFARParams" = field(default_factory=lambda: SensingCFARParams())
    source: SensingSourceParams = field(default_factory=SensingSourceParams)

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
        src_dict = sensing_dict.get("source")
        return cls(
            windows=SensingWindowsParams(delay_window=delay, doppler_window=doppler),
            cfar=SensingCFARParams.from_dict(cfar_dict if isinstance(cfar_dict, dict) else {}),
            source=SensingSourceParams.from_dict(src_dict if isinstance(src_dict, dict) else {}),
        )
