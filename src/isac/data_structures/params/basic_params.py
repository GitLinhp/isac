"""基础系统参数：信源、流管理与 OFDM 网格。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class SourceParams:
    """OFDM 信源配置（binary / ZC）"""

    type: Literal["binary", "zc"] = "binary"
    root_index: int = 1
    normalize: bool = True
    num_bits_per_symbol: Optional[int] = None
    cache_file: Optional[str] = None
    """发射波形缓存路径（相对 PROJECT_ROOT 或绝对路径）；None 表示不缓存"""

    def __post_init__(self) -> None:
        if self.type not in ("binary", "zc"):
            raise ValueError("source.type must be 'binary' or 'zc'")
        # ZC 占位比特形状对齐 QPSK：未配置时默认 2
        if self.type == "zc" and self.num_bits_per_symbol is None:
            self.num_bits_per_symbol = 2

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SourceParams":
        raw_type = config_dict.get("type", "binary")
        if not isinstance(raw_type, str):
            raise ValueError(f"source.type must be a string, got {type(raw_type)!r}")
        n_bps_raw = config_dict.get("num_bits_per_symbol")
        num_bits_per_symbol = int(n_bps_raw) if n_bps_raw is not None else None
        raw_cache = config_dict.get("cache_file")
        if raw_cache is None or (isinstance(raw_cache, str) and not raw_cache.strip()):
            cache_file = None
        else:
            cache_file = str(raw_cache).strip()
        return cls(
            type=raw_type.strip().lower(),
            root_index=int(config_dict.get("root_index", 1)),
            normalize=bool(config_dict.get("normalize", True)),
            num_bits_per_symbol=num_bits_per_symbol,
            cache_file=cache_file,
        )


@dataclass
class StreamManagementParams:
    """资源网格解映射流管理配置"""

    rx_tx_association: list = field(default_factory=lambda: [[1]])
    num_streams: int = 1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StreamManagementParams":
        assoc = config_dict.get("rx_tx_association", [[1]])
        return cls(
            rx_tx_association=assoc,
            num_streams=int(config_dict.get("num_streams", 1)),
        )


@dataclass
class OFDMParams:
    """OFDM 网格与调制参数"""

    num_symbols: int = 512
    fft_size: int = 2048
    subcarrier_spacing: float = 30000.0
    cyclic_prefix_length: int = 0
    l_min: int = -6
    dc_null: bool = True

    @property
    def samp_rate(self) -> int:
        return int(self.subcarrier_spacing * self.fft_size)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        raw_l_min = config_dict.get("l_min", -6)
        l_min = int(-6 if raw_l_min is None else raw_l_min)
        return cls(
            num_symbols=int(config_dict.get("num_symbols", 1024)),
            fft_size=int(config_dict.get("fft_size", 1024)),
            subcarrier_spacing=float(config_dict.get("subcarrier_spacing", 30000.0)),
            cyclic_prefix_length=int(config_dict.get("cyclic_prefix_length", 0)),
            l_min=l_min,
            dc_null=bool(config_dict.get("dc_null", True)),
        )
