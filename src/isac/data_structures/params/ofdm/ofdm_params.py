"""
OFDM 与调制相关参数
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class OFDMSourceParams:
    """发射参考信号类型（与 ``[ofdm.source]`` 对应）。"""

    type: Literal["binary", "zc"] = "binary"
    """``binary``：比特流 + QAM；``zc``：Zadoff-Chu（不经 Mapper）。"""
    root_index: int = 1
    """ZC 根索引 ``u``（须满足 ``gcd(u, N)=1``）。"""
    normalize: bool = True
    """ZC 是否按 ``sqrt(N)`` 归一化。"""

    def __post_init__(self) -> None:
        if self.type not in ("binary", "zc"):
            raise ValueError("ofdm.source.type must be 'binary' or 'zc'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMSourceParams":
        if not isinstance(config_dict, dict):
            config_dict = {}
        raw_type = config_dict.get("type", "binary")
        if not isinstance(raw_type, str):
            raise ValueError(f"ofdm.source.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            root_index=int(config_dict.get("root_index", 1)),
            normalize=bool(config_dict.get("normalize", True)),
        )


@dataclass
class OFDMParams:
    """OFDM 与载波配置。"""

    carrier_frequency: float = 2.6e9
    """载波频率 (Hz)"""
    num_bits_per_symbol: int = 2
    """每个符号的比特数"""
    num_symbols: int = 1024
    """符号数"""
    num_subcarriers: int = 1024
    """子载波数"""
    num_valid_subcarriers: int = 1024
    """有效子载波数"""
    subcarrier_spacing: float = 30000.0
    """子载波间隔 (Hz)"""
    cyclic_prefix_length: int = 0
    """循环前缀长度"""
    l_min: int = -6
    """OFDM 解调器最小时延抽头索引"""
    source: OFDMSourceParams = field(default_factory=OFDMSourceParams)
    """发射参考源配置"""

    @property
    def qam_order(self) -> int:
        return 2 ** self.num_bits_per_symbol

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        if not isinstance(config_dict, dict):
            config_dict = {}
        raw_l_min = config_dict.get("l_min", -6)
        l_min = int(-6 if raw_l_min is None else raw_l_min)
        cp = config_dict.get(
            "cyclic_prefix_length",
            config_dict.get("num_cyclic_prefix", 0),
        )
        src_dict = config_dict.get("source")
        return cls(
            carrier_frequency=float(config_dict.get("carrier_frequency", 2.6e9)),
            num_bits_per_symbol=int(config_dict.get("num_bits_per_symbol", 2)),
            num_symbols=config_dict.get("num_symbols", 1024),
            num_subcarriers=config_dict.get("num_subcarriers", 1024),
            num_valid_subcarriers=config_dict.get("num_valid_subcarriers", 1024),
            subcarrier_spacing=config_dict.get("subcarrier_spacing", 30000.0),
            cyclic_prefix_length=int(cp),
            l_min=l_min,
            source=OFDMSourceParams.from_dict(
                src_dict if isinstance(src_dict, dict) else {}
            ),
        )
