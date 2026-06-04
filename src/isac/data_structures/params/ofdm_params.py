"""
OFDM 参数数据结构和配置类
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OFDMParams:
    """OFDM配置"""

    num_symbols: int = 1024
    """符号数"""
    num_subcarriers: int = 1024
    """子载波数"""
    num_valid_subcarriers: int = 1024
    """有效子载波数"""
    subcarrier_spacing: float = 30000.0
    """子载波间隔(Hz)"""
    cyclic_prefix_length: int = 0
    """循环前缀长度"""
    l_min: int = -6
    """OFDM 解调器最小时延抽头索引（Sionna ``OFDMDemodulator``）。"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        """从字典创建配置对象"""
        raw_l_min = config_dict.get("l_min", -6)
        l_min = int(-6 if raw_l_min is None else raw_l_min)
        return cls(
            num_symbols=config_dict.get("num_symbols", 1024),
            num_subcarriers=config_dict.get("num_subcarriers", 1024),
            num_valid_subcarriers=config_dict.get("num_valid_subcarriers", 1024),
            subcarrier_spacing=config_dict.get("subcarrier_spacing", 30000.0),
            cyclic_prefix_length=config_dict.get("cyclic_prefix_length", 0),
            l_min=l_min,
        )
