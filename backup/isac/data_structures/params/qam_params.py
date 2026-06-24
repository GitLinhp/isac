"""
QAM 调制相关参数
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class QAMParams:
    """QAM 调制配置（比特/符号与阶数）。"""

    num_bits_per_symbol: int = 2
    """每个符号的比特数"""
    qam_order: int = 4
    """QAM 阶数（典型为 ``2 ** num_bits_per_symbol``）"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QAMParams":
        """从根级配置读取 ``num_bits_per_symbol``；``qam_order`` 与原先逻辑一致为 ``n ** 2``。"""
        n = config_dict.get("num_bits_per_symbol", 2)
        return cls(num_bits_per_symbol=n, qam_order=n**2)
