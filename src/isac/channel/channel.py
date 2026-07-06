"""ISAC 信道抽象基类：子类实现无噪施加，基类统一 AWGN。"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from sionna.phy.config import Precision

from .awgn import AWGN


class Channel(ABC):
    """ISAC 信道基类：``_apply_channel`` 施加无噪信道，``__call__`` 可选 AWGN。"""

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        self._awgn = AWGN(precision=precision, device=device)

    def __call__(
        self,
        x_rg: torch.Tensor,
        x_time: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """按 ``domain`` 选择 ``x_rg``（frequency）或 ``x_time``（time）经信道；``snr_db`` 为数值时加 AWGN。"""
        if domain == "frequency":
            inputs = x_rg
        elif domain == "time":
            inputs = x_time
        else:
            raise ValueError(f"不支持的域: {domain}。支持的值: 'time', 'frequency'")
        y_clean = self._apply_channel(inputs, domain)
        if snr_db is None:
            return y_clean
        return self._awgn(y_clean, snr_db)

    @abstractmethod
    def _apply_channel(self, inputs: torch.Tensor, domain: str) -> torch.Tensor:
        """施加无噪信道（子类实现）。"""
