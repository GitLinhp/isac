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
        inputs: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """经信道；``snr_db`` 为数值时加 AWGN，默认 ``None`` 不加噪。"""
        y_clean = self._apply_channel(inputs, domain)
        if snr_db is None:
            return y_clean
        else:
            return self._awgn(y_clean, snr_db)

    @abstractmethod
    def _apply_channel(self, inputs: torch.Tensor, domain: str) -> torch.Tensor:
        """施加无噪信道（子类实现）。"""
