"""OFDM 频域 LS 信道估计。"""

import torch
from sionna.phy.ofdm import ResourceGrid


class LSChannelEstimator:
    """OFDM 频域 LS 信道估计：``h = y * conj(x) / (|x|^2 + eps)``。"""

    def __init__(self, rg: ResourceGrid) -> None:
        self.rg = rg

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        """估计频域信道（LS）。

        对 ``x``/``y`` 做 ``squeeze`` 后，``x`` 为 ``(num_ofdm_symbols, fft_size)``；
        ``y`` 为 ``(rx_num, num_ofdm_symbols, fft_size)``，``rx_num=1`` 时退化为 2D。
        假定 ``batch_size=1``；squeeze 后 ``y.ndim > 3`` 将报错。
        """
        s, f = self.rg.num_ofdm_symbols, self.rg.fft_size

        x = x.squeeze()
        y = y.squeeze()

        if x.shape[-2:] != (s, f):
            raise ValueError(f"x 末两维须为 ({s}, {f})，收到 {tuple(x.shape)}")
        x = x.reshape(s, f)

        if y.shape[-2:] != (s, f):
            raise ValueError(f"y 末两维须为 ({s}, {f})，收到 {tuple(y.shape)}")
        if y.ndim == 2:
            y = y.reshape(s, f)
        elif y.ndim == 3:
            y = y.reshape(y.shape[0], s, f)
            if y.shape[0] == 1:
                y = y.squeeze(0)
        else:
            raise ValueError(
                f"y squeeze 后须为 2D (S,F) 或 3D (rx_num,S,F)，收到 ndim={y.ndim}"
            )

        denom = torch.abs(x) ** 2 + eps
        h = y * torch.conj(x) / denom

        return h
