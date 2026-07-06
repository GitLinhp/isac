"""OFDM 频域 LS 信道估计。"""

import torch
from sionna.phy.ofdm import ResourceGrid


class LSChannelEstimator:
    """OFDM 频域 LS 信道估计：``h = y * conj(x) / (|x|^2 + eps)``。"""

    def __init__(self, rg: ResourceGrid) -> None:
        self.rg = rg

    def _normalize_grid(self, t: torch.Tensor, *, name: str) -> torch.Tensor:
        """
        ``x``/``y`` 经相同维度检测：squeeze 后末两维为 ``(S, F)``；
        允许 2D ``(S, F)`` 或 3D ``(N, S, F)``（``N=1`` 时退化为 2D）。
        squeeze 后 ``ndim > 3`` 将报错。
        """
        s, f = self.rg.num_ofdm_symbols, self.rg.fft_size
        t = t.squeeze()

        if t.shape[-2:] != (s, f):
            raise ValueError(f"{name} 末两维须为 ({s}, {f})，收到 {tuple(t.shape)}")
        if t.ndim == 2:
            return t.reshape(s, f)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], s, f)
            if t.shape[0] == 1:
                return t.squeeze(0)
            return t
        raise ValueError(
            f"{name} squeeze 后须为 2D (S,F) 或 3D (N,S,F)，收到 ndim={t.ndim}"
        )

    def __call__(
        self, x_rg: torch.Tensor, y_rg: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        """估计频域信道（LS）。

        ``x_rg``/``y_rg`` 经相同维度检测：squeeze 后末两维为 ``(S, F)``；
        允许 2D ``(S, F)`` 或 3D ``(N, S, F)``（``N=1`` 时退化为 2D）。
        squeeze 后 ``ndim > 3`` 将报错。

        参数:
        ----------
        - x_rg : torch.Tensor
            - 发射频域资源网格
        - y_rg : torch.Tensor
            - 接收频域资源网格
        - eps : float
            - 平滑因子

        返回:
        -------
        - h_freq : torch.Tensor
            - 频域信道响应，形状为 ``(num_ofdm_symbols, fft_size)`` 或
            ``(rx_num, num_ofdm_symbols, fft_size)``。
        """
        x_rg = self._normalize_grid(x_rg, name="x_rg")
        y_rg = self._normalize_grid(y_rg, name="y_rg")

        denom = torch.abs(x_rg) ** 2 + eps
        h_freq = y_rg * torch.conj(x_rg) / denom

        return h_freq
