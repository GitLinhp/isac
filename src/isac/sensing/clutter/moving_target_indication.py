"""动目标显示（Moving Target Indication）：脉冲对消，``__call__`` 仅接受 ``torch.Tensor``。"""

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..spectrum.sensing_performance import SensingPerformance


class MovingTargetIndication:
    """动目标显示：``order`` 阶脉冲对消，系数为 ``(1 - z^{-1})^{order}``；PRF 由 ``ResourceGrid.ofdm_symbol_duration`` 推导。"""

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        filter_order: int = 1,
        prf: Optional[float] = None,
    ):
        self.sensing_performance = sensing_performance
        self.rg = sensing_performance.rg
        self.filter_order = int(filter_order)

        if prf is None:
            self.prf = 1.0 / float(self.rg.ofdm_symbol_duration)
        else:
            self.prf = float(prf)

        self.filter_coefficients = self._coefficients_from_order(self.filter_order)

    @staticmethod
    def _coefficients_from_order(order: int) -> np.ndarray:
        """``(1 - z^{-1})^{order}`` 的 FIR 分子系数 ``b[k] = (-1)^k * C(order, k)``，``k=0…order``。"""
        n = int(order)
        if n < 1:
            raise ValueError(f"filter_order 须为 >= 1 的整数，收到: {order!r}")
        if n > 64:
            raise ValueError(f"filter_order 过大（>64），请减小阶数以控制计算量")
        return np.array(
            [((-1) ** k) * math.comb(n, k) for k in range(n + 1)],
            dtype=np.float64,
        )

    @staticmethod
    def _fir_last_dim(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """因果 FIR：``y[n] = sum_k b[k] * x[n-k]``（``n-k<0`` 视为 0），与 ``scipy.signal.lfilter(b, [1], x)`` 零初态沿最后一维一致。"""
        b = b.flatten()
        L = int(b.numel())
        *lead, t = x.shape
        if L == 0:
            return torch.zeros_like(x)
        if t == 0:
            return x
        x2 = x.reshape(-1, t)
        b = b.to(device=x2.device, dtype=x2.dtype)
        xpad = F.pad(x2, (L - 1, 0))
        windows = xpad.unfold(1, L, 1).flip(-1)
        y2 = (windows * b).sum(dim=-1)
        return y2.reshape(*lead, t)

    def __call__(
        self,
        signal_data: torch.Tensor,
        axis: Optional[int] = None,
    ) -> torch.Tensor:
        """沿慢时间/符号维应用 MTI；仅支持 ``torch.Tensor``。

        输入形状为 ``(num_ofdm_symbols, fft_size)`` 或
        ``(rx_num, num_ofdm_symbols, fft_size)``。``axis=None`` 时自动选符号维：
        2D 用 ``0``，3D 用 ``-2``。
        """
        if not isinstance(signal_data, torch.Tensor):
            raise TypeError("signal_data 须为 torch.Tensor")
        ndim = signal_data.ndim
        if ndim not in (2, 3):
            raise ValueError(
                f"signal_data 须为 2D (S,F) 或 3D (rx_num,S,F)，收到 ndim={ndim}"
            )
        s, f = self.rg.num_ofdm_symbols, self.rg.fft_size
        if signal_data.shape[-2:] != (s, f):
            raise ValueError(
                f"signal_data 末两维须为 ({s}, {f})，收到 {tuple(signal_data.shape)}"
            )
        if axis is None:
            axis = -2 if ndim == 3 else 0
        axis_norm = axis % ndim
        x = torch.movedim(signal_data, axis_norm, -1)
        b = torch.as_tensor(self.filter_coefficients, device=x.device, dtype=x.dtype)
        y = self._fir_last_dim(x, b)
        return torch.movedim(y, -1, axis_norm)

    def frequency_response(
        self,
        num_points: int = 1024,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIR 频率响应 ``H(e^{j\\omega})``，与 ``scipy.signal.freqz(b, a=[1], worN=...)`` 在 ``\\omega\\in[0,\\pi)`` 上等价。

        返回
        -----
        frequencies_hz
            物理频率 (Hz)，``\\omega * prf / (2\\pi)``。
        magnitude
            线性幅度 ``|H|``。
        """
        dev = device if device is not None else torch.device("cpu")
        # 与 ``scipy.signal.freqz(..., worN=num_points)`` 的 ``\\omega`` 采样一致：``[0, \\pi)`` 上均匀 ``num_points`` 点。
        omega = torch.arange(num_points, device=dev, dtype=dtype) * (
            torch.pi / num_points
        )
        b = torch.as_tensor(self.filter_coefficients, device=dev, dtype=dtype)
        k = torch.arange(b.numel(), device=dev, dtype=dtype)
        om = omega.unsqueeze(1)
        kk = k.unsqueeze(0)
        phase = -(om * kk)
        z = torch.polar(torch.ones_like(phase), phase)
        b_c = b.to(z.dtype)
        H = (z * b_c.unsqueeze(0)).sum(dim=1)
        magnitude = torch.abs(H).to(dtype)
        frequencies_hz = omega * (self.prf / (2.0 * torch.pi))
        return frequencies_hz, magnitude

    def plot_frequency_response(
        self,
        num_points: int = 1024,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        frequencies, magnitude = self.frequency_response(num_points)
        freq_np = frequencies.detach().cpu().numpy()
        mag_db = (20 * torch.log10(magnitude + 1e-10)).detach().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(freq_np, mag_db)
        plt.xlabel("频率 (Hz)")
        plt.ylabel("幅度 (dB)")
        plt.title(f"MTI滤波器频率响应 (order={self.filter_order})")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"MTI频率响应图已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
