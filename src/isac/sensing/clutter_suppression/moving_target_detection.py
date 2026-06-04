"""动目标检测（Moving Target Detection）：沿脉冲 / 符号维多普勒 FFT；``__call__`` 仅接受 ``torch.Tensor``。"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..sensing_performance import SensingPerformance
from ...utils import get_logger
from ...utils.windows import get_named_window_tensor_1d

logger = get_logger(__name__)


class MovingTargetDetection:
    """动目标检测：多普勒 FFT；默认 FFT 长度为 ``ResourceGrid.num_ofdm_symbols``。"""

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        carrier_frequency: float,
        num_filters: Optional[int] = None,
    ):
        self.sensing_performance = sensing_performance
        self.rg = sensing_performance.rg
        self.carrier_frequency = float(carrier_frequency)
        if num_filters is None:
            self.num_filters = int(self.rg.num_ofdm_symbols)
        else:
            self.num_filters = int(num_filters)

    def __call__(
        self,
        signal_data: torch.Tensor,
        axis: int = -1,
        window: Optional[str] = "hann",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """多普勒向 FFT；返回 ``(谱, 多普勒频率_Hz)``，二者与输入同 device。"""
        if not isinstance(signal_data, torch.Tensor):
            raise TypeError("signal_data 须为 torch.Tensor")
        dev = signal_data.device
        in_dtype = signal_data.dtype
        ndim = signal_data.ndim
        axis_norm = axis % ndim

        x = torch.movedim(signal_data.to(torch.complex128), axis_norm, -1)
        n_last = x.shape[-1]

        if window is not None:
            w = get_named_window_tensor_1d(
                window, n_last, device=dev, dtype=torch.float64, sym=True
            ).reshape((1,) * (x.ndim - 1) + (n_last,))
            x = x * w.to(torch.complex128)

        spec = torch.fft.fftshift(
            torch.fft.fft(x, n=self.num_filters, dim=-1),
            dim=-1,
        )
        spec = torch.movedim(spec, -1, axis_norm).to(in_dtype)

        doppler_resolution = self.sensing_performance.doppler_resolution
        freq_np = np.fft.fftshift(np.fft.fftfreq(self.num_filters, 1.0 / doppler_resolution))
        doppler_frequencies = torch.as_tensor(freq_np, device=dev, dtype=torch.float64)

        return spec, doppler_frequencies

    def plot_doppler_spectrum(
        self,
        doppler_spectrum: torch.Tensor,
        doppler_frequencies: torch.Tensor,
        show: bool = True,
        save_path: Optional[str] = None,
        title: str = "MTD多普勒频谱",
    ) -> None:
        spectrum_np = doppler_spectrum.detach().cpu().numpy()
        freq_np = doppler_frequencies.detach().cpu().numpy()

        power_spectrum = 20 * np.log10(np.abs(spectrum_np) + 1e-10)
        if spectrum_np.ndim > 1:
            power_spectrum = np.mean(power_spectrum, axis=tuple(range(spectrum_np.ndim - 1)))

        plt.figure(figsize=(10, 6))
        plt.plot(freq_np, power_spectrum)
        plt.xlabel("多普勒频率 (Hz)")
        plt.ylabel("功率 (dB)")
        plt.title(title)
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("MTD多普勒频谱图已保存到: %s", save_path)
        if show:
            plt.show()
        else:
            plt.close()
