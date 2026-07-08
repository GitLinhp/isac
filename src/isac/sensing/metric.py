"""时延-多普勒 / 距离-速度 坐标换算（spectrum 与 detection 共用）。"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .geometry import delay_to_range, doppler_to_velocity
from .spectrum.sensing_performance import SensingPerformance
from ..data_structures.types import SensMode
from ..utils.type_converter import convert

ArrayLike = Union[np.ndarray, torch.Tensor, float, int]


class SpectrumMetric:
    """DD/RV 谱 bin 与物理量之间的统一换算。

    **坐标约定**

    - **全局 bin**：全 FFT 网格索引，``SensingPerformance.*_bins[k]`` 查表。
    - **局部 bin**：ROI 裁切后谱的 ``(delay, doppler)``；delay 从 0 起，
      doppler 以裁切后 ``doppler_center(num_doppler)`` 为零点。
    - 对称 ROI 下，局部 bin 换算结果等于全局轴在 ``roi_slices`` 对应位置的值。
    """

    def __init__(self, sensing_performance: SensingPerformance) -> None:
        self.sensing_performance = sensing_performance

    @staticmethod
    def doppler_center(num_doppler_bins: int) -> float:
        """多普勒 fftshift 网格中心（bin 坐标，可为半整数）。"""
        return num_doppler_bins / 2.0

    def local_bins_to_tau_fd(
        self,
        delay_bin: ArrayLike,
        doppler_bin: ArrayLike,
        *,
        num_doppler_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ROI 局部 bin → 时延 τ(s) 与多普勒 f_d(Hz)。"""
        sp = self.sensing_performance
        dt = float(sp.delay_resolution)
        dres = float(sp.doppler_resolution)
        center = self.doppler_center(num_doppler_bins)

        d = convert(
            delay_bin, "torch", dtype=torch.float64, device=torch.device("cpu")
        ).reshape(-1)
        dop = convert(
            doppler_bin, "torch", dtype=torch.float64, device=torch.device("cpu")
        ).reshape(-1)
        tau_s = d * dt
        fd_hz = (dop - center) * dres
        return tau_s, fd_hz

    def local_bins_to_range_velocity(
        self,
        delay_bin: ArrayLike,
        doppler_bin: ArrayLike,
        *,
        num_doppler_bins: int,
        sens_mode: SensMode = "monostatic",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ROI 局部 bin → (τ, f_d, 距离 m, 速度 m/s)。"""
        tau_s, fd_hz = self.local_bins_to_tau_fd(
            delay_bin,
            doppler_bin,
            num_doppler_bins=num_doppler_bins,
        )
        if tau_s.numel() == 0:
            return tau_s, fd_hz, tau_s, tau_s

        fc = float(self.sensing_performance.carrier_frequency)
        range_m = delay_to_range(tau_s, fc, sens_mode)
        v_mps = doppler_to_velocity(fd_hz, fc, sens_mode)
        return tau_s, fd_hz, range_m.reshape(-1), v_mps.reshape(-1)

    def physical_to_local_bins(
        self,
        range_m: ArrayLike,
        velocity_mps: ArrayLike,
        *,
        num_doppler_bins: int,
        sens_mode: SensMode = "monostatic",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """物理量 → ROI 局部 bin（与 :meth:`local_bins_to_range_velocity` 互逆）。"""
        sp = self.sensing_performance
        range_res = float(getattr(sp, f"range_resolution_{sens_mode}"))
        vel_res = float(getattr(sp, f"velocity_resolution_{sens_mode}"))
        center = self.doppler_center(num_doppler_bins)

        r = convert(
            range_m, "torch", dtype=torch.float64, device=torch.device("cpu")
        ).reshape(-1)
        v = convert(
            velocity_mps, "torch", dtype=torch.float64, device=torch.device("cpu")
        ).reshape(-1)
        delay_bin = r / range_res
        doppler_bin = center - v / vel_res
        return delay_bin, doppler_bin

    def global_bin_to_tau_fd(
        self,
        delay_bin: ArrayLike,
        doppler_bin: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """全谱全局 bin → τ(s)、f_d(Hz)（查轴数组，用于 roundtrip 测试）。"""
        sp = self.sensing_performance
        d = np.asarray(delay_bin, dtype=np.float64).reshape(-1)
        dop = np.asarray(doppler_bin, dtype=np.float64).reshape(-1)
        delay_ns = sp.delay_bins[d.astype(int)]
        tau_s = delay_ns * 1e-9
        fd_hz = sp.doppler_bins[dop.astype(int)]
        return tau_s, fd_hz
