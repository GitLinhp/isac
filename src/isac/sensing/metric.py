"""时延-多普勒 / 距离-速度 坐标换算（spectrum 与 detection 共用）。"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

from .geometry import delay_to_range, doppler_to_velocity
from .spectrum.sensing_performance import SensingPerformance
from ..data_structures.types import MetricMode, RoiSlices, SensMode

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

        d = _to_float64_tensor(delay_bin)
        dop = _to_float64_tensor(doppler_bin)
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

    def roi_delay_bin_count(self, max_range_m: float) -> int:
        """物理最大距离 (m) → 时延维 bin 数。"""
        dr = self.sensing_performance.range_resolution
        return max(1, int(max_range_m / dr) + 1)

    def roi_doppler_half_bins(self, max_velocity_mps: float) -> int:
        """物理最大速度 (m/s) → 多普勒半宽 bin 数。"""
        dv = self.sensing_performance.velocity_resolution
        return max(1, int(round(max_velocity_mps / dv)))

    def bin_slices(
        self,
        n_doppler: int,
        n_delay: int,
        max_range_m: float,
        max_velocity_mps: float,
    ) -> RoiSlices:
        """由 ROI 物理量与谱尺寸计算 ``(dop_start, dop_end, delay_start, delay_end)``。"""
        delay_bins = min(n_delay, self.roi_delay_bin_count(max_range_m))
        dop_half = min(n_doppler // 2, self.roi_doppler_half_bins(max_velocity_mps))
        dop_center = n_doppler // 2
        dop_start = max(0, dop_center - dop_half)
        dop_end = min(n_doppler, dop_center + dop_half)
        return dop_start, dop_end, 0, delay_bins

    def axes_for_roi(
        self,
        roi_slices: RoiSlices,
        metric_mode: MetricMode,
        sens_mode: SensMode = "monostatic",
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """为 ROI 裁切谱返回 ``(x_axis, y_axis, x_label, y_label)``。"""
        dop_start, dop_end, delay_start, delay_end = roi_slices
        sp = self.sensing_performance

        if metric_mode == "dd":
            x_axis = sp.delay_bins[delay_start:delay_end]
            y_axis = sp.doppler_bins[dop_start:dop_end]
            return x_axis, y_axis, "Delay (ns)", "Doppler (Hz)"

        x_axis = sp.range_bins_for(sens_mode)[delay_start:delay_end]
        y_axis = sp.velocity_bins_for(sens_mode)[dop_start:dop_end]
        return x_axis, y_axis, "Range (m)", "Velocity (m/s)"


def _to_float64_tensor(value: ArrayLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64).reshape(-1)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return torch.from_numpy(arr)
