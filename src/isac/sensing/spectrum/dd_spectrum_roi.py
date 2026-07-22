"""时延–多普勒谱 ROI：物理量配置、bin 切片与裁切。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from ...data_structures.params.sensing_params import DelayDopplerRoiParams
from ...data_structures.types import MetricMode, RoiSlices, SensMode
from .sensing_performance import SensingPerformance


@dataclass
class DelayDopplerRoi:
    """DD 谱 ROI 配置与裁切管理。

    持有 ``max_range_m`` / ``max_velocity_mps`` 物理上界，负责 bin 计数、切片计算、
    张量裁切及 ``slices`` 缓存；可视化轴标定通过 :meth:`axes` 提供。
    """

    max_range_m: float
    max_velocity_mps: float
    sensing_performance: SensingPerformance
    slices: Optional[RoiSlices] = field(default=None, repr=False)

    @classmethod
    def from_params(
        cls,
        params: DelayDopplerRoiParams,
        sensing_performance: SensingPerformance,
    ) -> DelayDopplerRoi:
        """由 TOML ``[dd_spectrum_roi]`` 参数构造。"""
        return cls(
            max_range_m=params.max_range_m,
            max_velocity_mps=params.max_velocity_mps,
            sensing_performance=sensing_performance,
        )

    def validate(self) -> None:
        """校验 ROI 物理量为正。"""
        if self.max_range_m <= 0:
            raise ValueError(f"max_range_m 须为正，收到 {self.max_range_m}")
        if self.max_velocity_mps <= 0:
            raise ValueError(
                f"max_velocity_mps 须为正，收到 {self.max_velocity_mps}"
            )

    def delay_bin_count(self, sens_mode: SensMode = "monostatic") -> int:
        """``max_range_m`` 对应的时延维 bin 数（含零时延 bin）。"""
        dr = getattr(self.sensing_performance, f"range_resolution_{sens_mode}")
        return max(1, int(self.max_range_m / dr) + 1)

    def doppler_half_bins(self, sens_mode: SensMode = "monostatic") -> int:
        """``max_velocity_mps`` 对应的多普勒半宽 bin 数。

        使用 ``round`` 对齐速度分辨网格；有效上界见 :meth:`effective_physical_limits`，
        可能略小于配置的 ``max_velocity_mps``。
        """
        dv = getattr(self.sensing_performance, f"velocity_resolution_{sens_mode}")
        return max(1, int(round(self.max_velocity_mps / dv)))

    def effective_physical_limits(
        self, sens_mode: SensMode = "monostatic"
    ) -> tuple[float, float]:
        """由 ROI bin 网格反推对齐后的有效 ``max_range_m`` / ``max_velocity_mps``。

        ``max_velocity_mps`` 为半宽 × 速度分辨率，是配置值对齐 bin 后的近似上界。
        """
        dr = getattr(self.sensing_performance, f"range_resolution_{sens_mode}")
        dv = getattr(self.sensing_performance, f"velocity_resolution_{sens_mode}")
        return (
            (self.delay_bin_count(sens_mode) - 1) * dr,
            self.doppler_half_bins(sens_mode) * dv,
        )

    def bin_slices(
        self,
        h_dd: torch.Tensor,
        sens_mode: SensMode = "monostatic",
    ) -> RoiSlices:
        """由 ROI 物理量与全尺寸谱形状计算裁切切片。

        多普勒维以零多普勒（``n_doppler // 2``）为中心对称裁切，长度为
        ``2 * dop_half + 1``（Python 切片右端开区间，故 ``dop_end = center + half + 1``）。
        """
        self.validate()
        n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
        delay_bins = min(n_delay, self.delay_bin_count(sens_mode=sens_mode))
        dop_half = min(
            n_doppler // 2,
            self.doppler_half_bins(sens_mode=sens_mode),
        )
        dop_center = n_doppler // 2
        dop_start = max(0, dop_center - dop_half)
        dop_end = min(n_doppler, dop_center + dop_half + 1)
        return dop_start, dop_end, 0, delay_bins

    def crop(
        self,
        h_dd: torch.Tensor,
        sens_mode: SensMode = "monostatic",
    ) -> torch.Tensor:
        """按 ROI 裁切 DD 谱并缓存 ``slices``。"""
        dop_start, dop_end, delay_start, delay_end = self.bin_slices(
            h_dd, sens_mode=sens_mode
        )
        self.slices = (dop_start, dop_end, delay_start, delay_end)
        return h_dd[..., dop_start:dop_end, delay_start:delay_end]

    @property
    def num_doppler_bins(self) -> int:
        """裁切后多普勒维长度（须先 :meth:`crop`）。"""
        if self.slices is None:
            raise ValueError("num_doppler_bins 要求先调用 crop()")
        dop_start, dop_end, _, _ = self.slices
        return dop_end - dop_start

    def axes(
        self,
        metric_mode: MetricMode,
        sens_mode: SensMode = "monostatic",
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """为 ROI 裁切谱返回 ``(x_axis, y_axis, x_label, y_label)``。"""
        if self.slices is None:
            raise ValueError("axes 要求先调用 crop()")
        dop_start, dop_end, delay_start, delay_end = self.slices
        sp = self.sensing_performance

        if metric_mode == "dd":
            x_axis = sp.delay_bins[delay_start:delay_end]
            y_axis = sp.doppler_bins[dop_start:dop_end]
            return x_axis, y_axis, "Delay (ns)", "Doppler (Hz)"

        x_axis = getattr(sp, f"range_bins_{sens_mode}")[delay_start:delay_end]
        y_axis = getattr(sp, f"velocity_bins_{sens_mode}")[dop_start:dop_end]
        return x_axis, y_axis, "Range (m)", "Velocity (m/s)"
