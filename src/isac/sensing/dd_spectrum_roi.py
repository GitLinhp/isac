"""时延–多普勒谱 ROI：物理量配置与 bin 裁剪。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isac.data_structures.params.sensing_params import DelayDopplerRoiParams
    from isac.sensing.sensing_performance import SensingPerformance


@dataclass(frozen=True)
class DelayDopplerRoi:
    """DD 谱 ROI（距离/速度物理量语义）。

    运行时结合 ``SensingPerformance`` 分辨率换算 bin 并裁剪谱图。
    """

    max_range_m: float
    max_velocity_mps: float

    def __post_init__(self) -> None:
        if self.max_range_m <= 0:
            raise ValueError(f"max_range_m 须为正，收到 {self.max_range_m}")
        if self.max_velocity_mps <= 0:
            raise ValueError(
                f"max_velocity_mps 须为正，收到 {self.max_velocity_mps}"
            )

    @classmethod
    def from_params(cls, params: DelayDopplerRoiParams) -> DelayDopplerRoi:
        return cls(
            max_range_m=params.max_range_m,
            max_velocity_mps=params.max_velocity_mps,
        )

    def delay_bins(self, sensing_performance: SensingPerformance) -> int:
        dr = sensing_performance.range_resolution
        return max(1, int(self.max_range_m / dr) + 1)

    def doppler_half_bins(self, sensing_performance: SensingPerformance) -> int:
        dv = sensing_performance.velocity_resolution
        return max(1, int(round(self.max_velocity_mps / dv)))

    def bin_slices(
        self,
        h_dd: torch.Tensor,
        sensing_performance: SensingPerformance,
    ) -> tuple[int, int, int, int]:
        """返回 ``(dop_start, dop_end, delay_start, delay_end)`` 切片索引。"""
        n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
        delay_bins = min(n_delay, self.delay_bins(sensing_performance))
        dop_half = min(n_doppler // 2, self.doppler_half_bins(sensing_performance))
        dop_center = n_doppler // 2
        dop_start = max(0, dop_center - dop_half)
        dop_end = min(n_doppler, dop_center + dop_half)
        return dop_start, dop_end, 0, delay_bins

    def crop(
        self,
        h_dd: torch.Tensor,
        sensing_performance: SensingPerformance,
    ) -> torch.Tensor:
        """裁剪 DD 谱 ROI；末两维为 ``(多普勒, 时延)``。"""
        dop_start, dop_end, delay_start, delay_end = self.bin_slices(
            h_dd, sensing_performance
        )
        return h_dd[..., dop_start:dop_end, delay_start:delay_end]

    def feature_shape(
        self,
        h_dd: torch.Tensor,
        sensing_performance: SensingPerformance,
    ) -> tuple[int, int]:
        """裁剪后 ``(多普勒, 时延)`` 尺寸。"""
        dop_start, dop_end, delay_start, delay_end = self.bin_slices(
            h_dd, sensing_performance
        )
        return dop_end - dop_start, delay_end - delay_start

    def limits(
        self,
        h_dd: torch.Tensor,
        sensing_performance: SensingPerformance,
    ) -> tuple[float, float]:
        """实际裁剪对应的 ``(max_range_m, max_velocity_mps)``。"""
        _, _, _, delay_end = self.bin_slices(h_dd, sensing_performance)
        dop_start, dop_end, _, _ = self.bin_slices(h_dd, sensing_performance)
        dr = sensing_performance.range_resolution
        dv = sensing_performance.velocity_resolution
        max_range_m = (delay_end - 1) * dr
        dop_half = (dop_end - dop_start) // 2
        max_velocity_mps = dop_half * dv
        return max_range_m, max_velocity_mps
