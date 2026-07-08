"""ISAC 共享数据类型：感知域别名与 MUSIC 检峰结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch

MetricMode = Literal["dd", "rv"]
"""展示/日志域：时延-多普勒 (dd) 或 距离-速度 (rv)。"""

SensMode = Literal["monostatic", "bistatic"]
"""物理换算尺度：单基地往返 / 双基地单程。"""

RoiSlices = tuple[int, int, int, int]
"""ROI 裁切切片 ``(dop_start, dop_end, delay_start, delay_end)``。"""


@dataclass
class SensingEstimate:
    """感知物理量估计结果（距离/速度/功率）。"""

    est_ranges: torch.Tensor
    est_velocities: torch.Tensor
    peaks_power: torch.Tensor


@dataclass
class MusicPeaks:
    """MUSIC 裁切谱 bin 检峰结果（delay/doppler 局部索引 + 功率）。"""

    peaks_delay: torch.Tensor
    peaks_doppler: torch.Tensor
    peaks_power: torch.Tensor
    num_doppler_bins: int
    """裁切谱多普勒维长度（squeeze 后 shape[0]），用于局部 bin → Hz 换算。"""

    @classmethod
    def empty(
        cls,
        device: Union[str, torch.device],
        *,
        num_doppler_bins: int = 0,
    ) -> MusicPeaks:
        """协方差分解失败或无峰时返回长度为 0 的空结果。"""
        dev = torch.device(device) if isinstance(device, str) else device
        return cls(
            peaks_delay=torch.empty(0, dtype=torch.float64, device=dev),
            peaks_doppler=torch.empty(0, dtype=torch.float64, device=dev),
            peaks_power=torch.empty(0, dtype=torch.float32, device=dev),
            num_doppler_bins=num_doppler_bins,
        )
