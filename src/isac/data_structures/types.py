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
    """感知物理量估计结果（距离/速度）。"""

    est_ranges: torch.Tensor
    est_velocities: torch.Tensor


@dataclass
class MusicPeaks:
    """MUSIC 裁切谱 bin 检峰结果（delay/doppler 局部索引）。"""

    peaks_delay: torch.Tensor
    peaks_doppler: torch.Tensor

    @classmethod
    def empty(cls, device: Union[str, torch.device]) -> MusicPeaks:
        """协方差分解失败或无峰时返回长度为 0 的空结果。"""
        dev = torch.device(device) if isinstance(device, str) else device
        return cls(
            peaks_delay=torch.empty(0, dtype=torch.float64, device=dev),
            peaks_doppler=torch.empty(0, dtype=torch.float64, device=dev),
        )

    @classmethod
    def from_local_bins(
        cls,
        delay_bin: Union[torch.Tensor, float],
        doppler_bin: Union[torch.Tensor, float],
        *,
        device: Union[str, torch.device],
    ) -> MusicPeaks:
        """ROI 局部 bin → :class:`MusicPeaks`。"""
        dev = torch.device(device) if isinstance(device, str) else device
        d = torch.as_tensor(delay_bin, dtype=torch.float64, device=dev).reshape(-1)
        dop = torch.as_tensor(doppler_bin, dtype=torch.float64, device=dev).reshape(-1)
        return cls(peaks_delay=d, peaks_doppler=dop)
