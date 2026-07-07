"""感知域共享类型别名。"""

from __future__ import annotations

from typing import Literal

MetricMode = Literal["dd", "rv"]
"""展示/日志域：时延-多普勒 (dd) 或 距离-速度 (rv)。"""

SensMode = Literal["monostatic", "bistatic"]
"""物理换算尺度：单基地往返 / 双基地单程。"""

RoiSlices = tuple[int, int, int, int]
"""ROI 裁切切片 ``(dop_start, dop_end, delay_start, delay_end)``。"""
