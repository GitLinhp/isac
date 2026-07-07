"""平面 ROI 内位置与速度采样（z=0，速度方向在 xy 平面）。"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from ..utils.numerical import cartesian_direction_to_yaw_pitch_roll

SamplingMode = Literal["uniform", "gaussian"]


class RoiKinematicsSampler:
    """平面 ROI 运动学采样器：构造时批量采样，逐条 ``pop()`` 消费。"""

    def __init__(
        self,
        *,
        roi: list[float] | tuple[float, ...],
        position_sampling_mode: SamplingMode,
        speed_range: list[float] | tuple[float, ...],
        speed_sampling_mode: SamplingMode,
        num_samples: int,
    ) -> None:
        x_lo, x_hi, y_lo, y_hi = self.parse_roi_xy(roi)
        smin, smax = self.parse_speed_range(speed_range)
        n = int(num_samples)
        positions = self._sample_positions(
            x_lo, x_hi, y_lo, y_hi, n, position_sampling_mode
        )
        velocities, orientations = self._sample_velocities(
            smin, smax, n, speed_sampling_mode
        )
        self._positions: deque[np.ndarray] = deque(positions)
        self._velocities: deque[np.ndarray] = deque(velocities)
        self._orientations: deque[np.ndarray] = deque(orientations)

    def pop(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """弹出一条 ``(position, velocity, orientation)``，均为 shape ``(3,)``。"""
        if not self._positions:
            raise IndexError("ROI 采样队列已空")
        return (
            self._positions.popleft(),
            self._velocities.popleft(),
            self._orientations.popleft(),
        )

    def __len__(self) -> int:
        """剩余可 pop 条数。"""
        return len(self._positions)

    @staticmethod
    def parse_roi_xy(
        roi4: list[float] | tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        """解析平面 ROI 四元组 ``XMIN XMAX YMIN YMAX``。"""
        if len(roi4) != 4:
            raise ValueError("平面 ROI 须为四元组：XMIN XMAX YMIN YMAX")
        x_lo, x_hi, y_lo, y_hi = (float(v) for v in roi4)
        for name, lo, hi in (("x", x_lo, x_hi), ("y", y_lo, y_hi)):
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"ROI 维度 `{name}` 非法：须为有限值")
            if lo > hi:
                raise ValueError(f"ROI 维度 `{name}` 非法：须满足 min <= max")
        return x_lo, x_hi, y_lo, y_hi

    @staticmethod
    def parse_speed_range(
        pair: list[float] | tuple[float, ...],
    ) -> tuple[float, float]:
        """解析速度模值范围 ``MIN MAX``。"""
        if len(pair) != 2:
            raise ValueError("speed_range 须为二元组：MIN MAX")
        smin, smax = float(pair[0]), float(pair[1])
        if not np.isfinite(smin) or not np.isfinite(smax):
            raise ValueError("speed_range 须为有限值")
        if smin < 0 or smax <= smin:
            raise ValueError("speed_range 须满足 0 <= min < max")
        return smin, smax

    @staticmethod
    def _sample_positions(
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
        num_samples: int,
        sampling_mode: SamplingMode,
    ) -> np.ndarray:
        """在平面 ROI 内采样位置，形状 ``(num_samples, 3)``。"""
        if num_samples <= 0:
            raise ValueError("num_samples 必须大于 0")

        n = int(num_samples)
        if sampling_mode == "uniform":
            x = np.random.uniform(x_lo, x_hi, size=n)
            y = np.random.uniform(y_lo, y_hi, size=n)
            z = np.zeros(n, dtype=np.float64)
            return np.column_stack((x, y, z)).astype(np.float64)
        if sampling_mode == "gaussian":
            center = np.array(
                [(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, 0.0], dtype=np.float64
            )
            std = np.array(
                [(x_hi - x_lo) / 6.0, (y_hi - y_lo) / 6.0, 0.0], dtype=np.float64
            )
            pts = np.random.normal(loc=center, scale=std, size=(n, 3)).astype(
                np.float64
            )
            return np.clip(pts, [x_lo, y_lo, 0.0], [x_hi, y_hi, 0.0])
        raise ValueError("sampling_mode 仅支持 'uniform' 或 'gaussian'")

    @staticmethod
    def _sample_speeds(
        smin: float,
        smax: float,
        sampling_mode: SamplingMode,
        num_samples: int,
    ) -> np.ndarray:
        """在 ``[smin, smax]`` 内采样速度模值。"""
        if num_samples <= 0:
            raise ValueError("num_samples 必须大于 0")

        n = int(num_samples)
        if sampling_mode == "uniform":
            return np.random.uniform(smin, smax, size=n).astype(np.float64)
        if sampling_mode == "gaussian":
            center = (smin + smax) / 2.0
            std = (smax - smin) / 6.0
            speeds = np.random.normal(loc=center, scale=std, size=n).astype(np.float64)
            return np.clip(speeds, smin, smax)
        raise ValueError("speed_sampling_mode 仅支持 'uniform' 或 'gaussian'")

    @staticmethod
    def _sample_planar_directions(num_samples: int) -> np.ndarray:
        """xy 平面均匀随机单位方向，形状 ``(num_samples, 3)``。"""
        n = int(num_samples)
        theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        return np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n))).astype(
            np.float64
        )

    @classmethod
    def _sample_velocities(
        cls,
        smin: float,
        smax: float,
        num_samples: int,
        sampling_mode: SamplingMode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """采样速度向量与朝向。"""
        speeds = cls._sample_speeds(smin, smax, sampling_mode, num_samples)
        dirs = cls._sample_planar_directions(num_samples)
        orientations = cartesian_direction_to_yaw_pitch_roll(dirs)
        velocities = (speeds[:, None] * dirs).astype(np.float64)
        return velocities, orientations

