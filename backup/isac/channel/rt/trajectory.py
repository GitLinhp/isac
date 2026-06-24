from dataclasses import dataclass, field
import numpy as np


@dataclass(kw_only=True)
class Trajectory:

    # 沿轨迹的距离 [米]
    distance: float = 0.0
    # 描述轨迹线段的控制点
    points: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # 运动速度 [米/秒]，默认采用步行速度 (4 km/h)
    velocity: float = 1.11

    # 沿轨迹的累计距离分布（从零开始），长度等于控制点数量
    _cumulative_distances: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.compute_cumulative_distances()

    def compute_cumulative_distances(self) -> list[float]:
        """根据 `points` 计算并缓存累计距离（从 0 开始）。"""
        pts = np.asarray(self.points, dtype=np.float64)
        self.points = pts

        if len(pts) == 0:
            self._cumulative_distances = []
            return self._cumulative_distances
        if len(pts) == 1:
            self._cumulative_distances = [0.0]
            return self._cumulative_distances

        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.zeros(len(pts), dtype=np.float64)
        cumulative[1:] = np.cumsum(segment_lengths)
        self._cumulative_distances = cumulative.tolist()
        return self._cumulative_distances

    def current_position_and_direction(self) -> tuple[np.ndarray, np.ndarray] | None:
        """根据当前 distance 返回世界坐标下的位置和方向。"""
        n_points = len(self.points)
        # 没有控制点时无法定义位置与方向
        if n_points == 0:
            return None
        # 只有一个点时位置固定，方向退化为零向量
        if n_points == 1:
            return self.points[0], np.zeros(3)

        # 将距离限制在轨迹总长度范围内，避免越界采样
        safe_dist = np.clip(self.distance, 0.0, self.total_distance())
        # 找到累计距离中第一个 >= safe_dist 的点作为插值终点
        end_idx = np.searchsorted(self._cumulative_distances, safe_dist, side="left")
        # 起点为终点前一个点（起始边界时保持为 0）
        start_idx = max(end_idx - 1, 0)
        # 恰好落在控制点上时，无需插值，直接返回该点
        if start_idx == end_idx:
            direction = np.zeros(3)
            return self.points[start_idx], direction

        start_dist = self._cumulative_distances[start_idx]
        end_dist = self._cumulative_distances[end_idx]
        # 计算当前距离在线段 [start_idx, end_idx] 内的归一化插值系数
        t = (safe_dist - start_dist) / (end_dist - start_dist)

        # 线性插值得到当前位置，并将方向归一化
        direction = self.points[end_idx] - self.points[start_idx]
        pos = self.points[start_idx] + t * direction

        return pos, direction / np.linalg.norm(direction)

    def total_distance(self) -> float:
        if len(self._cumulative_distances) == 0:
            return 0.0
        return self._cumulative_distances[-1]

    def __len__(self) -> int:
        return len(self.points)
