"""z=0 平面目标定位：由单基地斜距 + 双基地折叠路径长（TX 与单基地 RX 共址）求 (x, y)。"""

import math
from typing import Sequence

import numpy as np

from .utils import MONOSTATIC_TX_RX_EPS_M


def ground_circle_radius_sq(
    slant_range_m: float,
    station_z_m: float,
    *,
    z_target_m: float = 0.0,
) -> float:
    """斜距在 ``z=z_target`` 平面上的水平圆半径平方 ``R^2 = r_slant^2 - (z_s - z_t)^2``。"""
    dz = float(station_z_m) - float(z_target_m)
    r_sq = float(slant_range_m) ** 2 - dz * dz
    if r_sq < 0.0:
        raise ValueError(
            f"斜距 {slant_range_m} m 在 z_target={z_target_m} 下无实水平半径 "
            f"(station_z={station_z_m})"
        )
    return r_sq


def intersect_circles_xy(
    center1_xy: Sequence[float],
    radius1_sq: float,
    center2_xy: Sequence[float],
    radius2_sq: float,
    *,
    tol: float = 1e-9,
) -> list[tuple[float, float]]:
    """两圆在 xy 平面求交，返回 0、1 或 2 个 ``(x, y)``。"""
    x1, y1 = float(center1_xy[0]), float(center1_xy[1])
    x2, y2 = float(center2_xy[0]), float(center2_xy[1])
    if radius1_sq < 0.0 or radius2_sq < 0.0:
        return []

    dx = x2 - x1
    dy = y2 - y1
    d_sq = dx * dx + dy * dy
    if d_sq < tol * tol:
        if abs(radius1_sq - radius2_sq) <= tol:
            raise ValueError("两圆同心且半径相同，无法在 xy 平面唯一确定交点")
        return []

    d = math.sqrt(d_sq)
    r1 = math.sqrt(radius1_sq)
    r2 = math.sqrt(radius2_sq)
    gap_tol = max(tol, 1e-2 * max(d, r1, r2, 1.0))
    if d > r1 + r2 + gap_tol or d < abs(r1 - r2) - gap_tol:
        return []

    a = (radius1_sq - radius2_sq + d_sq) / (2.0 * d)
    h_sq = radius1_sq - a * a
    if h_sq < -gap_tol:
        return []
    h = math.sqrt(max(h_sq, 0.0))

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    if h <= tol:
        return [(xm, ym)]

    rx = -dy * h / d
    ry = dx * h / d
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def _xy_from_linearized_circles(
    center1_xy: Sequence[float],
    radius1_sq: float,
    center2_xy: Sequence[float],
    radius2_sq: float,
    *,
    y_hint: float | None = None,
) -> tuple[float, float]:
    """两圆方程相减得线性方程；无几何交点时仍给出最小二乘意义的近似 ``(x,y)``。"""
    x1, y1 = float(center1_xy[0]), float(center1_xy[1])
    x2, y2 = float(center2_xy[0]), float(center2_xy[1])
    a = 2.0 * (x2 - x1)
    b = 2.0 * (y2 - y1)
    c = radius1_sq - radius2_sq - (x1 * x1 - x2 * x2 + y1 * y1 - y2 * y2)

    if abs(a) < 1e-12 and abs(b) < 1e-12:
        raise ValueError("两圆心重合，无法线性化求交")

    if abs(b) < 1e-12:
        x = c / a
        y_sq = radius1_sq - (x - x1) ** 2
    elif abs(a) < 1e-12:
        y = c / b
        x_sq = radius1_sq - (y - y1) ** 2
        if x_sq < 0.0:
            x = x1
        else:
            x_off = math.sqrt(x_sq)
            x = x1 + x_off if (y_hint is None or y_hint >= 0) else x1 - x_off
        return (x, y)
    else:
        x = c / a
        y_sq = radius1_sq - (x - x1) ** 2

    if y_sq < 0.0:
        y = 0.0
    else:
        y_abs = math.sqrt(y_sq)
        y = -y_abs if (y_hint is not None and y_hint < 0) else y_abs
    return (x, y)


def select_xy_solution(
    solutions: list[tuple[float, float]],
    *,
    y_hint: float | None = None,
) -> tuple[float, float]:
    """在 1 或 2 个交点中择一；``y_hint`` 用于消解 ±y 镜像歧义。"""
    if not solutions:
        raise ValueError("无有效 (x, y) 交点")
    if len(solutions) == 1:
        return solutions[0]
    if y_hint is not None:
        return min(solutions, key=lambda p: abs(p[1] - float(y_hint)))
    return min(solutions, key=lambda p: abs(p[1]))


def localize_xy_z0_colocated_tx_mono_bistatic(
    mono_rx_pos: Sequence[float],
    bistatic_rx_pos: Sequence[float],
    *,
    r_mono_slant_m: float,
    r_bistatic_sum_m: float,
    z_target_m: float = 0.0,
    y_hint: float | None = None,
    tx_pos: Sequence[float] | None = None,
    tx_rx_colocated_eps_m: float = MONOSTATIC_TX_RX_EPS_M,
) -> tuple[float, float]:
    """已知 ``z=z_target``，由单基地斜距与双基地折叠路径长求目标 ``(x, y)``。

    假定发射机与单基地接收机共址（或间距 ≤ ``tx_rx_colocated_eps_m``），故
    ``r_bistatic_sum = ||X-T|| + ||R_bi-T||`` 且 ``||X-T|| ≈ r_mono_slant``，
    第二圆心为 ``bistatic_rx`` 在 xy 上的投影，半径为 ``r_bistatic_sum - r_mono_slant`` 的水平分量。
    """
    mono = np.asarray(mono_rx_pos, dtype=np.float64).reshape(3)
    bi = np.asarray(bistatic_rx_pos, dtype=np.float64).reshape(3)

    if tx_pos is not None:
        tx = np.asarray(tx_pos, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(tx - mono)) > tx_rx_colocated_eps_m:
            raise ValueError(
                "发射机与单基地接收机未共址，本闭式解不适用；"
                f"间距 {float(np.linalg.norm(tx - mono)):.6f} m > eps {tx_rx_colocated_eps_m}"
            )

    r_leg2 = float(r_bistatic_sum_m) - float(r_mono_slant_m)
    if r_leg2 <= 0.0:
        raise ValueError(
            f"双基地折叠路径长 {r_bistatic_sum_m} m 须大于单基地斜距 {r_mono_slant_m} m"
        )

    r1_sq = ground_circle_radius_sq(
        r_mono_slant_m, float(mono[2]), z_target_m=z_target_m
    )
    r2_sq = ground_circle_radius_sq(r_leg2, float(bi[2]), z_target_m=z_target_m)

    c1 = (float(mono[0]), float(mono[1]))
    c2 = (float(bi[0]), float(bi[1]))
    solutions = intersect_circles_xy(c1, r1_sq, c2, r2_sq)
    if solutions:
        return select_xy_solution(solutions, y_hint=y_hint)
    return _xy_from_linearized_circles(c1, r1_sq, c2, r2_sq, y_hint=y_hint)


def position_rmse_xy(
    est_xy: Sequence[float],
    true_xy: Sequence[float],
) -> float:
    """平面位置 RMSE（m），仅 x、y 分量。"""
    e = np.asarray(est_xy, dtype=np.float64).reshape(2)
    t = np.asarray(true_xy, dtype=np.float64).reshape(2)
    return float(np.linalg.norm(e - t))
