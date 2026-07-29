"""z=0 平面目标定位：由单基地斜距 + 双基地折叠路径长（TX 与单基地 RX 共址）求 (x, y)。"""

import math
from typing import Sequence

import numpy as np

from .geometry import MONOSTATIC_TX_RX_EPS_M


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


def _unit_xy(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-12:
        return 0.0, 0.0
    return dx / n, dy / n


def horn_beam_score_xy(
    point_xy: Sequence[float],
    station0_xy: Sequence[float],
    station1_xy: Sequence[float],
    *,
    aim_xy: Sequence[float] = (0.0, 0.0),
    cos_power: float = 2.0,
) -> float:
    """两站 Horn 方向图代理增益之和：``G=max(cos θ,0)^p``，boresight 指向 ``aim_xy``。

    θ 为站→目标相对站→瞄准点的离轴角。用于双圆两解消歧（朝瞄准点一侧得分更高）。
    """
    ax, ay = float(aim_xy[0]), float(aim_xy[1])
    px, py = float(point_xy[0]), float(point_xy[1])
    p = float(cos_power)
    score = 0.0
    for station in (station0_xy, station1_xy):
        sx, sy = float(station[0]), float(station[1])
        bx, by = _unit_xy(ax - sx, ay - sy)
        vx, vy = _unit_xy(px - sx, py - sy)
        cos_t = bx * vx + by * vy
        score += max(cos_t, 0.0) ** p
    return score


def select_xy_solution(
    solutions: list[tuple[float, float]],
    *,
    y_hint: float | None = None,
    station0_xy: Sequence[float] | None = None,
    station1_xy: Sequence[float] | None = None,
    horn_aim_xy: Sequence[float] = (0.0, 0.0),
    horn_cos_power: float = 2.0,
) -> tuple[float, float]:
    """在 1 或 2 个交点中择一。

    优先级：``y_hint`` → 两站已知时的 Horn 波束打分（指向 ``horn_aim_xy``，默认原点）
    → ``|y|`` 最小。
    """
    if not solutions:
        raise ValueError("无有效 (x, y) 交点")
    if len(solutions) == 1:
        return solutions[0]
    if y_hint is not None:
        return min(solutions, key=lambda p: abs(p[1] - float(y_hint)))
    if station0_xy is not None and station1_xy is not None:
        return max(
            solutions,
            key=lambda p: horn_beam_score_xy(
                p,
                station0_xy,
                station1_xy,
                aim_xy=horn_aim_xy,
                cos_power=horn_cos_power,
            ),
        )
    return min(solutions, key=lambda p: abs(p[1]))


def _midpoint_xy(
    a_xy: Sequence[float],
    b_xy: Sequence[float],
) -> tuple[float, float]:
    return (
        0.5 * (float(a_xy[0]) + float(b_xy[0])),
        0.5 * (float(a_xy[1]) + float(b_xy[1])),
    )


def path_sum_xy(
    point_xy: Sequence[float],
    tx_xy: Sequence[float],
    rx_xy: Sequence[float],
) -> float:
    """折叠路径长 ``||P-TX|| + ||P-RX||``。"""
    px, py = float(point_xy[0]), float(point_xy[1])
    return math.hypot(px - float(tx_xy[0]), py - float(tx_xy[1])) + math.hypot(
        px - float(rx_xy[0]), py - float(rx_xy[1])
    )


def _ellipse_point_xy(
    tx_xy: Sequence[float],
    rx_xy: Sequence[float],
    path_sum_m: float,
    theta: float,
) -> tuple[float, float] | None:
    """以 TX/RX 为焦点、路径和为 ``path_sum_m`` 的椭圆参数点；退化时返回 ``None``。"""
    tx0, tx1 = float(tx_xy[0]), float(tx_xy[1])
    rx0, rx1 = float(rx_xy[0]), float(rx_xy[1])
    mx, my = 0.5 * (tx0 + rx0), 0.5 * (tx1 + rx1)
    dx, dy = tx0 - rx0, tx1 - rx1
    dist = math.hypot(dx, dy)
    a = 0.5 * float(path_sum_m)
    c = 0.5 * dist
    if a < c - 1e-12:
        return None
    b = math.sqrt(max(a * a - c * c, 0.0))
    if dist < 1e-12:
        return (mx + a * math.cos(theta), my + a * math.sin(theta))
    ux, uy = dx / dist, dy / dist
    vx, vy = -uy, ux
    return (
        mx + a * math.cos(theta) * ux + b * math.sin(theta) * vx,
        my + a * math.cos(theta) * uy + b * math.sin(theta) * vy,
    )


def intersect_ellipses_xy(
    tx0_xy: Sequence[float],
    rx0_xy: Sequence[float],
    path_sum0_m: float,
    tx1_xy: Sequence[float],
    rx1_xy: Sequence[float],
    path_sum1_m: float,
    *,
    n_theta: int = 720,
    dedupe_tol_m: float = 1e-4,
) -> list[tuple[float, float]]:
    """两椭圆交点（各以 TX/RX 为焦点、路径和为 ``path_sum``）。

    在椭圆 0 上参数扫描，对椭圆 1 残差过零做二分；通常返回 0–2 个点。
    """
    s0 = float(path_sum0_m)
    s1 = float(path_sum1_m)
    if not np.isfinite(s0) or not np.isfinite(s1) or s0 <= 0.0 or s1 <= 0.0:
        return []

    solutions: list[tuple[float, float]] = []
    g_prev: float | None = None
    th_prev = 0.0
    n = max(int(n_theta), 64)

    for i in range(n + 1):
        theta = 2.0 * math.pi * i / n
        p = _ellipse_point_xy(tx0_xy, rx0_xy, s0, theta)
        if p is None:
            g_prev = None
            continue
        g = path_sum_xy(p, tx1_xy, rx1_xy) - s1
        if g_prev is not None and g_prev * g <= 0.0 and (abs(g_prev) + abs(g)) > 0.0:
            lo, hi = th_prev, theta
            glo, ghi = g_prev, g
            for _ in range(48):
                mid_t = 0.5 * (lo + hi)
                p_mid = _ellipse_point_xy(tx0_xy, rx0_xy, s0, mid_t)
                if p_mid is None:
                    break
                g_mid = path_sum_xy(p_mid, tx1_xy, rx1_xy) - s1
                if glo * g_mid <= 0.0:
                    hi, ghi = mid_t, g_mid
                else:
                    lo, glo = mid_t, g_mid
            p_sol = _ellipse_point_xy(tx0_xy, rx0_xy, s0, 0.5 * (lo + hi))
            if p_sol is not None:
                if all(
                    math.hypot(p_sol[0] - q[0], p_sol[1] - q[1]) > dedupe_tol_m
                    for q in solutions
                ):
                    solutions.append(p_sol)
        g_prev = g
        th_prev = theta

    return solutions


def localize_xy_two_quasi_monostatic_path_sums(
    tx0_xy: Sequence[float],
    rx0_xy: Sequence[float],
    r0_m: float,
    tx1_xy: Sequence[float],
    rx1_xy: Sequence[float],
    r1_m: float,
    *,
    y_hint: float | None = None,
    use_horn_disambiguation: bool = True,
    horn_aim_xy: Sequence[float] = (0.0, 0.0),
    horn_cos_power: float = 2.0,
) -> tuple[float, float]:
    """两站准单基地：测距 ``r`` 对应路径和 ``2r = ||P-TX||+||P-RX||``，椭圆交会求 ``(x,y)``。

    Horn 消歧使用各站 TX/RX 中点；无实交点时回退为中点圆交会/线性化。
    """
    r0 = float(r0_m)
    r1 = float(r1_m)
    if not np.isfinite(r0) or not np.isfinite(r1) or r0 <= 0.0 or r1 <= 0.0:
        raise ValueError(f"invalid monostatic ranges: r0={r0_m}, r1={r1_m}")

    mid0 = _midpoint_xy(tx0_xy, rx0_xy)
    mid1 = _midpoint_xy(tx1_xy, rx1_xy)
    solutions = intersect_ellipses_xy(
        tx0_xy,
        rx0_xy,
        2.0 * r0,
        tx1_xy,
        rx1_xy,
        2.0 * r1,
    )
    if not solutions:
        # 回退：中点共址圆近似
        solutions = intersect_circles_xy(mid0, r0 * r0, mid1, r1 * r1)
        if not solutions:
            return _xy_from_linearized_circles(
                mid0, r0 * r0, mid1, r1 * r1, y_hint=y_hint
            )

    return select_xy_solution(
        solutions,
        y_hint=y_hint,
        station0_xy=mid0 if use_horn_disambiguation else None,
        station1_xy=mid1 if use_horn_disambiguation else None,
        horn_aim_xy=horn_aim_xy,
        horn_cos_power=horn_cos_power,
    )


def localize_xy_two_monostatic_ranges(
    pos0_xy: Sequence[float],
    r0_m: float,
    pos1_xy: Sequence[float],
    r1_m: float,
    *,
    y_hint: float | None = None,
    use_horn_disambiguation: bool = True,
    horn_aim_xy: Sequence[float] = (0.0, 0.0),
    horn_cos_power: float = 2.0,
) -> tuple[float, float]:
    """两单基地斜距在 z=0 平面圆交会，得到目标 ``(x, y)``。

    各站与目标均假定在同一水平面（z=0），故 ``r`` 即水平距离。

    无 ``y_hint`` 时默认用 Horn 波束打分消歧（两站均朝 ``horn_aim_xy``，默认原点）；
    ``use_horn_disambiguation=False`` 时回退为 ``|y|`` 最小。
    """
    r0 = float(r0_m)
    r1 = float(r1_m)
    if not np.isfinite(r0) or not np.isfinite(r1) or r0 <= 0.0 or r1 <= 0.0:
        raise ValueError(f"invalid monostatic ranges: r0={r0_m}, r1={r1_m}")

    c0 = (float(pos0_xy[0]), float(pos0_xy[1]))
    c1 = (float(pos1_xy[0]), float(pos1_xy[1]))
    solutions = intersect_circles_xy(c0, r0 * r0, c1, r1 * r1)
    if solutions:
        return select_xy_solution(
            solutions,
            y_hint=y_hint,
            station0_xy=c0 if use_horn_disambiguation else None,
            station1_xy=c1 if use_horn_disambiguation else None,
            horn_aim_xy=horn_aim_xy,
            horn_cos_power=horn_cos_power,
        )
    return _xy_from_linearized_circles(c0, r0 * r0, c1, r1 * r1, y_hint=y_hint)


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
