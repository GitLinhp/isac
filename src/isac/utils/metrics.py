import torch
from scipy.constants import c
from scipy.optimize import linear_sum_assignment

from .type_converter import convert


def match_peaks_and_compute_radial_rmse(
    *,
    est_ranges: torch.Tensor,
    est_velocities: torch.Tensor,
    true_ranges: torch.Tensor,
    true_velocities: torch.Tensor,
    label: str = "单基地感知",
    distance_axis_label: str = "径向距离",
    velocity_axis_label: str = "径向速度",
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用匈牙利算法在 MUSIC 峰与真值格点间做一对一最小代价匹配，并计算径向 RMSE。

    代价 ``C[i,j] = (er_i-tr_j)^2 + (ev_i-tv_j)^2``（联合平方误差）。``N!=M`` 时 SciPy 给出
    ``min(N,M)`` 条最优部分匹配；输出中会说明峰数、真值点数与匹配条数。

    返回 ``(rmse_range_m, rmse_velocity_mps, est_range_m, est_velocity_mps, music_peak_db)``，
    ``est_*`` / ``music_peak_db`` 为零维 ``float64`` 张量；``music_peak_db`` 固定为 NaN。
    ``est_range_m`` / ``est_velocity_mps`` 取 **匹配对内联合误差最小** 的那一对的估计值。
    """
    dtype = torch.float64
    device = est_ranges.device
    er = convert(est_ranges.reshape(-1), "torch", dtype=dtype, device=device)
    ev = convert(est_velocities.reshape(-1), "torch", dtype=dtype, device=device)
    n = er.numel()
    if n == 0 or ev.numel() != n:
        raise ValueError("est_ranges、est_velocities 须等长且非空")

    tr_conv = convert(true_ranges, "torch", dtype=dtype, device=device)
    tv_conv = convert(true_velocities, "torch", dtype=dtype, device=device)
    if tr_conv.shape != tv_conv.shape:
        raise ValueError("true_ranges 与 true_velocities 形状须一致")
    tr_raw = tr_conv.reshape(-1)
    tv_raw = tv_conv.reshape(-1)
    m = tr_raw.numel()
    if m == 0 or tv_raw.numel() != m:
        raise ValueError("true_ranges、true_velocities 须同形状或可广播为等长且非空")

    # 代价矩阵 (N, M)，在 CPU 上跑匈牙利以避免 GPU 与 SciPy 边界问题
    er_c = er.detach().cpu()
    ev_c = ev.detach().cpu()
    tr_c = tr_raw.detach().cpu()
    tv_c = tv_raw.detach().cpu()
    diff_r = er_c.unsqueeze(1) - tr_c.unsqueeze(0)
    diff_v = ev_c.unsqueeze(1) - tv_c.unsqueeze(0)
    cost_np = (diff_r.square() + diff_v.square()).numpy()

    row_ind, col_ind = linear_sum_assignment(cost_np)
    k = int(row_ind.shape[0])
    if verbose:
        print(f"{label} — 匈牙利匹配: MUSIC 峰数 N={n}, 真值点数 M={m}, 匹配条数 K={k}")

    if k == 0:
        raise RuntimeError("匈牙利匹配未产生任何配对")

    ri = torch.from_numpy(row_ind).to(dtype=torch.long, device=device)
    cj = torch.from_numpy(col_ind).to(dtype=torch.long, device=device)
    er_m = er[ri]
    ev_m = ev[ri]
    tr_m = tr_raw[cj]
    tv_m = tv_raw[cj]

    rmse_range = torch.sqrt(torch.mean((er_m - tr_m) ** 2))
    rmse_velocity = torch.sqrt(torch.mean((ev_m - tv_m) ** 2))

    joint_sq = (er_m - tr_m) ** 2 + (ev_m - tv_m) ** 2
    best = torch.argmin(joint_sq)
    est_range_m = er_m[best].detach()
    est_velocity_mps = ev_m[best].detach()
    true_r_show = tr_m[best].detach()
    true_v_show = tv_m[best].detach()
    music_peak_db = torch.tensor(float("nan"), dtype=dtype, device=device)

    if verbose:
        print(
            f"{label} — {distance_axis_label} 真值: {convert(true_r_show, 'float'):.2f} m, "
            f"估计: {convert(est_range_m, 'float'):.2f} m, RMSE: {convert(rmse_range, 'float'):.2f} m"
        )
        print(
            f"{label} — {velocity_axis_label} 真值: {convert(true_v_show, 'float'):.2f} m/s, "
            f"估计: {convert(est_velocity_mps, 'float'):.2f} m/s, "
            f"RMSE: {convert(rmse_velocity, 'float'):.2f} m/s"
        )
    return rmse_range, rmse_velocity, est_range_m, est_velocity_mps, music_peak_db


def compute_rmse(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算均方根误差。"""
    est = convert(estimate, "torch", dtype=torch.float64, device=estimate.device)
    tgt = convert(target, "torch", dtype=torch.float64, device=target.device)
    return torch.sqrt(torch.mean((est - tgt) ** 2))


def compute_mse(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算均方误差。"""
    est = convert(estimate, "torch", dtype=torch.float64, device=estimate.device)
    tgt = convert(target, "torch", dtype=torch.float64, device=target.device)
    return torch.mean((est - tgt) ** 2)
