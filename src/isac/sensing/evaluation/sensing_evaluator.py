"""MUSIC 感知 RMSE 评估：估计物理量与几何真值之间的匈牙利匹配。

典型流水线：

1. :class:`~isac.sensing.detection.music_estimator.MUSICEstimator` 检峰；
2. :class:`~isac.sensing.evaluation.sensing_estimator.SensingEstimator` 将
   :class:`~isac.data_structures.types.MusicPeaks` 换算为
   :class:`~isac.data_structures.types.SensingEstimate`；
3. 本模块 ``match_peaks_and_compute_radial_rmse`` 将 ``est_ranges`` /
   ``est_velocities`` 与几何真值对齐并计算径向 RMSE。

估计量亦可来自 CNN 等非 MUSIC 路径，只要提供等长的距离/速度向量即可。
真值通常取自 :class:`~isac.channel.rt.rx_target_tx_geometric.RxTargetTxGeometric`
的 ``range_tensor`` / ``vel_tensor``（可 ``reshape(-1)`` 传入）。
参见 ``script/simulation/sensing/rt/run_sensing_monostatic.py``。
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from ...utils.type_converter import convert


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
    """用匈牙利算法在估计峰与真值格点间做一对一最小代价匹配，并计算径向 RMSE。

    代价 ``C[i,j] = (er_i-tr_j)^2 + (ev_i-tv_j)^2``（联合平方误差）。
    ``N != M`` 时 SciPy :func:`~scipy.optimize.linear_sum_assignment` 返回
    ``min(N, M)`` 条最优部分匹配；``verbose=True`` 时会打印峰数、真值点数与匹配条数。

    参数
    ----
    - est_ranges :
        估计径向距离 (m)，1D ``torch.Tensor``，与 ``est_velocities`` 等长非空。
    - est_velocities :
        估计径向速度 (m/s)，1D ``torch.Tensor``，与 ``est_ranges`` 等长非空。
    - true_ranges :
        真值距离 (m)，与 ``true_velocities`` 形状一致；多维张量会在内部 ``reshape(-1)``。
    - true_velocities :
        真值速度 (m/s)，与 ``true_ranges`` 形状一致。
    - label :
        日志行前缀，用于区分仿真场景或 episode。
    - distance_axis_label / velocity_axis_label :
        verbose 输出中的轴名称；双基地场景可改为「路径长」等。
    - verbose :
        是否打印匈牙利匹配统计与 RMSE 行。

    返回
    ----
    五元组 ``(rmse_range_m, rmse_velocity_mps, est_range_m, est_velocity_mps, music_peak_db)``，
    dtype 均为 ``float64``，device 与 ``est_ranges`` 一致。

    - ``rmse_range_m`` / ``rmse_velocity_mps``：在 **K 条匈牙利匹配对** 上分别对距离、速度求 RMS。
    - ``est_range_m`` / ``est_velocity_mps``：零维张量，取匹配对内联合误差最小的那一对的估计值
      （仅用于 verbose 展示，不代表 RMSE 计算所用的单一配对）。
    - ``music_peak_db``：历史兼容占位，恒为 NaN；调用方通常解包为 ``_``。

    异常
    ----
    - ``ValueError``：估计或真值为空，或 ``est_*`` / ``true_*`` 形状不一致。
    - ``RuntimeError``：匈牙利匹配未产生任何配对（``k == 0``）。
    """
    dtype = torch.float64
    device = est_ranges.device
    # 统一 float64 与 device，拉平估计向量并校验峰数 N
    er = convert(est_ranges.reshape(-1), "torch", dtype=dtype, device=device)
    ev = convert(est_velocities.reshape(-1), "torch", dtype=dtype, device=device)
    n = er.numel()
    if n == 0 or ev.numel() != n:
        raise ValueError("est_ranges、est_velocities 须等长且非空")

    # 真值张量形状对齐后拉平为 M 个格点
    tr_conv = convert(true_ranges, "torch", dtype=dtype, device=device)
    tv_conv = convert(true_velocities, "torch", dtype=dtype, device=device)
    if tr_conv.shape != tv_conv.shape:
        raise ValueError("true_ranges 与 true_velocities 形状须一致")
    tr_raw = tr_conv.reshape(-1)
    tv_raw = tv_conv.reshape(-1)
    m = tr_raw.numel()
    if m == 0 or tv_raw.numel() != m:
        raise ValueError("true_ranges、true_velocities 须同形状或可广播为等长且非空")

    # 代价矩阵 (N, M)，在 CPU 上跑匈牙利以避免 SciPy 与 GPU 交互问题
    er_c = er.detach().cpu()
    ev_c = ev.detach().cpu()
    tr_c = tr_raw.detach().cpu()
    tv_c = tv_raw.detach().cpu()
    diff_r = er_c.unsqueeze(1) - tr_c.unsqueeze(0)
    diff_v = ev_c.unsqueeze(1) - tv_c.unsqueeze(0)
    cost_np = (diff_r.square() + diff_v.square()).numpy()

    # 匈牙利求最小联合平方误差的一对一（或 N!=M 时的部分）匹配
    row_ind, col_ind = linear_sum_assignment(cost_np)
    k = int(row_ind.shape[0])
    if verbose:
        print(f"{label} — 匈牙利匹配: MUSIC 峰数 N={n}, 真值点数 M={m}, 匹配条数 K={k}")

    if k == 0:
        raise RuntimeError("匈牙利匹配未产生任何配对")

    # 按匹配索引回取 K 对估计/真值子集
    ri = torch.from_numpy(row_ind).to(dtype=torch.long, device=device)
    cj = torch.from_numpy(col_ind).to(dtype=torch.long, device=device)
    er_m = er[ri]
    ev_m = ev[ri]
    tr_m = tr_raw[cj]
    tv_m = tv_raw[cj]

    # RMSE 在 K 条匹配对上分别对距离、速度求 RMS
    rmse_range = torch.sqrt(torch.mean((er_m - tr_m) ** 2))
    rmse_velocity = torch.sqrt(torch.mean((ev_m - tv_m) ** 2))

    # verbose 展示用：取联合误差最小的那一对
    joint_sq = (er_m - tr_m) ** 2 + (ev_m - tv_m) ** 2
    best = torch.argmin(joint_sq)
    est_range_m = er_m[best].detach()
    est_velocity_mps = ev_m[best].detach()
    true_r_show = tr_m[best].detach()
    true_v_show = tv_m[best].detach()
    # 历史兼容占位，保留旧 API 五元组签名
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
