"""MUSIC 感知评估：bin 峰 → 物理量换算 → 匈牙利 RMSE 匹配。

bin 检峰由上游 :class:`~isac.sensing.detection.music_estimator.MUSICEstimator` 完成；
本模块接收其输出的 ``(delay_bin, doppler_bin, power)``，不做 ROI 裁切或全谱检峰。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from ...utils.numerical import linear_to_db
from ...utils.type_converter import convert
from ..metric import SpectrumMetric
from ..types import MetricMode, SensMode
from ..spectrum.sensing_performance import SensingPerformance

_PEAK_TABLE_HEADERS: dict[MetricMode, list[str]] = {
    "dd": [
        "峰值",
        "时延索引",
        "多普勒索引",
        "时延 (ns)",
        "多普勒 (Hz)",
        "Score (dB)",
    ],
    "rv": [
        "峰值",
        "时延bin",
        "多普勒bin",
        "距离 (m)",
        "速度 (m/s)",
        "Score (dB)",
    ],
}


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


@dataclass
class MusicEvaluationResult:
    """MUSIC 估计与匈牙利 RMSE 匹配结果。"""

    est_ranges: torch.Tensor
    est_velocities: torch.Tensor
    peaks_power: torch.Tensor
    rmse_range_m: torch.Tensor
    rmse_velocity_mps: torch.Tensor
    est_range_m: torch.Tensor
    est_velocity_mps: torch.Tensor
    music_peak_db: torch.Tensor


class MusicSensingEvaluator:
    """MUSIC 感知评估器：bin 峰 → 物理量换算 → 可选 RMSE。

    bin 检峰由外部 :class:`~isac.sensing.detection.music_estimator.MUSICEstimator` 完成；
    ``estimate`` 负责物理量换算与日志，``evaluate`` 额外做匈牙利 RMSE 匹配。

    **estimate 返回值恒为** ``(distance_m, velocity_mps, peaks_power)``；
    ``metric_mode`` 仅影响日志表头与展示列，不改变返回值。
    """

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        device: Union[str, torch.device],
    ):
        """初始化 MUSIC 感知评估器。

        参数
        ----
        - sensing_performance :
            感知性能对象，用于 bin→物理量换算。
        - device :
            张量计算设备（用于换算结果与日志）。
        """
        self.sensing_performance = sensing_performance
        self._metric = SpectrumMetric(sensing_performance)
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )

    def estimate(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        peaks_power: torch.Tensor,
        *,
        num_doppler_bins: int,
        sens_mode: SensMode = "monostatic",
        metric_mode: MetricMode = "rv",
        log_peaks: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 MUSIC bin 检峰结果换算为距离/速度。

        参数
        ----
        - peaks_delay / peaks_doppler / peaks_power :
            :class:`MUSICEstimator` 输出的裁切谱局部 bin 索引与功率。
        - num_doppler_bins :
            裁切谱多普勒维长度（与输入谱 ``shape[0]`` 一致）。
        - sens_mode :
            物理换算尺度（``monostatic`` 往返 / ``bistatic`` 单程）。
        - metric_mode :
            仅影响日志列名与单位（bin+ns/Hz 或 m/m/s）。
        - log_peaks :
            是否打印谱峰表格。

        返回
        ----
        ``(distance_m, velocity_mps, peaks_power)``：距离/速度 ``float64``，功率 ``float32``。
        """
        tau_s, fd_hz, range_m, v_mps = self._metric.local_bins_to_range_velocity(
            peaks_delay,
            peaks_doppler,
            num_doppler_bins=num_doppler_bins,
            sens_mode=sens_mode,
        )
        if peaks_delay.numel() > 0:
            dev = self.device
            tau_s = tau_s.to(device=dev)
            fd_hz = fd_hz.to(device=dev)
            range_m = range_m.to(device=dev)
            v_mps = v_mps.to(device=dev)

        if log_peaks:
            if peaks_delay.numel() > 0:
                self._log_peak_table(
                    peaks_delay=peaks_delay,
                    peaks_doppler=peaks_doppler,
                    peaks_power=peaks_power,
                    metric_mode=metric_mode,
                    physics=(tau_s, fd_hz, range_m, v_mps),
                )
            else:
                print("MUSIC算法未检测到谱峰")

        return range_m.reshape(-1), v_mps.reshape(-1), peaks_power

    def evaluate(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        peaks_power: torch.Tensor,
        *,
        num_doppler_bins: int,
        true_ranges: torch.Tensor,
        true_velocities: torch.Tensor,
        sens_mode: SensMode = "monostatic",
        metric_mode: MetricMode = "rv",
        log_peaks: bool = True,
        label: str = "单基地感知",
        distance_axis_label: str = "径向距离",
        velocity_axis_label: str = "径向速度",
        verbose: bool = True,
    ) -> MusicEvaluationResult:
        """bin 峰换算为物理量并与真值格点做匈牙利匹配，返回 RMSE 与最佳配对估计。"""
        est_ranges, est_velocities, peaks_power_out = self.estimate(
            peaks_delay,
            peaks_doppler,
            peaks_power,
            num_doppler_bins=num_doppler_bins,
            sens_mode=sens_mode,
            metric_mode=metric_mode,
            log_peaks=log_peaks,
        )
        (
            rmse_range_m,
            rmse_velocity_mps,
            est_range_m,
            est_velocity_mps,
            music_peak_db,
        ) = match_peaks_and_compute_radial_rmse(
            est_ranges=est_ranges,
            est_velocities=est_velocities,
            true_ranges=true_ranges,
            true_velocities=true_velocities,
            label=label,
            distance_axis_label=distance_axis_label,
            velocity_axis_label=velocity_axis_label,
            verbose=verbose,
        )
        return MusicEvaluationResult(
            est_ranges=est_ranges,
            est_velocities=est_velocities,
            peaks_power=peaks_power_out,
            rmse_range_m=rmse_range_m,
            rmse_velocity_mps=rmse_velocity_mps,
            est_range_m=est_range_m,
            est_velocity_mps=est_velocity_mps,
            music_peak_db=music_peak_db,
        )

    def _log_peak_table(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        peaks_power: torch.Tensor,
        metric_mode: MetricMode,
        *,
        physics: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """将估计谱峰格式化为表格并打印。"""
        print(f"使用MUSIC算法检测到 {peaks_delay.numel()} 个谱峰:")

        peaks_delay_np = peaks_delay.cpu().numpy()
        peaks_doppler_np = peaks_doppler.cpu().numpy()
        peaks_power_np = peaks_power.cpu().numpy()
        tau_s, fd_hz, range_m, v_mps = physics
        tau_ns_np = (tau_s * 1e9).detach().cpu().numpy().reshape(-1)
        fd_hz_np = fd_hz.detach().cpu().numpy().reshape(-1)
        range_m_np = range_m.detach().cpu().numpy().reshape(-1)
        v_mps_np = v_mps.detach().cpu().numpy().reshape(-1)

        use_dd = metric_mode == "dd"
        phys_a_np = tau_ns_np if use_dd else range_m_np
        phys_b_np = fd_hz_np if use_dd else v_mps_np
        headers = _PEAK_TABLE_HEADERS[metric_mode]

        table_data = [
            [
                i + 1,
                int(round(float(delay_idx))),
                int(round(float(doppler_idx))),
                f"{float(phys_a_np[i]):.2f}",
                f"{float(phys_b_np[i]):.2f}",
                f"{linear_to_db(float(power), is_power=True, return_type='float'):.2f}",
            ]
            for i, (delay_idx, doppler_idx, power) in enumerate(
                zip(peaks_delay_np, peaks_doppler_np, peaks_power_np)
            )
        ]
        print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))
