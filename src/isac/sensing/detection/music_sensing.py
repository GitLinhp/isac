"""MUSIC 感知编排：裁切谱 bin → 物理量换算 → 可选日志。

输入须为上游 :class:`~isac.sensing.spectrum.DelayDopplerSpectrum` 按
``[dd_spectrum_roi]`` 裁切后的时延多普勒谱；本模块不做 ROI 裁切或全谱坐标映射。
检峰委托 :class:`~isac.sensing.detection.music_estimator.MUSICEstimator`。
"""

from __future__ import annotations

from typing import Tuple

import torch
from tabulate import tabulate

from ...utils.numerical import linear_to_db
from ..metric import SpectrumMetric
from ..types import MetricMode, SensMode
from ..spectrum.sensing_performance import SensingPerformance
from .music_estimator import MUSICEstimator

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


class MusicSensingEstimator:
    """MUSIC 感知编排层：ROI 裁切谱 → 物理量换算与可选日志。

    调用 :class:`MUSICEstimator` 完成 bin 检峰；本类仅负责 ROI 局部坐标下的
    物理量换算与 ``metric_mode`` 日志展示。

    **返回值恒为** ``(distance_m, velocity_mps, peaks_power)``；``metric_mode`` 仅影响
    日志表头与展示列，不改变返回值。
    """

    def __init__(
        self,
        music_estimator: MUSICEstimator,
        sensing_performance: SensingPerformance,
    ):
        """初始化 MUSIC 感知编排器。

        参数
        ----
        - music_estimator :
            bin 检峰器。
        - sensing_performance :
            感知性能对象，用于 bin→物理量换算。
        """
        self.music_estimator = music_estimator
        self.sensing_performance = sensing_performance
        self._metric = SpectrumMetric(sensing_performance)
        self.device = music_estimator.device

    def __call__(
        self,
        spectrum_tensor: torch.Tensor,
        *,
        num_sources: int | None = None,
        threshold: float = 0.1,
        cfar: torch.Tensor | None = None,
        sens_mode: SensMode = "monostatic",
        metric_mode: MetricMode = "rv",
        log_peaks: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D-MUSIC 谱峰估计并换算为距离/速度。

        参数
        ----
        - spectrum_tensor :
            ROI 裁切后的时延多普勒谱，squeeze 后须为 ``(num_doppler, num_delay)``。
        - num_sources :
            信号源数量；``None`` 时自动估计（协方差特征值）且默认输出 2 个峰。
        - threshold :
            自动估计信号源数量时的归一化特征值阈值。
        - cfar :
            与谱同形状的二维阈值面；给定则候选门限为 ``|X| > cfar``。
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
        spectrum = torch.squeeze(spectrum_tensor)
        if spectrum.ndim != 2:
            raise ValueError(
                f"spectrum_tensor 需要是二维矩阵，当前形状: {tuple(spectrum.shape)}"
            )
        num_doppler_bins = int(spectrum.shape[0])

        peaks_delay, peaks_doppler, peaks_power = self.music_estimator(
            spectrum_tensor,
            num_sources=num_sources,
            threshold=threshold,
            cfar=cfar,
        )

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
