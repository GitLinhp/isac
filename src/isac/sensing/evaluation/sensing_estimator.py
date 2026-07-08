"""MUSIC bin 峰 → 物理量换算与日志。

本模块是感知流水线中的 **估计** 阶段：将裁切谱局部 bin 索引换算为时延/多普勒或距离/速度，
不负责检峰、ROI 裁切或与真值对齐的 RMSE 评估。

典型流水线：

1. :class:`~isac.sensing.detection.music_estimator.MUSICEstimator` 对裁切 DD 谱检峰，
   输出 :class:`~isac.data_structures.types.MusicPeaks`；
2. 本模块 :class:`SensingEstimator` 将 ``MusicPeaks`` 换算为
   :class:`~isac.data_structures.types.SensingEstimate`；
3. 下游 :func:`~isac.sensing.evaluation.sensing_evaluator.match_peaks_and_compute_radial_rmse`
   与几何真值做匈牙利匹配。

参见 ``script/simulation/sensing/rt/run_sensing_monostatic.py``。
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tabulate import tabulate

from ...data_structures.types import MetricMode, MusicPeaks, SensingEstimate, SensMode
from ..metric import SpectrumMetric
from ..spectrum.dd_spectrum_roi import DelayDopplerRoi
from ..spectrum.sensing_performance import SensingPerformance

# metric_mode → 谱峰日志表头（dd: 时延-多普勒；rv: 距离-速度）
_PEAK_TABLE_HEADERS: dict[MetricMode, list[str]] = {
    "dd": [
        "峰值",
        "时延索引",
        "多普勒索引",
        "时延 (ns)",
        "多普勒 (Hz)",
    ],
    "rv": [
        "峰值",
        "时延bin",
        "多普勒bin",
        "距离 (m)",
        "速度 (m/s)",
    ],
}


class SensingEstimator:
    """MUSIC bin 峰 → 物理量换算与日志。

    bin 检峰由外部 :class:`~isac.sensing.detection.music_estimator.MUSICEstimator` 完成；
    对实例直接调用 ``estimator(peaks, ...)`` 将 :class:`~isac.data_structures.types.MusicPeaks`
    换算为 :class:`~isac.data_structures.types.SensingEstimate`。

    换算委托 :class:`~isac.sensing.metric.SpectrumMetric`：
    局部 bin → (τ, f_d) → (距离 m, 速度 m/s)，其中多普勒中心由 ROI 裁切后的
    ``num_doppler_bins`` 或全谱时的 ``rg.num_ofdm_symbols`` 决定。

    ``metric_mode`` 仅影响日志表头与展示列，不改变返回值。
    ``sens_mode`` 影响单基地往返 / 双基地单程的物理尺度。
    """

    def __init__(
        self,
        sensing_performance: SensingPerformance,
        device: Union[str, torch.device],
        dd_spectrum_roi: Optional[DelayDopplerRoi] = None,
    ):
        """初始化感知估计器。

        参数
        ----
        - sensing_performance :
            感知性能对象，提供 ``delay_resolution``、``doppler_resolution``、
            载频等，供 :class:`~isac.sensing.metric.SpectrumMetric` 做 bin→物理量换算。
        - device :
            换算结果张量的目标设备（``SpectrumMetric`` 默认在 CPU 换算，此处再 ``.to(device)``）。
        - dd_spectrum_roi :
            与 :class:`~isac.sensing.spectrum.DelayDopplerSpectrum` 共享的 ROI 实例；
            ``None`` 时表示全谱，``__call__`` 使用 ``rg.num_ofdm_symbols`` 作多普勒 bin 数。
        """
        self.sensing_performance = sensing_performance
        self._dd_spectrum_roi = dd_spectrum_roi
        self._metric = SpectrumMetric(sensing_performance)
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )

    def __call__(
        self,
        peaks: MusicPeaks,
        *,
        sens_mode: SensMode = "monostatic",
        metric_mode: MetricMode = "rv",
        log_peaks: bool = True,
    ) -> SensingEstimate:
        """将 MUSIC bin 检峰结果换算为距离/速度。

        参数
        ----
        - peaks :
            :class:`~isac.data_structures.types.MusicPeaks`：``peaks_delay`` /
            ``peaks_doppler`` 为裁切谱局部 bin 索引。
        - sens_mode :
            物理换算尺度：``monostatic`` 为往返路径，``bistatic`` 为单程折叠路径长。
        - metric_mode :
            仅影响 ``log_peaks`` 表格列名与单位（``dd``: ns/Hz；``rv``: m/m/s）。
        - log_peaks :
            是否打印谱峰表格；无峰时打印提示行。

        返回
        ----
        :class:`~isac.data_structures.types.SensingEstimate`：
        ``est_ranges`` / ``est_velocities`` 为 1D ``float64`` 张量。
        """
        num_doppler_bins = (
            self._dd_spectrum_roi.num_doppler_bins
            if self._dd_spectrum_roi is not None
            else self.sensing_performance.rg.num_ofdm_symbols
        )
        tau_s, fd_hz, range_m, v_mps = self._metric.local_bins_to_range_velocity(
            peaks.peaks_delay,
            peaks.peaks_doppler,
            num_doppler_bins=num_doppler_bins,
            sens_mode=sens_mode,
        )
        if peaks.peaks_delay.numel() > 0:
            dev = self.device
            tau_s = tau_s.to(device=dev)
            fd_hz = fd_hz.to(device=dev)
            range_m = range_m.to(device=dev)
            v_mps = v_mps.to(device=dev)

        if log_peaks:
            if peaks.peaks_delay.numel() > 0:
                self._log_peak_table(
                    peaks=peaks,
                    metric_mode=metric_mode,
                    physics=(tau_s, fd_hz, range_m, v_mps),
                )
            else:
                print("MUSIC算法未检测到谱峰\n")

        return SensingEstimate(
            est_ranges=range_m.reshape(-1),
            est_velocities=v_mps.reshape(-1),
        )

    def _log_peak_table(
        self,
        peaks: MusicPeaks,
        metric_mode: MetricMode,
        *,
        physics: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """将估计谱峰格式化为表格并打印。"""
        print(f"使用MUSIC算法检测到 {peaks.peaks_delay.numel()} 个谱峰:")

        peaks_delay_np = peaks.peaks_delay.cpu().numpy()
        peaks_doppler_np = peaks.peaks_doppler.cpu().numpy()
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
            ]
            for i, (delay_idx, doppler_idx) in enumerate(
                zip(peaks_delay_np, peaks_doppler_np)
            )
        ]
        print(tabulate(table_data, headers=headers, tablefmt="simple_grid") + "\n")
