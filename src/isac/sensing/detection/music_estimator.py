"""2D-MUSIC 时延-多普勒谱峰估计。

流程概览
--------
1. 裁剪搜索窗（近距保护 + 可选 ``search_range``）
2. 子阵快拍 → 样本协方差 → 噪声子空间（``eigh`` + 对角加载）
3. ``|谱|`` 局部极大值筛候选点（可选 CFAR 门限）
4. 对候选点算 MUSIC 伪谱 × 幅度，贪心去重选峰
5. bin → τ, f_d → 距离/速度；``metric_mode`` 仅影响日志列

术语
----
- **sens_mode**（``monostatic`` / ``bistatic``）：物理换算尺度，作用于 ``delay_to_range`` /
  ``doppler_to_velocity``。
- **metric_mode**（``delay_doppler`` / ``range_velocity`` 及别名）：日志表头与展示单位，
  **不改变** ``__call__`` 返回值。

常数（定义于 :mod:`~isac.sensing.music_kernels`）
-------------------------------------------------
``SUBARRAY_SIZE``、``NUM_SNAPSHOTS``、``MAX_CANDIDATES``、``MAX_PEAKS`` 等。
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from tabulate import tabulate

from ...utils import linear_to_db
from ..spectrum.delay_doppler_spectrum import compute_dd_roi_slices
from .metric_mode import SensMode, canonical_metric_mode, metric_mode
from .music_kernels import (
    COV_DIAG_LOAD_MIN,
    COV_DIAG_LOAD_REL,
    MAX_CANDIDATES,
    MIN_SEARCH_DIMENSION,
    MUSIC_EPS,
    NUM_SNAPSHOTS,
    SubarrayGeometry,
    batch_music_scores,
    greedy_select_peaks,
    local_maxima_candidates,
    noise_subspace_from_loaded_covariance,
    resolve_num_output_peaks,
    subarray_geometry,
)
from ..spectrum.sensing_performance import SensingPerformance
from ..geometry import delay_to_range, doppler_to_velocity

_PEAK_TABLE_HEADERS: dict[metric_mode, list[str]] = {
    "delay_doppler": [
        "峰值",
        "时延索引",
        "多普勒索引",
        "时延 (ns)",
        "多普勒 (Hz)",
        "功率 (dBm)",
    ],
    "range_velocity": [
        "峰值",
        "时延bin",
        "多普勒bin",
        "距离 (m)",
        "速度 (m/s)",
        "功率 (dBm)",
    ],
}


def _linear_mag_to_dbm_scalar(power: torch.Tensor | float) -> float:
    """线性幅度 → dBm 标量，供日志表格式化。"""
    power_dbm = linear_to_db(power, is_power=False)
    if isinstance(power_dbm, (int, float)):
        return float(power_dbm)
    return float(power_dbm.item())


class _CallContext(NamedTuple):
    """``__call__`` 准备阶段产物。"""

    search_region: torch.Tensor
    cfar_region: Optional[torch.Tensor]
    delay_start: int
    doppler_start: int
    num_doppler_bins: int
    num_delay_bins: int
    bin_origin: Tuple[int, int]
    metric_mode_canon: metric_mode
    search_too_small: bool


class MUSICEstimator:
    """MUSIC 算法估计器（编排层）。

    使用 2D-MUSIC（子阵/空间平滑 + 候选点扫描）估计谱峰；对实例直接调用
    ``estimator(spectrum_tensor, ...)`` 即可。

    **返回值恒为** ``(distance_m, velocity_mps, peaks_power)``；``metric_mode`` 仅影响
    日志表头与展示列，不改变返回值。

    ``__call__`` 流程：**检峰 (bin+功率) → 物理量 (τ, f_d, r, v) → 按 metric_mode 打日志**。

    设计目标：

    - 避免对整张 (M×N) 展开构造巨大协方差矩阵导致爆内存；
    - 用小子阵 (Md×Nd) 的快拍构造协方差，大小为 (Md·Nd)×(Md·Nd)；
    - 不对全网格扫描：先在 |X| 上做局部极大值筛候选点，再用 MUSIC 伪谱打分选峰。
    """

    def __init__(
        self,
        device: torch.device,
        sensing_performance: Optional[SensingPerformance] = None,
        near_range_guard_m: float = 0.0,
        max_range_m: Optional[float] = None,
        max_velocity_mps: Optional[float] = None,
    ):
        """初始化 MUSIC 估计器。

        参数
        ----
        device :
            计算设备。
        sensing_performance :
            感知性能对象；``__call__`` 须注入以完成 bin→物理量换算与日志。
        near_range_guard_m :
            默认近距保护物理距离 (m)；跳过 ``[0, guard)`` 对应时延 bin。
        max_range_m, max_velocity_mps :
            ``[dd_spectrum_roi]`` 物理 ROI；二者均给定且已注入 ``sensing_performance`` 时，
            预计算裁剪谱在全网格中的 ``bin_origin``。
        """
        self.device = device
        self.sensing_performance = sensing_performance
        self.near_range_guard_m = near_range_guard_m
        self._bin_origin: Tuple[int, int] = (0, 0)
        if (
            sensing_performance is not None
            and max_range_m is not None
            and max_velocity_mps is not None
        ):
            rg = sensing_performance.rg
            dop_start, _, delay_start, _ = compute_dd_roi_slices(
                sensing_performance,
                max_range_m,
                max_velocity_mps,
                rg.num_ofdm_symbols,
                rg.fft_size,
            )
            self._bin_origin = (dop_start, delay_start)

    def __call__(
        self,
        spectrum_tensor: torch.Tensor,
        num_sources: Optional[int] = None,
        search_range: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.1,
        cfar: Optional[torch.Tensor] = None,
        metric_mode: metric_mode = "delay_doppler",
        *,
        sens_mode: SensMode = "monostatic",
        near_range_guard_m: Optional[float] = None,
        log_peaks: bool = True,
        bin_origin: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D-MUSIC 谱峰估计。

        准备 → 检峰 → 物理量 → 日志（四段）；``bin_origin`` 为 ``None`` 时使用构造时
        由 ``[dd_spectrum_roi]`` 预计算的 ``self._bin_origin``。

        参数
        ----
        spectrum_tensor :
            时延多普勒谱，squeeze 后须为 ``(num_symbols, num_subcarriers)``。
        num_sources :
            信号源数量；``None`` 时自动估计（协方差特征值）且默认输出 2 个峰。
        search_range :
            ``(delay_start, delay_end, doppler_start, doppler_end)``。
        threshold :
            自动估计信号源数量时的归一化特征值阈值。
        cfar :
            与谱同形状的二维阈值面；给定则候选门限为 ``|X| > cfar``。
        metric_mode :
            仅影响日志列名与单位（bin+ns/Hz 或 m/m/s）。
        sens_mode :
            物理换算尺度（``monostatic`` 往返 / ``bistatic`` 单程）。
        near_range_guard_m :
            单次调用覆盖近距保护距离；``None`` 用构造默认值。
        log_peaks :
            是否打印谱峰表格。
        bin_origin :
            裁剪谱在全网格中的 ``(doppler_start, delay_start)`` 偏移。

        返回
        ----
        ``(distance_m, velocity_mps, peaks_power)``：距离/速度 ``float64``，功率 ``float32``。
        """
        if self.sensing_performance is None:
            raise ValueError(
                "MUSICEstimator 须在构造时注入 sensing_performance，以将谱峰 bin 换为距离/速度物理量"
            )

        ctx = self._prepare_call_context(
            spectrum_tensor,
            search_range=search_range,
            cfar=cfar,
            metric_mode=metric_mode,
            sens_mode=sens_mode,
            near_range_guard_m=near_range_guard_m,
            bin_origin=bin_origin,
        )
        if ctx.search_too_small:
            return self._return_empty_peaks()

        noise_subspace = self._build_covariance_and_noise_subspace(
            ctx.search_region,
            ctx.num_doppler_bins,
            ctx.num_delay_bins,
            num_sources,
            threshold,
        )
        if noise_subspace is None:
            print("MUSIC算法：协方差特征分解失败，本次无谱峰输出")
            return self._return_empty_peaks()

        peaks_delay, peaks_doppler, peaks_power = self._score_and_select_peaks(
            ctx.search_region,
            noise_subspace,
            ctx.num_doppler_bins,
            ctx.num_delay_bins,
            ctx.delay_start,
            ctx.doppler_start,
            num_sources,
            cfar_region=ctx.cfar_region,
        )

        tau_s, fd_hz, range_m, v_mps = self._compute_peak_physics(
            peaks_delay,
            peaks_doppler,
            sens_mode=sens_mode,
            bin_origin=ctx.bin_origin,
        )

        if log_peaks:
            if peaks_delay.numel() > 0:
                self._log_peak_table(
                    peaks_delay=peaks_delay,
                    peaks_doppler=peaks_doppler,
                    peaks_power=peaks_power,
                    metric_mode=ctx.metric_mode_canon,
                    physics=(tau_s, fd_hz, range_m, v_mps),
                )
            else:
                print("MUSIC算法未检测到谱峰")

        return self._finalize_music_return(range_m, v_mps, peaks_power)

    def _prepare_call_context(
        self,
        spectrum_tensor: torch.Tensor,
        *,
        search_range: Optional[Tuple[int, int, int, int]],
        cfar: Optional[torch.Tensor],
        metric_mode: metric_mode,
        sens_mode: SensMode,
        near_range_guard_m: Optional[float],
        bin_origin: Optional[Tuple[int, int]],
    ) -> _CallContext:
        """校验输入、裁剪搜索窗，返回后续检峰所需上下文。"""
        origin = self._bin_origin if bin_origin is None else bin_origin
        metric_mode_canon = canonical_metric_mode(metric_mode)

        spectrum = torch.squeeze(
            spectrum_tensor.clone().to(device=self.device, dtype=torch.complex64)
        )
        if spectrum.ndim != 2:
            raise ValueError(
                f"spectrum_tensor 需要是二维矩阵，当前形状: {tuple(spectrum.shape)}"
            )
        num_symbols, num_subcarriers = spectrum.shape

        cfar_full: Optional[torch.Tensor] = None
        if cfar is not None:
            cfar_full = cfar.to(device=self.device)
            if cfar_full.shape != spectrum.shape:
                raise ValueError(
                    "cfar 须与 squeeze 后的 spectrum_tensor 同形状，"
                    f"当前 cfar {tuple(cfar_full.shape)}，谱 {tuple(spectrum.shape)}"
                )

        guard_m = (
            self.near_range_guard_m
            if near_range_guard_m is None
            else near_range_guard_m
        )
        delay_start, delay_end, doppler_start, doppler_end = self._get_search_range(
            search_range,
            num_subcarriers,
            num_symbols,
            sens_mode=sens_mode,
            near_range_guard_m=guard_m,
        )

        num_doppler_bins = int(doppler_end - doppler_start)
        num_delay_bins = int(delay_end - delay_start)
        search_too_small = (
            num_doppler_bins < MIN_SEARCH_DIMENSION
            or num_delay_bins < MIN_SEARCH_DIMENSION
        )

        search_region = spectrum[doppler_start:doppler_end, delay_start:delay_end]
        cfar_region = None
        if cfar_full is not None:
            cfar_region = cfar_full[doppler_start:doppler_end, delay_start:delay_end]

        return _CallContext(
            search_region=search_region,
            cfar_region=cfar_region,
            delay_start=delay_start,
            doppler_start=doppler_start,
            num_doppler_bins=num_doppler_bins,
            num_delay_bins=num_delay_bins,
            bin_origin=origin,
            metric_mode_canon=metric_mode_canon,
            search_too_small=search_too_small,
        )

    def _compute_peak_physics(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        *,
        sens_mode: SensMode,
        bin_origin: Tuple[int, int] = (0, 0),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 bin 批量得到 τ(s)、f_d(Hz)、距离 (m)、速度 (m/s)。"""
        sp = self.sensing_performance
        assert sp is not None
        dev = self.device
        dop_origin, delay_origin = bin_origin
        d = peaks_delay.detach().to(device=dev, dtype=torch.float64).reshape(-1)
        dop = peaks_doppler.detach().to(device=dev, dtype=torch.float64).reshape(-1)
        if d.numel() == 0:
            z = torch.empty(0, dtype=torch.float64, device=dev)
            return z, z, z, z

        dt = float(sp.delay_resolution)
        dres = float(sp.doppler_resolution)
        half = float(sp.rg.num_ofdm_symbols // 2)
        tau_s = (d + delay_origin) * dt
        fd_hz = (dop + dop_origin - half) * dres
        fc = float(sp.carrier_frequency)
        range_m = delay_to_range(tau_s, fc, sens_mode)
        v_mps = doppler_to_velocity(fd_hz, fc, sens_mode)
        return tau_s, fd_hz, range_m.reshape(-1), v_mps.reshape(-1)

    def _finalize_music_return(
        self,
        range_m: torch.Tensor,
        velocity_mps: torch.Tensor,
        peaks_power: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一返回出口：``(distance_m, velocity_mps, peaks_power)``。"""
        return range_m.reshape(-1), velocity_mps.reshape(-1), peaks_power

    def _return_empty_peaks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """搜索域过小或协方差失败时返回空三元组。"""
        _, _, empty_power = self._empty_peak_tensors()
        dev = self.device
        empty_phys = torch.empty(0, dtype=torch.float64, device=dev)
        return self._finalize_music_return(empty_phys, empty_phys, empty_power)

    def _log_peak_table(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        peaks_power: torch.Tensor,
        metric_mode: metric_mode,
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

        use_dd = metric_mode == "delay_doppler"
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
                f"{_linear_mag_to_dbm_scalar(power):.2f}",
            ]
            for i, (delay_idx, doppler_idx, power) in enumerate(
                zip(peaks_delay_np, peaks_doppler_np, peaks_power_np)
            )
        ]
        print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))

    def _get_search_range(
        self,
        search_range: Optional[Tuple[int, int, int, int]],
        num_subcarriers: int,
        num_symbols: int,
        *,
        sens_mode: SensMode = "monostatic",
        near_range_guard_m: float = 1.0,
    ) -> Tuple[int, int, int, int]:
        """返回 ``(delay_start, delay_end, doppler_start, doppler_end)``。"""
        if search_range is None:
            sp = self.sensing_performance
            if sp is None:
                raise ValueError(
                    "search_range 为 None 时须注入 sensing_performance，"
                    "以按 near_range_guard_m 换算近距保护 bin"
                )
            delay_guard = sp.near_delay_guard_bins(near_range_guard_m, sens_mode)
            delay_guard = min(max(0, delay_guard), num_subcarriers - 1)
            return delay_guard, num_subcarriers, 0, num_symbols
        return search_range

    def _empty_peak_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """无谱峰时返回 ``self.device`` 侧三个空一维张量。"""
        dev = self.device
        return (
            torch.empty(0, dtype=torch.float64, device=dev),
            torch.empty(0, dtype=torch.float64, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
        )

    def _build_covariance_and_noise_subspace(
        self,
        search_region: torch.Tensor,
        num_doppler_bins: int,
        num_delay_bins: int,
        num_sources: Optional[int],
        threshold: float,
    ) -> Optional[torch.Tensor]:
        """子阵快拍 → 样本协方差 → 噪声子空间；分解失败返回 ``None``。"""
        geometry = subarray_geometry(num_doppler_bins, num_delay_bins, self.device)
        max_doppler_offset = num_doppler_bins - geometry.doppler_size
        max_delay_offset = num_delay_bins - geometry.delay_size

        if max_doppler_offset <= 0 or max_delay_offset <= 0:
            raise ValueError("搜索区域太小，无法构建子阵")

        doppler_indices = torch.randint(
            0, max_doppler_offset + 1, (NUM_SNAPSHOTS,), device=self.device
        )
        delay_indices = torch.randint(
            0, max_delay_offset + 1, (NUM_SNAPSHOTS,), device=self.device
        )

        # 随机子阵偏移的 patch 提取难以向量化且不影响主热点，保留逐快拍循环。
        snapshots = torch.empty(
            (geometry.size, NUM_SNAPSHOTS),
            dtype=torch.complex64,
            device=self.device,
        )
        for t in range(NUM_SNAPSHOTS):
            patch = search_region[
                doppler_indices[t] : doppler_indices[t] + geometry.doppler_size,
                delay_indices[t] : delay_indices[t] + geometry.delay_size,
            ]
            snapshots[:, t] = patch.reshape(-1)

        covariance_matrix = (snapshots @ snapshots.conj().T) / NUM_SNAPSHOTS
        covariance_matrix = (
            covariance_matrix + covariance_matrix.conj().transpose(-2, -1)
        ) * 0.5

        identity = torch.eye(geometry.size, dtype=torch.complex64, device=self.device)
        trace_real = torch.real(torch.diagonal(covariance_matrix).sum()).clamp(
            min=MUSIC_EPS
        )
        base_load = (trace_real / float(geometry.size)) * COV_DIAG_LOAD_REL
        base_load_f = max(float(base_load.item()), COV_DIAG_LOAD_MIN)

        return noise_subspace_from_loaded_covariance(
            covariance_matrix.to(torch.complex128),
            identity.to(torch.complex128),
            base_load_f,
            num_sources,
            threshold,
            geometry.size,
        )

    def _score_and_select_peaks(
        self,
        search_region: torch.Tensor,
        noise_subspace: torch.Tensor,
        num_doppler_bins: int,
        num_delay_bins: int,
        delay_start: int,
        doppler_start: int,
        num_sources: Optional[int],
        *,
        cfar_region: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """候选检峰 → 批量 MUSIC 评分 → 贪心去重 → 全局 bin 坐标。"""
        magnitude = torch.abs(search_region)
        candidates = local_maxima_candidates(
            magnitude,
            cfar=cfar_region,
            max_candidates=MAX_CANDIDATES,
        )
        geometry = subarray_geometry(num_doppler_bins, num_delay_bins, self.device)
        scores = batch_music_scores(
            candidates,
            magnitude,
            noise_subspace,
            num_doppler_bins=num_doppler_bins,
            num_delay_bins=num_delay_bins,
            geometry=geometry,
        )

        num_output = resolve_num_output_peaks(num_sources)
        sel_scores, sel_dop, sel_delay = greedy_select_peaks(
            scores,
            candidates[:, 0],
            candidates[:, 1],
            num_output,
        )

        peaks_doppler = sel_dop + doppler_start
        peaks_delay = sel_delay + delay_start
        return peaks_delay, peaks_doppler, sel_scores
