"""MUSIC 算法实现模块

提供 2D-MUSIC 算法用于时延多普勒谱的峰值估计。
"""

import torch
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

from tabulate import tabulate

from ..utils import linear_to_db
from .sensing_performance import SensingPerformance
from .utils import delay_to_range, doppler_to_velocity

# 常量定义
_MIN_SEARCH_DIMENSION = 8  # 最小搜索维度
_SUBARRAY_SIZE = 16  # 子阵大小
_NUM_SNAPSHOTS = 2048  # 快拍数量
_MAX_CANDIDATES = 200  # 最大候选点数
_MIN_PEAK_THRESHOLD = 0.05  # 最小峰值阈值（相对于最大值）
_MAX_PEAKS = 10  # 最大峰值数量
_MUSIC_EPS = 1e-12  # 数值稳定项
# 协方差对角加载：trace/subarray_size 的比例基准，病态矩阵时按倍数加重试 eigh
_COV_DIAG_LOAD_REL = 1e-8
_COV_DIAG_LOAD_MIN = 1e-20

MusicMode = Literal["delay_doppler", "range_velocity", "dd", "rv"]
SensMode = Literal["monostatic", "bistatic"]

_MODE_CANONICAL: dict[str, MusicMode] = {
    "delay_doppler": "delay_doppler",
    "dd": "delay_doppler",
    "range_velocity": "range_velocity",
    "rv": "range_velocity",
}


def _canonical_music_mode(metric_mode: str) -> MusicMode:
    """将 ``metric_mode`` 别名解析为 ``delay_doppler`` / ``range_velocity``。"""
    key = metric_mode.strip().lower()
    try:
        return _MODE_CANONICAL[key]
    except KeyError as exc:
        raise ValueError(
            "metric_mode 须为 'delay_doppler'、'dd'、'range_velocity' 或 'rv'，"
            f"当前为: {metric_mode!r}"
        ) from exc


def _linear_mag_to_dbm_scalar(power: torch.Tensor | float) -> float:
    """线性幅度 → dBm 标量，供日志表格式化（与原先分支语义一致）。"""
    power_dbm = linear_to_db(power, is_power=False)
    if isinstance(power_dbm, (int, float)):
        return float(power_dbm)
    return float(power_dbm.item())


def _steering_vector_2d(
    doppler_idx: int,
    delay_idx: int,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    doppler_row_idx: torch.Tensor,
    delay_row_idx: torch.Tensor,
) -> torch.Tensor:
    """2D 子阵导向向量（Kronecker）；多普勒维对应 fftshift 后以网格中心为 0 的归一化频率。"""
    normalized_doppler_freq = (float(doppler_idx) - num_doppler_bins / 2.0) / num_doppler_bins
    normalized_delay_freq = float(delay_idx) / num_delay_bins
    doppler_steering = torch.exp(1j * 2 * torch.pi * normalized_doppler_freq * doppler_row_idx).to(
        torch.complex64
    )
    delay_steering = torch.exp(1j * 2 * torch.pi * normalized_delay_freq * delay_row_idx).to(
        torch.complex64
    )
    return torch.kron(doppler_steering, delay_steering)


def _noise_subspace_from_loaded_covariance(
    cov64: torch.Tensor,
    id64: torch.Tensor,
    base_load_f: float,
    num_sources: Optional[int],
    threshold: float,
    subarray_size: int,
) -> Optional[torch.Tensor]:
    """对已 ``complex128`` 化的样本协方差做对角加载并重试 ``eigh``，得到噪声子空间（``complex64``）。"""
    last_exc: Optional[BaseException] = None
    for scale in (1.0, 1e2, 1e4, 1e6, 1e8):
        load = base_load_f * scale
        r_mat = cov64 + load * id64
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(r_mat)
        except RuntimeError as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            if "eigh" not in err_lower and "converge" not in err_lower:
                raise
            continue

        sorted_indices = torch.argsort(eigenvalues.real, descending=True)
        eigenvalues = eigenvalues.real[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if num_sources is None:
            normalized_eigenvalues = eigenvalues / (eigenvalues[0] + _MUSIC_EPS)
            num_signal_sources = int(torch.sum(normalized_eigenvalues > threshold).item())
            num_signal_sources = max(1, min(num_signal_sources, subarray_size - 1))
        else:
            num_signal_sources = max(1, min(int(num_sources), subarray_size - 1))

        return eigenvectors[:, num_signal_sources:].to(torch.complex64)

    print(
        f"MUSIC: torch.linalg.eigh 在加重对角加载后仍未收敛，跳过噪声子空间估计。"
        f"末次异常: {last_exc}"
    )
    return None


class MUSICEstimator:
    """MUSIC 算法估计器

    使用 2D-MUSIC（子阵/空间平滑 + 候选点扫描）估计谱峰；对实例直接调用 ``estimator(spectrum_tensor, ...)`` 即可。
    ``sensing_performance`` 在构造时注入；``__call__`` 流程为 **检峰 (bin+功率) → 统一物理量 (τ,f_d,r,v，带符号) → 按 metric_mode 打日志**，
    并返回 **距离 (m)、速度 (m/s)** 与伪谱功率；``metric_mode`` 仅影响日志表头与选用列，不改变返回值。

    设计目标：
    - 避免对整张 (M*N) 展开构造巨大协方差矩阵导致爆内存；
    - 用小子阵 (Md×Nd) 的快拍构造协方差，大小为 (Md*Nd)×(Md*Nd)；
    - 为提速，不对全网格扫描：先在 |X| 上做局部极大值筛候选点，再用 MUSIC 伪谱打分选峰。
    """

    def __init__(
        self,
        device: torch.device,
        sensing_performance: Optional[SensingPerformance] = None,
    ):
        """初始化 MUSIC 估计器

        参数:
        ----------
        - device : torch.device
            计算设备
        - sensing_performance : SensingPerformance, 可选
            感知性能对象；用于 ``__call__`` 成功检出峰时打印表格中的物理量列。
        """
        self.device = device
        self.sensing_performance = sensing_performance

    def __call__(
        self,
        spectrum_tensor: torch.Tensor,
        num_sources: Optional[int] = None,
        search_range: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.1,
        cfar: Optional[torch.Tensor] = None,
        metric_mode: MusicMode = "delay_doppler",
        *,
        sens_mode: SensMode = "monostatic",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用 Torch 实现的 2D-MUSIC 估计谱峰（``__call__``，可直接 ``estimator(...)``）。

        参数:
        ----------
        - spectrum_tensor : torch.Tensor
            时延多普勒谱；会在本方法内 ``clone`` 后转到 ``self.device`` / ``complex64``，
            再 ``squeeze`` 去掉尺寸为 1 的维度，最终须为 (num_symbols, num_subcarriers)
        - num_sources : int, 可选
            信号源数量，如果为 None 则自动估计
        - search_range : tuple[int, int, int, int], 可选
            搜索范围 (delay_start, delay_end, doppler_start, doppler_end)
        - threshold : float
            阈值，用于自动估计信号源数量（归一化特征值），默认 0.1
        - cfar : torch.Tensor | None
            与 squeeze 后的 ``spectrum_tensor`` 同形状 ``(num_symbols, num_subcarriers)``
            的二维阈值面（通常即 ``cfar_detector(|h_dd|)``）。不为 ``None`` 时，
            候选点幅度门限改为逐点 ``|X| > cfar``；为 ``None`` 时仍用相对峰值比例门限。
        - metric_mode : {'delay_doppler', 'dd', 'range_velocity', 'rv'}
            仅影响谱峰 **日志** 的列名与展示单位（bin 索引 + ns/Hz 或 m/m/s 列），不改变返回值。
        - sens_mode : {'monostatic', 'bistatic'}
            同时作用于 ``delay_to_range`` 与 ``doppler_to_velocity``：``monostatic`` 对应径向网格往返尺度
            （``tau·c/2``、``v∝f_d/(2f_c)``）；``bistatic`` 对应单程几何路径尺度（``tau·c``、``v∝f_d/f_c``）。

        返回:
        ----------
        ``(distance_m, velocity_mps, peaks_power)``：前两维 ``float64`` 一维张量，``peaks_power`` 为 ``float32``。
        须构造时已注入 ``sensing_performance``。
        """
        if self.sensing_performance is None:
            raise ValueError(
                "MUSICEstimator 须在构造时注入 sensing_performance，以将谱峰 bin 换为距离/速度物理量"
            )

        metric_mode_canon = _canonical_music_mode(metric_mode)
        # --- 谱矩阵与搜索窗 ---
        spectrum_tensor = torch.squeeze(
            spectrum_tensor.clone().to(device=self.device, dtype=torch.complex64)
        )
        if spectrum_tensor.ndim != 2:
            raise ValueError(
                f"spectrum_tensor 需要是二维矩阵，当前形状: {tuple(spectrum_tensor.shape)}"
            )
        num_symbols, num_subcarriers = spectrum_tensor.shape

        cfar_full: Optional[torch.Tensor] = None
        if cfar is not None:
            cfar_full = cfar.to(device=self.device)
            if cfar_full.shape != spectrum_tensor.shape:
                raise ValueError(
                    "cfar 须与 squeeze 后的 spectrum_tensor 同形状，"
                    f"当前 cfar {tuple(cfar_full.shape)}，谱 {tuple(spectrum_tensor.shape)}"
                )

        # 设置搜索范围并提取搜索区域
        delay_start, delay_end, doppler_start, doppler_end = self._get_search_range(
            search_range, num_subcarriers, num_symbols
        )

        search_region = spectrum_tensor[doppler_start:doppler_end, delay_start:delay_end]
        num_doppler_bins = int(doppler_end - doppler_start)
        num_delay_bins = int(delay_end - delay_start)

        if num_doppler_bins < _MIN_SEARCH_DIMENSION or num_delay_bins < _MIN_SEARCH_DIMENSION:
            return self._return_empty_peaks(sens_mode=sens_mode)

        cfar_region: Optional[torch.Tensor] = None
        if cfar_full is not None:
            cfar_region = cfar_full[doppler_start:doppler_end, delay_start:delay_end]

        # --- 检峰：协方差 / 噪声子空间 ---
        noise_subspace = self._build_covariance_and_noise_subspace(
            search_region, num_doppler_bins, num_delay_bins, num_sources, threshold
        )
        if noise_subspace is None:
            print("MUSIC算法：协方差特征分解失败，本次无谱峰输出")
            return self._return_empty_peaks(sens_mode=sens_mode)

        candidate_coords = self._extract_candidate_coordinates(
            search_region, num_delay_bins, cfar_region=cfar_region
        )

        peaks_delay, peaks_doppler, peaks_power = self._score_and_select_peaks(
            candidate_coords,
            search_region,
            noise_subspace,
            num_doppler_bins,
            num_delay_bins,
            delay_start,
            doppler_start,
            num_sources,
        )

        # --- 物理量：τ、f_d、r、v（与返回值同源）---
        tau_s, fd_hz, range_m, v_mps = self._compute_peak_physics(
            peaks_delay,
            peaks_doppler,
            sens_mode=sens_mode,
        )

        # --- 日志：按 metric_mode 仅影响表格列 ---
        if peaks_delay.numel() > 0:
            self._log_peak_table(
                peaks_delay=peaks_delay,
                peaks_doppler=peaks_doppler,
                peaks_power=peaks_power,
                sensing_performance=self.sensing_performance,
                metric_mode=metric_mode_canon,
                physics=(tau_s, fd_hz, range_m, v_mps),
            )
        else:
            print("MUSIC算法未检测到谱峰")

        return self._finalize_music_return(range_m, v_mps, peaks_power)

    def _compute_peak_physics(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        *,
        sens_mode: SensMode,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """由 bin 批量得到 τ(s)、f_d(Hz)、距离 (m)、速度 (m/s)。

        ``metric_mode``（``__call__``）不参与换算；``sens_mode`` 同时传入 ``delay_to_range`` 与 ``doppler_to_velocity``。
        """
        sp = self.sensing_performance
        assert sp is not None
        dev = self.device
        d = peaks_delay.detach().to(device=dev, dtype=torch.float64).reshape(-1)
        dop = peaks_doppler.detach().to(device=dev, dtype=torch.float64).reshape(-1)
        if d.numel() == 0:
            z = torch.empty(0, dtype=torch.float64, device=dev)
            return z, z, z, z

        dt = float(sp.delay_resolution)
        dres = float(sp.doppler_resolution)
        half = float(sp.rg.num_ofdm_symbols // 2)
        tau_s = d * dt
        fd_hz = (dop - half) * dres
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
        """与 ``_compute_peak_physics`` 输出的距离/速度对齐后返回三元组。"""
        return range_m.reshape(-1), velocity_mps.reshape(-1), peaks_power

    def _return_empty_peaks(
        self,
        *,
        sens_mode: SensMode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """搜索域过小或协方差失败时：空峰 → 物理量占位 → 与主路径相同的三元组。"""
        pd, dop, pp = self._empty_peak_tensors()
        _, _, rm, vm = self._compute_peak_physics(pd, dop, sens_mode=sens_mode)
        return self._finalize_music_return(rm, vm, pp)

    def _log_peak_table(
        self,
        peaks_delay: torch.Tensor,
        peaks_doppler: torch.Tensor,
        peaks_power: torch.Tensor,
        sensing_performance: Optional[SensingPerformance],
        metric_mode: MusicMode = "delay_doppler",
        *,
        physics: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """将估计谱峰格式化为表格并打印。``physics`` 为 (τ_s, f_d_hz, range_m, v_mps) 时与返回值同源。"""
        print(f"使用MUSIC算法检测到 {peaks_delay.numel()} 个谱峰:")

        # torch.Tensor -> numpy，方便格式化为标量输出
        peaks_delay_np = peaks_delay.cpu().numpy()
        peaks_doppler_np = peaks_doppler.cpu().numpy()
        peaks_power_np = peaks_power.cpu().numpy()

        table_data = []
        if sensing_performance is not None and physics is not None:
            tau_s, fd_hz, range_m, v_mps = physics
            tau_ns_np = (tau_s * 1e9).detach().cpu().numpy().reshape(-1)
            fd_hz_np = fd_hz.detach().cpu().numpy().reshape(-1)
            range_m_np = range_m.detach().cpu().numpy().reshape(-1)
            v_mps_np = v_mps.detach().cpu().numpy().reshape(-1)
            if metric_mode == "delay_doppler":
                headers = [
                    "峰值",
                    "时延索引",
                    "多普勒索引",
                    "时延 (ns)",
                    "多普勒 (Hz)",
                    "功率 (dBm)",
                ]
            else:
                headers = [
                    "峰值",
                    "时延bin",
                    "多普勒bin",
                    "距离 (m)",
                    "速度 (m/s)",
                    "功率 (dBm)",
                ]
            for i, (delay_idx, doppler_idx, power) in enumerate(
                zip(peaks_delay_np, peaks_doppler_np, peaks_power_np)
            ):
                if metric_mode == "delay_doppler":
                    phys_a = float(tau_ns_np[i])
                    phys_b = float(fd_hz_np[i])
                else:
                    phys_a = float(range_m_np[i])
                    phys_b = float(v_mps_np[i])

                power_dbm_value = _linear_mag_to_dbm_scalar(power)

                table_data.append(
                    [
                        i + 1,
                        int(round(float(delay_idx))),
                        int(round(float(doppler_idx))),
                        f"{phys_a:.2f}",
                        f"{phys_b:.2f}",
                        f"{power_dbm_value:.2f}",
                    ]
                )
        else:
            idx_a, idx_b = (
                ("时延索引", "多普勒索引") if metric_mode == "delay_doppler" else ("距离索引", "速度索引")
            )
            headers = ["峰值", idx_a, idx_b, "功率 (dBm)"]
            for i, (delay_idx, doppler_idx, power) in enumerate(
                zip(peaks_delay_np, peaks_doppler_np, peaks_power_np)
            ):
                power_dbm_value = _linear_mag_to_dbm_scalar(power)
                table_data.append(
                    [
                        i + 1,
                        int(delay_idx),
                        int(doppler_idx),
                        f"{power_dbm_value:.2f}",
                    ]
                )

        table_str = tabulate(table_data, headers=headers, tablefmt="simple_grid")
        print(f"\n{table_str}")

    # ==================== 辅助方法 ====================
    def _get_search_range(
        self,
        search_range: Optional[Tuple[int, int, int, int]],
        num_subcarriers: int,
        num_symbols: int,
    ) -> Tuple[int, int, int, int]:
        """获取搜索范围

        参数:
        ----------
        - search_range : tuple[int, int, int, int] | None
            用户指定的搜索范围 (delay_start, delay_end, doppler_start, doppler_end)
        - num_subcarriers : int
            子载波数量
        - num_symbols : int
            符号数量

        返回:
        ----------
        - tuple[int, int, int, int]
            (delay_start, delay_end, doppler_start, doppler_end)
        """
        if search_range is None:
            # 跳过近零时延 bin，减轻直达波/泄漏在 MUSIC 候选中的占优
            delay_guard = min(8, max(0, num_subcarriers // 64))
            return delay_guard, num_subcarriers, 0, num_symbols
        return search_range

    def _empty_peak_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """无谱峰或搜索域过小时返回 ``self.device`` 侧三个空一维张量。"""
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
        """构建协方差矩阵并提取噪声子空间

        对样本协方差做 Hermitian 对称化、相对迹对角加载，并在 ``complex128`` 上调用 ``eigh``；
        若仍不收敛则逐级加大加载重试。全部失败时返回 ``None``（由 ``__call__`` 返回空峰）。

        参数:
        ----------
        - search_region : torch.Tensor
            搜索区域的时延多普勒谱，形状为 (num_doppler_bins, num_delay_bins)
        - num_doppler_bins : int
            多普勒维度大小
        - num_delay_bins : int
            时延维度大小
        - num_sources : int | None
            信号源数量，如果为 None 则自动估计
        - threshold : float
            用于自动估计信号源数量的阈值

        返回:
        ----------
        - noise_subspace : torch.Tensor | None
            噪声子空间，形状为 (subarray_size, num_noise_modes)；分解失败时为 ``None``
        """
        # 确定子阵大小
        subarray_doppler_size = min(_SUBARRAY_SIZE, num_doppler_bins)
        subarray_delay_size = min(_SUBARRAY_SIZE, num_delay_bins)
        subarray_size = subarray_doppler_size * subarray_delay_size

        max_doppler_offset = num_doppler_bins - subarray_doppler_size
        max_delay_offset = num_delay_bins - subarray_delay_size

        if max_doppler_offset <= 0 or max_delay_offset <= 0:
            raise ValueError("搜索区域太小，无法构建子阵")

        # 随机采样子阵快拍
        doppler_indices = torch.randint(
            0, max_doppler_offset + 1, (_NUM_SNAPSHOTS,), device=self.device
        )
        delay_indices = torch.randint(
            0, max_delay_offset + 1, (_NUM_SNAPSHOTS,), device=self.device
        )

        snapshots = torch.empty(
            (subarray_size, _NUM_SNAPSHOTS), dtype=torch.complex64, device=self.device
        )
        for t in range(_NUM_SNAPSHOTS):
            patch = search_region[
                doppler_indices[t] : doppler_indices[t] + subarray_doppler_size,
                delay_indices[t] : delay_indices[t] + subarray_delay_size,
            ]
            snapshots[:, t] = patch.reshape(-1)

        # 样本协方差；Hermitian 对称化减轻浮点非 Hermitian 导致的 eigh 不稳定
        covariance_matrix = (snapshots @ snapshots.conj().T) / _NUM_SNAPSHOTS
        covariance_matrix = (covariance_matrix + covariance_matrix.conj().transpose(-2, -1)) * 0.5

        identity = torch.eye(subarray_size, dtype=torch.complex64, device=self.device)
        trace_real = torch.real(torch.diagonal(covariance_matrix).sum()).clamp(min=_MUSIC_EPS)
        base_load = (trace_real / float(subarray_size)) * _COV_DIAG_LOAD_REL
        base_load_f = max(float(base_load.item()), _COV_DIAG_LOAD_MIN)

        cov64 = covariance_matrix.to(torch.complex128)
        id64 = identity.to(torch.complex128)

        return _noise_subspace_from_loaded_covariance(
            cov64,
            id64,
            base_load_f,
            num_sources,
            threshold,
            subarray_size,
        )

    def _extract_candidate_coordinates(
        self,
        search_region: torch.Tensor,
        num_delay_bins: int,
        cfar_region: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """从幅度谱的局部最大值中提取候选坐标

        参数:
        ----------
        - search_region : torch.Tensor
            搜索区域的时延多普勒谱
        - num_delay_bins : int
            时延维度大小
        - cfar_region : torch.Tensor | None
            与 ``search_region`` 同形状的阈值面；若给定则幅度门限为 ``|X| > cfar_region``

        返回:
        ----------
        - candidate_coords : torch.Tensor
            候选坐标，形状为 (num_candidates, 2)，每行为 (doppler_idx, delay_idx)
        """
        magnitude = torch.abs(search_region)
        magnitude_4d = magnitude.unsqueeze(0).unsqueeze(0)  # (1, 1, M, N)

        # 使用最大池化检测局部最大值
        pooled = F.max_pool2d(magnitude_4d, kernel_size=3, stride=1, padding=1)
        if cfar_region is None:
            amp_gate = magnitude_4d > magnitude_4d.max() * _MIN_PEAK_THRESHOLD
        else:
            cfar_4d = (
                cfar_region.to(device=magnitude.device, dtype=magnitude.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            amp_gate = magnitude_4d > cfar_4d
        is_peak = (magnitude_4d == pooled) & amp_gate

        candidate_coords = torch.nonzero(is_peak[0, 0], as_tuple=False)

        # 如果没有找到峰值，使用幅度最大的前 k 个点
        if candidate_coords.numel() == 0:
            magnitude_flat = magnitude.flatten()
            num_top_candidates = min(_MAX_CANDIDATES, magnitude_flat.numel())
            top_indices = torch.topk(magnitude_flat, k=num_top_candidates).indices
            candidate_coords = torch.stack(
                [top_indices // num_delay_bins, top_indices % num_delay_bins], dim=1
            )

        # 选择幅度最大的前 k 个候选点
        candidate_magnitudes = magnitude[candidate_coords[:, 0], candidate_coords[:, 1]]
        num_selected = min(_MAX_CANDIDATES, candidate_coords.shape[0])
        top_candidate_indices = torch.topk(candidate_magnitudes, k=num_selected).indices
        candidate_coords = candidate_coords[top_candidate_indices]

        return candidate_coords

    def _score_and_select_peaks(
        self,
        candidate_coords: torch.Tensor,
        search_region: torch.Tensor,
        noise_subspace: torch.Tensor,
        num_doppler_bins: int,
        num_delay_bins: int,
        delay_start: int,
        doppler_start: int,
        num_sources: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算候选点的 MUSIC 分数并输出最终峰值集合。

        参数:
        ----------
        - candidate_coords : torch.Tensor
            候选坐标，形状为 (num_candidates, 2)
        - search_region : torch.Tensor
            搜索区域的时延多普勒谱
        - noise_subspace : torch.Tensor
            噪声子空间，形状为 (subarray_size, num_noise_modes)
        - num_doppler_bins : int
            多普勒维度大小
        - num_delay_bins : int
            时延维度大小
        - delay_start : int
            时延起始索引
        - doppler_start : int
            多普勒起始索引
        - num_sources : int | None
            信号源数量

        返回:
        ----------
        - tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - peaks_delay : torch.Tensor
                估计的时延索引数组
            - peaks_doppler : torch.Tensor
                估计的多普勒索引数组
            - peaks_power : torch.Tensor
                对应的 MUSIC 加权分数
            (peaks_delay, peaks_doppler, peaks_power)
        """
        # 搜索区域幅度图。后续会与 MUSIC 伪谱相乘，降低纯数值尖峰的影响。
        magnitude = torch.abs(search_region)
        # 子阵尺寸不超过可用维度，避免越界。
        subarray_doppler_size = min(_SUBARRAY_SIZE, num_doppler_bins)
        subarray_delay_size = min(_SUBARRAY_SIZE, num_delay_bins)

        # 构建子阵导向向量所需的一维索引。
        doppler_indices = torch.arange(
            subarray_doppler_size, device=self.device, dtype=torch.float32
        )
        delay_indices = torch.arange(subarray_delay_size, device=self.device, dtype=torch.float32)

        # 对每个候选坐标计算 MUSIC 伪谱值，并结合局部幅度形成最终排序分数。
        scores = []
        for doppler_idx, delay_idx in candidate_coords.tolist():
            doppler_idx = int(doppler_idx)
            delay_idx = int(delay_idx)

            steering_vector = _steering_vector_2d(
                doppler_idx,
                delay_idx,
                num_doppler_bins=num_doppler_bins,
                num_delay_bins=num_delay_bins,
                doppler_row_idx=doppler_indices,
                delay_row_idx=delay_indices,
            )
            # a^H En En^H a 的等价实现：先投影到噪声子空间，再求能量。
            projection = noise_subspace.conj().T @ steering_vector
            denominator = torch.sum(torch.abs(projection) ** 2)
            pseudospectrum = (1.0 / (denominator + _MUSIC_EPS)).real

            # 使用“伪谱 * 原谱幅度”作为综合评分，抑制孤立数值尖峰。
            score = (pseudospectrum * magnitude[doppler_idx, delay_idx]).item()
            scores.append((score, doppler_idx, delay_idx))

        # 分数降序，优先选择最可信的候选点。
        scores.sort(key=lambda t: t[0], reverse=True)

        # 输出峰值数量：默认 2，且限制在 [1, _MAX_PEAKS]。
        num_output_peaks = int(num_sources if num_sources is not None else 2)
        num_output_peaks = max(1, min(num_output_peaks, _MAX_PEAKS))

        # 贪心去重：要求多普勒索引和时延索引分别唯一，避免同一行/列重复选峰。
        selected_peaks = []
        used_doppler_indices = set()
        used_delay_indices = set()

        for score, doppler_idx, delay_idx in scores:
            if doppler_idx in used_doppler_indices or delay_idx in used_delay_indices:
                continue

            selected_peaks.append((score, doppler_idx, delay_idx))
            used_doppler_indices.add(doppler_idx)
            used_delay_indices.add(delay_idx)

            if len(selected_peaks) >= num_output_peaks:
                break

        # 将局部窗口索引映射回全局 delay-doppler 坐标。
        peaks_doppler = torch.tensor(
            [doppler_idx + doppler_start for _, doppler_idx, _ in selected_peaks],
            dtype=torch.int64,
            device=self.device,
        )
        peaks_delay = torch.tensor(
            [delay_idx + delay_start for _, _, delay_idx in selected_peaks],
            dtype=torch.int64,
            device=self.device,
        )
        peaks_power = torch.tensor(
            [score for score, _, _ in selected_peaks],
            dtype=torch.float32,
            device=self.device,
        )

        return peaks_delay, peaks_doppler, peaks_power
