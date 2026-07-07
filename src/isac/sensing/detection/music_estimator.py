"""2D-MUSIC 时延-多普勒谱峰估计（bin 检峰层）。

输入须为上游 :class:`~isac.sensing.spectrum.DelayDopplerSpectrum` 按
``[dd_spectrum_roi]`` 裁切后的时延多普勒谱；本模块不做 ROI 裁切。

流程概览
--------
1. 在裁切谱上检峰
2. 子阵快拍 → 样本协方差 → 噪声子空间（``eigh`` + 对角加载）
3. ``|谱|`` 局部极大值筛候选点（可选 CFAR 门限）
4. 对候选点算 MUSIC 伪谱 × 幅度，贪心去重选峰
5. 返回裁切谱坐标系下的 ``(delay_bin, doppler_bin, power)``

物理量换算、日志与 RMSE 评估见 :class:`~isac.sensing.detection.music_sensing.MusicSensingEvaluator`。

采用子阵空间平滑 + 候选点扫描，避免对全 (M×N) 网格构造巨型协方差矩阵。
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from ..metric import SpectrumMetric

# --- 算法常数 ---
SUBARRAY_SIZE = 16  # 子阵单边上限，实际取 min(16, num_bins)
NUM_SNAPSHOTS = 2048  # 随机滑窗子阵快拍数，用于样本协方差
MAX_CANDIDATES = 200  # 局部极大值候选上限，控制 MUSIC 批量评分规模
MIN_PEAK_THRESHOLD = 0.05  # 无 CFAR 时相对谱峰的幅度门限比例
MAX_PEAKS = 10  # 最大输出峰数
MUSIC_EPS = 1e-12  # 伪谱分母稳定项
COV_DIAG_LOAD_REL = 1e-8  # eigh 数值稳定：对角加载相对 trace 的比例
COV_DIAG_LOAD_MIN = 1e-20  # 对角加载下限

# --- 模块级 2D-MUSIC 辅助算子（不对外导出）---


class _SubarrayGeometry(NamedTuple):
    """子阵尺寸与导向向量相位索引。

    - ``size`` = ``doppler_size`` × ``delay_size``
    - ``doppler_row_idx`` / ``delay_row_idx``：子阵内行/列索引，用于构造导向向量相位
    """

    doppler_size: int
    delay_size: int
    size: int
    doppler_row_idx: torch.Tensor
    delay_row_idx: torch.Tensor


def _subarray_geometry(
    num_doppler_bins: int,
    num_delay_bins: int,
    device: torch.device,
) -> _SubarrayGeometry:
    """由搜索窗维度确定子阵几何（不超过 ``SUBARRAY_SIZE``）。"""
    doppler_size = min(SUBARRAY_SIZE, num_doppler_bins)
    delay_size = min(SUBARRAY_SIZE, num_delay_bins)
    return _SubarrayGeometry(
        doppler_size=doppler_size,
        delay_size=delay_size,
        size=doppler_size * delay_size,
        doppler_row_idx=torch.arange(doppler_size, device=device, dtype=torch.float32),
        delay_row_idx=torch.arange(delay_size, device=device, dtype=torch.float32),
    )


def _steering_vector_2d(
    doppler_idx: int,
    delay_idx: int,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: _SubarrayGeometry,
) -> torch.Tensor:
    """单点 2D 导向向量，形状 ``(geometry.size,)``。"""
    return _batch_steering_vectors(
        torch.tensor(
            [[doppler_idx, delay_idx]], device=geometry.doppler_row_idx.device
        ),
        num_doppler_bins=num_doppler_bins,
        num_delay_bins=num_delay_bins,
        geometry=geometry,
    )[:, 0]


def _batch_steering_vectors(
    candidates: torch.Tensor,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: _SubarrayGeometry,
) -> torch.Tensor:
    """批量 2D 导向向量，形状 ``(geometry.size, num_candidates)``。

    ``candidates`` 每行为裁切谱局部坐标 ``(doppler_idx, delay_idx)``。
    多普勒归一化频率以 fftshift 网格中心为 0：``(idx - M/2) / M``；
    时延归一化为 ``idx / N``。返回 Kronecker 积 ``a_dop ⊗ a_delay``。
    """
    doppler_idx = candidates[:, 0].to(dtype=torch.float32)
    delay_idx = candidates[:, 1].to(dtype=torch.float32)
    center = SpectrumMetric.doppler_center(num_doppler_bins)
    norm_dop = (doppler_idx - center) / num_doppler_bins
    norm_delay = delay_idx / num_delay_bins

    dop_phase = (
        2 * torch.pi * norm_dop.unsqueeze(0) * geometry.doppler_row_idx.unsqueeze(1)
    )
    delay_phase = (
        2 * torch.pi * norm_delay.unsqueeze(0) * geometry.delay_row_idx.unsqueeze(1)
    )
    dop_steering = torch.exp(1j * dop_phase).to(torch.complex64)
    delay_steering = torch.exp(1j * delay_phase).to(torch.complex64)
    return (dop_steering.unsqueeze(1) * delay_steering.unsqueeze(0)).reshape(
        geometry.size, -1
    )


def _noise_subspace_from_loaded_covariance(
    cov64: torch.Tensor,
    id64: torch.Tensor,
    base_load_f: float,
    num_sources: Optional[int],
    threshold: float,
    subarray_size: int,
) -> Optional[torch.Tensor]:
    """对已加载样本协方差做特征分解，返回噪声子空间。

    逐级加大对角加载（``1, 1e2, …, 1e8``）重试 ``eigh``。
    ``num_sources is None`` 时用归一化特征值 > ``threshold`` 估计信号维数。
    成功时返回噪声子空间列向量，形状 ``(subarray_size, K)``；全部失败返回 ``None``。
    """
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
            normalized_eigenvalues = eigenvalues / (eigenvalues[0] + MUSIC_EPS)
            num_signal_sources = int(
                torch.sum(normalized_eigenvalues > threshold).item()
            )
            num_signal_sources = max(1, min(num_signal_sources, subarray_size - 1))
        else:
            num_signal_sources = max(1, min(int(num_sources), subarray_size - 1))

        return eigenvectors[:, num_signal_sources:].to(torch.complex64)

    print(
        "MUSIC: torch.linalg.eigh 在加重对角加载后仍未收敛，跳过噪声子空间估计。"
        f"末次异常: {last_exc}"
    )
    return None


def _local_maxima_candidates(
    magnitude: torch.Tensor,
    *,
    cfar: Optional[torch.Tensor] = None,
    max_candidates: int = MAX_CANDIDATES,
    min_peak_ratio: float = MIN_PEAK_THRESHOLD,
) -> torch.Tensor:
    """从幅度谱局部极大值提取候选坐标，形状 ``(K, 2)``。

    3×3 邻域局部极大 + 幅度门限（``cfar`` 给定时为 ``|X| > cfar``）。
    若无候选则回退为全谱幅度 ``topk`` 点，再按幅度取前 ``max_candidates`` 个。
    """
    num_delay_bins = magnitude.shape[1]
    magnitude_4d = magnitude.unsqueeze(0).unsqueeze(0)

    pooled = F.max_pool2d(magnitude_4d, kernel_size=3, stride=1, padding=1)
    if cfar is None:
        amp_gate = magnitude_4d > magnitude_4d.max() * min_peak_ratio
    else:
        cfar_4d = (
            cfar.to(device=magnitude.device, dtype=magnitude.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        amp_gate = magnitude_4d > cfar_4d
    is_peak = (magnitude_4d == pooled) & amp_gate

    candidate_coords = torch.nonzero(is_peak[0, 0], as_tuple=False)

    if candidate_coords.numel() == 0:
        magnitude_flat = magnitude.flatten()
        num_top = min(max_candidates, magnitude_flat.numel())
        top_indices = torch.topk(magnitude_flat, k=num_top).indices
        candidate_coords = torch.stack(
            [top_indices // num_delay_bins, top_indices % num_delay_bins], dim=1
        )

    candidate_magnitudes = magnitude[candidate_coords[:, 0], candidate_coords[:, 1]]
    num_selected = min(max_candidates, candidate_coords.shape[0])
    top_indices = torch.topk(candidate_magnitudes, k=num_selected).indices
    return candidate_coords[top_indices]


def _batch_music_scores(
    candidates: torch.Tensor,
    magnitude: torch.Tensor,
    noise_subspace: torch.Tensor,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: _SubarrayGeometry,
) -> torch.Tensor:
    """批量 MUSIC 综合评分：伪谱 × 局部幅度。

    伪谱 ``1 / (a^H E_n E_n^H a + eps)``，再乘以候选点处 ``|X|``，形状 ``(num_candidates,)``。
    """
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=magnitude.device)

    steering = _batch_steering_vectors(
        candidates,
        num_doppler_bins=num_doppler_bins,
        num_delay_bins=num_delay_bins,
        geometry=geometry,
    )
    projection = noise_subspace.conj().T @ steering
    denominator = projection.abs().square().sum(dim=0)
    pseudospectrum = 1.0 / (denominator + MUSIC_EPS)
    local_amp = magnitude[candidates[:, 0], candidates[:, 1]]
    return (pseudospectrum.real * local_amp).to(torch.float32)


def _greedy_select_peaks(
    scores: torch.Tensor,
    doppler_idx: torch.Tensor,
    delay_idx: torch.Tensor,
    num_output_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按分数降序贪心选峰，抑制栅瓣。

    同一多普勒 bin 或同一时延 bin 各最多保留一个峰。
    返回 ``(scores, doppler, delay)``，长度 ≤ ``num_output_peaks``。
    """
    order = torch.argsort(scores, descending=True)
    selected_scores: list[float] = []
    selected_dop: list[int] = []
    selected_delay: list[int] = []
    used_doppler: set[int] = set()
    used_delay: set[int] = set()

    for idx in order.tolist():
        d_idx = int(doppler_idx[idx].item())
        t_idx = int(delay_idx[idx].item())
        if d_idx in used_doppler or t_idx in used_delay:
            continue
        selected_scores.append(float(scores[idx].item()))
        selected_dop.append(d_idx)
        selected_delay.append(t_idx)
        used_doppler.add(d_idx)
        used_delay.add(t_idx)
        if len(selected_scores) >= num_output_peaks:
            break

    device = scores.device
    if not selected_scores:
        return (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )
    return (
        torch.tensor(selected_scores, dtype=torch.float32, device=device),
        torch.tensor(selected_dop, dtype=torch.int64, device=device),
        torch.tensor(selected_delay, dtype=torch.int64, device=device),
    )


def _resolve_num_output_peaks(num_sources: Optional[int]) -> int:
    """输出峰数：``num_sources is None`` 时默认 2，限制在 ``[1, MAX_PEAKS]``。"""
    n = int(num_sources if num_sources is not None else 2)
    return max(1, min(n, MAX_PEAKS))


# --- MUSICEstimator：裁切谱 bin 检峰入口 ---


class MUSICEstimator:
    """MUSIC 算法估计器（纯 bin 检峰）。

    使用 2D-MUSIC（子阵/空间平滑 + 候选点扫描）在 **ROI 裁切后的** 时延多普勒谱上
    估计谱峰；对实例直接调用 ``estimator(spectrum_tensor, ...)`` 即可。

    **返回值恒为** ``(delay_bin, doppler_bin, peaks_power)``，坐标相对输入裁切谱。
    不做物理量换算与日志（见 :class:`~isac.sensing.detection.music_sensing.MusicSensingEvaluator`）。
    """

    def __init__(self, device: torch.device):
        """初始化 MUSIC 估计器。

        参数
        ----
        - device :
            计算设备。
        """
        self.device = device

    def __call__(
        self,
        spectrum_tensor: torch.Tensor,
        num_sources: Optional[int] = None,
        threshold: float = 0.1,
        cfar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D-MUSIC 谱峰 bin 估计。

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

        返回
        ----
        ``(delay_bin, doppler_bin, peaks_power)``：
        ``delay_bin``/``doppler_bin`` 为整型索引张量（``int64``），``power`` 为 ``float32``。
        """
        # 1. 校验并准备裁切谱
        spectrum, cfar_mask, num_doppler_bins, num_delay_bins = self._prepare_spectrum(
            spectrum_tensor, cfar=cfar
        )

        # 2. 子阵快拍 → 噪声子空间
        noise_subspace = self._noise_subspace_from_spectrum(
            spectrum,
            num_doppler_bins,
            num_delay_bins,
            num_sources,
            threshold,
        )
        if noise_subspace is None:
            return self._return_empty_peaks()

        # 3. 候选检峰 → MUSIC 评分 → 贪心选峰
        return self._score_and_select_peaks(
            spectrum,
            noise_subspace,
            num_doppler_bins,
            num_delay_bins,
            num_sources,
            cfar_mask=cfar_mask,
        )

    def _prepare_spectrum(
        self,
        spectrum_tensor: torch.Tensor,
        *,
        cfar: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """校验裁切谱输入。

        返回 ``(spectrum, cfar_mask, num_doppler_bins, num_delay_bins)``。
        """
        spectrum = torch.squeeze(
            spectrum_tensor.clone().to(device=self.device, dtype=torch.complex64)
        )
        if spectrum.ndim != 2:
            raise ValueError(
                f"spectrum_tensor 需要是二维矩阵，当前形状: {tuple(spectrum.shape)}"
            )
        num_symbols, num_subcarriers = spectrum.shape

        cfar_mask: Optional[torch.Tensor] = None
        if cfar is not None:
            cfar_mask = cfar.to(device=self.device)
            if cfar_mask.shape != spectrum.shape:
                raise ValueError(
                    "cfar 须与 squeeze 后的 spectrum_tensor 同形状，"
                    f"当前 cfar {tuple(cfar_mask.shape)}，谱 {tuple(spectrum.shape)}"
                )

        num_doppler_bins = num_symbols
        num_delay_bins = num_subcarriers

        return (
            spectrum,
            cfar_mask,
            num_doppler_bins,
            num_delay_bins,
        )

    def _return_empty_peaks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """协方差分解失败时返回三个长度为 0 的空张量。"""
        dev = self.device
        return (
            torch.empty(0, dtype=torch.float64, device=dev),
            torch.empty(0, dtype=torch.float64, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
        )

    def _noise_subspace_from_spectrum(
        self,
        spectrum: torch.Tensor,
        num_doppler_bins: int,
        num_delay_bins: int,
        num_sources: Optional[int],
        threshold: float,
    ) -> Optional[torch.Tensor]:
        """子阵快拍 → 样本协方差 → 噪声子空间。

        谱尺寸不足以容纳子阵时抛出 ``ValueError``；``eigh`` 失败返回 ``None``。
        """
        geometry = _subarray_geometry(num_doppler_bins, num_delay_bins, self.device)
        max_doppler_offset = num_doppler_bins - geometry.doppler_size
        max_delay_offset = num_delay_bins - geometry.delay_size

        if max_doppler_offset < 0 or max_delay_offset < 0:
            raise ValueError("搜索区域太小，无法构建子阵")

        doppler_indices = torch.randint(
            0, max_doppler_offset + 1, (NUM_SNAPSHOTS,), device=self.device
        )
        delay_indices = torch.randint(
            0, max_delay_offset + 1, (NUM_SNAPSHOTS,), device=self.device
        )

        snapshots = torch.empty(
            (geometry.size, NUM_SNAPSHOTS),
            dtype=torch.complex64,
            device=self.device,
        )
        for t in range(NUM_SNAPSHOTS):
            patch = spectrum[
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

        return _noise_subspace_from_loaded_covariance(
            covariance_matrix.to(torch.complex128),
            identity.to(torch.complex128),
            base_load_f,
            num_sources,
            threshold,
            geometry.size,
        )

    def _score_and_select_peaks(
        self,
        spectrum: torch.Tensor,
        noise_subspace: torch.Tensor,
        num_doppler_bins: int,
        num_delay_bins: int,
        num_sources: Optional[int],
        *,
        cfar_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """候选检峰 → 批量 MUSIC 评分 → 贪心去重。

        返回 ``(delay_bin, doppler_bin, power)``，坐标相对输入裁切谱。
        """
        magnitude = torch.abs(spectrum)
        candidates = _local_maxima_candidates(
            magnitude,
            cfar=cfar_mask,
            max_candidates=MAX_CANDIDATES,
        )
        geometry = _subarray_geometry(num_doppler_bins, num_delay_bins, self.device)
        scores = _batch_music_scores(
            candidates,
            magnitude,
            noise_subspace,
            num_doppler_bins=num_doppler_bins,
            num_delay_bins=num_delay_bins,
            geometry=geometry,
        )

        num_output = _resolve_num_output_peaks(num_sources)
        sel_scores, sel_dop, sel_delay = _greedy_select_peaks(
            scores,
            candidates[:, 0],
            candidates[:, 1],
            num_output,
        )

        return sel_delay, sel_dop, sel_scores
