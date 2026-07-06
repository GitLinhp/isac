"""2D-MUSIC 底层算子：子阵几何、协方差分解、候选检峰与批量伪谱评分。

供 :class:`~isac.sensing.music_estimator.MUSICEstimator` 编排调用；不对外导出到
``isac.sensing.__init__``。
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

# --- 算法常数 ---
MIN_SEARCH_DIMENSION = 8
SUBARRAY_SIZE = 16
NUM_SNAPSHOTS = 2048
MAX_CANDIDATES = 200
MIN_PEAK_THRESHOLD = 0.05
MAX_PEAKS = 10
MUSIC_EPS = 1e-12
COV_DIAG_LOAD_REL = 1e-8
COV_DIAG_LOAD_MIN = 1e-20


class SubarrayGeometry(NamedTuple):
    """子阵尺寸与导向向量的一维索引。

    - ``doppler_size`` × ``delay_size`` → ``size = doppler_size * delay_size``
    - ``doppler_row_idx`` / ``delay_row_idx``：形状 ``(doppler_size,)``、``(delay_size,)``
    """

    doppler_size: int
    delay_size: int
    size: int
    doppler_row_idx: torch.Tensor
    delay_row_idx: torch.Tensor


def subarray_geometry(
    num_doppler_bins: int,
    num_delay_bins: int,
    device: torch.device,
) -> SubarrayGeometry:
    """由搜索窗维度确定子阵几何（不超过 ``SUBARRAY_SIZE``）。"""
    doppler_size = min(SUBARRAY_SIZE, num_doppler_bins)
    delay_size = min(SUBARRAY_SIZE, num_delay_bins)
    return SubarrayGeometry(
        doppler_size=doppler_size,
        delay_size=delay_size,
        size=doppler_size * delay_size,
        doppler_row_idx=torch.arange(
            doppler_size, device=device, dtype=torch.float32
        ),
        delay_row_idx=torch.arange(delay_size, device=device, dtype=torch.float32),
    )


def steering_vector_2d(
    doppler_idx: int,
    delay_idx: int,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: SubarrayGeometry,
) -> torch.Tensor:
    """单点 2D 导向向量 ``a``，Kronecker 积 ``a_dop ⊗ a_delay``，形状 ``(geometry.size,)``。"""
    return batch_steering_vectors(
        torch.tensor([[doppler_idx, delay_idx]], device=geometry.doppler_row_idx.device),
        num_doppler_bins=num_doppler_bins,
        num_delay_bins=num_delay_bins,
        geometry=geometry,
    )[:, 0]


def batch_steering_vectors(
    candidates: torch.Tensor,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: SubarrayGeometry,
) -> torch.Tensor:
    """批量 2D 导向向量，形状 ``(geometry.size, num_candidates)``。

    ``candidates`` 每行为 ``(doppler_idx, delay_idx)``（搜索窗局部坐标）。
    多普勒归一化频率以 fftshift 网格中心为 0：``(idx - M/2) / M``。
    """
    doppler_idx = candidates[:, 0].to(dtype=torch.float32)
    delay_idx = candidates[:, 1].to(dtype=torch.float32)
    norm_dop = (doppler_idx - num_doppler_bins / 2.0) / num_doppler_bins
    norm_delay = delay_idx / num_delay_bins

    dop_phase = (
        2 * torch.pi * norm_dop.unsqueeze(0) * geometry.doppler_row_idx.unsqueeze(1)
    )
    delay_phase = (
        2 * torch.pi * norm_delay.unsqueeze(0) * geometry.delay_row_idx.unsqueeze(1)
    )
    dop_steering = torch.exp(1j * dop_phase).to(torch.complex64)
    delay_steering = torch.exp(1j * delay_phase).to(torch.complex64)
    # Kronecker：a_dop[i] * a_delay[j] 展平为 (Md*Nd, N)
    return (dop_steering.unsqueeze(1) * delay_steering.unsqueeze(0)).reshape(
        geometry.size, -1
    )


def noise_subspace_from_loaded_covariance(
    cov64: torch.Tensor,
    id64: torch.Tensor,
    base_load_f: float,
    num_sources: Optional[int],
    threshold: float,
    subarray_size: int,
) -> Optional[torch.Tensor]:
    """对已 ``complex128`` 样本协方差做对角加载并重试 ``eigh``，返回噪声子空间 ``(size, K)``。"""
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


def local_maxima_candidates(
    magnitude: torch.Tensor,
    *,
    cfar: Optional[torch.Tensor] = None,
    max_candidates: int = MAX_CANDIDATES,
    min_peak_ratio: float = MIN_PEAK_THRESHOLD,
) -> torch.Tensor:
    """从幅度谱局部极大值提取候选坐标，形状 ``(K, 2)``，每行 ``(doppler_idx, delay_idx)``。"""
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


def batch_music_scores(
    candidates: torch.Tensor,
    magnitude: torch.Tensor,
    noise_subspace: torch.Tensor,
    *,
    num_doppler_bins: int,
    num_delay_bins: int,
    geometry: SubarrayGeometry,
) -> torch.Tensor:
    """批量 MUSIC 综合评分：伪谱 × 局部幅度，形状 ``(num_candidates,)``。"""
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=magnitude.device)

    steering = batch_steering_vectors(
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


def greedy_select_peaks(
    scores: torch.Tensor,
    doppler_idx: torch.Tensor,
    delay_idx: torch.Tensor,
    num_output_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按分数降序贪心选峰，多普勒/时延索引各自唯一。

    返回 ``(selected_scores, selected_doppler, selected_delay)``，长度 ≤ ``num_output_peaks``。
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


def resolve_num_output_peaks(num_sources: Optional[int]) -> int:
    """输出峰数：默认 2，限制在 ``[1, MAX_PEAKS]``。"""
    n = int(num_sources if num_sources is not None else 2)
    return max(1, min(n, MAX_PEAKS))
