"""时延–多普勒谱预处理与单基地标签计算。"""

import numpy as np
import torch

from ..sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ..sensing.sensing_performance import SensingPerformance
def squeeze_cfr_to_sf(cfr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """将 RT/HDF5 堆叠 CFR 规整为 ``(S, F)`` 复数张量。

    典型输入形状为 ``(1, 1, 1, 1, S, F)`` 或 ``(S, F)``。
    """
    h = torch.as_tensor(cfr)
    if h.ndim == 2:
        return h.to(dtype=torch.complex64)
    while h.ndim > 2 and h.shape[0] == 1:
        h = h.squeeze(0)
    if h.ndim != 2:
        h = h.reshape(-1, h.shape[-2], h.shape[-1])[0]
    return h.to(dtype=torch.complex64)


def crop_dd_roi(
    h_dd: torch.Tensor,
    *,
    offset: int,
) -> torch.Tensor:
    """裁剪近场 ROI，与 ``DelayDopplerSpectrum.visualize(offset=...)`` 一致。

    ``h_dd`` 末两维为 ``(多普勒, 时延)``；返回 ``(2*offset, offset)``。
    """
    if offset <= 0:
        raise ValueError(f"offset 须为正整数，收到 {offset}")
    n_doppler, n_delay = h_dd.shape[-2], h_dd.shape[-1]
    dop_center = n_doppler // 2
    dop_start = max(0, dop_center - offset)
    dop_end = min(n_doppler, dop_center + offset)
    delay_end = min(n_delay, offset)
    return h_dd[..., dop_start:dop_end, 0:delay_end]


def dd_spectrum_to_features(
    h_dd: torch.Tensor,
    *,
    offset: int = 128,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """将复数时延–多普勒谱转为 CNN 输入特征 ``(C, H, W)``。

    - 通道 0：幅度 dB（逐样本零均值、单位方差）
    - 通道 1（可选）：相位，映射到 ``[-1, 1]``
    """
    roi = crop_dd_roi(h_dd, offset=offset)
    mag = torch.abs(roi).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)

    channels = [mag_db]
    if use_phase:
        phase = torch.angle(roi) / np.pi
        channels.append(phase)

    return torch.stack(channels, dim=0)


def monostatic_labels_from_kinematics(
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    bs_position: np.ndarray,
    *,
    bs_velocity: np.ndarray | None = None,
) -> tuple[float, float]:
    """单基地几何真值：斜距 (m) 与 RX 视线径向速度 (m/s)。"""
    t_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    t_vel = np.asarray(target_velocity, dtype=np.float64).reshape(3)
    r_pos = np.asarray(bs_position, dtype=np.float64).reshape(3)
    r_vel = (
        np.zeros(3, dtype=np.float64)
        if bs_velocity is None
        else np.asarray(bs_velocity, dtype=np.float64).reshape(3)
    )

    los = t_pos - r_pos
    range_m = float(np.linalg.norm(los))
    los_unit = los / (range_m + 1e-12)
    radial_vel = float(np.dot(t_vel - r_vel, los_unit))
    return range_m, radial_vel


def compute_dd_spectrum(
    cfr: np.ndarray | torch.Tensor,
    sensing_performance: SensingPerformance,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """由 CFR 计算时延–多普勒谱 ``(多普勒, 时延)``。"""
    dev = device or torch.device("cpu")
    dd = DelayDopplerSpectrum(sensing_performance, device=dev)
    h_sf = squeeze_cfr_to_sf(cfr)
    return dd(h_sf)
