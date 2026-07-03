"""时延–多普勒谱预处理与单基地标签计算。"""

from __future__ import annotations

import numpy as np
import torch

from isac.sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from isac.sensing.sensing_performance import SensingPerformance


def crop_dd_roi(
    h_dd: torch.Tensor,
    *,
    max_range_m: float,
    max_velocity_mps: float,
    sensing_performance: SensingPerformance,
) -> torch.Tensor:
    """按物理 ROI 裁剪 DD 谱；``h_dd`` 末两维为 ``(多普勒, 时延)``。

    若输入已在 ``DelayDopplerSpectrum.__call__`` 中裁剪，则原样返回。
    """
    dd = DelayDopplerSpectrum(
        sensing_performance,
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
    )
    sp = sensing_performance
    full_ref = torch.empty(
        *h_dd.shape[:-2],
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        device=h_dd.device,
        dtype=h_dd.dtype,
    )
    expected = dd.feature_shape(full_ref)
    if h_dd.shape[-2:] == expected:
        return h_dd
    return dd.crop(h_dd)


def roi_sensing_limits(
    sensing_performance: SensingPerformance,
    *,
    max_range_m: float,
    max_velocity_mps: float,
    h_dd: torch.Tensor | None = None,
) -> tuple[float, float]:
    """裁切 ROI 对应的 ``(max_range_m, max_velocity_mps)``。"""
    dd = DelayDopplerSpectrum(
        sensing_performance,
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
    )
    if h_dd is not None:
        return dd.roi_limits(h_dd)
    eff_max_range_m = (dd.roi_delay_bins() - 1) * (
        sensing_performance.range_resolution
    )
    eff_max_velocity_mps = dd.roi_doppler_half_bins() * (
        sensing_performance.velocity_resolution
    )
    return eff_max_range_m, eff_max_velocity_mps


def physical_to_bins(
    range_m: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    range_resolution: float,
    velocity_resolution: float,
) -> torch.Tensor:
    """物理量转为连续 bin 监督目标 ``(B, 2)``。

    - 第 0 维：距离 bin ``range_m / range_resolution``
    - 第 1 维：多普勒 bin（有符号）``velocity_mps / velocity_resolution``
    """
    range_bin = range_m / range_resolution
    vel_bin = velocity_mps / velocity_resolution
    return torch.stack([range_bin, vel_bin], dim=-1)


def bins_to_physical(
    bins: torch.Tensor,
    *,
    range_resolution: float,
    velocity_resolution: float,
) -> torch.Tensor:
    """连续 bin 还原为 ``(range_m, velocity_mps)``。"""
    range_m = bins[..., 0] * range_resolution
    velocity = bins[..., 1] * velocity_resolution
    return torch.stack([range_m, velocity], dim=-1)


def dd_spectrum_to_features(
    h_dd: torch.Tensor,
    *,
    max_range_m: float,
    max_velocity_mps: float,
    sensing_performance: SensingPerformance,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """将复数时延–多普勒谱转为 CNN 输入特征 ``(C, H, W)``。

    - 通道 0：幅度 dB（逐样本零均值、单位方差）
    - 通道 1（可选）：相位，映射到 ``[-1, 1]``
    """
    cropped = crop_dd_roi(
        h_dd,
        max_range_m=max_range_m,
        max_velocity_mps=max_velocity_mps,
        sensing_performance=sensing_performance,
    )
    mag = torch.abs(cropped).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)

    channels = [mag_db]
    if use_phase:
        phase = torch.angle(cropped) / np.pi
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
