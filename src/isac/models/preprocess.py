"""时延–多普勒谱预处理：特征提取与运动学标签生成。

CNN / MUSIC 共用 ``spectrum_tensor``（复数裁切谱）作为估计器输入；
训练标签由 ``kinematics_to_target_bins`` 从运动学统一生成。
"""

from __future__ import annotations

import numpy as np
import torch

from ..sensing.geometry import monostatic_range_velocity
from ..sensing.metric import SpectrumMetric
from ..sensing.spectrum.sensing_performance import SensingPerformance


def dd_spectrum_to_features(
    h_dd: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """将单条 ROI 裁切复数谱转为 CNN 特征 ``(C, H, W)``。

    - 通道 0：幅度 dB（逐样本零均值、单位方差）
    - 通道 1（可选）：相位，映射到 ``[-1, 1]``
    """
    mag = torch.abs(h_dd).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)

    channels = [mag_db]
    if use_phase:
        phase = torch.angle(h_dd) / np.pi
        channels.append(phase)

    return torch.stack(channels, dim=0)


def normalize_spectrum_batch(spectrum_tensor: torch.Tensor) -> torch.Tensor:
    """将复数谱规范为 ``(B, H, W)``。"""
    if spectrum_tensor.ndim == 2:
        return spectrum_tensor.unsqueeze(0)
    if spectrum_tensor.ndim == 3:
        return spectrum_tensor
    raise ValueError(
        "spectrum_tensor 须为 (H, W) 或 (B, H, W)，"
        f"收到 {tuple(spectrum_tensor.shape)}"
    )


def spectrum_tensor_to_features(
    spectrum_tensor: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """复数裁切谱 → CNN 特征 ``(B, C, H, W)`` float32。"""
    batch = normalize_spectrum_batch(spectrum_tensor)
    return torch.stack(
        [
            dd_spectrum_to_features(batch[i], eps=eps, use_phase=use_phase)
            for i in range(batch.shape[0])
        ],
        dim=0,
    ).to(dtype=torch.float32)


def kinematics_to_range_velocity(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    bs_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运动学 → 单基地斜距与径向速度 ``(B,)`` float32。"""
    pos = target_position.reshape(-1, 3)
    vel = target_velocity.reshape(-1, 3)
    bs = bs_pos.reshape(-1).detach().cpu().numpy()

    ranges: list[float] = []
    velocities: list[float] = []
    for i in range(pos.shape[0]):
        r, v = monostatic_range_velocity(
            pos[i].detach().cpu().numpy(),
            vel[i].detach().cpu().numpy(),
            bs,
        )
        ranges.append(r)
        velocities.append(v)

    device = target_position.device
    dtype = torch.float32
    return (
        torch.tensor(ranges, dtype=dtype, device=device),
        torch.tensor(velocities, dtype=dtype, device=device),
    )


def kinematics_to_target_bins(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    bs_pos: torch.Tensor,
    *,
    sensing_performance: SensingPerformance,
    num_doppler_bins: int,
) -> torch.Tensor:
    """运动学 → ROI 局部 bin 监督 ``(B, 2)`` = ``[peaks_delay, peaks_doppler]``。"""
    range_m, vel_mps = kinematics_to_range_velocity(
        target_position, target_velocity, bs_pos
    )
    metric = SpectrumMetric(sensing_performance)
    delay_bin, doppler_bin = metric.physical_to_local_bins(
        range_m,
        vel_mps,
        num_doppler_bins=num_doppler_bins,
        sens_mode="monostatic",
    )
    return torch.stack([delay_bin, doppler_bin], dim=-1).to(
        dtype=range_m.dtype,
        device=range_m.device,
    )
