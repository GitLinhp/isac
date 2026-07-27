"""时延–多普勒谱预处理：特征提取与运动学标签生成。

CNN / MUSIC 共用 ``spectrum_tensor``（复数裁切谱）作为估计器输入；
训练标签由 ``kinematics_to_target_bins`` 从运动学统一生成。
"""

from __future__ import annotations

import numpy as np
import torch

from ..data_structures.types import SensMode
from ..sensing.geometry import compute_range, compute_vel, monostatic_range_velocity
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


def _bistatic_range_velocity_batch(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运动学 → 双基地折叠路径长与路径变化率 ``(B,)`` float32。"""
    pos = target_position.reshape(-1, 3)
    vel = target_velocity.reshape(-1, 3)
    tx = tx_pos.reshape(-1).detach().cpu().numpy()
    rx = rx_pos.reshape(-1).detach().cpu().numpy()

    device = target_position.device
    t_pos = pos.to(dtype=torch.float64)
    t_vel = vel.to(dtype=torch.float64)
    r_pos = torch.as_tensor(rx, dtype=torch.float64, device=device).reshape(1, 3)
    x_stack = torch.as_tensor(tx, dtype=torch.float64, device=device).reshape(1, 3)
    r_vel = torch.zeros(1, 3, dtype=torch.float64, device=device)
    x_vel = torch.zeros(1, 3, dtype=torch.float64, device=device)

    is_bistatic = torch.ones(1, pos.shape[0], 1, dtype=torch.bool, device=device)
    range_m = compute_range(is_bistatic, t_pos, r_pos, x_stack)[0, :, 0]
    vel_mps = compute_vel(
        is_bistatic, t_pos, t_vel, r_pos, r_vel, x_stack, x_vel
    )[0, :, 0]

    device = target_position.device
    dtype = torch.float32
    return (
        range_m.to(dtype=dtype, device=device),
        vel_mps.to(dtype=dtype, device=device),
    )


def kinematics_to_range_velocity(
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    bs_pos: torch.Tensor,
    *,
    sens_mode: SensMode = "monostatic",
    tx_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """运动学 → 距离与速度真值 ``(B,)`` float32。

    - ``monostatic``：斜距与 RX 视线径向速度（``tx_pos`` 缺省为 ``bs_pos`` 共址）
    - ``bistatic``：折叠路径长与路径变化率（须提供 ``tx_pos``，``bs_pos`` 为 RX）
    """
    if sens_mode == "bistatic":
        if tx_pos is None:
            raise ValueError("sens_mode='bistatic' 时须提供 tx_pos")
        return _bistatic_range_velocity_batch(
            target_position, target_velocity, tx_pos, bs_pos
        )

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
    sens_mode: SensMode = "monostatic",
    tx_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """运动学 → ROI 局部 bin 监督 ``(B, 2)`` = ``[peaks_delay, peaks_doppler]``。"""
    range_m, vel_mps = kinematics_to_range_velocity(
        target_position,
        target_velocity,
        bs_pos,
        sens_mode=sens_mode,
        tx_pos=tx_pos,
    )
    metric = SpectrumMetric(sensing_performance)
    delay_bin, doppler_bin = metric.physical_to_local_bins(
        range_m,
        vel_mps,
        num_doppler_bins=num_doppler_bins,
        sens_mode=sens_mode,
    )
    return torch.stack([delay_bin, doppler_bin], dim=-1).to(
        dtype=range_m.dtype,
        device=range_m.device,
    )
