"""RT 感知多普勒符号校正（无 sensing/channel 依赖，避免循环导入）。"""

import torch


def align_rt_monostatic_doppler_phase(
    h: torch.Tensor,
    *,
    axis: int = -2,
) -> torch.Tensor:
    """RT 单基地：沿 OFDM 符号（慢时间）轴翻转相位，使 DD→MUSIC 速度与 ``geom.vel_tensor`` 同号。

    Sionna RT 频域信道的慢时间相位约定与 ``doppler_to_velocity`` / 几何径向速度（远离为正）
    相反；对符号维做 ``flip`` 仅反转多普勒符号，时延 bin 保持不变。仅用于 ``channel.type='rt'``
    且 ``sens_mode='monostatic'`` 的感知链路，勿用于静态目标等非 RT 信道。
    """
    if not isinstance(h, torch.Tensor):
        raise TypeError("h 须为 torch.Tensor")
    ndim = h.ndim
    if ndim not in (2, 3):
        raise ValueError(f"h 须为 2D (S,F) 或 3D (rx_num,S,F)，收到 ndim={ndim}")
    axis_norm = int(axis) % ndim
    return torch.flip(h, dims=[axis_norm])
