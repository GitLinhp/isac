"""感知几何与物理量转换工具函数。"""

import numpy as np
import torch
from scipy.constants import c


# 判定 TX/RX 是否视为共址（monostatic）：间距小于等于该阈值 (m)
MONOSTATIC_TX_RX_EPS_M = 1e-3


def stack_state_field(
    states: dict[str, dict[str, np.ndarray]],
    names: list[str],
    field: str,
    device: torch.device,
) -> torch.Tensor:
    """将 ``states[name][field]`` 按 ``names`` 顺序堆成 ``(len(names), 3)`` 的 float64 张量。"""
    return torch.stack(
        [
            torch.as_tensor(
                states[n][field], dtype=torch.float64, device=device
            ).reshape(3)
            for n in names
        ],
        dim=0,
    )


def compute_path_type(
    r_stack: torch.Tensor,
    x_stack: torch.Tensor,
    n_targets: int,
    *,
    eps_m: float = MONOSTATIC_TX_RX_EPS_M,
) -> torch.Tensor:
    """``True``=bistatic，``False``=monostatic；形状 ``(n_rx, n_target, n_tx)``。"""
    sep = torch.linalg.vector_norm(
        r_stack[:, None, :] - x_stack[None, :, :],
        dim=-1,
    )
    is_bistatic = sep > eps_m
    return is_bistatic.unsqueeze(1).expand(-1, n_targets, -1).clone()


def compute_range(
    is_bistatic: torch.Tensor,
    t_stack: torch.Tensor,
    r_stack: torch.Tensor,
    x_stack: torch.Tensor,
) -> torch.Tensor:
    """按路径类型计算几何路径长度 (m)。

    - **bistatic**：双基地「折叠」路径长 ``||T-X|| + ||R-T||``（直射几何，不经遮挡判定）。
    - **monostatic**：TX/RX 共址近似下采用单基地量程 ``||R-T||``，与 ``||T-X||+||R-T||`` 在 ``X≈R`` 时一致。

    张量形状均为 ``(n_rx, n_target, n_tx)``，与 ``compute_path_type`` 一致。
    """
    n_tx = x_stack.shape[0]
    d_tx = torch.linalg.vector_norm(
        t_stack[None, :, None, :] - x_stack[None, None, :, :],
        dim=-1,
    )
    d_rx = torch.linalg.vector_norm(
        r_stack[:, None, None, :] - t_stack[None, :, None, :],
        dim=-1,
    )
    range_bi = d_tx + d_rx
    d_mono = torch.linalg.vector_norm(
        r_stack[:, None, None, :] - t_stack[None, :, None, :],
        dim=-1,
    )
    range_mono = d_mono.expand(-1, -1, n_tx)
    return torch.where(is_bistatic, range_bi, range_mono)


def compute_vel(
    is_bistatic: torch.Tensor,
    t_pos: torch.Tensor,
    t_vel: torch.Tensor,
    r_pos: torch.Tensor,
    r_vel: torch.Tensor,
    x_stack: torch.Tensor,
    x_vel: torch.Tensor,
) -> torch.Tensor:
    """几何距离变化率 (m/s)，与 ``compute_range`` 的路径定义配套。

    - **monostatic**：目标相对 RX 在 ``T-R`` 连线方向上的径向速度投影（与原 ``RX 视线径向速度`` 一致）。
    - **bistatic**：双基地路径长 ``||T-X||+||R-T||`` 对时间的一阶导数，
      ``(v_T-v_X)·(T-X)/||T-X|| + (v_R-v_T)·(R-T)/||R-T||``（TX/RX/目标均可运动）。

    返回形状 ``(n_rx, n_target, n_tx)``；``x_vel`` 与 ``x_stack`` 同为 ``(n_tx, 3)``。
    """
    n_tx = x_stack.shape[0]
    n_rx = r_pos.shape[0]
    eps = torch.tensor(1e-12, dtype=t_pos.dtype, device=t_pos.device)

    # --- monostatic：沿用 RX–目标视线投影 ---
    los_tr = t_pos[None, :, :] - r_pos[:, None, :]
    dist_tr = torch.linalg.vector_norm(los_tr, dim=-1, keepdim=True).clamp_min(eps)
    u_tr = los_tr / dist_tr
    vel_mono_2d = ((t_vel[None, :, :] - r_vel[:, None, :]) * u_tr).sum(dim=-1)
    vel_mono = vel_mono_2d.unsqueeze(-1).expand(-1, -1, max(n_tx, 1)).clone()

    # --- bistatic：TX 腿 + RX 腿 ---
    diff_tx = t_pos[:, None, :] - x_stack[None, :, :]  # (n_target, n_tx, 3)
    L_tx = torch.linalg.vector_norm(diff_tx, dim=-1, keepdim=True).clamp_min(eps)
    u_tx = diff_tx / L_tx
    rate_tx = ((t_vel[:, None, :] - x_vel[None, :, :]) * u_tx).sum(
        dim=-1
    )  # (n_target, n_tx)
    rate_tx = rate_tx.unsqueeze(0).expand(n_rx, -1, -1)

    diff_rt = (
        r_pos[:, None, :] - t_pos[None, :, :]
    )  # (n_rx, n_target, 3)，与 range 中 ‖R-T‖ 一致
    L_rx = torch.linalg.vector_norm(diff_rt, dim=-1, keepdim=True).clamp_min(eps)
    u_rt = diff_rt / L_rx
    rate_rx = ((r_vel[:, None, :] - t_vel[None, :, :]) * u_rt).sum(
        dim=-1
    )  # (n_rx, n_target)
    rate_rx = rate_rx.unsqueeze(-1).expand(-1, -1, n_tx)

    vel_bi = rate_tx + rate_rx
    return torch.where(is_bistatic, vel_bi, vel_mono)


# 感知指标转换
def delay_to_range(
    tau_s: torch.Tensor,
    carrier_frequency: float,
    sens_mode: str = "monostatic",
) -> torch.Tensor:
    r"""由时延 \(\tau\)（s）换算距离 (m)，调用形态与 ``doppler_to_velocity`` 一致。

    ``doppler_to_velocity(doppler_hz, carrier_frequency, sens_mode)``
    ``delay_to_range(tau_s, carrier_frequency, sens_mode)``

    ``carrier_frequency`` 与速度换算共用同一 ``SensingPerformance`` 标量；本函数当前换算**不依赖**频率，
    仅校验为正数，以保持参数列表与物理配置对齐。

    - ``monostatic``：``tau_s * c / 2``，与 ``SensingPerformance.range_resolution = c * delay_resolution / 2``
      及 ``k · Δr`` 网格一致（\(Δτ\) 对应往返）。
    - ``bistatic``：``tau_s * c``，折叠路径单程几何长度 \(cτ\)（与 MUSIC ``sens_mode='bistatic'`` 配套）。

    ``sens_mode`` 由调用方（如 ``MUSICEstimator(..., sens_mode=...)``）指定，勿与谱图 **metric_mode**
    （``delay_doppler`` / ``range_velocity``）混淆。
    """
    fc = float(carrier_frequency)
    if fc <= 0:
        raise ValueError("carrier_frequency 必须为正数")
    if sens_mode == "monostatic":
        return tau_s * (c / 2.0)
    elif sens_mode == "bistatic":
        return tau_s * c
    else:
        raise ValueError(
            f"不支持的 sens_mode: {sens_mode}，须为 'monostatic' 或 'bistatic'"
        )


def doppler_to_velocity(
    doppler_hz: torch.Tensor,
    carrier_frequency: float,
    sens_mode: str = "monostatic",
) -> torch.Tensor:
    r"""由多普勒频移 \(f_d\)（Hz）反推与 MUSIC/OFDM 网格配套的标量速度（m/s）。

    传统单基地雷达约定：目标远离雷达（距离增大）为正径向速度，对应正多普勒频移。
    省略 ``sens_mode`` 时默认为 ``monostatic``（\(v=f_d c/(2f_c)\)）。

    - ``monostatic``：\(v=f_d c/(2f_c)\)，适用于双程/colocated。
    - ``bistatic``：\(v=f_d c/f_c\)，假定 \(f_d\) 已对应理想「单程」多普勒。

    ``sens_mode`` 须与 ``delay_to_range``、``MUSICEstimator`` 等处一致；勿与 MUSIC **metric_mode**（``mode`` 参数）混淆。
    """

    fc = float(carrier_frequency)

    if sens_mode == "monostatic":
        return (doppler_hz * c) / (2.0 * fc)
    elif sens_mode == "bistatic":
        return (doppler_hz * c) / fc
    else:
        raise ValueError(f"不支持的速度模型: {sens_mode}")
