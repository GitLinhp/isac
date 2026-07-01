"""感知几何、物理量转换与雷达信号处理工具函数。"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.constants import c

from isac.utils.type_converter import convert

# 判定 TX/RX 是否视为共址（monostatic）：间距小于等于该阈值 (m)
MONOSTATIC_TX_RX_EPS_M = 1e-3


_STATE_FIELD_IDX = {"pos": 0, "vel": 1}


def stack_state_field(
    states: dict[str, list[np.ndarray]],
    names: list[str],
    field: str,
    device: torch.device,
) -> torch.Tensor:
    """将 ``states[name][field]`` 按 ``names`` 顺序堆成 ``(len(names), 3)`` 的 float64 张量。"""
    idx = _STATE_FIELD_IDX[field]
    return torch.stack(
        [
            torch.as_tensor(
                states[n][idx], dtype=torch.float64, device=device
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

    速度语义：**靠近为负，远离为正**（与 ``geom.vel_tensor`` 一致）。
    Sionna RT / OFDM DD 谱上的 \(f_d\) 符号与上述约定相反，故取负：

    - ``monostatic``：\(v = -f_d c/(2f_c)\)，适用于双程/colocated。
    - ``bistatic``：\(v = -f_d c/f_c\)，假定 \(f_d\) 已对应理想「单程」多普勒。

    省略 ``sens_mode`` 时默认为 ``monostatic``。
    ``sens_mode`` 须与 ``delay_to_range``、``MUSICEstimator`` 等处一致；勿与 MUSIC **metric_mode** 混淆。
    """

    fc = float(carrier_frequency)

    if sens_mode == "monostatic":
        return -(doppler_hz * c) / (2.0 * fc)
    elif sens_mode == "bistatic":
        return -(doppler_hz * c) / fc
    else:
        raise ValueError(f"不支持的速度模型: {sens_mode}")


# --- 雷达信号处理（radarsimpy processing 接口兼容迁移） ---


def _same_type_output(data: torch.Tensor, like: Union[NDArray, torch.Tensor]):
    """按 like 的类型返回 torch 或 numpy。"""
    if isinstance(like, torch.Tensor):
        return data
    return convert(data, "numpy")


def _find_peaks_1d(x: torch.Tensor) -> torch.Tensor:
    """
    简单 1D 峰值检测（替代 scipy.signal.find_peaks 的核心用途）。

    峰值定义：x[i-1] < x[i] >= x[i+1]。
    """
    if x.numel() < 3:
        return torch.empty((0,), dtype=torch.long, device=x.device)
    left = x[1:-1] > x[:-2]
    right = x[1:-1] >= x[2:]
    mask = left & right
    return torch.where(mask)[0] + 1


def range_fft(
    data: Union[NDArray, torch.Tensor],
    rwin: Optional[Union[NDArray, torch.Tensor]] = None,
    n: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    计算距离向 FFT（距离像矩阵）。

    :param data:
        基带数据，形状 ``[channels, pulses, adc_samples]``
    :param rwin:
        距离向窗函数，长度需等于 ``adc_samples``。默认矩形窗
    :param n:
        FFT 点数。若 ``n > adc_samples``，自动零填充
    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。
    :return:
        距离像矩阵，形状 ``[channels, pulses, range]``
    """
    x = convert(data, "torch", device=device)
    shape = x.shape

    if rwin is None:
        win = 1
    else:
        win = convert(rwin, "torch", dtype=x.real.dtype, device=device)
        win = win.reshape(1, 1, -1).repeat(shape[0], shape[1], 1)

    out = torch.fft.fft(x * win, n=n, dim=2)
    return _same_type_output(out, data)


def doppler_fft(
    data: Union[NDArray, torch.Tensor],
    dwin: Optional[Union[NDArray, torch.Tensor]] = None,
    n: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    计算多普勒向 FFT（距离-多普勒矩阵）。

    :param data:
        距离像矩阵，形状 ``[channels, pulses, adc_samples]``
    :param dwin:
        多普勒向窗函数，长度需等于 ``pulses``。默认矩形窗
    :param n:
        FFT 点数。若 ``n > pulses``，自动零填充
    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。
    :return:
        距离-多普勒矩阵，形状 ``[channels, Doppler, range]``
    """
    x = convert(data, "torch", device=device)
    shape = x.shape

    if dwin is None:
        win = 1
    else:
        win = convert(dwin, "torch", dtype=x.real.dtype, device=device)
        win = win.reshape(1, -1, 1).repeat(shape[0], 1, shape[2])

    out = torch.fft.fft(x * win, n=n, dim=1)
    return _same_type_output(out, data)


def range_doppler_fft(
    data: Union[NDArray, torch.Tensor],
    rwin: Optional[Union[NDArray, torch.Tensor]] = None,
    dwin: Optional[Union[NDArray, torch.Tensor]] = None,
    rn: Optional[int] = None,
    dn: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    距离-多普勒联合处理。

    先做距离向 FFT，再做多普勒向 FFT。
    """
    return doppler_fft(
        range_fft(data, rwin=rwin, n=rn, device=device),
        dwin=dwin,
        n=dn,
        device=device,
    )


def doa_music(
    covmat: Union[NDArray, torch.Tensor],
    nsig: int,
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray, torch.Tensor] = range(-90, 91),
    device: Optional[torch.device] = None,
) -> tuple[list, list, Union[NDArray, torch.Tensor]]:
    """
    使用 MUSIC 算法估计 ULA 阵列来波方向。

    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。

    :return:
        (doa角度(度), 峰值索引, 伪谱dB)
    """
    cov = convert(covmat, "torch", dtype=torch.complex128, device=device)
    dev = cov.device
    n_array = cov.shape[0]
    array = torch.linspace(
        0, (n_array - 1) * spacing, n_array, dtype=torch.float64, device=dev
    )
    scan_t = convert(np.array(scanangles), "torch", dtype=torch.float64, device=device)

    _, eig_vects = torch.linalg.eigh(cov)
    noise_subspace = eig_vects[:, :-nsig]

    array_grid, angle_grid = torch.meshgrid(
        array, torch.deg2rad(scan_t), indexing="ij"
    )
    steering_vect = torch.exp(1j * 2 * torch.pi * array_grid * torch.sin(angle_grid)) / math.sqrt(
        n_array
    )

    pseudo_spectrum = 1.0 / torch.linalg.norm(
        noise_subspace.transpose(0, 1).conj() @ steering_vect, dim=0
    )
    ps_db = 10.0 * torch.log10(pseudo_spectrum / torch.min(pseudo_spectrum))
    doa_idx = _find_peaks_1d(ps_db.real)
    if doa_idx.numel() > nsig:
        order = torch.argsort(ps_db[doa_idx])[-nsig:]
        doa_idx = doa_idx[order]
    return (
        convert(scan_t[doa_idx], "numpy").tolist(),
        convert(doa_idx, "numpy").tolist(),
        _same_type_output(ps_db.real, covmat),
    )


def doa_root_music(
    covmat: Union[NDArray, torch.Tensor],
    nsig: int,
    spacing: float = 0.5,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    使用 Root-MUSIC 算法估计 ULA 阵列来波方向。

    说明：多项式求根暂用 NumPy（Torch 暂无等价高层接口）。
    """
    cov = convert(covmat, "torch", dtype=torch.complex128, device=device)
    dev = cov.device
    n_covmat = cov.shape[0]

    _, eig_vects = torch.linalg.eigh(cov)
    noise_subspace = eig_vects[:, :-nsig]
    noise_mat = noise_subspace @ noise_subspace.transpose(0, 1).conj()

    coeff = torch.zeros((n_covmat - 1,), dtype=torch.complex128, device=dev)
    for i in range(1, n_covmat):
        coeff[i - 1] = torch.diagonal(noise_mat, offset=i).sum()
    coeff = torch.hstack((torch.flip(coeff, dims=[0]), torch.trace(noise_mat), coeff.conj()))

    roots = np.roots(convert(coeff, "numpy"))
    mask = np.abs(roots) <= 1
    for i in np.where(np.abs(roots) == 1)[0]:
        mask_idx = np.argsort(np.abs(roots - roots[i]))[1]
        mask[mask_idx] = False
    roots = roots[mask]
    sorted_indices = np.argsort(1.0 - np.abs(roots))
    sin_vals = np.angle(roots[sorted_indices[:nsig]]) / (2 * np.pi * spacing)
    out = torch.as_tensor(
        np.degrees(np.arcsin(sin_vals)), dtype=torch.float64, device=dev
    )
    return _same_type_output(out, covmat)


def doa_esprit(
    covmat: Union[NDArray, torch.Tensor],
    nsig: int,
    spacing: float = 0.5,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    使用 ESPRIT 算法估计 ULA 阵列来波方向。

    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。
    """
    cov = convert(covmat, "torch", dtype=torch.complex128, device=device)
    _, eig_vects = torch.linalg.eigh(cov)
    signal_subspace = eig_vects[:, -nsig:]
    phi = torch.linalg.pinv(signal_subspace[0:-1, :]) @ signal_subspace[1:, :]
    eigs = torch.linalg.eigvals(phi)
    out = torch.rad2deg(torch.asin(torch.angle(eigs) / torch.pi / (spacing / 0.5)))
    return _same_type_output(out.real, covmat)


def doa_iaa(
    beam_vect: Union[NDArray, torch.Tensor],
    steering_vect: Union[NDArray, torch.Tensor],
    num_it: int = 15,
    p_init: Optional[Union[NDArray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    IAA-APES 算法（迭代自适应幅相估计）。

    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。

    :return:
        每个扫描栅格上的功率谱（dB）
    """
    y = convert(beam_vect, "torch", dtype=torch.complex128, device=device)
    a_mat = convert(steering_vect, "torch", dtype=torch.complex128, device=device)
    dev = y.device
    num_grid = a_mat.shape[1]

    if p_init is None:
        spectrum_k = torch.zeros(num_grid, dtype=torch.complex128, device=dev)
        for ik in range(0, num_grid):
            a_vect = a_mat[:, ik].conj().reshape(1, -1)
            spectrum_k[ik] = (
                (1.0 / ((a_vect @ a_vect.conj().transpose(0, 1)) ** 2))
                * torch.mean(torch.abs(a_vect @ y) ** 2)
            ).item()
    else:
        spectrum_k = convert(p_init, "torch", dtype=torch.complex128, device=device)

    for _ in range(0, num_it - 1):
        p_diag = torch.diag(spectrum_k.flatten())
        r_mat = a_mat @ p_diag @ a_mat.conj().transpose(0, 1)
        r_mat_inv = torch.linalg.inv(r_mat)
        for ik in range(0, num_grid):
            a_vect = a_mat[:, ik].conj().reshape(1, -1)
            spec = a_vect @ r_mat_inv @ y / (a_vect @ r_mat_inv @ a_vect.conj().transpose(0, 1))
            spectrum_k[ik] = torch.mean(torch.abs(spec) ** 2)

    out = 10 * torch.log10(torch.real(spectrum_k))
    return _same_type_output(out, beam_vect)


def doa_bartlett(
    covmat: Union[NDArray, torch.Tensor],
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray, torch.Tensor] = range(-90, 91),
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    Bartlett 波束形成（ULA）。

    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。
    """
    cov = convert(covmat, "torch", dtype=torch.complex128, device=device)
    dev = cov.device
    n_array = cov.shape[0]
    array = torch.linspace(
        0, (n_array - 1) * spacing, n_array, dtype=torch.float64, device=dev
    )
    scan_t = convert(np.array(scanangles), "torch", dtype=torch.float64, device=device)

    array_grid, angle_grid = torch.meshgrid(
        array, torch.deg2rad(scan_t), indexing="ij"
    )
    steering_vect = torch.exp(1j * 2 * torch.pi * array_grid * torch.sin(angle_grid)) / math.sqrt(
        n_array
    )

    ps = torch.sum(steering_vect.conj() * (cov @ steering_vect), dim=0).real
    return _same_type_output(10 * torch.log10(ps), covmat)


def doa_capon(
    covmat: Union[NDArray, torch.Tensor],
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray, torch.Tensor] = range(-90, 91),
    device: Optional[torch.device] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    Capon(MVDR) 波束形成（ULA）。

    :param device:
        转 torch 时的目标设备；``None`` 时由 ``convert`` 自动选择。
    """
    cov = convert(covmat, "torch", dtype=torch.complex128, device=device)
    dev = cov.device
    n_array = cov.shape[0]
    array = torch.linspace(
        0, (n_array - 1) * spacing, n_array, dtype=torch.float64, device=dev
    )
    scan_t = convert(np.array(scanangles), "torch", dtype=torch.float64, device=device)

    array_grid, angle_grid = torch.meshgrid(
        array, torch.deg2rad(scan_t), indexing="ij"
    )
    steering_vect = torch.exp(1j * 2 * torch.pi * array_grid * torch.sin(angle_grid)) / math.sqrt(
        n_array
    )

    cov = cov + torch.eye(n_array, dtype=torch.complex128, device=dev) * 1e-9
    inv_covmat = torch.linalg.pinv(cov)
    ps = torch.zeros_like(scan_t, dtype=torch.float64)
    for idx, _ in enumerate(scan_t):
        s_vect = steering_vect[:, idx]
        denom = s_vect.conj() @ inv_covmat @ s_vect
        weight = inv_covmat @ s_vect / denom
        ps[idx] = torch.abs(weight.conj() @ cov @ weight).real
    return _same_type_output(10 * torch.log10(ps), covmat)
