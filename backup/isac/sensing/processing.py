"""
雷达信号处理（Torch 实现）

说明：
- 该模块对 `src/radarsimpy/processing.py` 做接口兼容迁移。
- 输入支持 `numpy.ndarray` / `torch.Tensor`，经 `isac.utils.type_converter.convert`
  转为 `torch.Tensor` 计算；`device` 为 ``None`` 时与 ``convert`` 一致（有 CUDA 则默认
  CUDA，否则 CPU）；亦可显式传入 ``torch.device("cpu")`` 等。
- 输出按主输入类型自动适配：
  - 主输入是 torch -> 返回 torch
  - 主输入是 numpy -> 返回 numpy
"""

from typing import List, Optional, Union
import math

import numpy as np
import torch
from numpy.typing import NDArray

from isac.utils.type_converter import convert


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
