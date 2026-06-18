"""射线信道路径：CFR/CIR 抽取、变长 CIR 堆叠，以及输出文件名用的场景 slug。"""

from __future__ import annotations

import numpy as np
import torch
from sionna.phy.channel import subcarrier_frequencies


from sionna.phy.ofdm import ResourceGrid


def scene_slug_from_rt_scene(scene: object) -> str:
    """输出文件名用：将 ``scene_params.filename`` 规范为合法片段（未配置或字面 ``None`` 时用 ``scene``）。"""
    raw = getattr(scene.scene_params, "filename", None)
    if raw is None:
        return "scene"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "scene"
    return s


def stack_ragged_cir_samples(
    cir_a_list: list[np.ndarray],
    cir_tau_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """路径条数随几何变化时，将各样本 CIR 在每一维上取上界后零填充，再堆成 ``(N,...)``。"""
    if not cir_a_list or len(cir_a_list) != len(cir_tau_list):
        raise ValueError("cir_a_list 与 cir_tau_list 须同长度且非空")
    n = len(cir_a_list)
    ndims_a = {x.ndim for x in cir_a_list}
    if len(ndims_a) != 1:
        raise ValueError(f"CIR_a 各样本秩不一致: {ndims_a}")
    nd_a = cir_a_list[0].ndim
    max_shape_a = list(cir_a_list[0].shape)
    for arr in cir_a_list[1:]:
        if arr.ndim != nd_a:
            raise ValueError("CIR_a 逐样本秩不一致")
        max_shape_a = [max(max_shape_a[i], arr.shape[i]) for i in range(nd_a)]

    ndims_t = {x.ndim for x in cir_tau_list}
    if len(ndims_t) != 1:
        raise ValueError(f"CIR_tau 各样本秩不一致: {ndims_t}")
    nd_t = cir_tau_list[0].ndim
    max_shape_t = list(cir_tau_list[0].shape)
    for arr in cir_tau_list[1:]:
        if arr.ndim != nd_t:
            raise ValueError("CIR_tau 逐样本秩不一致")
        max_shape_t = [max(max_shape_t[i], arr.shape[i]) for i in range(nd_t)]

    out_a = np.zeros((n,) + tuple(max_shape_a), dtype=np.float64)
    out_tau = np.zeros((n,) + tuple(max_shape_t), dtype=np.float64)
    for i, (a, t) in enumerate(zip(cir_a_list, cir_tau_list, strict=True)):
        out_a[(i,) + tuple(slice(0, s) for s in a.shape)] = a
        out_tau[(i,) + tuple(slice(0, s) for s in t.shape)] = t
    return out_a, out_tau


def paths_cfr_per_tx_torch(
    rg: "ResourceGrid",
    rt_scene: object,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> dict[str, torch.Tensor]:
    """按发射机在 RT ``Paths.cfr`` 上切片，得到各 TX 到唯一 RX 的频域信道 ``(S, F)``。

    形状约定（与 Sionna RT 一致）：``cfr`` 为
    ``(batch, num_rx, num_tx, num_rx_ant, num_time_steps, fft_size)``。
    """
    freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    h = rt_scene.paths.cfr(
        frequencies=freqs,
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
        normalize_delays=False,
        normalize=True,
        out_type="torch",
    )
    if h.ndim != 6:
        raise ValueError(
            f"paths.cfr 须为 6D (batch, rx, tx, rx_ant, S, F)，收到 {tuple(h.shape)}"
        )
    tx_names = list(rt_scene.tx_states.keys())
    num_tx = int(h.shape[2])
    if num_tx != len(tx_names):
        raise ValueError(
            f"CFR 的 tx 维 ({num_tx}) 与 tx_states 数量 ({len(tx_names)}) 不一致"
        )
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(tx_names):
        slab = h[0, 0, i, 0]
        if device is not None or dtype != slab.dtype:
            slab = slab.to(device=device, dtype=dtype)
        out[name] = slab
    return out


def paths_cfr_numpy(rg: object, rt_scene: object) -> np.ndarray:
    """与 ``Channel.cfr`` 一致：在 OFDM 子载波频率网格上取射线追踪 CFR（numpy）。"""
    freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    return rt_scene.paths.cfr(
        frequencies=freqs,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
        num_time_steps=rg.num_ofdm_symbols,
        out_type="numpy",
    )


def paths_cir_numpy(rg: object, rt_scene: object) -> tuple[np.ndarray, np.ndarray]:
    """与 ``Channel`` OFDM 采样一致的路径 CIR（numpy）：``cir_a`` 最后一维 ``[Re,Im]``，`tau` 为时延 (s)。"""
    a_cpx, tau = rt_scene.paths.cir(
        num_time_steps=rg.num_ofdm_symbols,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
        normalize_delays=False,
        out_type="numpy",
    )
    tau_np = np.asarray(tau, dtype=np.float64)
    a_np = np.asarray(a_cpx)
    cir_a = np.stack(
        [
            np.asarray(a_np.real, dtype=np.float64),
            np.asarray(a_np.imag, dtype=np.float64),
        ],
        axis=-1,
    )
    return cir_a, tau_np
