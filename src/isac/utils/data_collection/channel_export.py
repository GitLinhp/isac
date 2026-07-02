"""射线信道路径：CFR 抽取与输出文件名 slug。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sionna.phy.channel import subcarrier_frequencies
from sionna.rt import Paths

if TYPE_CHECKING:
    from sionna.phy.ofdm import ResourceGrid

    from ...channel.rt.rt_channel import RTChannel
    from ...channel.rt.rt_simulator import RTSimulator
    from ...channel.rt.rt_target import RTTarget


def scene_slug_from_rt_simulator(rt_simulator: RTSimulator) -> str:
    """输出文件名用：取 ``rt_simulator_params.filename``；未配置或为空时用 ``\"None\"``。"""
    raw = getattr(rt_simulator.rt_simulator_params, "filename", None)
    if raw is None:
        return "None"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "None"
    return s


def paths_intersect_object(paths: Paths, object_id: int) -> bool:
    """任一路径在任一 bounce 深度与 ``object_id`` 相交则返回 True。"""
    return bool(np.any(np.asarray(paths.objects) == object_id))


def count_paths_intersect_object(paths: Paths, object_id: int) -> int:
    """与 ``object_id`` 有交互的路径条数（沿 depth 轴聚合）。"""
    objs = np.asarray(paths.objects)
    return int(np.sum(np.any(objs == object_id, axis=0)))


def paths_intersect_target(rt_simulator: RTSimulator, target: RTTarget) -> bool:
    """目标位姿更新后，判断是否存在与该目标 mesh 相交的路径。"""
    return paths_intersect_object(rt_simulator.paths, int(target.object_id))


def paths_cfr_numpy(rg: ResourceGrid, rt_simulator: RTSimulator) -> np.ndarray:
    """在 OFDM 子载波频率网格上取射线追踪 CFR（numpy）。"""
    freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    return rt_simulator.paths.cfr(
        frequencies=freqs,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
        num_time_steps=rg.num_ofdm_symbols,
        out_type="numpy",
    )


def cfr_numpy_to_h_freq(
    cfr: np.ndarray,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """将 HDF5 单条 ``paths.cfr`` numpy 转为 ``ApplyOFDMChannel`` 所需的 7D 张量。

    目标 layout：``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, S, F]``。
    6D 输入按 ``RTChannel.get_cfr`` / ``cfr`` 属性单天线约定
    ``[batch, num_rx, num_tx, num_rx_ant, S, F]`` 重排并插入 ``num_tx_ant=1``。
    """
    h = torch.as_tensor(cfr, device=device, dtype=torch.complex64)
    if h.ndim == 7:
        return h
    if h.ndim == 6:
        # [batch, rx, tx, rx_ant, S, F] -> [batch, rx, rx_ant, tx, tx_ant, S, F]
        h = h.permute(0, 1, 3, 2, 4, 5).unsqueeze(4)
        return h
    raise ValueError(
        f"CFR 须为 6D 或 7D 以适配 ApplyOFDMChannel，收到 ndim={h.ndim}, shape={tuple(cfr.shape)}"
    )


def apply_stored_cfr_frequency(
    x_rg: torch.Tensor,
    cfr: np.ndarray,
    channel: RTChannel,
    *,
    snr_db: float | None,
) -> torch.Tensor:
    """用 HDF5 存储 CFR 在频域施加信道（``ApplyOFDMChannel`` + 可选 AWGN）。"""
    h = cfr_numpy_to_h_freq(cfr, device=x_rg.device)
    y_clean = channel.channel_freq(x_rg, h)
    if y_clean.shape[-2:] != x_rg.shape[-2:]:
        raise ValueError(
            f"存储 CFR 施加后末两维须与 x_rg 一致："
            f"y {tuple(y_clean.shape)} vs x {tuple(x_rg.shape)}"
        )
    if snr_db is not None:
        y_clean = channel._awgn(y_clean, snr_db)
    return y_clean
