"""射线信道路径：CFR 抽取与输出文件名 slug。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sionna.phy.channel import subcarrier_frequencies

if TYPE_CHECKING:
    from sionna.phy.ofdm import ResourceGrid
    from ...channel.rt.rt_simulator import RTSimulator


def scene_slug_from_rt_simulator(rt_simulator: RTSimulator) -> str:
    """输出文件名用：将 ``rt_simulator_params.filename`` 规范为合法片段。"""
    raw = getattr(rt_simulator.rt_simulator_params, "filename", None)
    if raw is None:
        return "scene"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "scene"
    return s


def paths_cfr_numpy(rg: ResourceGrid, rt_simulator: RTSimulator) -> np.ndarray:
    """在 OFDM 子载波频率网格上取射线追踪 CFR（numpy）。"""
    freqs = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
    return rt_simulator.paths.cfr(
        frequencies=freqs,
        sampling_frequency=1 / rg.ofdm_symbol_duration,
        num_time_steps=rg.num_ofdm_symbols,
        out_type="numpy",
    )
