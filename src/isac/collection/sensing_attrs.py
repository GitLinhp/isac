"""从 System 提取 CNN 训练 / 推理所需的感知属性。"""

from __future__ import annotations

from typing import Any

import torch

from ..system import System


def sensing_attrs_from_system(system: System) -> dict[str, Any]:
    """返回训练标签与日志用的感知属性（含 ``num_doppler_bins``）。"""
    comps = system.components
    sp = comps.sensing_performance
    dd_roi = comps.dd_spectrum_roi
    if sp is None or dd_roi is None:
        raise ValueError(
            "sensing_attrs_from_system 要求 [ofdm]、[carrier_frequency] 与 [dd_spectrum_roi]"
        )

    h_full = torch.zeros(
        sp.rg.num_ofdm_symbols,
        sp.rg.fft_size,
        dtype=torch.complex64,
    )
    _ = dd_roi.crop(h_full, sens_mode="monostatic")

    return {
        "range_resolution": sp.range_resolution_monostatic,
        "velocity_resolution": sp.velocity_resolution_monostatic,
        "max_range_m": dd_roi.max_range_m,
        "max_velocity_mps": dd_roi.max_velocity_mps,
        "num_doppler_bins": dd_roi.num_doppler_bins,
    }
