"""从 System 提取 CNN 训练 / 推理所需的感知属性。"""

from __future__ import annotations

from typing import Any

import torch

from ..system import System


def sensing_attrs_from_system(
    system: System, *, sens_mode: str = "monostatic"
) -> dict[str, Any]:
    """返回训练标签与日志用的感知属性（含 ``num_doppler_bins``）。

    配置 ``[dd_spectrum_roi]`` 时，``num_doppler_bins`` 与 ``max_range_m`` /
    ``max_velocity_mps`` 为裁切后 ROI 网格的有效上界；未配置时使用全谱 FFT 网格。

    Parameters
    ----------
    sens_mode
        ``monostatic`` 或 ``bistatic``；影响 ROI 裁切尺度与分辨率字段。
    """
    comps = system.components
    sp = comps.sensing_performance
    dd_roi = comps.dd_spectrum_roi
    if sp is None:
        raise ValueError(
            "sensing_attrs_from_system 要求 [ofdm] 与 [carrier_frequency]"
        )

    if dd_roi is not None:
        h_full = torch.zeros(
            sp.rg.num_ofdm_symbols,
            sp.rg.fft_size,
            dtype=torch.complex64,
        )
        _ = dd_roi.crop(h_full, sens_mode=sens_mode)
        max_range_m, max_velocity_mps = dd_roi.effective_physical_limits(
            sens_mode=sens_mode
        )
        num_doppler_bins = dd_roi.num_doppler_bins
    else:
        max_range_m = getattr(sp, f"max_range_{sens_mode}")
        max_velocity_mps = getattr(sp, f"max_velocity_{sens_mode}")
        num_doppler_bins = sp.rg.num_ofdm_symbols

    return {
        "range_resolution": getattr(sp, f"range_resolution_{sens_mode}"),
        "velocity_resolution": getattr(sp, f"velocity_resolution_{sens_mode}"),
        "max_range_m": max_range_m,
        "max_velocity_mps": max_velocity_mps,
        "num_doppler_bins": num_doppler_bins,
    }
