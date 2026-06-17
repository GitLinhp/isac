"""Flowgraph 感知性能：由 merge_config 计算，更新 Qt 滑块范围（谱图轴在 grcc 时用默认值）。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from gr_config import EffectiveConfig, grc_overrides_from_grc_vars, merge_config, resolve_config_path

if TYPE_CHECKING:
    pass


def effective_for_flowgraph(tb) -> EffectiveConfig:
    cfg = str(resolve_config_path(tb.config_file))
    overrides = grc_overrides_from_grc_vars(
        fft_len=int(tb.fft_len),
        ofdm_symbols=int(tb.ofdm_symbols),
        cp_len=int(tb.cp_len),
        subcarrier_spacing=float(tb.subcarrier_spacing),
        center_freq=float(tb.center_freq),
        seed=42,
        device=str(tb.tx_device),
    )
    return merge_config(cfg, overrides)


def apply_sensing_perf_ui(tb) -> EffectiveConfig:
    """根据 GRC 物理参数 merge 后刷新 target_range / target_velocity 滑块范围。"""
    eff = effective_for_flowgraph(tb)
    tb.sensing_perf = eff

    r_rng = tb._target_range_range
    v_rng = tb._target_velocity_range
    r_rng.max = float(eff.R_max)
    v_rng.min = -float(eff.v_max)
    v_rng.max = float(eff.v_max)

    r_val = min(float(tb.target_range), float(eff.R_max))
    r_val = max(0.1, r_val)
    v_val = float(tb.target_velocity)
    v_val = max(-float(eff.v_max), min(float(eff.v_max), v_val))
    if abs(r_val - tb.target_range) > 1e-6:
        tb.set_target_range(r_val)
    if abs(v_val - tb.target_velocity) > 1e-6:
        tb.set_target_velocity(v_val)

    burst = float(getattr(tb, "burst_pri_sec", 0.0))
    spec_ms = int(getattr(tb, "spectrogram_interval", 5000))
    if burst > 0:
        print(
            f"burst 模式: PRI={burst:.3f}s, 谱图刷新间隔={spec_ms} ms "
            f"(须晚于首帧 GPU DD 完成，否则 gr-radar 谱图可能崩溃)"
        )

    return eff
