"""Flowgraph 感知性能：由 merge_config 计算，更新 Qt 滑块范围（谱图轴在 grcc 时用默认值）。"""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import setup_gnuradio_paths_from

setup_gnuradio_paths_from(__file__)

from gr_config import EffectiveConfig, grc_overrides_from_grc_vars, merge_config, resolve_config_path

if TYPE_CHECKING:
    pass


def recommended_spectrogram_interval_ms(tb, eff: EffectiveConfig) -> int:
    """gr-radar 谱图在首包 DD 矩阵到达前 refresh 会崩溃，大 CPI 须留足 GPU 时间。"""
    n_sym = int(getattr(tb, "ofdm_symbols", eff.ofdm_symbols))
    fft_len = int(getattr(tb, "fft_len", eff.fft_len))
    cp_len = int(getattr(tb, "cp_len", eff.cp_len))
    samp_rate = int(getattr(tb, "samp_rate", eff.samp_rate))
    gpu_floor = max(180_000, n_sym * 180)

    idle_ms = float(getattr(tb, "idle_ms", 0.0))
    if idle_ms > 0 and samp_rate > 0:
        cpi_ms = 1000.0 * n_sym * (fft_len + cp_len) / samp_rate
        pri_ms = idle_ms + cpi_ms
        return max(gpu_floor, int(pri_ms * 600))
    return gpu_floor


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
    """根据 GRC 物理参数 merge 后刷新 target_range / target_velocity 滑块范围（若存在）。"""
    eff = effective_for_flowgraph(tb)
    tb.sensing_perf = eff

    if hasattr(tb, "_target_range_range") and hasattr(tb, "target_range"):
        r_rng = tb._target_range_range
        r_rng.max = float(eff.R_max)
        r_val = min(float(tb.target_range), float(eff.R_max))
        r_val = max(0.1, r_val)
        if abs(r_val - tb.target_range) > 1e-6:
            tb.set_target_range(r_val)

    if hasattr(tb, "_target_velocity_range") and hasattr(tb, "target_velocity"):
        v_rng = tb._target_velocity_range
        v_rng.min = -float(eff.v_max)
        v_rng.max = float(eff.v_max)
        v_val = float(tb.target_velocity)
        v_val = max(-float(eff.v_max), min(float(eff.v_max), v_val))
        if abs(v_val - tb.target_velocity) > 1e-6:
            tb.set_target_velocity(v_val)

    if hasattr(tb, "radar_qtgui_spectrogram_plot_0"):
        plot = tb.radar_qtgui_spectrogram_plot_0
        for method, args in (
            ("set_axis_x", (0.0, float(eff.R_max))),
            ("set_axis_y", (-float(eff.v_max), float(eff.v_max))),
        ):
            fn = getattr(plot, method, None)
            if callable(fn):
                fn(*args)

    spec_ms = recommended_spectrogram_interval_ms(tb, eff)
    ensure_spectrogram_interval(tb, spec_ms, eff)

    idle_ms = float(getattr(tb, "idle_ms", 0.0))
    if idle_ms > 0:
        cpi_ms = (
            1000.0 * int(tb.ofdm_symbols) * (int(tb.fft_len) + int(tb.cp_len))
            / int(tb.samp_rate)
            if int(getattr(tb, "samp_rate", 0)) > 0
            else 0.0
        )
        print(
            f"Style 1 burst: idle_ms={idle_ms:.0f}, CPI≈{cpi_ms:.1f} ms, "
            f"谱图刷新间隔={spec_ms} ms"
        )
    else:
        print(
            f"谱图刷新间隔: {spec_ms} ms "
            f"（首帧 GPU RT+DD 完成后再 refresh）"
        )

    print(
        f"感知性能 (EffectiveConfig): Δr={eff.range_resolution:.2f} m, "
        f"R_max={eff.R_max:.1f} m, v_max={eff.v_max:.1f} m/s"
    )

    return eff


def ensure_spectrogram_interval(tb, spec_ms: int, eff: EffectiveConfig) -> None:
    """gr-radar 谱图 interval 仅在构造时生效；过短时重建块以免空缓冲崩溃。"""
    if not hasattr(tb, "radar_qtgui_spectrogram_plot_0"):
        return
    current = int(getattr(tb, "spectrogram_interval", spec_ms))
    if current >= spec_ms:
        return

    from gnuradio import radar

    old = tb.radar_qtgui_spectrogram_plot_0
    tb.disconnect((tb.sionna_dd_rx_0, 1), (old, 0))
    tb.radar_qtgui_spectrogram_plot_0 = radar.qtgui_spectrogram_plot(
        int(tb.fft_len),
        int(spec_ms),
        "target_range",
        "target_velocity",
        "ISAC Baseline DD",
        [0.0, float(eff.R_max)],
        [-float(eff.v_max), float(eff.v_max)],
        [-15.0, -12.0],
        True,
        "packet_len",
    )
    tb.connect((tb.sionna_dd_rx_0, 1), (tb.radar_qtgui_spectrogram_plot_0, 0))
    if hasattr(tb, "set_spectrogram_interval"):
        tb.set_spectrogram_interval(int(spec_ms))
    print(f"已重建谱图块: refresh 间隔 {current} ms → {spec_ms} ms")
