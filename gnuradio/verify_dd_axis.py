#!/usr/bin/env python3
"""验证 DD 谱峰值与 GRC 轴标定：对比配置 range/velocity 与 Sionna 物理 bin。"""
import os
import sys
from pathlib import Path

_GRC = Path(__file__).resolve().parent
_REPO = _GRC.parent
for _p in (_GRC, str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from gr_config import grc_overrides_from_grc_vars, merge_config
from sionna_rx import build_rx_context, compute_delay_doppler_matrix, prepare_dd_outputs
from sionna_tx import get_tx_packet
from static_target_simulator import apply_grc_default_channel

# 与 simulator_ofdm.grc 一致
FFT_LEN = 2048
OFDM_SYMBOLS = 512
CP_LEN = 512
SUBCARRIER_SPACING = 15000.0
CENTER_FREQ = 6e9
CONFIG = "config/simulation/sensing/sensing_monostatic.toml"
SEED = 42
DEVICE = "cuda:0" if os.environ.get("ISAC_VERIFY_CPU") != "1" else "cpu"

TARGET_RANGE_M = 1110.0
TARGET_VELOCITY_MPS = 88.0
C = 299_792_458.0


def _grc_kw() -> dict:
    return dict(
        fft_len=FFT_LEN,
        ofdm_symbols=OFDM_SYMBOLS,
        cp_len=CP_LEN,
        subcarrier_spacing=SUBCARRIER_SPACING,
        center_freq=CENTER_FREQ,
    )


def _axis_map_2pt(bin_idx: int, n_bins: int, axis: tuple[float, float]) -> float:
    """gr-radar 两点线性轴 [min, max]。"""
    lo, hi = axis
    if n_bins <= 1:
        return lo
    return lo + bin_idx / (n_bins - 1) * (hi - lo)


def main() -> int:
    cfg_path = str(_REPO / CONFIG)
    gr_kw = _grc_kw()
    effective = merge_config(
        cfg_path,
        grc_overrides_from_grc_vars(seed=SEED, device=DEVICE, **gr_kw),
    )
    samp_rate = effective.samp_rate

    ctx = build_rx_context(cfg_path, seed=SEED, device=DEVICE, **gr_kw)
    sp = ctx.delay_doppler.sensing_performance

    pkt = get_tx_packet(cfg_path, seed=SEED, device=DEVICE, **gr_kw)
    tx = np.asarray(pkt.time, dtype=np.complex64).reshape(-1)
    rx = apply_grc_default_channel(
        tx,
        range_m=TARGET_RANGE_M,
        velocity_mps=TARGET_VELOCITY_MPS,
        center_freq=CENTER_FREQ,
        samp_rate=samp_rate,
        device=DEVICE,
        rndm_phaseshift=False,
    )

    h_dd_raw = compute_delay_doppler_matrix(rx, ctx, DEVICE)
    iq, _log_mag = prepare_dd_outputs(h_dd_raw)
    peak = np.unravel_index(int(np.argmax(np.abs(iq) ** 2)), iq.shape)
    dop_bin, delay_bin = peak

    range_res = float(sp.range_resolution)
    dop_res = float(sp.doppler_resolution)
    half = OFDM_SYMBOLS // 2
    dop_bin_raw = OFDM_SYMBOLS - 1 - dop_bin
    fd_hz = (dop_bin_raw - half) * dop_res
    v_phys = (fd_hz * C) / (2.0 * CENTER_FREQ)
    r_phys = delay_bin * range_res

    v_max = effective.v_max
    r_max = effective.R_max

    v_old = _axis_map_2pt(dop_bin, OFDM_SYMBOLS, (0.0, v_max))
    v_new = _axis_map_2pt(dop_bin, OFDM_SYMBOLS, (-v_max, v_max))
    v_bad = _axis_map_2pt(dop_bin, OFDM_SYMBOLS, (v_max, -v_max))
    r_disp = _axis_map_2pt(delay_bin, FFT_LEN, (0.0, r_max))

    print("=== DD 轴标定验证 (merge_config + GRC overrides) ===")
    if effective.override_log:
        for line in effective.override_log:
            print(f"  [config] {line}")
    print(f"配置目标: range={TARGET_RANGE_M:.1f} m, velocity={TARGET_VELOCITY_MPS:.1f} m/s")
    print(f"峰值 bin: delay={delay_bin}, doppler={dop_bin}")
    print(f"Sionna 物理换算: range={r_phys:.1f} m, velocity={v_phys:.2f} m/s")
    print(f"旧 spectrogram 轴 [0,v_max]:         velocity_display={v_old:.1f} m/s")
    print(f"新 spectrogram 轴 [-v_max,v_max]:     velocity_display={v_new:.1f} m/s")
    print(f"错误轴 [v_max,-v_max] (Qwt 非法):   velocity_display={v_bad:.1f} m/s")
    print(f"距离轴 [0,R_max]:               range_display={r_disp:.1f} m")
    print(
        f"分辨率: Δr={range_res:.2f} m, Δv={float(sp.velocity_resolution):.3f} m/s, "
        f"v_max={v_max:.1f}"
    )

    err_r = abs(r_phys - TARGET_RANGE_M)
    err_v = abs(v_phys - TARGET_VELOCITY_MPS)
    err_v_old = abs(v_old - TARGET_VELOCITY_MPS)
    err_v_new = abs(v_new - TARGET_VELOCITY_MPS)

    ok_phys = err_r <= 10 * range_res and abs(err_v) <= 2 * float(sp.velocity_resolution)
    ok_axis = err_v_new <= 2 * float(sp.velocity_resolution) and err_v_new < err_v_old

    print()
    if ok_phys:
        print("OK: Sionna DD 峰值与配置目标在 2 bin 容差内一致")
    else:
        print(f"WARN: 物理峰值偏差 range={err_r:.1f}m velocity={err_v:.2f}m/s（检查多普勒符号/延迟）")

    if ok_axis:
        print("OK: [-v_max,v_max] 轴（flip 后）与配置速度一致")
    else:
        print(f"WARN: 新轴速度显示误差 {err_v_new:.1f} m/s（旧轴 {err_v_old:.1f} m/s）")

    return 0 if ok_phys and ok_axis else 1


if __name__ == "__main__":
    raise SystemExit(main())
