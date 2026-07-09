#!/usr/bin/env python3
"""验证 GRC GrSystemContext 与 run_sensing_baseline.py (--domain time) 数值对齐。"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import ensure_isac_importable, setup_gnuradio_paths_from

_, _REPO = setup_gnuradio_paths_from(__file__)
ensure_isac_importable()

import numpy as np

from gr_config import grc_overrides_from_grc_vars, merge_config
from gr_system import get_gr_system_context
from sionna_rx import build_rx_context, compute_delay_doppler_matrix, prepare_dd_outputs

CONFIG = "config/simulation/sensing/sensing_baseline.toml"
FFT_LEN = 1024
OFDM_SYMBOLS = 1024
CP_LEN = 0
SUBCARRIER_SPACING = 30000.0
CENTER_FREQ = 3.5e9
SEED = 42
DEVICE = "cuda:0" if os.environ.get("ISAC_VERIFY_CPU") != "1" else "cpu"


def _grc_kw() -> dict:
    return dict(
        fft_len=FFT_LEN,
        ofdm_symbols=OFDM_SYMBOLS,
        cp_len=CP_LEN,
        subcarrier_spacing=SUBCARRIER_SPACING,
        center_freq=CENTER_FREQ,
    )


def _run_baseline_script() -> tuple[np.ndarray, tuple[int, int]]:
    from isac.system import System
    from isac.utils import set_random_seed

    set_random_seed(SEED)
    system = System(
        CONFIG,
        device=DEVICE,
    )
    system.components.rt_simulator.get("reflector").velocity = [0, 0, -20]
    system.components.rt_simulator.get("bs1_tx").velocity = [30, 0, 0]

    _, x_rg, x_time = system.transmit()
    y_time = system.components.channel(x_rg, x_time, domain="time", snr_db=None)
    h_dd = system.compute_sensing_spectrum(
        x_rg, system.components.demodulator(y_time).squeeze()
    )
    h_np = h_dd.detach().cpu().numpy().astype(np.complex64)
    peak = np.unravel_index(int(np.argmax(np.abs(h_np) ** 2)), h_np.shape)
    return h_np, peak


def _run_grc_pipeline() -> tuple[np.ndarray, tuple[int, int]]:
    cfg_path = str(_REPO / CONFIG)
    gr_kw = _grc_kw()
    ctx = get_gr_system_context(cfg_path, seed=SEED, device=DEVICE, **gr_kw)
    pkt = ctx.transmit_packet()
    y_time = ctx.apply_channel_time(pkt.time, snr_db=None)
    rx_ctx = build_rx_context(cfg_path, seed=SEED, device=DEVICE, tx_packet=pkt, **gr_kw)
    h_np = compute_delay_doppler_matrix(y_time, rx_ctx, DEVICE)
    peak = np.unravel_index(int(np.argmax(np.abs(h_np) ** 2)), h_np.shape)
    return h_np, peak


def main() -> int:
    cfg_path = str(_REPO / CONFIG)
    effective = merge_config(
        cfg_path,
        grc_overrides_from_grc_vars(seed=SEED, device=DEVICE, **_grc_kw()),
    )

    print("=== sensing_baseline GRC vs 脚本对齐验证 (无噪信道) ===")
    if effective.override_log:
        for line in effective.override_log:
            print(f"  [config] {line}")

    ctx = get_gr_system_context(cfg_path, seed=SEED, device=DEVICE, **_grc_kw())
    ctx.print_rt_paths()

    h_script, peak_script = _run_baseline_script()
    h_grc, peak_grc = _run_grc_pipeline()

    _, log_script = prepare_dd_outputs(h_script, flip_doppler=True)
    _, log_grc = prepare_dd_outputs(h_grc, flip_doppler=True)

    corr = np.abs(np.vdot(h_script.ravel(), h_grc.ravel())) / (
        np.linalg.norm(h_script.ravel()) * np.linalg.norm(h_grc.ravel()) + 1e-30
    )

    print(f"脚本 DD 峰值 bin: delay={peak_script[1]}, doppler={peak_script[0]}")
    print(f"GRC  DD 峰值 bin: delay={peak_grc[1]}, doppler={peak_grc[0]}")
    print(f"DD 矩阵相关系数: {corr:.6f}")

    peak_match = peak_script == peak_grc
    corr_ok = corr > 0.99

    if peak_match and corr_ok:
        print("OK: GRC 三段链路与 run_sensing_baseline (--domain time) 一致")
        return 0

    if not peak_match:
        print("WARN: 峰值 bin 不一致（检查随机种子 / RT 场景速度）")
    if not corr_ok:
        print("WARN: DD 矩阵相关系数偏低")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
