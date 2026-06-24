#!/usr/bin/env python3
"""sensing_baseline.grc 入口：启动前 GPU 预热 + merge_config 刷新感知 UI。"""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import ensure_isac_importable, setup_gnuradio_paths_from

_, _REPO = setup_gnuradio_paths_from(__file__)
ensure_isac_importable()
os.chdir(_REPO)

import sensing_baseline as sim
from flowgraph_perf import apply_sensing_perf_ui
from gr_system import prewarm_gr_system

_BASELINE_KW = dict(
    fft_len=1024,
    ofdm_symbols=1024,
    cp_len=0,
    subcarrier_spacing=30000.0,
    center_freq=3.5e9,
)

_orig_init = sim.sensing_baseline.__init__


def _init_with_perf_ui(self):
    _orig_init(self)
    apply_sensing_perf_ui(self)


sim.sensing_baseline.__init__ = _init_with_perf_ui


def _prewarm_before_gui() -> None:
    print("=== sensing_baseline：启动前 GPU 预热 ===")
    prewarm_gr_system(
        "config/simulation/sensing/sensing_baseline.toml",
        seed=42,
        device="cuda:0",
        **_BASELINE_KW,
    )


if __name__ == "__main__":
    _prewarm_before_gui()
    sim.main(top_block_cls=sim.sensing_baseline)
