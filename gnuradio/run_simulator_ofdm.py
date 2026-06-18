#!/usr/bin/env python3
"""simulator_ofdm 入口：启动后由 Python 根据 merge_config 刷新感知性能 UI。"""
import simulator_ofdm as sim
from flowgraph_perf import apply_sensing_perf_ui

_orig_init = sim.simulator_ofdm.__init__


def _init_with_perf_ui(self):
    _orig_init(self)
    apply_sensing_perf_ui(self)


sim.simulator_ofdm.__init__ = _init_with_perf_ui

if __name__ == "__main__":
    sim.main(top_block_cls=sim.simulator_ofdm)
