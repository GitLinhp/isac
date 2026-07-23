"""GNU Radio 块：CPI 复数距离谱 1D ESPRIT 闭式估距。

接在 ``blocks_integrate_xx`` (complex) 之后，与录制 selector 并行；
结果推送至 :mod:`isac_imp.range_profile_plot` 的 PyQtGraph 竖线标注，并打印日志。
"""

from __future__ import annotations

import queue
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from gnuradio import gr

for _p in [Path.cwd(), *Path.cwd().parents]:
    _src = _p / "src"
    if (_src / "isac").is_dir():
        sys.path.insert(0, str(_src))
        break

from isac.sensing.detection.range_esprit_estimator import RangeEspritEstimator
from isac_imp.range_profile_plot import publish_range_estimates

_LOG_PREFIX = "[RangeESPRIT]"


class RangeEspritBlock(gr.sync_block):
    """复数 CPI 距离谱 → 1D ESPRIT 距离估计（无流输出）。"""

    def __init__(
        self,
        vlen_in: int = 4096,
        range_bin_step: float = 0.305,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: int = 1,
        esprit_enable: bool = True,
        subarray_size: int = 16,
    ) -> None:
        self._vlen_in = int(vlen_in)
        self._range_bin_step = float(range_bin_step)
        self._range_roi = (float(range_roi[0]), float(range_roi[1]))
        self._num_sources = int(num_sources)
        self._esprit_enable = bool(esprit_enable)
        self._subarray_size = int(subarray_size)

        gr.sync_block.__init__(
            self,
            name="Range ESPRIT",
            in_sig=[(np.complex64, self._vlen_in)],
            out_sig=None,
        )

        self._estimator = RangeEspritEstimator()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_busy = False
        self._result_queue: queue.Queue[list[float]] = queue.Queue()
        self._frame_count = 0

    @property
    def esprit_enable(self) -> bool:
        return self._esprit_enable

    @esprit_enable.setter
    def esprit_enable(self, value: bool) -> None:
        self._esprit_enable = bool(value)
        if not self._esprit_enable:
            publish_range_estimates([], method_name="ESPRIT")

    def start(self) -> bool:
        self._shutdown_worker()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="range_esprit")
        self._worker_busy = False
        self._frame_count = 0
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        return True

    def stop(self) -> bool:
        self._shutdown_worker()
        publish_range_estimates([], method_name="ESPRIT")
        return True

    def _shutdown_worker(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._worker_busy = False

    def _drain_results(self) -> None:
        while True:
            try:
                ranges = self._result_queue.get_nowait()
            except queue.Empty:
                break
            publish_range_estimates(ranges, method_name="ESPRIT")

    def _run_esprit(self, profile: np.ndarray, frame_idx: int) -> None:
        try:
            peaks = self._estimator(
                profile,
                range_bin_step=self._range_bin_step,
                range_roi=self._range_roi,
                num_sources=self._num_sources,
                subarray_size=self._subarray_size,
            )
            ranges = peaks.peak_ranges_m.tolist()
            self._result_queue.put(ranges)
            if ranges:
                print(f"{_LOG_PREFIX} frame #{frame_idx} — 1D ESPRIT 距离估计 (m):")
                for i, r in enumerate(ranges):
                    print(f"  峰 {i + 1}: {r:.3f} m")
            else:
                print(f"{_LOG_PREFIX} frame #{frame_idx} — 未检测到谱峰")
        except Exception:
            traceback.print_exc()
        finally:
            self._worker_busy = False

    def work(self, input_items, output_items) -> int:
        del output_items
        self._drain_results()

        if not self._esprit_enable:
            return len(input_items[0])

        if self._worker_busy or self._executor is None:
            return len(input_items[0])

        profile = np.asarray(input_items[0][0], dtype=np.complex64).copy()
        self._frame_count += 1
        frame_idx = self._frame_count
        self._worker_busy = True
        self._executor.submit(self._run_esprit, profile, frame_idx)
        return 1
