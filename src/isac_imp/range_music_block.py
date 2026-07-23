"""GNU Radio 块：CPI 复数距离谱 1D MUSIC 超分辨估距。

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

from isac.sensing.detection.range_music_estimator import RangeMusicEstimator
from isac_imp.range_profile_plot import publish_music_ranges

_LOG_PREFIX = "[RangeMusic]"


class RangeMusicBlock(gr.sync_block):
    """复数 CPI 距离谱 → 1D MUSIC 距离估计（无流输出）。"""

    def __init__(
        self,
        vlen_in: int = 4096,
        range_bin_step: float = 0.305,
        range_roi: tuple[float, float] = (0.0, 30.0),
        num_sources: int = 1,
        music_enable: bool = True,
        subarray_size: int = 16,
        threshold: float = 0.1,
    ) -> None:
        self._vlen_in = int(vlen_in)
        self._range_bin_step = float(range_bin_step)
        self._range_roi = (float(range_roi[0]), float(range_roi[1]))
        self._num_sources = int(num_sources)
        self._music_enable = bool(music_enable)
        self._subarray_size = int(subarray_size)
        self._threshold = float(threshold)

        gr.sync_block.__init__(
            self,
            name="Range MUSIC",
            in_sig=[(np.complex64, self._vlen_in)],
            out_sig=None,
        )

        self._estimator = RangeMusicEstimator()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_busy = False
        self._result_queue: queue.Queue[list[float]] = queue.Queue()
        self._frame_count = 0

    @property
    def music_enable(self) -> bool:
        return self._music_enable

    @music_enable.setter
    def music_enable(self, value: bool) -> None:
        self._music_enable = bool(value)
        if not self._music_enable:
            publish_music_ranges([])

    def start(self) -> bool:
        self._shutdown_worker()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="range_music")
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
        publish_music_ranges([])
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
            publish_music_ranges(ranges)

    def _run_music(self, profile: np.ndarray, frame_idx: int) -> None:
        try:
            peaks = self._estimator(
                profile,
                range_bin_step=self._range_bin_step,
                range_roi=self._range_roi,
                num_sources=self._num_sources,
                subarray_size=self._subarray_size,
                threshold=self._threshold,
            )
            ranges = peaks.peak_ranges_m.tolist()
            self._result_queue.put(ranges)
            if ranges:
                rows = [
                    [i + 1, f"{r:.3f}"] for i, r in enumerate(ranges)
                ]
                print(f"{_LOG_PREFIX} frame #{frame_idx} — 1D MUSIC 距离估计 (m):")
                for row in rows:
                    print(f"  峰 {row[0]}: {row[1]} m")
            else:
                print(f"{_LOG_PREFIX} frame #{frame_idx} — 未检测到谱峰")
        except Exception:
            traceback.print_exc()
        finally:
            self._worker_busy = False

    def work(self, input_items, output_items) -> int:
        self._drain_results()

        if not self._music_enable:
            return len(input_items[0])

        if self._worker_busy or self._executor is None:
            return len(input_items[0])

        profile = np.asarray(input_items[0][0], dtype=np.complex64).copy()
        self._frame_count += 1
        frame_idx = self._frame_count
        self._worker_busy = True
        self._executor.submit(self._run_music, profile, frame_idx)
        return 1
