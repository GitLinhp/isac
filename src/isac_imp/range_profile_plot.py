"""GNU Radio 块：零多普勒距离谱 PyQtGraph 折线图显示。

接在 ``blocks_nlog10_ff`` 之后，输入全谱 float32 dB 向量；块内按
``compute_range_roi`` 裁剪 ROI，经 Qt 信号在主线程刷新独立窗口。
"""

from __future__ import annotations

import time
import weakref

import numpy as np
import pyqtgraph as pg
from gnuradio import gr
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from isac_imp.range_profile_roi_slice import compute_range_roi

_BG = "w"
_FG = "#333333"
_LINE = "#0066cc"
_MUSIC_LINE = "#cc0000"

_display_registry: list[weakref.ReferenceType[_RangeProfileDisplay]] = []


def _register_display(display: _RangeProfileDisplay) -> None:
    _display_registry.append(weakref.ref(display))
    dead = [r for r in _display_registry if r() is None]
    for r in dead:
        _display_registry.remove(r)


def publish_music_ranges(ranges_m: list[float]) -> None:
    """向已注册的 ``RangeProfilePlotBlock`` 窗口推送 MUSIC 距离标注。"""
    for ref in list(_display_registry):
        display = ref()
        if display is None:
            continue
        display.post_music_ranges(ranges_m)


class _RangeProfileDisplay(QObject):
    """Qt 主线程上的 PyQtGraph 距离谱折线图窗口。"""

    profile_ready = pyqtSignal(object, object)  # x_m, y_db
    music_ranges_ready = pyqtSignal(object)  # list[float]
    show_requested = pyqtSignal()

    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        axis_x: tuple[float, float],
    ) -> None:
        super().__init__()
        self._axis_x = axis_x
        self._frame_count = 0
        self._title = str(title)

        self._win = pg.GraphicsLayoutWidget(show=False, title=self._title)
        self._plot = self._win.addPlot(title=title)
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._curve = self._plot.plot(pen=pg.mkPen(_LINE, width=1))
        self._music_lines: list[pg.InfiniteLine] = []
        self._apply_light_theme()
        self._apply_x_axis()

        self.profile_ready.connect(self._on_profile, Qt.QueuedConnection)
        self.music_ranges_ready.connect(self._on_music_ranges, Qt.QueuedConnection)
        self.show_requested.connect(self._do_show, Qt.QueuedConnection)
        _register_display(self)

    def _apply_light_theme(self) -> None:
        self._win.setBackground(_BG)
        self._plot.getViewBox().setBackgroundColor(_BG)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(_FG)
            axis.setTextPen(_FG)
        self._plot.titleLabel.setAttr("color", _FG)
        self._set_plot_title(self._title)

    def _set_plot_title(self, text: str) -> None:
        self._plot.setTitle(text, color=_FG)

    def request_show(self) -> None:
        self.show_requested.emit()

    def _do_show(self) -> None:
        self._win.show()
        try:
            self._win.raise_()
        except Exception:
            pass

    def show(self) -> None:
        self.request_show()

    def close(self) -> None:
        self._win.close()

    def set_axis_x(self, xmin: float, xmax: float) -> None:
        self._axis_x = (float(xmin), float(xmax))
        self._apply_x_axis()

    def _apply_x_axis(self) -> None:
        xmin, xmax = self._axis_x
        self._plot.setXRange(xmin, xmax, padding=0)

    def post_profile(self, x_m: np.ndarray, y_db: np.ndarray) -> None:
        self.profile_ready.emit(x_m, y_db)

    def post_music_ranges(self, ranges_m: list[float]) -> None:
        self.music_ranges_ready.emit(list(ranges_m))

    @pyqtSlot(object)
    def _on_music_ranges(self, ranges_m: list[float]) -> None:
        for line in self._music_lines:
            self._plot.removeItem(line)
        self._music_lines.clear()
        for r in ranges_m:
            line = pg.InfiniteLine(
                pos=float(r),
                angle=90,
                pen=pg.mkPen(_MUSIC_LINE, width=1.5, style=Qt.DashLine),
                label=f"{float(r):.2f} m",
                labelOpts={"position": 0.9, "color": _MUSIC_LINE},
            )
            self._plot.addItem(line)
            self._music_lines.append(line)

    @pyqtSlot(object, object)
    def _on_profile(self, x_m: np.ndarray, y_db: np.ndarray) -> None:
        x = np.ascontiguousarray(x_m, dtype=np.float64)
        y = np.ascontiguousarray(y_db, dtype=np.float32)
        self._curve.setData(x, y)

        finite = y[np.isfinite(y)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                lo, hi = float(finite.min()), float(finite.max())
            if hi > lo:
                pad = max(3.0, 0.05 * (hi - lo))
                self._plot.setYRange(lo - pad, hi + pad, padding=0)

        self._frame_count += 1
        self._set_plot_title(f"{self._title}  [#{self._frame_count}]")


class RangeProfilePlotBlock(gr.sync_block):
    """全谱 dB 距离谱 → ROI 切片 → PyQtGraph 折线图（独立窗口）。"""

    def __init__(
        self,
        vlen_in: int = 4096,
        range_roi: tuple[float, float] = (0.0, 30.0),
        range_bin_step: float = 0.305,
    ) -> None:
        self._vlen_in = int(vlen_in)
        self._start_bin = 0
        self._num_bins = 1
        self._x_start_m = 0.0
        self._update_period_s = 0.10
        self._last_update = 0.0

        gr.sync_block.__init__(
            self,
            name="Range Profile Plot",
            in_sig=[(np.float32, self._vlen_in)],
            out_sig=None,
        )

        self._range_roi = (float(range_roi[0]), float(range_roi[1]))
        self._range_bin_step = float(range_bin_step)
        self._recompute_roi()

        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        self._display = _RangeProfileDisplay(
            title="Range Profile",
            xlabel="Range (m)",
            ylabel="Power (dB)",
            axis_x=(self._x_start_m, x_max),
        )

    @property
    def range_roi(self) -> tuple[float, float]:
        return self._range_roi

    @range_roi.setter
    def range_roi(self, value: tuple[float, float]) -> None:
        self._range_roi = (float(value[0]), float(value[1]))
        self._recompute_roi()
        self._update_display_x_axis()

    @property
    def range_bin_step(self) -> float:
        return self._range_bin_step

    @range_bin_step.setter
    def range_bin_step(self, value: float) -> None:
        self._range_bin_step = float(value)
        self._recompute_roi()
        self._update_display_x_axis()

    def _recompute_roi(self) -> None:
        start_bin, num_bins, x_start_m = compute_range_roi(
            range_roi=self._range_roi,
            range_bin_step=self._range_bin_step,
            vlen_in=self._vlen_in,
        )
        self._start_bin = start_bin
        self._num_bins = num_bins
        self._x_start_m = x_start_m

    def _update_display_x_axis(self) -> None:
        x_max = self._x_start_m + (self._num_bins - 1) * self._range_bin_step
        self._display.set_axis_x(self._x_start_m, x_max)

    def start(self) -> bool:
        self._display.show()
        return super().start()

    def stop(self) -> bool:
        self._display.close()
        return super().stop()

    def work(self, input_items, output_items) -> int:
        now = time.monotonic()
        if now - self._last_update < self._update_period_s:
            return 1

        self._last_update = now
        vec = np.asarray(input_items[0][0], dtype=np.float32)
        s, n = self._start_bin, self._num_bins
        y = vec[s : s + n]
        x = self._x_start_m + np.arange(n, dtype=np.float64) * self._range_bin_step
        self._display.post_profile(x, y)
        return 1
