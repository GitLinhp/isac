"""事件驱动 DD 谱图：每帧 packet_len 矩阵到达后立即刷新（无 QTimer）。"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import setup_gnuradio_paths_from

setup_gnuradio_paths_from(__file__)

import numpy as np
import pmt
import pyqtgraph as pg
from gnuradio import gr
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class _DDSpectrogramDisplay(QObject):
    """Qt 主线程上的 PyQtGraph 谱图窗口。"""

    frame_ready = pyqtSignal(object)
    show_requested = pyqtSignal()

    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        axis_x: tuple[float, float],
        axis_y: tuple[float, float],
        axis_z: tuple[float, float],
        autoscale_z: bool,
    ) -> None:
        super().__init__()
        self._axis_x = axis_x
        self._axis_y = axis_y
        self._axis_z = axis_z
        self._autoscale_z = bool(autoscale_z)
        self._frame_count = 0
        self._title = str(title)

        self._win = pg.GraphicsLayoutWidget(show=False, title=self._title)
        self._plot = self._win.addPlot(title=title)
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)
        self._img = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._img)
        self._colorbar = pg.ColorBarItem(
            values=(-15.0, -12.0),
            colorMap=pg.colormap.get("viridis"),
        )
        self._colorbar.setImageItem(self._img, insert_in=self._plot)
        self._apply_axes()
        self.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self.show_requested.connect(self._do_show, Qt.QueuedConnection)

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
        self._apply_axes()

    def set_axis_y(self, ymin: float, ymax: float) -> None:
        self._axis_y = (float(ymin), float(ymax))
        self._apply_axes()

    def set_axis_z(self, zmin: float, zmax: float) -> None:
        self._axis_z = (float(zmin), float(zmax))
        self._autoscale_z = False

    def set_autoscale_z(self, enabled: bool) -> None:
        self._autoscale_z = bool(enabled)

    def post_frame(self, matrix: np.ndarray) -> None:
        self.frame_ready.emit(matrix)

    def _apply_axes(self) -> None:
        xmin, xmax = self._axis_x
        ymin, ymax = self._axis_y
        self._plot.setXRange(xmin, xmax, padding=0)
        self._plot.setYRange(ymin, ymax, padding=0)

    @pyqtSlot(object)
    def _on_frame(self, matrix: np.ndarray) -> None:
        data = np.ascontiguousarray(matrix, dtype=np.float32)
        self._img.setImage(data, autoLevels=False)
        xmin, xmax = self._axis_x
        ymin, ymax = self._axis_y
        self._img.setRect(xmin, ymin, xmax - xmin, ymax - ymin)

        if self._autoscale_z:
            finite = data[np.isfinite(data)]
            if finite.size:
                lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
                if hi <= lo:
                    lo, hi = float(finite.min()), float(finite.max())
                if hi > lo:
                    self._img.setLevels((lo, hi))
                    self._colorbar.setLevels((lo, hi))
        else:
            self._img.setLevels(self._axis_z)
            self._colorbar.setLevels(self._axis_z)

        self._frame_count += 1
        self._plot.setTitle(f"{self._title}  [#{self._frame_count}]")


class SionnaDDSpectrogramPlot(gr.basic_block):
    """DD log-magnitude 谱图：按 packet_len tag 帧边界事件刷新。"""

    def __init__(
        self,
        vlen: int = 1024,
        xlabel: str = "target_range",
        ylabel: str = "target_velocity",
        label: str = "DD Spectrogram",
        axis_x: list[float] | tuple[float, float] = (0.0, 10000.0),
        axis_y: list[float] | tuple[float, float] = (-150.0, 150.0),
        axis_z: list[float] | tuple[float, float] = (-15.0, -12.0),
        autoscale_z: bool = True,
        len_key: str = "packet_len",
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna DD Spectrogram",
            in_sig=[(np.float32, int(vlen))],
            out_sig=None,
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self._vlen = int(vlen)
        self._tag_key = pmt.intern(len_key)
        self._expected_rows = 0
        self._row_buf: list[np.ndarray] = []

        ax_x = (float(axis_x[0]), float(axis_x[1]))
        ax_y = (float(axis_y[0]), float(axis_y[1]))
        ax_z = (float(axis_z[0]), float(axis_z[1]))

        self._display = _DDSpectrogramDisplay(
            title=str(label),
            xlabel=str(xlabel),
            ylabel=str(ylabel),
            axis_x=ax_x,
            axis_y=ax_y,
            axis_z=ax_z,
            autoscale_z=bool(autoscale_z),
        )

    def start(self) -> bool:
        self._display.show()
        return super().start()

    def stop(self) -> bool:
        return super().stop()

    def set_axis_x(self, xmin: float, xmax: float) -> None:
        self._display.set_axis_x(xmin, xmax)

    def set_axis_y(self, ymin: float, ymax: float) -> None:
        self._display.set_axis_y(ymin, ymax)

    def set_axis_z(self, zmin: float, zmax: float) -> None:
        self._display.set_axis_z(zmin, zmax)

    def set_autoscale_z(self, enabled: bool) -> None:
        self._display.set_autoscale_z(enabled)

    def forecast(self, noutput_items, ninputs):
        return [4096]

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        n = len(inp)
        if n == 0:
            return gr.WORK_DONE

        base = self.nitems_read(0)
        for i in range(n):
            abs_idx = base + i
            for tag in self.get_tags_in_range(0, abs_idx, abs_idx + 1):
                if tag.key == self._tag_key:
                    self._row_buf.clear()
                    self._expected_rows = int(pmt.to_long(tag.value))

            if self._expected_rows <= 0:
                continue

            self._row_buf.append(np.array(inp[i], dtype=np.float32, copy=True))
            if len(self._row_buf) >= self._expected_rows:
                matrix = np.stack(self._row_buf, axis=0)
                self._row_buf.clear()
                self._expected_rows = 0
                self._display.post_frame(matrix)

        self.consume(0, n)
        return gr.WORK_DONE
