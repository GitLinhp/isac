"""GNU Radio 块：事件驱动 Delay-Doppler (DD) 谱图显示。

上游（如 OFDM Burst Sensing RX）输出 float32 向量流，每行对应一个多普勒 bin
的时延维 log 幅度；帧首行携带 ``packet_len`` tag，值为本帧多普勒行数。

本块按 tag 攒齐一行矩阵 ``(n_doppler, n_delay)`` 后，经 Qt 信号投递到主线程，
由 PyQtGraph 立即刷新（无 QTimer 轮询）。GNU Radio ``work`` 在调度线程运行，
GUI 更新必须经 ``QueuedConnection`` 回到 Qt 主线程。
"""

from __future__ import annotations

import numpy as np
import pmt
import pyqtgraph as pg
from gnuradio import gr
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


# ---------------------------------------------------------------------------
# Qt 显示层（主线程）
# ---------------------------------------------------------------------------


class _DDSpectrogramDisplay(QObject):
    """Qt 主线程上的 PyQtGraph 谱图窗口。

    GR 工作线程通过 ``post_frame`` / ``request_show`` 发信号；槽函数以
    ``QueuedConnection`` 连接，保证 ``setImage`` / ``show`` 只在 GUI 线程执行。
    """

    frame_ready = pyqtSignal(object)  # 投递一帧 float32 矩阵
    show_requested = pyqtSignal()

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

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
        # 跨线程：emit 在 GR 线程，槽在 Qt 主线程排队执行
        self.frame_ready.connect(self._on_frame, Qt.QueuedConnection)
        self.show_requested.connect(self._do_show, Qt.QueuedConnection)

    # -----------------------------------------------------------------------
    # 窗口显示
    # -----------------------------------------------------------------------

    def request_show(self) -> None:
        """请求在主线程显示窗口。"""
        self.show_requested.emit()

    def _do_show(self) -> None:
        """槽：实际 show / raise（仅主线程）。"""
        self._win.show()
        try:
            self._win.raise_()
        except Exception:
            pass

    def show(self) -> None:
        """对外入口：等价于 ``request_show``。"""
        self.request_show()

    def close(self) -> None:
        """关闭谱图窗口。"""
        self._win.close()

    # -----------------------------------------------------------------------
    # 轴与色标
    # -----------------------------------------------------------------------

    def set_axis_x(self, xmin: float, xmax: float) -> None:
        """更新横轴范围（物理量，如距离）。"""
        self._axis_x = (float(xmin), float(xmax))
        self._apply_axes()

    def set_axis_y(self, ymin: float, ymax: float) -> None:
        """更新纵轴范围（物理量，如速度）。"""
        self._axis_y = (float(ymin), float(ymax))
        self._apply_axes()

    def set_axis_z(self, zmin: float, zmax: float) -> None:
        """固定色标范围，并关闭 z 轴自适应。"""
        self._axis_z = (float(zmin), float(zmax))
        self._autoscale_z = False

    def set_autoscale_z(self, enabled: bool) -> None:
        """是否按每帧数据分位数自动调整色标。"""
        self._autoscale_z = bool(enabled)

    def _apply_axes(self) -> None:
        """将当前 axis_x / axis_y 应用到 plot 视口。"""
        xmin, xmax = self._axis_x
        ymin, ymax = self._axis_y
        self._plot.setXRange(xmin, xmax, padding=0)
        self._plot.setYRange(ymin, ymax, padding=0)

    # -----------------------------------------------------------------------
    # 帧刷新
    # -----------------------------------------------------------------------

    def post_frame(self, matrix: np.ndarray) -> None:
        """从任意线程投递一帧；实际绘制在 ``_on_frame``。"""
        self.frame_ready.emit(matrix)

    @pyqtSlot(object)
    def _on_frame(self, matrix: np.ndarray) -> None:
        """槽：更新 ImageItem、色标与标题帧计数。"""
        data = np.ascontiguousarray(matrix, dtype=np.float32)
        self._img.setImage(data, autoLevels=False)
        xmin, xmax = self._axis_x
        ymin, ymax = self._axis_y
        # 将像素网格映射到物理轴矩形 (x, y, w, h)
        self._img.setRect(xmin, ymin, xmax - xmin, ymax - ymin)

        if self._autoscale_z:
            finite = data[np.isfinite(data)]
            if finite.size:
                # 2%/98% 分位抑制离群点；退化时回退 min/max
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


# ---------------------------------------------------------------------------
# GNU Radio sync_block
# ---------------------------------------------------------------------------


class DDSpectrogramPlot(gr.sync_block):
    """DD log-magnitude 谱图：按 ``packet_len`` tag 界定帧边界并事件刷新。

    工作流程：读入向量行 → 遇 tag 重置缓冲并记录行数 → 攒满后 ``post_frame``。
    """

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

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
        """
        参数:
            - vlen: 输入向量长度（时延维列数，须与上游 ROI 一致）
            - xlabel / ylabel / label: 轴标签与窗口标题
            - axis_x / axis_y: 横纵轴物理范围 ``[min, max]``
            - axis_z: 固定色标范围；``autoscale_z=True`` 时每帧覆盖
            - autoscale_z: 是否按帧内分位数自适应色标
            - len_key: 帧长度 stream tag 名（默认 ``packet_len``）
        """
        gr.sync_block.__init__(
            self,
            name="DD Spectrogram",
            in_sig=[(np.float32, int(vlen))],
            out_sig=None,
        )
        self._vlen = int(vlen)
        self._tag_key = pmt.intern(len_key)
        self._expected_rows = 0  # 当前帧尚需攒齐的多普勒行数；0 表示未对齐
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

    # -----------------------------------------------------------------------
    # 生命周期与轴代理
    # -----------------------------------------------------------------------

    def start(self) -> bool:
        """流图启动时弹出谱图窗口。"""
        self._display.show()
        return super().start()

    def stop(self) -> bool:
        """流图停止（窗口由调用方或进程退出关闭）。"""
        return super().stop()

    def set_axis_x(self, xmin: float, xmax: float) -> None:
        """转发至显示层横轴。"""
        self._display.set_axis_x(xmin, xmax)

    def set_axis_y(self, ymin: float, ymax: float) -> None:
        """转发至显示层纵轴。"""
        self._display.set_axis_y(ymin, ymax)

    def set_axis_z(self, zmin: float, zmax: float) -> None:
        """转发至显示层固定色标。"""
        self._display.set_axis_z(zmin, zmax)

    def set_autoscale_z(self, enabled: bool) -> None:
        """转发至显示层色标自适应开关。"""
        self._display.set_autoscale_z(enabled)

    # -----------------------------------------------------------------------
    # work
    # -----------------------------------------------------------------------

    def work(self, input_items, output_items):
        """消费输入向量：tag 对齐帧 → 攒行 → 投递显示。

        无有效 ``packet_len`` 时丢弃行（不刷新）；返回已消费样点数。
        """
        inp = input_items[0]
        n = len(inp)
        base = self.nitems_read(0)

        for i in range(n):
            abs_idx = base + i
            # 帧首 tag：清空缓冲，记录本帧多普勒行数
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

        return n
