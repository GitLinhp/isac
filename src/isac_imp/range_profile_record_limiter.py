"""GNU Radio 块：CPI 复数距离谱透传 + 录制帧数上限。

接在 ``OfdmRangeProfileBlock`` out1 与录制 selector 之间；``record_enable`` 时计数，
达到 ``record_max_frames`` 后通知流图自动关闭录制。
"""

from __future__ import annotations

import inspect
import sys
import weakref
from pathlib import Path

import numpy as np
from gnuradio import gr
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

_LOG_PREFIX = "[RangeProfileRecordLimiter]"

_handler_registry: list[weakref.ReferenceType] = []


class _RecordLimitBridge(QObject):
    """GR 工作线程 → Qt 主线程：安全调用 ``set_record_enable(False)``。"""

    stop_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.stop_requested.connect(self._on_stop_requested, Qt.QueuedConnection)

    @pyqtSlot(object)
    def _on_stop_requested(self, tb_ref) -> None:
        tb = tb_ref() if isinstance(tb_ref, weakref.ReferenceType) else tb_ref
        if tb is None:
            return
        if hasattr(tb, "_apply_record_limit_stop"):
            tb._apply_record_limit_stop()
            return
        if hasattr(tb, "set_record_enable"):
            tb.set_record_enable(False)


_bridge = _RecordLimitBridge()


def _flowgraph_has_record_limiter(tb: object) -> bool:
    for name in (
        "range_profile_record_limiter_0",
        "range_profile_record_limiter",
        "range_profile_record_limiter_dev0",
        "range_profile_record_limiter_dev1",
    ):
        if hasattr(tb, name):
            return True
    if hasattr(tb, "record_output_dir") and hasattr(tb, "blocks_file_sink_0"):
        return True
    return hasattr(tb, "blocks_file_sink_dev0") or hasattr(tb, "blocks_file_sink_dev1")


def _find_flowgraph_from_stack() -> object | None:
    frame = inspect.currentframe()
    try:
        f = frame.f_back
        while f is not None:
            loc = f.f_locals.get("self")
            if (
                loc is not None
                and hasattr(loc, "set_record_enable")
                and _flowgraph_has_record_limiter(loc)
            ):
                return loc
            f = f.f_back
    finally:
        del frame
    return None


def bind_record_limit_handler(top_block) -> None:
    """注册流图；录满帧数时在主线程调用 ``set_record_enable(False)``。"""
    for ref in _handler_registry:
        if ref() is top_block:
            return
    _handler_registry.append(weakref.ref(top_block))
    dead = [r for r in _handler_registry if r() is None]
    for r in dead:
        _handler_registry.remove(r)


def allocate_next_record_path(
    output_dir: str,
    base_name: str = "range_profiles",
) -> str:
    """扫描 output_dir 中已有文件，返回 ``base_name_NNN`` 下一个可用路径。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_n = 0
    prefix = f"{base_name}_"
    for entry in out.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name == base_name:
            continue
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                n = int(suffix)
                if n == 0:
                    continue
                max_n = max(max_n, n)
    next_n = max(max_n + 1, 1)
    return str(out / f"{base_name}_{next_n:03d}")


def _open_record_file_if_available(
    top_block,
    limiter: RangeProfileRecordLimiter | None = None,
) -> None:
    open_fn = getattr(top_block, "open_new_record_file", None)
    if callable(open_fn) and limiter is None:
        open_fn()
        return

    file_sink_attr = "blocks_file_sink_0"
    output_dir = getattr(top_block, "record_output_dir", None)
    record_path_attr = "record_file_path"
    if limiter is not None:
        file_sink_attr = limiter._file_sink_attr
        record_path_attr = limiter._record_file_path_attr
        if limiter._record_output_dir_override:
            output_dir = limiter._record_output_dir_override

    file_sink = getattr(top_block, file_sink_attr, None)
    if file_sink is not None and output_dir:
        path = allocate_next_record_path(output_dir)
        if hasattr(top_block, record_path_attr):
            setattr(top_block, record_path_attr, path)
        file_sink.open(path)
        print(f"{_LOG_PREFIX} recording → {path}", file=sys.stderr)
        return
    record_path = getattr(top_block, record_path_attr, None)
    if file_sink is not None and record_path is not None:
        file_sink.open(record_path)


def _request_stop_on_main_thread(top_block) -> None:
    _bridge.stop_requested.emit(weakref.ref(top_block))


def notify_record_limit_reached(frames_written: int, max_frames: int) -> None:
    """触发已注册流图的录满回调。"""
    print(
        f"{_LOG_PREFIX} recorded {frames_written} frames (limit={max_frames}), stopping",
        file=sys.stderr,
    )
    for ref in list(_handler_registry):
        tb = ref()
        if tb is None:
            continue
        handler = getattr(tb, "_on_record_limit_reached", None)
        if callable(handler):
            handler()
        else:
            _request_stop_on_main_thread(tb)


class RangeProfileRecordLimiter(gr.sync_block):
    """CPI 复数距离谱 1:1 透传；record_enable 时计数，达到上限触发 disable。"""

    def __init__(
        self,
        vlen_in: int = 4096,
        record_enable: bool = False,
        record_max_frames: int = 100,
        record_output_dir_override: str | None = None,
        file_sink_attr: str = "blocks_file_sink_0",
        record_file_path_attr: str = "record_file_path",
    ) -> None:
        self._vlen_in = int(vlen_in)
        self._record_enable = bool(record_enable)
        self._record_max_frames = max(1, int(record_max_frames))
        self._record_output_dir_override = (
            str(record_output_dir_override) if record_output_dir_override else None
        )
        self._file_sink_attr = str(file_sink_attr)
        self._record_file_path_attr = str(record_file_path_attr)
        self._frames_written = 0
        self._limit_notified = False

        gr.sync_block.__init__(
            self,
            name="Range Profile Record Limiter",
            in_sig=[(np.complex64, self._vlen_in)],
            out_sig=[(np.complex64, self._vlen_in)],
        )

    @property
    def record_enable(self) -> bool:
        return self._record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        enabled = bool(value)
        if enabled and not self._record_enable:
            self._frames_written = 0
            self._limit_notified = False
            top_block = _find_flowgraph_from_stack()
            if top_block is not None:
                bind_record_limit_handler(top_block)
                if not callable(getattr(top_block, "open_new_record_file", None)):
                    _open_record_file_if_available(top_block, self)
        self._record_enable = enabled

    @property
    def record_max_frames(self) -> int:
        return self._record_max_frames

    @record_max_frames.setter
    def record_max_frames(self, value: int) -> None:
        self._record_max_frames = max(1, int(value))

    def work(self, input_items: list, output_items: list) -> int:
        n = len(input_items[0])
        if n <= 0:
            return 0

        out = output_items[0]
        for i in range(n):
            out[i][:] = input_items[0][i]
            if self._record_enable and not self._limit_notified:
                self._frames_written += 1
                if self._frames_written >= self._record_max_frames:
                    self._limit_notified = True
                    notify_record_limit_reached(
                        self._frames_written, self._record_max_frames
                    )
        return n
