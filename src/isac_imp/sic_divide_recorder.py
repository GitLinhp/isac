"""GNU Radio block: record OFDM Divide vector stream H(f) to complex64 .dat by CPI tag."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pmt
from gnuradio import gr

_COMPLEX_DTYPE = np.complex64


class _TaggedVectorPacketQueue:
    """Split a vector tagged stream into CPI packets (each packet = N vectors)."""

    def __init__(self, length_tag_key: str, vlen: int) -> None:
        self._vlen = int(vlen)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._vectors: list[np.ndarray] = []
        self._packets: deque[np.ndarray] = deque()
        self.tags_seen = 0
        self.packets_produced = 0

    def reset_stats(self) -> None:
        self.tags_seen = 0
        self.packets_produced = 0

    @property
    def depth(self) -> int:
        return len(self._packets)

    def add_work(self, vectors: np.ndarray, tags: list, stream_base: int) -> None:
        arr = np.asarray(vectors, dtype=_COMPLEX_DTYPE)
        if arr.ndim == 1:
            arr = arr.reshape(1, self._vlen)
        n = arr.shape[0]
        if n == 0:
            return

        base = len(self._vectors)
        for i in range(n):
            self._vectors.append(arr[i].copy())

        starts: list[tuple[int, int]] = []
        for tag in tags:
            if not pmt.eq(tag.key, self._length_tag_key):
                continue
            pkt_len = int(pmt.to_long(tag.value))
            if pkt_len <= 0:
                continue
            abs_start = int(tag.offset) - stream_base + base
            if abs_start < 0:
                continue
            self.tags_seen += 1
            starts.append((abs_start, pkt_len))

        if not starts:
            return

        starts.sort(key=lambda item: item[0])
        merged: list[tuple[int, int]] = []
        for start, length in starts:
            if merged and start == merged[-1][0]:
                merged[-1] = (start, length)
            elif not merged or start > merged[-1][0]:
                merged.append((start, length))

        trim_end = 0
        for start, length in merged:
            end = start + length
            if end > len(self._vectors):
                continue
            self._packets.append(np.stack(self._vectors[start:end], axis=0))
            self.packets_produced += 1
            trim_end = max(trim_end, end)

        if trim_end > 0:
            self._vectors = self._vectors[trim_end:]

    def has_packet(self) -> bool:
        return bool(self._packets)

    def pop_packet(self) -> np.ndarray:
        return self._packets.popleft()


class SicDivideRecorder(gr.sync_block):
    """Record tagged OFDM Divide output vectors; one file row per CPI (all symbols flattened)."""

    def __init__(
        self,
        path: str = "",
        vlen: int = 4096,
        length_tag_key: str = "packet_len",
        record_enable: bool = False,
    ) -> None:
        self._vlen = int(vlen)
        gr.sync_block.__init__(
            self,
            name="SicDivideRecorder",
            in_sig=[(np.complex64, self._vlen)],
            out_sig=[],
        )
        self._path = str(path)
        self._length_tag_key = str(length_tag_key)
        self._record_enable = bool(record_enable)
        self._queue = _TaggedVectorPacketQueue(self._length_tag_key, self._vlen)
        self._file = None
        self._cpis_written = 0
        self._record_start_time: float | None = None

    @property
    def record_enable(self) -> bool:
        return self._record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        enabled = bool(value)
        if enabled and not self._record_enable:
            self._open_file(truncate=True)
            self._cpis_written = 0
            self._queue.reset_stats()
            self._record_start_time = time.monotonic()
            print("[SicDivideRecorder] recording started", flush=True)
        elif not enabled and self._record_enable:
            self._flush_pending()
            duration_ms = (
                (time.monotonic() - self._record_start_time) * 1000.0
                if self._record_start_time is not None
                else 0.0
            )
            self._close_file()
            print(self._format_session_summary("recording stopped"), flush=True)
            if duration_ms < 100.0 and self._cpis_written == 0:
                print(
                    "[SicDivideRecorder] WARNING: recording window <100 ms with 0 CPIs; "
                    "warm up 2–5 s with Cal Record off, then record ≥2 s",
                    flush=True,
                )
            self._record_start_time = None
        self._record_enable = enabled

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str) -> None:
        self._path = str(value)
        if self._record_enable:
            self._close_file()
            self._open_file(truncate=True)

    @property
    def length_tag_key(self) -> str:
        return self._length_tag_key

    @length_tag_key.setter
    def length_tag_key(self, value: str) -> None:
        self._length_tag_key = str(value)
        self._queue = _TaggedVectorPacketQueue(self._length_tag_key, self._vlen)

    def _format_session_summary(self, event: str) -> str:
        return (
            f"[SicDivideRecorder] {event}, cpis_written={self._cpis_written}, "
            f"tags={self._queue.tags_seen}, cpis_queued={self._queue.packets_produced}, "
            f"q={self._queue.depth}"
        )

    def _open_file(self, *, truncate: bool) -> None:
        self._close_file()
        out = Path(self._path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._file = out.open("wb" if truncate else "ab")

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _write_cpi(self, packet: np.ndarray) -> None:
        if self._file is None:
            self._open_file(truncate=self._cpis_written == 0)
        assert self._file is not None
        packet.astype(_COMPLEX_DTYPE, copy=False).ravel().tofile(self._file)
        self._cpis_written += 1

    def _flush_pending(self) -> None:
        while self._queue.has_packet():
            self._write_cpi(self._queue.pop_packet())

    def stop(self) -> int:
        if self._record_enable:
            self._flush_pending()
            self._close_file()
            print(self._format_session_summary("stop"), flush=True)
        return super().stop()

    def work(self, input_items, _output_items):
        n = len(input_items[0])
        if n == 0:
            return 0

        stream_base = self.nitems_read(0)
        tags = self.get_tags_in_window(0, 0, n)
        self._queue.add_work(np.asarray(input_items[0][:n]), tags, stream_base)

        if self._record_enable:
            self._flush_pending()

        return n
