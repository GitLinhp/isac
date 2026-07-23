"""GNU Radio block: pair TX/RX calibration IQ by ``packet_len`` tag and write complex64 .dat."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pmt
from gnuradio import gr

_COMPLEX_DTYPE = np.complex64


class _TaggedPacketQueue:
    """Accumulate a sample stream and split out packets at ``length_tag_key`` boundaries."""

    def __init__(self, length_tag_key: str) -> None:
        self._length_tag_key = pmt.intern(length_tag_key)
        self._buffer = np.empty(0, dtype=_COMPLEX_DTYPE)
        self._packets: deque[np.ndarray] = deque()
        self.tags_seen = 0
        self.packets_produced = 0

    def clear(self) -> None:
        self._buffer = np.empty(0, dtype=_COMPLEX_DTYPE)
        self._packets.clear()

    def reset_stats(self) -> None:
        self.tags_seen = 0
        self.packets_produced = 0

    @property
    def depth(self) -> int:
        return len(self._packets)

    def add_work(self, samples: np.ndarray, tags: list, stream_base: int) -> None:
        if samples.size == 0:
            return

        base = self._buffer.size
        self._buffer = np.concatenate((self._buffer, samples.astype(_COMPLEX_DTYPE, copy=False)))

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
            if end > self._buffer.size:
                continue
            self._packets.append(self._buffer[start:end].copy())
            self.packets_produced += 1
            trim_end = max(trim_end, end)

        if trim_end > 0:
            self._buffer = self._buffer[trim_end:]

    def has_packet(self) -> bool:
        return bool(self._packets)

    def pop_packet(self) -> np.ndarray:
        return self._packets.popleft()


class SicCalRecorder(gr.sync_block):
    """Pair TX/RX bursts by FIFO tag order and append to calibration ``.dat`` files."""

    def __init__(
        self,
        tx_path: str = "",
        rx_path: str = "",
        length_tag_key: str = "packet_len",
        record_enable: bool = False,
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="SicCalRecorder",
            in_sig=[np.complex64, np.complex64],
            out_sig=[],
        )
        self._tx_path = str(tx_path)
        self._rx_path = str(rx_path)
        self._length_tag_key = str(length_tag_key)
        self._record_enable = bool(record_enable)
        self._tx_queue = _TaggedPacketQueue(self._length_tag_key)
        self._rx_queue = _TaggedPacketQueue(self._length_tag_key)
        self._tx_file = None
        self._rx_file = None
        self._packets_written = 0
        self._record_start_time: float | None = None

    @property
    def record_enable(self) -> bool:
        return self._record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        enabled = bool(value)
        if enabled and not self._record_enable:
            self._open_files(truncate=True)
            self._packets_written = 0
            self._tx_queue.reset_stats()
            self._rx_queue.reset_stats()
            self._record_start_time = time.monotonic()
            print("[SicCalRecorder] recording started", flush=True)
        elif not enabled and self._record_enable:
            self._flush_pending_pairs()
            duration_ms = (
                (time.monotonic() - self._record_start_time) * 1000.0
                if self._record_start_time is not None
                else 0.0
            )
            self._close_files()
            print(self._format_session_summary("recording stopped"), flush=True)
            if duration_ms < 100.0 and self._packets_written == 0:
                print(
                    "[SicCalRecorder] WARNING: recording window <100 ms with 0 packets; "
                    "warm up 2–5 s with Cal Record off, then record ≥2 s",
                    flush=True,
                )
            self._record_start_time = None
        self._record_enable = enabled

    @property
    def tx_path(self) -> str:
        return self._tx_path

    @tx_path.setter
    def tx_path(self, value: str) -> None:
        self._tx_path = str(value)
        if self._record_enable:
            self._close_files()
            self._open_files(truncate=True)

    @property
    def rx_path(self) -> str:
        return self._rx_path

    @rx_path.setter
    def rx_path(self, value: str) -> None:
        self._rx_path = str(value)
        if self._record_enable:
            self._close_files()
            self._open_files(truncate=True)

    @property
    def length_tag_key(self) -> str:
        return self._length_tag_key

    @length_tag_key.setter
    def length_tag_key(self, value: str) -> None:
        self._length_tag_key = str(value)
        self._tx_queue = _TaggedPacketQueue(self._length_tag_key)
        self._rx_queue = _TaggedPacketQueue(self._length_tag_key)

    def _format_session_summary(self, event: str) -> str:
        return (
            f"[SicCalRecorder] {event}, packets_written={self._packets_written}, "
            f"tx_tags={self._tx_queue.tags_seen} rx_tags={self._rx_queue.tags_seen}, "
            f"tx_pkts={self._tx_queue.packets_produced} rx_pkts={self._rx_queue.packets_produced}, "
            f"tx_q={self._tx_queue.depth} rx_q={self._rx_queue.depth}"
        )

    def _open_files(self, *, truncate: bool) -> None:
        self._close_files()
        tx = Path(self._tx_path)
        rx = Path(self._rx_path)
        tx.parent.mkdir(parents=True, exist_ok=True)
        rx.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if truncate else "ab"
        self._tx_file = tx.open(mode)
        self._rx_file = rx.open(mode)

    def _close_files(self) -> None:
        if self._tx_file is not None:
            self._tx_file.flush()
            self._tx_file.close()
            self._tx_file = None
        if self._rx_file is not None:
            self._rx_file.flush()
            self._rx_file.close()
            self._rx_file = None

    def _flush_pending_pairs(self) -> None:
        while self._tx_queue.has_packet() and self._rx_queue.has_packet():
            self._write_pair(self._tx_queue.pop_packet(), self._rx_queue.pop_packet())

    def stop(self) -> int:
        if self._record_enable:
            self._flush_pending_pairs()
            self._close_files()
            print(self._format_session_summary("stop"), flush=True)
        return super().stop()

    def _write_pair(self, tx_pkt: np.ndarray, rx_pkt: np.ndarray) -> None:
        if self._tx_file is None or self._rx_file is None:
            self._open_files(truncate=self._packets_written == 0)
        assert self._tx_file is not None and self._rx_file is not None
        tx_pkt.astype(_COMPLEX_DTYPE, copy=False).tofile(self._tx_file)
        rx_pkt.astype(_COMPLEX_DTYPE, copy=False).tofile(self._rx_file)
        self._packets_written += 1

    def work(self, input_items, _output_items):
        n = min(len(input_items[0]), len(input_items[1]))
        if n == 0:
            return 0

        tx_off = self.nitems_read(0)
        rx_off = self.nitems_read(1)
        tx_tags = self.get_tags_in_window(0, 0, n)
        rx_tags = self.get_tags_in_window(1, 0, n)

        self._tx_queue.add_work(np.asarray(input_items[0][:n]), tx_tags, tx_off)
        self._rx_queue.add_work(np.asarray(input_items[1][:n]), rx_tags, rx_off)

        if self._record_enable:
            self._flush_pending_pairs()

        return n
