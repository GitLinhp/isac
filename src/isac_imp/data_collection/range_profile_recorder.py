"""GNU Radio 块：双 USRP 线性功率距离谱成对录制（全帧率，无降采样）。

每台 USRP 使用独立的单输入 ``DevRangeProfileRecorder``（dev1/dev0），
经 ``RangeProfileSession`` 按到达顺序成对写入 ``shard_XXXX.npz``。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gnuradio import gr

DeviceId = Literal["dev1", "dev0"]

_SESSIONS: dict[str, RangeProfileSession] = {}
_SESSIONS_LOCK = threading.Lock()


def get_range_profile_session(
    output_dir: str,
    *,
    vlen: int,
    label: str,
    flush_every: int,
    meta_static: dict[str, Any] | None,
    record_enable: bool,
) -> RangeProfileSession:
    """按 output_dir 复用会话（GRC 双块共享同一目录）。"""
    key = str(output_dir)
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = RangeProfileSession(
                output_dir=key,
                vlen=vlen,
                label=label,
                flush_every=flush_every,
                meta_static=meta_static,
                record_enable=record_enable,
            )
            _SESSIONS[key] = session
        else:
            session.update_runtime(
                vlen=vlen,
                label=label,
                flush_every=flush_every,
                meta_static=meta_static,
                record_enable=record_enable,
            )
        return session


class RangeProfileSession:
    """成对缓冲、分片落盘与 meta 管理。"""

    _MAX_PENDING_SHARDS = 4

    def __init__(
        self,
        output_dir: str,
        vlen: int,
        label: str,
        flush_every: int,
        meta_static: dict[str, Any] | None,
        record_enable: bool,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._vlen = int(vlen)
        self._label = str(label)
        self._flush_every = max(1, int(flush_every))
        self._meta_static = dict(meta_static or {})
        self._record_enable = bool(record_enable)

        self._dev1_q: deque[np.ndarray] = deque()
        self._dev0_q: deque[np.ndarray] = deque()
        self._dev1_buf: list[np.ndarray] = []
        self._dev0_buf: list[np.ndarray] = []
        self._ts_buf: list[float] = []
        self._idx_buf: list[int] = []

        self._global_frame = 0
        self._shard_idx = 0
        self._total_frames = 0
        self._session_started = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="range_rec")
        self._pending: Future | None = None
        self._slow_io_warned = False

        if self._record_enable:
            self._prepare_session()

    def update_runtime(
        self,
        *,
        vlen: int,
        label: str,
        flush_every: int,
        meta_static: dict[str, Any] | None,
        record_enable: bool,
    ) -> None:
        self._vlen = int(vlen)
        self._label = str(label)
        self._flush_every = max(1, int(flush_every))
        self._meta_static = dict(meta_static or {})
        enabled = bool(record_enable)
        if enabled and not self._record_enable:
            self._prepare_session()
        self._record_enable = enabled

    @property
    def record_enable(self) -> bool:
        return self._record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        enabled = bool(value)
        if enabled and not self._record_enable:
            self._prepare_session()
        self._record_enable = enabled

    @property
    def output_dir(self) -> str:
        return str(self._output_dir)

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self._output_dir = Path(value)

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = str(value)

    @property
    def flush_every(self) -> int:
        return self._flush_every

    @flush_every.setter
    def flush_every(self, value: int) -> None:
        self._flush_every = max(1, int(value))

    def push(self, device_id: DeviceId, vector: np.ndarray) -> None:
        if not self._record_enable:
            return
        vec = np.asarray(vector, dtype=np.float32).copy()
        with self._lock:
            if device_id == "dev1":
                self._dev1_q.append(vec)
            else:
                self._dev0_q.append(vec)
            self._pair_locked()

    def finalize(self) -> None:
        if not self._record_enable:
            return
        with self._lock:
            if self._dev1_q or self._dev0_q:
                print(
                    f"[range_recorder] WARN: unmatched frames at stop "
                    f"(dev1={len(self._dev1_q)}, dev0={len(self._dev0_q)})",
                    file=sys.stderr,
                )
        self._flush_buffers()
        self._wait_pending()
        self._write_meta(final=True)
        print(
            f"[range_recorder] stopped: {self._total_frames} frames, "
            f"{self._shard_idx} shards → {self._output_dir.resolve()}",
            file=sys.stderr,
        )

    def _pair_locked(self) -> None:
        while self._dev1_q and self._dev0_q:
            dev1 = self._dev1_q.popleft()
            dev0 = self._dev0_q.popleft()
            self._dev1_buf.append(dev1)
            self._dev0_buf.append(dev0)
            self._ts_buf.append(time.time())
            self._idx_buf.append(self._global_frame)
            self._global_frame += 1
            if len(self._dev0_buf) >= self._flush_every:
                self._flush_buffers_locked()

    def _prepare_session(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if not self._session_started:
            self._write_meta(final=False)
            self._session_started = True
            print(
                f"[range_recorder] recording → {self._output_dir.resolve()} "
                f"label={self._label!r} flush_every={self._flush_every}",
                file=sys.stderr,
            )

    def _meta_payload(self, *, final: bool) -> dict[str, Any]:
        cp_len = self._meta_static.get("cp_len")
        if cp_len is None and "fft_len" in self._meta_static:
            cp_len = int(self._meta_static["fft_len"]) // 4
        samp_rate = float(self._meta_static.get("samp_rate", 0))
        transpose_len = int(self._meta_static.get("transpose_len", 4))
        fft_len = int(self._meta_static.get("fft_len", 2048))
        sym_samples = fft_len + int(cp_len or fft_len // 4)
        frame_rate = (
            samp_rate / (transpose_len * sym_samples) if samp_rate > 0 else 0.0
        )
        meta: dict[str, Any] = {
            "vlen": self._vlen,
            "label": self._label,
            "data_type": "linear_power",
            "input0_device": "dev1",
            "input1_device": "dev0",
            "total_frames": self._total_frames,
            "shard_count": self._shard_idx,
            "flush_every": self._flush_every,
            "frame_rate_hz": frame_rate,
            "finalized": final,
        }
        meta.update(self._meta_static)
        return meta

    def _write_meta(self, *, final: bool) -> None:
        path = self._output_dir / "meta.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._meta_payload(final=final), f, indent=2, ensure_ascii=False)
            f.write("\n")

    def _wait_pending(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def _submit_shard(
        self,
        dev1: np.ndarray,
        dev0: np.ndarray,
        timestamps: np.ndarray,
        frame_idx: np.ndarray,
        shard_idx: int,
    ) -> None:
        out_path = self._output_dir / f"shard_{shard_idx:04d}.npz"

        def _write() -> None:
            t0 = time.perf_counter()
            np.savez(
                out_path,
                profiles_dev1=dev1,
                profiles_dev0=dev0,
                timestamps=timestamps,
                frame_idx=frame_idx,
            )
            elapsed = time.perf_counter() - t0
            if elapsed > 0.15 and not self._slow_io_warned:
                self._slow_io_warned = True
                print(
                    f"[range_recorder] WARN: slow disk flush "
                    f"{elapsed:.2f}s for {dev1.shape[0]} frames → {out_path.name}",
                    file=sys.stderr,
                )

        self._wait_pending()
        self._pending = self._executor.submit(_write)

    def _flush_buffers(self) -> None:
        with self._lock:
            self._flush_buffers_locked()

    def _flush_buffers_locked(self) -> None:
        if not self._dev0_buf:
            return
        dev1 = np.stack(self._dev1_buf, axis=0)
        dev0 = np.stack(self._dev0_buf, axis=0)
        timestamps = np.asarray(self._ts_buf, dtype=np.float64)
        frame_idx = np.asarray(self._idx_buf, dtype=np.int64)
        n = dev0.shape[0]
        self._dev1_buf.clear()
        self._dev0_buf.clear()
        self._ts_buf.clear()
        self._idx_buf.clear()
        shard = self._shard_idx
        self._shard_idx += 1
        self._total_frames += n
        self._submit_shard(dev1, dev0, timestamps, frame_idx, shard)


def build_range_meta_static(
    *,
    fft_len: int = 2048,
    samp_rate: float = 122880000.0,
    R_max: float = 2500.0,
    range_bin_step: float = 0.61,
    zeropadding_fac: int = 2,
    transpose_len: int = 4,
    freq0: float = 6.03e9,
    freq1: float = 5.97e9,
    num_delay_samp0: int = 161,
    num_delay_samp1: int = 161,
) -> dict[str, Any]:
    """构建写入 meta.json 的静态参数字典（GRC 友好）。"""
    return {
        "fft_len": int(fft_len),
        "samp_rate": float(samp_rate),
        "R_max": float(R_max),
        "range_bin_step": float(range_bin_step),
        "zeropadding_fac": int(zeropadding_fac),
        "transpose_len": int(transpose_len),
        "freq0": float(freq0),
        "freq1": float(freq1),
        "num_delay_samp0": int(num_delay_samp0),
        "num_delay_samp1": int(num_delay_samp1),
    }


class DevRangeProfileRecorder(gr.sync_block):
    """单路距离谱输入；经 dummy 输出接入 null_sink。"""

    def __init__(
        self,
        device_id: DeviceId = "dev1",
        vlen: int = 4096,
        record_enable: bool = False,
        output_dir: str = "dataset/run_001",
        label: str = "",
        flush_every: int = 1200,
        fft_len: int = 2048,
        samp_rate: float = 122880000.0,
        R_max: float = 2500.0,
        range_bin_step: float = 0.61,
        zeropadding_fac: int = 2,
        transpose_len: int = 4,
        freq0: float = 6.03e9,
        freq1: float = 5.97e9,
        num_delay_samp0: int = 161,
        num_delay_samp1: int = 161,
        meta_static: dict[str, Any] | None = None,
    ) -> None:
        if meta_static is None:
            meta_static = build_range_meta_static(
                fft_len=fft_len,
                samp_rate=samp_rate,
                R_max=R_max,
                range_bin_step=range_bin_step,
                zeropadding_fac=zeropadding_fac,
                transpose_len=transpose_len,
                freq0=freq0,
                freq1=freq1,
                num_delay_samp0=num_delay_samp0,
                num_delay_samp1=num_delay_samp1,
            )
        gr.sync_block.__init__(
            self,
            name=f"DevRangeProfileRecorder_{device_id}",
            in_sig=[(np.float32, int(vlen))],
            out_sig=[(np.float32, 1)],
        )
        self._device_id: DeviceId = device_id
        self._session = get_range_profile_session(
            output_dir,
            vlen=vlen,
            label=label,
            flush_every=flush_every,
            meta_static=meta_static,
            record_enable=record_enable,
        )

    @property
    def record_enable(self) -> bool:
        return self._session.record_enable

    @record_enable.setter
    def record_enable(self, value: bool) -> None:
        self._session.record_enable = value

    @property
    def output_dir(self) -> str:
        return self._session.output_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self._session.output_dir = value

    @property
    def label(self) -> str:
        return self._session.label

    @label.setter
    def label(self, value: str) -> None:
        self._session.label = value

    @property
    def flush_every(self) -> int:
        return self._session.flush_every

    @flush_every.setter
    def flush_every(self, value: int) -> None:
        self._session.flush_every = value

    def work(self, input_items: list, output_items: list) -> int:
        n = len(input_items[0])
        if n <= 0:
            return 0
        output_items[0][:n] = 0.0
        if self._session.record_enable:
            if (
                self._session._pending is not None
                and self._session._pending.running()
                and len(self._session._dev0_buf)
                >= self._session._flush_every * RangeProfileSession._MAX_PENDING_SHARDS
                and not self._session._slow_io_warned
            ):
                self._session._slow_io_warned = True
                print(
                    "[range_recorder] WARN: disk flush lagging; "
                    "memory buffer growing (no frames dropped)",
                    file=sys.stderr,
                )
            for i in range(n):
                self._session.push(self._device_id, input_items[0][i])
        return n

    def stop(self) -> int:
        # 仅 dev0 块负责 finalize，避免重复写 meta
        if self._device_id == "dev0" and self._session.record_enable:
            self._session.finalize()
        return super().stop()


# 兼容 GRC epy 单块命名（device_id 由子类 wrapper 传入）
PairedRangeProfileRecorder = DevRangeProfileRecorder
blk = DevRangeProfileRecorder
