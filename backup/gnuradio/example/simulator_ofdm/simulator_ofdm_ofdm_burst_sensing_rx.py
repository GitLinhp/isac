"""
GNU Radio 嵌入式 Python 块：OFDM 突发感知接收（GR 同步后 LS/DD）

上游 ``header_payload_demux`` 在 SC 同步 + CFO 补偿后按 OFDM 符号向量（vlen=fft_len）
输出载荷；本块按符号计数攒齐 ``burst_len`` 样点后异步执行 LS/DD。

System / burst_len / OFDM 几何一律来自 TOML（config_file）。
注意：__init__ 形参默认值须与 GRC 变量保持同步。
"""

from __future__ import annotations

import queue
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pmt
import torch
from gnuradio import gr

from isac.system import System
from isac.utils import set_random_seed
from isac_imp.burst_pack import TPP_DONT
from isac_imp.gr_setup import (
    resolve_dd_output_vlen,
    resolve_ofdm_burst_len,
    resolve_ofdm_fft_cp,
    resolve_ofdm_num_symbols,
)

# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_TAG_PACKET_LEN = pmt.intern("packet_len")
_TAG_FRAME_LEN = pmt.intern("frame_len")
_LOG_EPS = 1e-20
_LOG_PREFIX = "[OFDM Burst Sensing RX]"


def _dd_log_magnitude(h_dd: np.ndarray, log_eps: float = _LOG_EPS) -> np.ndarray:
    """|·|² → +ε → log10，形状保持 (n_doppler, n_delay)。"""
    mag2 = np.abs(h_dd.astype(np.complex128, copy=False)) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


class blk(gr.sync_block):
    """GR 同步后按 OFDM 符号向量计数攒包的感知接收。"""

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        device="cpu",
        seed=42,
    ):
        self._config_file = str(config_file)
        self._device = str(device)
        self._seed = int(seed)
        self._fft_size, self._cp_len = resolve_ofdm_fft_cp(self._config_file)
        self._payload_syms = int(resolve_ofdm_num_symbols(self._config_file))
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)

        gr.sync_block.__init__(
            self,
            name="OFDM Burst Sensing RX",
            in_sig=[(np.complex64, self._fft_size)],
            out_sig=[(np.float32, self._dd_vlen)],
        )

        self._system: System | None = None
        self._x_rg: Any = None

        self._sym_buf = np.zeros(
            (self._payload_syms, self._fft_size), dtype=np.complex64
        )
        self._sym_idx = 0

        self._executor: ThreadPoolExecutor | None = None
        self._worker_busy = False
        self._result_queue: queue.Queue[
            list[tuple[np.ndarray, bool, int]]
        ] = queue.Queue()
        self._pending_rows: deque[tuple[np.ndarray, bool, int]] = deque()

        self.set_tag_propagation_policy(TPP_DONT)
        self._apply_min_input_buffer()

    def _apply_min_input_buffer(self) -> None:
        if hasattr(self, "set_min_input_buffer"):
            self.set_min_input_buffer(max(4096, self._payload_syms))

    @property
    def burst_len(self) -> int:
        return self._burst_len

    @property
    def dd_vlen(self) -> int:
        return self._dd_vlen

    def start(self):
        self._shutdown_worker()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sensing_rx"
        )
        self._worker_busy = False
        self._sym_idx = 0
        return True

    def stop(self):
        self._shutdown_worker()
        return True

    def _shutdown_worker(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._worker_busy = False
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def _ensure_system(self) -> System:
        if self._system is None:
            torch.set_num_threads(1)
            set_random_seed(self._seed)
            self._system = System(self._config_file, device=self._device)
            ofdm = self._system.params.ofdm
            if ofdm is not None:
                self._fft_size = int(ofdm.fft_size)
                self._cp_len = int(ofdm.cyclic_prefix_length)
                self._payload_syms = int(ofdm.num_symbols)
                self._burst_len = int(
                    ofdm.num_symbols * (ofdm.fft_size + ofdm.cyclic_prefix_length)
                )
                self._sym_buf = np.zeros(
                    (self._payload_syms, self._fft_size), dtype=np.complex64
                )
                self._apply_min_input_buffer()
        return self._system

    def _ensure_waveform(self) -> bool:
        if self._x_rg is not None:
            return True
        system = self._ensure_system()
        cache = system.components.transmit_cache
        if cache is None:
            return False
        try:
            self._x_rg = cache.load_x_rg()
        except (FileNotFoundError, ValueError):
            return False
        return True

    def _invalidate(self) -> None:
        self._shutdown_worker()
        self._system = None
        self._x_rg = None
        self._sym_idx = 0
        self._pending_rows.clear()
        self._fft_size, self._cp_len = resolve_ofdm_fft_cp(self._config_file)
        self._payload_syms = int(resolve_ofdm_num_symbols(self._config_file))
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)
        self._sym_buf = np.zeros(
            (self._payload_syms, self._fft_size), dtype=np.complex64
        )
        self._apply_min_input_buffer()

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        self._config_file = str(value)
        self._invalidate()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = str(value)
        self._invalidate()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)
        self._invalidate()

    def _rows_from_h_dd(self, h_dd: torch.Tensor) -> list[tuple[np.ndarray, bool, int]]:
        arr = h_dd.detach().cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        log_mag = _dd_log_magnitude(arr)
        n_rows, n_cols = int(log_mag.shape[0]), int(log_mag.shape[1])
        vlen = self._dd_vlen
        rows: list[tuple[np.ndarray, bool, int]] = []
        for i in range(n_rows):
            row = np.full(vlen, np.nan, dtype=np.float32)
            n = min(n_cols, vlen)
            row[:n] = log_mag[i, :n]
            rows.append((row, i == 0, n_rows))
        return rows

    def _compute_burst_rows(
        self, y_np: np.ndarray
    ) -> list[tuple[np.ndarray, bool, int]]:
        burst_len = self._burst_len
        if y_np.size < burst_len:
            return []
        y_np = y_np[:burst_len]

        system = self._ensure_system()
        if self._x_rg is None:
            return []
        x_rg = self._x_rg

        comps = system.components
        if comps.demodulator is None:
            return []

        y_time = torch.from_numpy(np.ascontiguousarray(y_np)).to(
            device=system.device,
            dtype=torch.complex64,
        )
        while y_time.dim() < 3:
            y_time = y_time.unsqueeze(0)

        with torch.inference_mode():
            y_rg = comps.demodulator(y_time)
            h_freq = comps.ls_channel_estimator(x_rg, y_rg)
            h_dd = comps.delay_doppler_spectrum(h_freq, sens_mode="monostatic")
            return self._rows_from_h_dd(h_dd)

    def _on_burst_done(self, future) -> None:
        try:
            rows = future.result()
            if rows:
                self._result_queue.put(rows)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        finally:
            self._worker_busy = False

    def _submit_burst_async(self, y_burst: np.ndarray) -> None:
        if self._executor is None or self._worker_busy:
            return
        self._worker_busy = True
        print(
            f"{_LOG_PREFIX} payload_len={y_burst.size} (GR sync)",
            file=sys.stderr,
            flush=True,
        )
        fut = self._executor.submit(self._compute_burst_rows, y_burst)
        fut.add_done_callback(self._on_burst_done)

    def _drain_worker_results(self) -> None:
        while True:
            try:
                rows = self._result_queue.get_nowait()
            except queue.Empty:
                break
            self._pending_rows.extend(rows)

    def _has_frame_len_tag(self, abs_offset: int) -> bool:
        for tag in self.get_tags_in_range(0, abs_offset, abs_offset + 1):
            if tag.key == _TAG_FRAME_LEN:
                return True
        return False

    def _finish_burst(self) -> None:
        y_burst = self._sym_buf.reshape(-1).astype(np.complex64, copy=False)
        self._sym_idx = 0
        self._submit_burst_async(y_burst)

    def _process_symbol(self, sym: np.ndarray, abs_offset: int) -> None:
        if self._sym_idx > 0 and self._has_frame_len_tag(abs_offset):
            self._sym_idx = 0
        self._sym_buf[self._sym_idx, :] = np.asarray(sym, dtype=np.complex64).ravel()[
            : self._fft_size
        ]
        self._sym_idx += 1
        if self._sym_idx >= self._payload_syms:
            self._finish_burst()

    def work(self, input_items, output_items):
        self._drain_worker_results()

        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)
        produced = 0

        if n_in > 0:
            if not self._ensure_waveform():
                pass
            else:
                base = int(self.nitems_read(0))
                for i in range(n_in):
                    self._process_symbol(inp[i], base + i)

        while produced < len(out) and self._pending_rows:
            row, is_first, n_rows = self._pending_rows.popleft()
            if is_first:
                abs_out = self.nitems_written(0) + produced
                self.add_item_tag(
                    0, abs_out, _TAG_PACKET_LEN, pmt.from_long(int(n_rows))
                )
            out[produced][:] = row
            produced += 1

        return produced
