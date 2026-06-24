"""Sionna RT 射线追踪信道：System.components.channel(domain=time) 的 GNU Radio 流块。"""
import queue
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import setup_gnuradio_paths_from

setup_gnuradio_paths_from(__file__)

import numpy as np
import pmt
from gnuradio import gr

from gr_config import resolve_config_path  # noqa: E402
from gr_system import GrSystemContext, get_gr_system_context, ofdm_packet_len_time  # noqa: E402


class _CudaChannelWorker:
    """专用 CUDA 线程，避免 GR 工作线程直接调 RT 信道。"""

    _instances: dict[Tuple[int, str], "_CudaChannelWorker"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, ctx: GrSystemContext) -> "_CudaChannelWorker":
        key = (id(ctx), ctx.device)
        with cls._instances_lock:
            worker = cls._instances.get(key)
            if worker is None:
                worker = cls(ctx)
                cls._instances[key] = worker
            return worker

    def __init__(self, ctx: GrSystemContext) -> None:
        self._ctx = ctx
        self._device = ctx.device
        self._snr_db: Optional[float] = None
        self._in_q: queue.Queue = queue.Queue(maxsize=4)
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="SionnaCudaRTChannel", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=180):
            raise RuntimeError("Sionna RT 信道 CUDA 线程初始化超时")

    def _loop(self) -> None:
        import torch

        if str(self._device).startswith("cuda"):
            idx = int(str(self._device).split(":")[1]) if ":" in str(self._device) else 0
            torch.cuda.set_device(idx)

        # 用真实发端包、无 AWGN 预热 CUDA（全零 dummy + snr_db 会在 AWGN 定标处失败）
        pkt = self._ctx.transmit_packet()
        self._ctx.apply_channel_time(pkt.time, snr_db=None)
        self._ready.set()

        while True:
            job = self._in_q.get()
            if job is None:
                break
            tx, snr_db, resp_q = job
            try:
                rx = self._ctx.apply_channel_time(tx, snr_db=snr_db)
                resp_q.put(rx)
            except Exception as exc:
                resp_q.put(exc)

    def apply(self, tx: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        resp_q: queue.Queue = queue.Queue(maxsize=1)
        self._in_q.put((tx, snr_db, resp_q))
        result = resp_q.get(timeout=180)
        if isinstance(result, Exception):
            raise result
        return result


class SionnaRTChannel(gr.basic_block):
    """RT 信道：整包时域 IQ 经 ``System.components.channel(time)`` 输出。"""

    def __init__(
        self,
        config_file: str = "config/simulation/sensing/sensing_baseline.toml",
        fft_len: int = 1024,
        ofdm_symbols: int = 1024,
        cp_len: int = 0,
        subcarrier_spacing: float = 30000.0,
        center_freq: float = 3.5e9,
        length_tag_key: str = "packet_len",
        seed: int = 42,
        device: str = "cuda:0",
        snr_db: float = -5.0,
        burst_mode: bool = False,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna RT Channel",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

        cfg = str(resolve_config_path(config_file))
        gr_kw = dict(
            fft_len=int(fft_len),
            ofdm_symbols=int(ofdm_symbols),
            cp_len=int(cp_len),
            subcarrier_spacing=float(subcarrier_spacing),
            center_freq=float(center_freq),
        )
        self._ctx = get_gr_system_context(
            cfg, seed=int(seed), device=str(device), **gr_kw
        )
        self._worker = _CudaChannelWorker.get(self._ctx)
        self._snr_db: Optional[float] = float(snr_db)
        self._device = str(device)
        self._tag_key = pmt.intern(length_tag_key)
        self._in_packet_len = ofdm_packet_len_time(ofdm_symbols, fft_len, cp_len)
        self._default_in_packet_len = self._in_packet_len
        self._out_packet_len = self._in_packet_len

        self._in_buf: List[np.complex64] = []
        self._out_buf: List[np.complex64] = []
        self._out_packet_sample_idx = 0
        self._burst_mode = bool(burst_mode)
        self._burst_armed = not self._burst_mode
        self._last_rx_len = self._in_packet_len

        self.set_min_output_buffer(max(4096, int(ofdm_symbols)))

    def set_snr_db(self, snr_db: float) -> None:
        """GRC 滑块回调。"""
        self._snr_db = float(snr_db)

    def _resolve_snr(self) -> Optional[float]:
        return self._snr_db

    def forecast(self, noutput_items: int, ninputs) -> List[int]:
        del noutput_items, ninputs
        return [1]

    def _process_packet(self, tx: np.ndarray) -> np.ndarray:
        rx = self._worker.apply(tx, snr_db=self._resolve_snr())
        self._last_rx_len = int(rx.size)
        return rx

    def _flush_input_packet(self) -> None:
        if not self._in_buf:
            return
        tx = np.ascontiguousarray(self._in_buf, dtype=np.complex64)
        self._out_buf.extend(self._process_packet(tx).tolist())
        self._out_packet_sample_idx = 0
        self._in_buf.clear()

    def _handle_packet_len_tag(self) -> None:
        if self._in_buf:
            self._flush_input_packet()
        self._in_buf.clear()
        self._out_packet_sample_idx = 0

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)
        n_out = len(out)
        consumed = 0
        produced = 0

        while produced < n_out and self._out_buf:
            if self._out_packet_sample_idx == 0:
                self.add_item_tag(
                    0,
                    self.nitems_written(0) + produced,
                    self._tag_key,
                    pmt.from_long(self._out_packet_len),
                )
            out[produced] = self._out_buf.pop(0)
            self._out_packet_sample_idx += 1
            if self._out_packet_sample_idx >= self._out_packet_len:
                self._out_packet_sample_idx = 0
            produced += 1

        while consumed < n_in and produced < n_out:
            abs_idx = self.nitems_read(0) + consumed
            for tag in self.get_tags_in_window(0, abs_idx, 1):
                if tag.key == self._tag_key:
                    self._handle_packet_len_tag()
                    self._in_packet_len = int(pmt.to_long(tag.value))
                    if self._burst_mode:
                        self._burst_armed = True

            if self._burst_mode and not self._burst_armed:
                out[produced] = inp[consumed]
                consumed += 1
                produced += 1
                continue

            self._in_buf.append(inp[consumed])
            consumed += 1

            if len(self._in_buf) >= self._in_packet_len:
                self._flush_input_packet()
                self._out_packet_len = self._last_rx_len
                if self._burst_mode:
                    self._burst_armed = False
                while produced < n_out and self._out_buf:
                    if self._out_packet_sample_idx == 0:
                        self.add_item_tag(
                            0,
                            self.nitems_written(0) + produced,
                            self._tag_key,
                            pmt.from_long(self._out_packet_len),
                        )
                    out[produced] = self._out_buf.pop(0)
                    self._out_packet_sample_idx += 1
                    if self._out_packet_sample_idx >= self._out_packet_len:
                        self._out_packet_sample_idx = 0
                    produced += 1

        while consumed < n_in:
            abs_idx = self.nitems_read(0) + consumed
            for tag in self.get_tags_in_window(0, abs_idx, 1):
                if tag.key == self._tag_key:
                    self._handle_packet_len_tag()
                    self._in_packet_len = int(pmt.to_long(tag.value))
                    if self._burst_mode:
                        self._burst_armed = True

            if self._burst_mode and not self._burst_armed:
                consumed += 1
                continue

            self._in_buf.append(inp[consumed])
            consumed += 1

            if len(self._in_buf) >= self._in_packet_len:
                self._flush_input_packet()
                self._out_packet_len = self._last_rx_len
                if self._burst_mode:
                    self._burst_armed = False

        if consumed:
            self.consume(0, consumed)
        if produced:
            self.produce(0, produced)
        if consumed or produced:
            return gr.WORK_CALLED_PRODUCE
        return gr.WORK_DONE
