"""Sionna OFDMModulator epy 块：频域符号向量 → 时域+CP 标量流。

替换 GNU Radio ``fft_vxx`` (IFFT) + ``digital_ofdm_cyclic_prefixer`` 链：

    SionnaResourceGridTx (fftshift 频域, vlen=fft_len)
      → SionnaOfdmModulatorBlock (标量时域+CP)
      → multiply_const / USRP

与 ``system_components.build_from_params`` 相同，使用
``OFDMModulator(cyclic_prefix_length=cp_len, device=device)``。
Sionna 内部已 ``ifftshift``；输出乘 ``1/sqrt(fft_len)`` 对齐 GR ``fft_vxx`` 幅度。

按 CPI 批量调用 modulator，并在独立 worker 线程执行 Torch，避免 GR 调度线程 SIGSEGV。
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pmt
import torch
from gnuradio import gr
from sionna.phy.ofdm import OFDMModulator

from isac_imp.burst_pack import TPP_DONT

_TORCH_LOCK = threading.Lock()


class SionnaOfdmModulatorBlock(gr.basic_block):
    """fftshift 频域 OFDM 符号 (vlen=fft_len) → 时域+CP 标量流。"""

    def __init__(
        self,
        fft_len: int = 2048,
        cp_len: int = 512,
        device: str = "cpu",
        length_tag_key: str = "packet_len",
    ) -> None:
        self._fft_len = int(fft_len)
        self._cp_len = int(cp_len)
        self._device = str(device)
        self._sym_len = self._fft_len + self._cp_len
        self._gr_scale = 1.0 / np.sqrt(float(self._fft_len))

        gr.basic_block.__init__(
            self,
            name="Sionna OFDM Modulator",
            in_sig=[(np.complex64, self._fft_len)],
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._modulator: OFDMModulator | None = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._mod_busy = False

        self._out_buf: np.ndarray | None = None
        self._out_idx = 0
        self._pending_tag: tuple[pmt.pmt, pmt.pmt] | None = None
        self._cpi_len: int | None = None
        self._freq_collect: list[np.ndarray] = []
        self._ready_queue: list[np.ndarray] = []

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._sym_len * 2)

    def forecast(self, noutput_items: int, ninputs: list) -> list:
        del ninputs
        remaining = 0
        if self._out_buf is not None:
            remaining = len(self._out_buf) - self._out_idx
        need_out = max(0, noutput_items - remaining)
        need_syms = (need_out + self._sym_len - 1) // self._sym_len
        if self._freq_collect and self._cpi_len is not None:
            need_syms = max(need_syms, self._cpi_len - len(self._freq_collect))
        elif self._out_buf is None or self._out_idx >= len(self._out_buf):
            need_syms = max(need_syms, self._cpi_len or 1)
        return [need_syms]

    def start(self) -> bool:
        torch.set_num_threads(1)
        self._modulator = OFDMModulator(
            cyclic_prefix_length=self._cp_len,
            device=self._device,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sionna_ofdm_mod")
        self._mod_busy = False
        self._reset_state()
        self._ready_queue = []
        return True

    def stop(self) -> bool:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._mod_busy = False
        self._modulator = None
        return True

    def _reset_state(self) -> None:
        self._out_buf = None
        self._out_idx = 0
        self._pending_tag = None
        self._cpi_len = None
        self._freq_collect = []

    def _tag_to_int(self, value: pmt.pmt) -> int:
        try:
            return int(pmt.to_long(value))
        except Exception:
            return int(pmt.to_python(value))

    def _modulate_cpi_sync(self, freq_cpi: np.ndarray) -> np.ndarray:
        assert self._modulator is not None
        cpi_len = int(freq_cpi.shape[0])
        x = torch.from_numpy(
            np.asarray(freq_cpi, dtype=np.complex64).reshape(1, cpi_len, self._fft_len)
        )
        if self._device != "cpu":
            x = x.to(self._device)
        with _TORCH_LOCK, torch.inference_mode():
            y = self._modulator(x)
        if self._device != "cpu":
            y = y.cpu()
        return (
            y.reshape(-1)
            .numpy()
            .astype(np.complex64, copy=False)
            * np.complex64(self._gr_scale)
        )

    def _on_cpi_ready(self, time_buf: np.ndarray) -> None:
        self._ready_queue.append(time_buf)
        self._mod_busy = False

    def _submit_cpi(self, freq_cpi: np.ndarray) -> None:
        assert self._executor is not None
        self._mod_busy = True
        buf = freq_cpi.copy()
        self._executor.submit(self._run_cpi_job, buf)

    def _run_cpi_job(self, freq_cpi: np.ndarray) -> None:
        try:
            time_buf = self._modulate_cpi_sync(freq_cpi)
        except Exception:
            self._mod_busy = False
            raise
        self._on_cpi_ready(time_buf)

    def general_work(self, input_items, output_items) -> int:
        out = output_items[0]
        n_produced = 0

        while n_produced < len(out):
            if self._out_buf is None or self._out_idx >= len(self._out_buf):
                if self._ready_queue:
                    self._out_buf = self._ready_queue.pop(0)
                    self._out_idx = 0
                else:
                    break

            if self._pending_tag is not None and self._out_idx == 0:
                key, val = self._pending_tag
                self.add_item_tag(
                    0,
                    self.nitems_written(0) + n_produced,
                    key,
                    val,
                )
                self._pending_tag = None

            out[n_produced] = self._out_buf[self._out_idx]
            self._out_idx += 1
            n_produced += 1

            if self._out_idx >= len(self._out_buf):
                self._out_buf = None
                self._out_idx = 0

        while (
            not self._mod_busy
            and len(input_items[0]) >= 1
            and (self._cpi_len is None or len(self._freq_collect) < self._cpi_len)
        ):
            in_offset = self.nitems_read(0)
            freq = np.asarray(input_items[0][0], dtype=np.complex64)
            for tag in self.get_tags_in_range(0, in_offset, in_offset + 1):
                if pmt.eq(tag.key, self._length_tag_key):
                    self._pending_tag = (tag.key, tag.value)
                    self._cpi_len = self._tag_to_int(tag.value)

            self._freq_collect.append(freq.copy())
            self.consume(0, 1)

            if self._cpi_len is not None and len(self._freq_collect) >= self._cpi_len:
                freq_cpi = np.stack(self._freq_collect[: self._cpi_len], axis=0)
                self._freq_collect = self._freq_collect[self._cpi_len :]
                self._submit_cpi(freq_cpi)

        return n_produced
