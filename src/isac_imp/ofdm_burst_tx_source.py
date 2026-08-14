"""统一 OFDM 突发发射源：ResourceGrid + IFFT/CP + 幅度 + Style1/调度。

无流输入；输出::

    out0  时域 IQ（含 Style1 tx_sob/tx_time/tx_eob）→ USRP Sink
    tx_schedule / tx_freq_cpi  消息口：计划 epoch + 频域 CPI 参考（后者仅首 CPI 发一次）

``start()`` 在 ``device``（默认 cpu；本测试图用 cuda）上用 Torch 预计算固定 CPI
时域/频域缓存，一次 D2H 后运行期仅 memcpy 重放，避免 USRP underflow。
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Optional

import numpy as np
import pmt
import torch
from gnuradio import gr

from isac_imp.burst_pack import (
    PORT_TX_FREQ_CPI,
    PORT_TX_SCHEDULE,
    TPP_DONT,
    add_style1_eob,
    add_style1_sob_time,
    make_tx_freq_cpi_msg,
    make_tx_schedule_msg,
    schedule_idle_delay_s,
)
from isac_imp.sionna_resource_grid_tx import _build_freq_grid_torch
from sionna.phy.ofdm import OFDMModulator

_CMD_Q_MAX = 8


def _normalize_torch_device(device: str) -> str:
    """Map ``cuda`` → first CUDA device for Sionna (expects ``cuda:0``)."""
    d = str(device).strip()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return "cuda:0"
    return d


def _modulate_symbol(freq_shifted: np.ndarray, cp_len: int) -> np.ndarray:
    """对齐 GR ``fft_vcc``(IFFT, shift=True) + ``ofdm_cyclic_prefixer``(rolloff=0)。"""
    td = np.fft.ifft(np.fft.ifftshift(freq_shifted), norm="forward")
    return np.concatenate((td[-cp_len:], td)).astype(np.complex64, copy=False)


def _precompute_cpi_torch(
    *,
    fft_len: int,
    num_symbols: int,
    subcarrier_spacing: float,
    cp_len: int,
    num_bits_per_symbol: int,
    device: str,
    seed: int,
    factor: float,
    burst_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """端到端 Torch 预计算，返回主机 ``(freq_cpi, td_unit, td_cpi)``。

    时域经 ``sionna.phy.ofdm.OFDMModulator``（ortho IFFT）再乘 ``sqrt(fft_len)``，
    与 GR ``fft_vcc`` / ``norm=\"forward\"`` 幅度对齐。
    """
    device = _normalize_torch_device(device)
    with torch.inference_mode():
        freq_t, _, _ = _build_freq_grid_torch(
            fft_len=fft_len,
            num_symbols=num_symbols,
            subcarrier_spacing=subcarrier_spacing,
            cp_len=cp_len,
            num_bits_per_symbol=num_bits_per_symbol,
            device=device,
            seed=seed,
        )
        modulator = OFDMModulator(cyclic_prefix_length=cp_len, device=device)
        # Sionna ortho (1/√N) → ×√N 对齐 GR forward IFFT
        td_unit_t = modulator(freq_t) * (float(fft_len) ** 0.5)
        if int(td_unit_t.numel()) != int(burst_len):
            raise RuntimeError(
                f"td burst length {td_unit_t.numel()} != expected {burst_len}"
            )
        td_cpi_t = td_unit_t * float(factor)
        if str(freq_t.device).startswith("cuda"):
            torch.cuda.synchronize()
        freq_cpi = freq_t.detach().cpu().numpy().astype(np.complex64, copy=False)
        td_unit = td_unit_t.detach().cpu().numpy().astype(np.complex64, copy=False)
        td_cpi = td_cpi_t.detach().cpu().numpy().astype(np.complex64, copy=False)
    return freq_cpi, td_unit, td_cpi


class OfdmBurstTxSourceBlock(gr.basic_block):
    """预计算 CPI → 时域 Style1 IQ；频域参考经消息一次性下发。"""

    def __init__(
        self,
        fft_len: int = 2048,
        cp_len: int = 512,
        num_symbols: int = 4,
        subcarrier_spacing: float = 60e3,
        num_bits_per_symbol: int = 2,
        seed: int = 42,
        device: str = "cpu",
        factor: float = 0.008,
        length_tag_key: str = "packet_len",
        time_lead_s: float = 0.5,
        idle_ms: float = 10.0,
        samp_rate: float = 122_880_000.0,
        scheduled_rx: bool = True,
        num_delay_samp: int = 0,
        **_ignored,
    ) -> None:
        del _ignored
        self._fft_len = int(fft_len)
        self._cp_len = int(cp_len)
        self._sym_len = self._fft_len + self._cp_len
        self._num_symbols = int(num_symbols)
        self._burst_len = self._num_symbols * self._sym_len
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._seed = int(seed)
        self._device = str(device)
        self._factor = float(factor)
        self._time_lead_s = float(time_lead_s)
        self._idle_ms = float(idle_ms)
        self._samp_rate = float(samp_rate)
        self._scheduled_rx = bool(scheduled_rx)
        self._num_delay_samp = int(num_delay_samp)
        self._usrp_source = None
        self._cmd_executor: Optional[ThreadPoolExecutor] = None
        self._cmd_busy = False
        self._cmd_q: Deque[tuple[float, int]] = deque()
        self._cmd_drop = 0

        self._idle_s = schedule_idle_delay_s(
            self._burst_len, self._samp_rate, self._idle_ms / 1000.0
        )
        self._burst_period_s = 0.0
        self._recompute_burst_period()

        gr.basic_block.__init__(
            self,
            name="OFDM Burst TX Source",
            in_sig=None,
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._srcid = pmt.intern("ofdm_burst_tx_source")
        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self._freq_port = pmt.intern(PORT_TX_FREQ_CPI)
        self.message_port_register_out(self._schedule_port)
        self.message_port_register_out(self._freq_port)

        self._freq_cpi: np.ndarray | None = None
        self._freq_msg = None
        self._freq_msg_sent = False
        self._td_unit: np.ndarray | None = None
        self._td_cpi: np.ndarray | None = None

        self._td_pos = 0
        self._burst_armed = False
        self._next_tx_epoch = 0.0
        self._burst_started = False
        self._burst_write_t0: float | None = None
        self._last_write_ms = 0.0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_relative_rate(1.0)
        self.set_min_output_buffer(0, self._burst_len * 2)
        self._log_last_epoch = None
        self._log_epoch_n = 0
        self._log_last_host_mono: float | None = None

    def _recompute_burst_period(self) -> None:
        burst_s = (
            float(self._burst_len) / float(self._samp_rate) if self._samp_rate > 0 else 0.0
        )
        self._burst_period_s = burst_s + self._idle_s

    def _apply_factor(self) -> None:
        if self._td_unit is None:
            return
        self._td_cpi = (self._td_unit * self._factor).astype(np.complex64, copy=False)

    @property
    def factor(self) -> float:
        return self._factor

    @factor.setter
    def factor(self, value: float) -> None:
        self._factor = float(value)
        self._apply_factor()

    @property
    def time_lead_s(self) -> float:
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value: float) -> None:
        self._time_lead_s = float(value)

    @property
    def idle_ms(self) -> float:
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value: float) -> None:
        self._idle_ms = float(value)
        self._idle_s = schedule_idle_delay_s(
            self._burst_len, self._samp_rate, self._idle_ms / 1000.0
        )
        self._recompute_burst_period()

    @property
    def samp_rate(self) -> float:
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value: float) -> None:
        self._samp_rate = float(value)
        self._idle_s = schedule_idle_delay_s(
            self._burst_len, self._samp_rate, self._idle_ms / 1000.0
        )
        self._recompute_burst_period()

    @property
    def num_delay_samp(self) -> int:
        return self._num_delay_samp

    @num_delay_samp.setter
    def num_delay_samp(self, value: int) -> None:
        self._num_delay_samp = int(value)

    @property
    def scheduled_rx(self) -> bool:
        return self._scheduled_rx

    @scheduled_rx.setter
    def scheduled_rx(self, value: bool) -> None:
        self._scheduled_rx = bool(value)

    def bind_scheduled_rx(self, usrp_source: Any) -> None:
        from isac_imp.scheduled_rx_registry import register_usrp_source

        self._usrp_source = usrp_source
        self._scheduled_rx = True
        register_usrp_source(usrp_source)

    def start(self) -> bool:
        torch.set_num_threads(1)
        freq, td_unit, td_cpi = _precompute_cpi_torch(
            fft_len=self._fft_len,
            num_symbols=self._num_symbols,
            subcarrier_spacing=self._subcarrier_spacing,
            cp_len=self._cp_len,
            num_bits_per_symbol=self._num_bits_per_symbol,
            device=self._device,
            seed=self._seed,
            factor=self._factor,
            burst_len=self._burst_len,
        )
        self._freq_cpi = freq
        self._freq_msg = make_tx_freq_cpi_msg(freq)
        self._freq_msg_sent = False
        self._td_unit = td_unit
        self._td_cpi = td_cpi

        self._td_pos = 0
        self._burst_armed = False
        self._next_tx_epoch = time.time() + self._time_lead_s
        self._burst_started = False
        self._burst_write_t0 = None
        self._last_write_ms = 0.0
        self._cmd_drop = 0
        self._cmd_q.clear()
        self._cmd_busy = False
        self._recompute_burst_period()
        if self._scheduled_rx and self._usrp_source is None:
            from isac_imp.scheduled_rx_registry import get_usrp_source

            self._usrp_source = get_usrp_source()
        if self._cmd_executor is not None:
            self._cmd_executor.shutdown(wait=False, cancel_futures=True)
        self._cmd_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ofdm_burst_tx_stream_cmd"
        )
        return True

    def stop(self) -> bool:
        if self._cmd_executor is not None:
            self._cmd_executor.shutdown(wait=False, cancel_futures=True)
            self._cmd_executor = None
        self._cmd_busy = False
        self._cmd_q.clear()
        return True

    def _kick_stream_cmd(self) -> None:
        if self._cmd_busy or not self._cmd_q or self._cmd_executor is None:
            return
        epoch, num_samps = self._cmd_q.popleft()
        self._cmd_busy = True
        self._cmd_executor.submit(self._run_stream_cmd, epoch, num_samps)

    def _run_stream_cmd(self, tx_epoch: float, num_samps: int) -> None:
        try:
            if self._usrp_source is None:
                return
            from gnuradio import uhd

            cmd = uhd.stream_cmd(uhd.stream_mode.STREAM_MODE_NUM_SAMPS_AND_DONE)
            cmd.num_samps = int(num_samps)
            cmd.stream_now = False
            cmd.time_spec = uhd.time_spec(float(tx_epoch))
            self._usrp_source.issue_stream_cmd(cmd)
        except Exception:
            traceback.print_exc()
        finally:
            self._cmd_busy = False
            self._kick_stream_cmd()

    def _issue_scheduled_recv(self, tx_epoch: float) -> None:
        """Enqueue RX stream_cmd; never block TX work on UHD/MPM RPC."""
        if not self._scheduled_rx or self._usrp_source is None:
            return
        if len(self._cmd_q) >= _CMD_Q_MAX:
            self._cmd_q.popleft()
            self._cmd_drop += 1
        self._cmd_q.append((float(tx_epoch), int(self._burst_len)))
        self._kick_stream_cmd()

    def _cmd_q_depth(self) -> int:
        return len(self._cmd_q) + int(self._cmd_busy)

    def _begin_burst(self, out_abs: int) -> None:
        # First CPI needs full time_lead for USRP arming. Later CPIs use a small
        # command lead and advance by whole burst periods if we fell behind.
        now = time.time()
        is_first = not self._burst_started
        if is_first:
            min_epoch = now + self._time_lead_s
            self._burst_started = True
            if self._next_tx_epoch < min_epoch:
                self._next_tx_epoch = min_epoch
        else:
            lead = min(0.02, max(0.0, self._time_lead_s))
            min_epoch = now + lead
            period = self._burst_period_s if self._burst_period_s > 0.0 else lead
            # Catch up on the period grid instead of jumping to now+lead once.
            while self._next_tx_epoch < min_epoch:
                self._next_tx_epoch += period
        epoch = self._next_tx_epoch
        add_style1_sob_time(self, 0, out_abs, epoch)
        self.message_port_pub(self._schedule_port, make_tx_schedule_msg(epoch))
        self._issue_scheduled_recv(epoch)
        self._burst_write_t0 = time.monotonic()
        host_now = self._burst_write_t0
        self._log_epoch_n += 1
        host_gap_ms = (
            (host_now - self._log_last_host_mono) * 1000.0
            if self._log_last_host_mono is not None
            else float("nan")
        )
        if self._log_last_epoch is not None and (
            self._log_epoch_n <= 3 or self._log_epoch_n % 500 == 0
        ):
            dt_ms = (epoch - self._log_last_epoch) * 1000.0
            msg = (
                f"[ofdm_burst_tx] epoch_dt_ms={dt_ms:.1f} "
                f"host_gap_ms={host_gap_ms:.1f} n={self._log_epoch_n}"
            )
            if self._cmd_drop > 0:
                msg += f" cmd_drop={self._cmd_drop}"
            print(msg, flush=True)
        self._log_last_epoch = epoch
        self._log_last_host_mono = host_now
        self._next_tx_epoch += self._burst_period_s

    def general_work(self, input_items, output_items) -> int:
        del input_items
        if self._td_cpi is None or self._freq_cpi is None:
            return 0

        # Do not host-side idle with return 0: that deschedules this source until
        # USRP Sink drains and self-locks near 1/epoch spacing. Inter-CPI gap is
        # encoded only in tx_time / _next_tx_epoch (burst_period).
        out_td = output_items[0]
        if len(out_td) <= 0:
            return 0

        # CPI 起始：首包发一次频域消息，再调度；后续 CPI 仅 schedule
        if self._td_pos == 0 and not self._burst_armed:
            if not self._freq_msg_sent and self._freq_msg is not None:
                self.message_port_pub(self._freq_port, self._freq_msg)
                self._freq_msg_sent = True
            self._begin_burst(self.nitems_written(0))
            self._burst_armed = True

        n = min(len(out_td), self._burst_len - self._td_pos)
        if n <= 0:
            return 0

        abs_td = self.nitems_written(0)
        out_td[:n] = self._td_cpi[self._td_pos : self._td_pos + n]

        end_pos = self._td_pos + n
        if end_pos >= self._burst_len:
            eob_abs = abs_td + (self._burst_len - 1 - self._td_pos)
            add_style1_eob(self, 0, eob_abs)

        self._td_pos += n
        if self._td_pos >= self._burst_len:
            if self._burst_write_t0 is not None:
                self._last_write_ms = (time.monotonic() - self._burst_write_t0) * 1000.0
                self._burst_write_t0 = None
            self._td_pos = 0
            self._burst_armed = False

        return n


blk = OfdmBurstTxSourceBlock
