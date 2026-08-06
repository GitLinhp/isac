"""USRP Source 连续 IQ 流 → ``packet_len`` 切包（``tx_schedule`` + ``rx_time`` 对齐）。

输入为 ``uhd.usrp_source`` 标量 complex64 流（含 ``rx_time`` tag）；
``tx_schedule`` 消息口接收 TX 计划 epoch，按 ``num_delay_samp`` 补偿后在
对应时刻切出 ``burst_len_samples`` 样点并打 ``packet_len`` tag。

非 collect 阶段对距下一 CPI 较远的区间做批量丢弃，避免 122 MHz 下逐样点 Python 循环。
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import (
    PORT_TX_SCHEDULE,
    TAG_RX_TIME,
    TPP_DONT,
    parse_uhd_time_pmt,
    schedule_idle_delay_s,
)


class BurstIqTagRxBlock(gr.basic_block):
    """``tx_schedule`` + ``rx_time`` → ``packet_len`` 标量 CPI 流。"""

    # USRP Source 单次输出缓冲上限约 8191；forecast 不得超过此值
    _FORECAST_MAX = 8191
    _MAX_SAMPLES_PER_WORK = 8191
    _BULK_DISCARD_MARGIN = 64
    _STATS_LOG_INTERVAL_S = 1.0

    def __init__(
        self,
        burst_len_samples: int = 10240,
        num_delay_samp: int = 0,
        length_tag_key: str = "packet_len",
        samp_rate: float = 122_880_000.0,
        idle_ms: float = 0.0,
        enable_stats_log: bool = True,
    ) -> None:
        self._burst_len = int(burst_len_samples)
        self._num_delay_samp = int(num_delay_samp)
        self._samp_rate = float(samp_rate)
        self._idle_ms = float(idle_ms)
        self._enable_stats_log = bool(enable_stats_log)
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )

        gr.basic_block.__init__(
            self,
            name="Burst IQ Tag RX",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_in(self._schedule_port)
        self.set_msg_handler(self._schedule_port, self._handle_tx_schedule)

        self._pending_epochs: deque[float] = deque()
        self._collecting = False
        self._out_queue: deque[np.ndarray] = deque()

        self._last_tag_epoch: float | None = None
        self._last_tag_abs: int | None = None
        self._collect_arr: np.ndarray | None = None
        self._collect_pos = 0

        self._cpi_count = 0
        self._stats_discarded = 0
        self._stats_bulk_discards = 0
        self._last_log_mono = 0.0
        self._last_log_cpi = 0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._burst_len * 2)
        self._recompute_timing()

    def _recompute_timing(self) -> None:
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )
        idle_s = schedule_idle_delay_s(
            self._burst_len, self._samp_rate, self._idle_ms / 1000.0
        )
        burst_s = float(self._burst_len) / self._samp_rate if self._samp_rate > 0 else 0.0
        self._burst_period_s = burst_s + idle_s

    @property
    def idle_ms(self) -> float:
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value: float) -> None:
        self._idle_ms = float(value)
        self._recompute_timing()

    @property
    def num_delay_samp(self) -> int:
        return self._num_delay_samp

    @num_delay_samp.setter
    def num_delay_samp(self, value: int) -> None:
        self._num_delay_samp = int(value)
        self._recompute_timing()

    @property
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len = int(value)
        self._recompute_timing()

    @property
    def samp_rate(self) -> float:
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value: float) -> None:
        self._samp_rate = float(value)
        self._recompute_timing()

    @property
    def cpi_count(self) -> int:
        return self._cpi_count

    def start(self) -> bool:
        self._pending_epochs.clear()
        self._collecting = False
        self._out_queue.clear()
        self._last_tag_epoch = None
        self._last_tag_abs = None
        self._collect_arr = np.zeros(self._burst_len, dtype=np.complex64)
        self._collect_pos = 0
        self._cpi_count = 0
        self._stats_discarded = 0
        self._stats_bulk_discards = 0
        self._last_log_mono = 0.0
        self._last_log_cpi = 0
        return True

    def _handle_tx_schedule(self, msg: pmt.pmt) -> None:
        if pmt.is_pair(msg):
            msg = pmt.cdr(msg)
        if not pmt.is_tuple(msg):
            return
        tx_epoch = parse_uhd_time_pmt(msg)
        self._pending_epochs.append(tx_epoch + self._delay_s)

    def _update_rx_time_tags(self, abs_start: int, abs_end: int) -> None:
        if abs_end <= abs_start:
            return
        for tag in self.get_tags_in_range(0, abs_start, abs_end):
            if pmt.eq(tag.key, TAG_RX_TIME):
                self._last_tag_epoch = parse_uhd_time_pmt(tag.value)
                self._last_tag_abs = tag.offset

    def _sample_epoch(self, abs_offset: int) -> float | None:
        if self._last_tag_epoch is None or self._last_tag_abs is None:
            return None
        if self._samp_rate <= 0:
            return self._last_tag_epoch
        delta = abs_offset - self._last_tag_abs
        return self._last_tag_epoch + float(delta) / self._samp_rate

    def _cpi_start_abs(self) -> int | None:
        if not self._pending_epochs:
            return None
        if (
            self._last_tag_epoch is None
            or self._last_tag_abs is None
            or self._samp_rate <= 0
        ):
            return None
        delta_s = self._pending_epochs[0] - self._last_tag_epoch
        return self._last_tag_abs + int(round(delta_s * self._samp_rate))

    def _samples_until_cpi(self, abs_offset: int) -> int | None:
        cpi_abs = self._cpi_start_abs()
        if cpi_abs is None:
            return None
        return cpi_abs - abs_offset

    def _drop_stale_epochs(self, sample_epoch: float | None) -> None:
        if sample_epoch is None or not self._pending_epochs:
            return
        while len(self._pending_epochs) > 1:
            if sample_epoch > self._pending_epochs[0] + self._burst_period_s:
                self._pending_epochs.popleft()
            else:
                break
        if sample_epoch > self._pending_epochs[0] + self._burst_period_s:
            self._pending_epochs[0] = sample_epoch

    def _finish_burst(self) -> None:
        if self._collect_pos >= self._burst_len and self._collect_arr is not None:
            self._out_queue.append(self._collect_arr.copy())
            self._cpi_count += 1
        self._collect_pos = 0
        self._collecting = False

    def _emit_from_queue(
        self,
        out: np.ndarray,
        out_base: int,
        produced: int,
    ) -> int:
        while produced < len(out) and self._out_queue:
            burst = self._out_queue.popleft()
            n_copy = min(len(burst), len(out) - produced)
            out[produced : produced + n_copy] = burst[:n_copy]
            self.add_item_tag(
                0,
                out_base + produced,
                self._length_tag_key,
                pmt.from_long(self._burst_len),
            )
            produced += n_copy
            if n_copy < len(burst):
                self._out_queue.appendleft(burst[n_copy:])
                break
        return produced

    def _maybe_log_stats(self) -> None:
        if not self._enable_stats_log:
            return
        now = time.monotonic()
        if now - self._last_log_mono < self._STATS_LOG_INTERVAL_S:
            return
        cpi_delta = self._cpi_count - self._last_log_cpi
        print(
            f"[burst_iq_tag_rx] CPI/s={cpi_delta:.1f} "
            f"total_cpi={self._cpi_count} "
            f"discarded={self._stats_discarded} "
            f"bulk_ops={self._stats_bulk_discards}",
            flush=True,
        )
        self._last_log_mono = now
        self._last_log_cpi = self._cpi_count

    def forecast(self, noutput_items: int, ninputs: list) -> list:
        del noutput_items
        if self._collecting:
            need = self._burst_len - self._collect_pos
        else:
            need = self._FORECAST_MAX
        cap = min(self._FORECAST_MAX, self._MAX_SAMPLES_PER_WORK)
        return [min(max(1, need), cap)]

    def general_work(self, input_items, output_items) -> int:
        inp = input_items[0]
        out = output_items[0]
        in_base = self.nitems_read(0)
        out_base = self.nitems_written(0)
        consumed = 0
        produced = self._emit_from_queue(out, out_base, 0)

        while consumed < len(inp) and consumed < self._MAX_SAMPLES_PER_WORK:
            budget = min(
                len(inp) - consumed,
                self._MAX_SAMPLES_PER_WORK - consumed,
            )
            abs_in = in_base + consumed

            if self._collecting:
                need = self._burst_len - self._collect_pos
                n_take = min(need, budget)
                if self._collect_arr is None:
                    self._collect_arr = np.zeros(self._burst_len, dtype=np.complex64)
                self._collect_arr[self._collect_pos : self._collect_pos + n_take] = inp[
                    consumed : consumed + n_take
                ]
                self._collect_pos += n_take
                consumed += n_take
                if self._collect_pos >= self._burst_len:
                    self._finish_burst()
                    produced = self._emit_from_queue(out, out_base, produced)
                continue

            self._update_rx_time_tags(abs_in, abs_in + 1)
            sample_epoch = self._sample_epoch(abs_in)
            self._drop_stale_epochs(sample_epoch)

            cpi_abs = self._cpi_start_abs()
            if cpi_abs is not None and abs_in >= cpi_abs:
                self._pending_epochs.popleft()
                self._collecting = True
                self._collect_pos = 0
                continue

            if self._pending_epochs:
                until = self._samples_until_cpi(abs_in)
                if until is not None and until > self._BULK_DISCARD_MARGIN:
                    chunk = min(until - self._BULK_DISCARD_MARGIN, budget)
                    chunk = max(1, chunk)
                    self._update_rx_time_tags(abs_in, abs_in + chunk)
                    end_epoch = self._sample_epoch(abs_in + chunk - 1)
                    self._drop_stale_epochs(end_epoch)
                    self._stats_discarded += chunk
                    self._stats_bulk_discards += 1
                    consumed += chunk
                    continue

            if not self._pending_epochs and budget > 1:
                chunk = budget
                self._update_rx_time_tags(abs_in, abs_in + chunk)
                self._stats_discarded += chunk
                self._stats_bulk_discards += 1
                consumed += chunk
                continue

            consumed += 1
            self._stats_discarded += 1

        if consumed:
            self.consume(0, consumed)
        self._maybe_log_stats()
        if produced:
            return produced
        if consumed:
            return gr.WORK_CALLED_PRODUCE
        return 0


blk = BurstIqTagRxBlock
