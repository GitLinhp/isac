"""仿 gr-radar ``usrp_echotimer_cc`` 的 RX 时域补偿。

完整 CPI 收齐后做 ``num_delay_samps`` 前移 + 尾补零，并在首样点打 ``packet_len`` / ``rx_time``。
使用 ``basic_block`` 避免 sync_block 1:1 约束下的 tag 丢失。
"""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import TAG_RX_TIME, TPP_DONT


class EchotimerRxCompensatorBlock(gr.basic_block):
    """``packet_len`` 标量 IQ 流 → 延迟补偿后的 CPI 流。"""

    # USRP Source / 上游块单次 work 缓冲上限约 8191；forecast 不得超过此值
    _FORECAST_MAX = 8191
    _MAX_TAIL_PAD = 32

    def __init__(
        self,
        burst_len_samples: int = 10240,
        length_tag_key: str = "packet_len",
        num_delay_samps: int = 0,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Echotimer RX Compensator",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._burst_len = int(burst_len_samples)
        self._num_delay = max(0, int(num_delay_samps))
        self._length_tag_key = pmt.intern(length_tag_key)
        self._srcid = pmt.intern("usrp_sink_source_echotimer")
        self.set_tag_propagation_policy(TPP_DONT)

        self._cpi_buf: np.ndarray | None = None
        self._cpi_pos = 0
        self._cpi_len = 0
        self._cpi_rx_time = None
        self._pending_emit: np.ndarray | None = None
        self._pending_emit_pos = 0
        self._pending_rx_time = None

        self.set_min_output_buffer(self._burst_len)

    @property
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len = int(value)

    @property
    def num_delay_samps(self) -> int:
        return self._num_delay

    @num_delay_samps.setter
    def num_delay_samps(self, value: int) -> None:
        self._num_delay = max(0, int(value))

    def start(self) -> bool:
        self._cpi_buf = None
        self._cpi_pos = 0
        self._cpi_len = 0
        self._cpi_rx_time = None
        self._pending_emit = None
        self._pending_emit_pos = 0
        self._pending_rx_time = None
        return True

    def _tag_to_int(self, value: pmt.pmt) -> int:
        try:
            return int(pmt.to_long(value))
        except Exception:
            return int(pmt.to_python(value))

    def _apply_shift(self, buf: np.ndarray, burst_len: int) -> None:
        delay = min(self._num_delay, burst_len)
        if delay <= 0:
            return
        tail = burst_len - delay
        if tail > 0:
            buf[:tail] = buf[delay:burst_len]
        if delay > 0:
            buf[tail:burst_len] = 0 + 0j

    def _begin_cpi(self, burst_len: int, rx_time_val) -> None:
        self._cpi_len = int(burst_len)
        self._cpi_buf = np.zeros(self._cpi_len, dtype=np.complex64)
        self._cpi_pos = 0
        self._cpi_rx_time = rx_time_val

    def _reset_cpi(self) -> None:
        self._cpi_buf = None
        self._cpi_pos = 0
        self._cpi_len = 0
        self._cpi_rx_time = None

    def _cpi_start_at(self, read_pos: int) -> tuple[int, object | None] | None:
        for tag in self.get_tags_in_range(0, read_pos, read_pos + 1):
            if not pmt.eq(tag.key, self._length_tag_key):
                continue
            burst_len = self._tag_to_int(tag.value)
            rx_time_val = None
            for rt in self.get_tags_in_range(0, read_pos, read_pos + 1):
                if pmt.eq(rt.key, TAG_RX_TIME):
                    rx_time_val = rt.value
                    break
            return burst_len, rx_time_val
        return None

    def _finalize_cpi(self) -> None:
        if self._cpi_buf is None or self._cpi_pos < self._cpi_len:
            return
        self._apply_shift(self._cpi_buf, self._cpi_len)
        self._pending_emit = self._cpi_buf
        self._pending_emit_pos = 0
        self._pending_rx_time = self._cpi_rx_time
        self._reset_cpi()

    def _flush_pending_emit(
        self,
        out: np.ndarray,
        out_produced: int,
        write_base: int,
    ) -> int:
        if self._pending_emit is None:
            return out_produced

        buf = self._pending_emit
        pos = self._pending_emit_pos
        space = len(out) - out_produced
        if space <= 0:
            return out_produced

        n_copy = min(len(buf) - pos, space)
        out[out_produced : out_produced + n_copy] = buf[pos : pos + n_copy]
        abs_out = write_base + out_produced
        if pos == 0:
            self.add_item_tag(
                0,
                abs_out,
                self._length_tag_key,
                pmt.from_long(len(buf)),
                self._srcid,
            )
            if self._pending_rx_time is not None:
                self.add_item_tag(
                    0, abs_out, TAG_RX_TIME, self._pending_rx_time, self._srcid
                )

        self._pending_emit_pos += n_copy
        if self._pending_emit_pos >= len(buf):
            self._pending_emit = None
            self._pending_emit_pos = 0
            self._pending_rx_time = None

        return out_produced + n_copy

    def _emit_cpi(
        self,
        out: np.ndarray,
        out_produced: int,
        write_base: int,
    ) -> int:
        self._finalize_cpi()
        return self._flush_pending_emit(out, out_produced, write_base)

    def forecast(self, noutput_items: int, ninputs) -> list:
        del noutput_items, ninputs
        if self._cpi_buf is not None:
            need = self._cpi_len - self._cpi_pos
        else:
            need = 1
        return [min(max(1, need), self._FORECAST_MAX), 0]

    def general_work(self, input_items, output_items) -> int:
        inp = input_items[0]
        out = output_items[0]
        if len(out) <= 0:
            return 0

        read_base = self.nitems_read(0)
        write_base = self.nitems_written(0)
        in_consumed = 0
        out_produced = self._flush_pending_emit(out, 0, write_base)

        while in_consumed < len(inp):
            if self._pending_emit is not None:
                out_produced = self._flush_pending_emit(out, out_produced, write_base)
                if self._pending_emit is not None:
                    break

            if self._cpi_buf is None:
                found = self._cpi_start_at(read_base + in_consumed)
                if found is None:
                    in_consumed += 1
                    continue
                burst_len, rx_time_val = found
                if burst_len <= 0:
                    in_consumed += 1
                    continue
                self._begin_cpi(burst_len, rx_time_val)

            self._cpi_buf[self._cpi_pos] = inp[in_consumed]
            self._cpi_pos += 1
            in_consumed += 1

            if self._cpi_pos < self._cpi_len:
                continue

            out_produced = self._emit_cpi(out, out_produced, write_base)
            if self._pending_emit is not None and out_produced >= len(out):
                break

        # 上游已空且 CPI 差少量样点：零填充以避免 basic_block 与 sync 链死锁
        if self._cpi_buf is not None and in_consumed == len(inp):
            shortfall = self._cpi_len - self._cpi_pos
            if 0 < shortfall <= self._MAX_TAIL_PAD:
                self._cpi_buf[self._cpi_pos : self._cpi_len] = 0
                self._cpi_pos = self._cpi_len
                out_produced = self._emit_cpi(out, out_produced, write_base)

        if in_consumed > 0:
            self.consume(0, in_consumed)
        if out_produced > 0:
            return out_produced
        if in_consumed > 0:
            return gr.WORK_CALLED_PRODUCE
        return 0


blk = EchotimerRxCompensatorBlock
