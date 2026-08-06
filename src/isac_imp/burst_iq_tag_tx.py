"""Style1 UHD tag 转换：GR ``packet_len`` 时域流 → ``tx_sob`` / ``tx_time`` / ``tx_eob``。

上游 ``digital_ofdm_cyclic_prefixer`` 在 CPI 首样点打 ``packet_len=burst_len_samples``；
本块 1:1 透传样点，并在 CPI 边界叠加 UHD Style1 stream tag，供 ``uhd.usrp_sink``
（``len_tag_name`` 留空）使用。每个 CPI 经 ``tx_schedule`` 消息口发布计划 epoch 供 RX 对齐。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pmt
from gnuradio import gr

from isac_imp.burst_pack import (
    PORT_TX_SCHEDULE,
    TAG_TX_EOB,
    TPP_DONT,
    add_style1_eob,
    add_style1_sob_time,
    make_tx_schedule_msg,
    schedule_idle_delay_s,
)


class BurstIqTagTxBlock(gr.sync_block):
    """``packet_len`` 标量 IQ 流 → 相同样本 + Style1 UHD 突发 tag + ``tx_schedule`` 消息。"""

    def __init__(
        self,
        burst_len_samples: int = 10240,
        length_tag_key: str = "packet_len",
        time_lead_s: float = 0.03,
        idle_ms: float = 0.0,
        samp_rate: float = 122_880_000.0,
        scheduled_rx: bool = False,
        num_delay_samp: int = 0,
    ) -> None:
        self._burst_len = int(burst_len_samples)
        self._time_lead_s = float(time_lead_s)
        self._idle_ms = float(idle_ms)
        self._samp_rate = float(samp_rate)
        self._scheduled_rx = bool(scheduled_rx)
        self._num_delay_samp = int(num_delay_samp)
        self._usrp_source = None
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )
        self._idle_s = schedule_idle_delay_s(
            self._burst_len, self._samp_rate, self._idle_ms / 1000.0
        )

        gr.sync_block.__init__(
            self,
            name="Burst IQ Tag TX",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._length_tag_key = pmt.intern(length_tag_key)
        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_out(self._schedule_port)

        self._sym_idx = 0
        self._active_burst_len = self._burst_len
        self._idle_until = 0.0
        self._next_tx_epoch = 0.0
        self._burst_period_s = 0.0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._burst_len * 2)
        self._recompute_burst_period()

    def _recompute_burst_period(self) -> None:
        burst_s = float(self._burst_len) / float(self._samp_rate) if self._samp_rate > 0 else 0.0
        self._burst_period_s = burst_s + self._idle_s

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
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len = int(value)
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
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )

    @property
    def scheduled_rx(self) -> bool:
        return self._scheduled_rx

    @scheduled_rx.setter
    def scheduled_rx(self, value: bool) -> None:
        self._scheduled_rx = bool(value)

    def bind_scheduled_rx(self, usrp_source: Any) -> None:
        """流图 ``main`` / Snippet 中绑定 USRP Source。"""
        from isac_imp.scheduled_rx_registry import register_usrp_source

        self._usrp_source = usrp_source
        self._scheduled_rx = True
        register_usrp_source(usrp_source)

    def start(self) -> bool:
        self._sym_idx = 0
        self._active_burst_len = self._burst_len
        self._idle_until = 0.0
        self._next_tx_epoch = time.time() + self._time_lead_s
        self._recompute_burst_period()
        if self._scheduled_rx and self._usrp_source is None:
            from isac_imp.scheduled_rx_registry import get_usrp_source

            self._usrp_source = get_usrp_source()
        return True

    def _issue_scheduled_recv(self, tx_epoch: float) -> None:
        if not self._scheduled_rx or self._usrp_source is None:
            return
        from gnuradio import uhd

        cmd = uhd.stream_cmd(uhd.stream_mode.STREAM_MODE_NUM_SAMPS_AND_DONE)
        cmd.num_samps = self._burst_len
        cmd.stream_now = False
        # 与 echotimer 一致：RX 调度时刻 = TX epoch（wait_to_start 已在 epoch 中）
        # num_delay_samp 由 echotimer_rx_compensator 在样点域补偿
        cmd.time_spec = uhd.time_spec(tx_epoch)
        self._usrp_source.issue_stream_cmd(cmd)

    def _tag_to_int(self, value: pmt.pmt) -> int:
        try:
            return int(pmt.to_long(value))
        except Exception:
            return int(pmt.to_python(value))

    def _begin_burst(self, out_abs: int) -> None:
        min_epoch = time.time() + self._time_lead_s
        if self._next_tx_epoch < min_epoch:
            self._next_tx_epoch = min_epoch
        epoch = self._next_tx_epoch
        add_style1_sob_time(self, 0, out_abs, epoch)
        self.message_port_pub(self._schedule_port, make_tx_schedule_msg(epoch))
        self._issue_scheduled_recv(epoch)
        self._next_tx_epoch += self._burst_period_s

    def work(self, input_items, output_items) -> int:
        if self._idle_s > 0.0 and time.monotonic() < self._idle_until:
            return 0

        inp = input_items[0]
        out = output_items[0]
        n = min(len(inp), len(out))
        if n <= 0:
            return 0

        in_base = self.nitems_read(0)
        out_base = self.nitems_written(0)

        for i in range(n):
            for tag in self.get_tags_in_range(0, in_base + i, in_base + i + 1):
                if pmt.eq(tag.key, self._length_tag_key):
                    burst_len = self._tag_to_int(tag.value)
                    if burst_len > 0:
                        self._active_burst_len = burst_len
                    self._sym_idx = 0
                    self._begin_burst(out_base + i)

            out[i] = inp[i]

            if self._sym_idx == self._active_burst_len - 1:
                add_style1_eob(self, 0, out_base + i)

            self._sym_idx += 1
            if self._sym_idx >= self._active_burst_len:
                self._sym_idx = 0
                if self._idle_s > 0.0:
                    self._idle_until = time.monotonic() + self._idle_s
                    return i + 1

        return n


blk = BurstIqTagTxBlock
