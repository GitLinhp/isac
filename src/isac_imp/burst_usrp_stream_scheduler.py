"""``tx_schedule`` → UHD ``STREAM_MODE_NUM_SAMPS_AND_DONE`` 按 CPI 定时拉流。

仿 echotimer：不在 GNU Radio 中维持 122 MHz 连续 RX，而是在每个 CPI 计划时刻
向 ``uhd.usrp_source`` 下发有限样点采集命令，从架构上消除 idle 期间 overflow。
"""

from __future__ import annotations

import time

import pmt
from gnuradio import gr, uhd

from isac_imp.burst_pack import PORT_TX_SCHEDULE, parse_uhd_time_pmt


class BurstUsrpStreamSchedulerBlock(gr.basic_block):
    """消息驱动：``tx_schedule`` epoch → ``usrp_source.issue_stream_cmd``。"""

    def __init__(
        self,
        usrp_source: uhd.usrp_source,
        burst_len_samples: int,
        num_delay_samp: int = 0,
        samp_rate: float = 122_880_000.0,
        enable_stats_log: bool = True,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Burst USRP Stream Scheduler",
            in_sig=None,
            out_sig=None,
        )
        self._source = usrp_source
        self._burst_len = int(burst_len_samples)
        self._num_delay_samp = int(num_delay_samp)
        self._samp_rate = float(samp_rate)
        self._enable_stats_log = bool(enable_stats_log)
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )

        self._schedule_port = pmt.intern(PORT_TX_SCHEDULE)
        self.message_port_register_in(self._schedule_port)
        self.set_msg_handler(self._schedule_port, self._handle_tx_schedule)

        self._cmd_count = 0
        self._last_log_mono = 0.0
        self._last_log_cmd = 0

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
    def burst_len_samples(self) -> int:
        return self._burst_len

    @burst_len_samples.setter
    def burst_len_samples(self, value: int) -> None:
        self._burst_len = int(value)

    @property
    def samp_rate(self) -> float:
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value: float) -> None:
        self._samp_rate = float(value)
        self._delay_s = (
            float(self._num_delay_samp) / self._samp_rate if self._samp_rate > 0 else 0.0
        )

    @property
    def cmd_count(self) -> int:
        return self._cmd_count

    def _issue_recv(self, rx_epoch: float) -> None:
        cmd = uhd.stream_cmd(uhd.stream_mode.STREAM_MODE_NUM_SAMPS_AND_DONE)
        cmd.num_samps = self._burst_len
        cmd.stream_now = False
        cmd.time_spec = uhd.time_spec(rx_epoch)
        self._source.issue_stream_cmd(cmd)
        self._cmd_count += 1

    def _handle_tx_schedule(self, msg: pmt.pmt) -> None:
        if pmt.is_pair(msg):
            msg = pmt.cdr(msg)
        if not pmt.is_tuple(msg):
            return
        tx_epoch = parse_uhd_time_pmt(msg)
        self._issue_recv(tx_epoch)
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        if not self._enable_stats_log:
            return
        now = time.monotonic()
        if now - self._last_log_mono < 1.0:
            return
        cmd_delta = self._cmd_count - self._last_log_cmd
        print(
            f"[burst_usrp_stream_scheduler] recv_cmd/s={cmd_delta:.1f} "
            f"total_cmd={self._cmd_count}",
            flush=True,
        )
        self._last_log_mono = now
        self._last_log_cmd = self._cmd_count


blk = BurstUsrpStreamSchedulerBlock
