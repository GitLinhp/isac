"""
GNU Radio 嵌入式 Python 块：Style 1 USRP 突发正弦波源

周期性输出复数正弦突发，并在流上附加 UHD 所需的 stream tag：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

Style 1 约定：下游 USRP Sink 的 len_tag_name 必须留空，由 tag 而非包长 tag 界定突发边界。
"""

from __future__ import annotations

import time

import numpy as np
import pmt
from gnuradio import gr

# UHD / USRP 发射侧标准 stream tag 键名
from isac_imp.burst_pack import TAG_TX_EOB, TAG_TX_SOB, TAG_TX_TIME


class blk(gr.basic_block):
    """
    Style 1 定时突发源：输出带相位连续的复数正弦波。

    工作流程：突发期内持续输出样本并打 tag；突发结束后进入 idle 静默期，
    静默期不产出样本（general_work 返回 0），到期再启动下一突发。
    """

    def __init__(
        self,
        samp_rate=1e6,
        tone_freq=100e3,
        burst_ms=100.0,
        idle_ms=400.0,
        tx_amp=0.3,
        time_lead_s=0.3,
        startup_delay_s=0.2,
    ):
        """
        参数:
            samp_rate:   采样率 (Hz)
            tone_freq:   正弦音调频率 (Hz)
            burst_ms:    单次突发持续时间 (毫秒)
            idle_ms:     两次突发之间的静默间隔 (毫秒)，不含突发本身时长
            tx_amp:      输出幅度，建议 ≤ 1.0 以免 USRP 饱和
            time_lead_s: tx_time 相对当前 wall-clock 的提前量 (秒)，
                         给 USRP 调度器预留缓冲，避免“已过期”时刻
        """
        gr.basic_block.__init__(
            self,
            name="Sine Burst Source",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._samp_rate = float(samp_rate)
        self._tone_freq = float(tone_freq)
        self._burst_ms = float(burst_ms)
        self._idle_ms = float(idle_ms)
        self._tx_amp = float(tx_amp)
        self._time_lead_s = float(time_lead_s)

        # 每样本相位增量；突发样本总数；静默期（秒）
        self._phase_inc = 2.0 * np.pi * self._tone_freq / self._samp_rate
        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

        # 突发状态机：是否正在输出、当前突发内样本索引、下次允许启动的 monotonic 时刻
        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = 0.0

        # 不向上游传播 tag；保证输出缓冲至少容纳两个完整突发，减少调度欠载
        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.set_min_output_buffer(self._burst_len * 2)

    def _recompute_timing(self) -> None:
        """采样率或突发/静默参数变更后，重新计算相位步进、突发长度与输出缓冲。"""
        self._phase_inc = 2.0 * np.pi * self._tone_freq / self._samp_rate
        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self.set_min_output_buffer(self._burst_len * 2)

    @property
    def samp_rate(self):
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value):
        self._samp_rate = float(value)
        self._recompute_timing()

    @property
    def tone_freq(self):
        return self._tone_freq

    @tone_freq.setter
    def tone_freq(self, value):
        self._tone_freq = float(value)
        self._recompute_timing()

    @property
    def burst_ms(self):
        return self._burst_ms

    @burst_ms.setter
    def burst_ms(self, value):
        self._burst_ms = float(value)
        self._recompute_timing()

    @property
    def idle_ms(self):
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        self._idle_ms = float(value)
        self._recompute_timing()

    @property
    def tx_amp(self):
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        self._tx_amp = float(value)

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

    def forecast(self, noutput_items, ninput_items):
        """无输入端口，不向调度器声明输入需求。"""
        del noutput_items, ninput_items
        return []

    def _schedule_delay_s(self) -> float:
        """
        计算当前突发结束后，到下一突发开始前的等待时间（秒）。

        周期 = max(idle + burst, burst)；返回值 = 周期 - burst，即纯 idle 时长。
        """
        burst_s = self._burst_len / self._samp_rate
        period_s = max(self._idle_s + burst_s, burst_s)
        return max(0.0, period_s - burst_s)

    def _tx_time_pmt(self):
        """
        构造 tx_time tag 的 PMT 值：(整数秒, 小数秒)。

        使用 time.time() + time_lead_s，与 UHD timed TX 的绝对时刻格式一致。
        """
        t = time.time() + self._time_lead_s
        sec = int(t)
        frac = t - sec
        return pmt.make_tuple(pmt.from_uint64(sec), pmt.from_double(frac))

    def general_work(self, input_items, output_items):
        """
        主工作函数：按块输出正弦样本，在突发首尾附加 UHD tag。

        静默期内返回 0（不消耗输出缓冲）；突发可分多次 general_work 完成。
        返回值 n 表示本次写入输出端口的有效样本数。
        """
        del input_items
        out = output_items[0]

        # 非突发态：未到调度时刻则暂不输出
        if not self._burst_active:
            if time.monotonic() < self._next_burst_at:
                return 0
            self._burst_active = True
            self._burst_idx = 0

        # 本次最多输出的样本数（受输出缓冲与剩余突发长度限制）
        n_remain = self._burst_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        # 按全局样本索引生成相位，保证跨 general_work 调用相位连续
        idx = np.arange(self._burst_idx, self._burst_idx + n, dtype=np.float64)
        phase = self._phase_inc * idx
        out[:n] = (self._tx_amp * (np.cos(phase) + 1j * np.sin(phase))).astype(
            np.complex64, copy=False
        )

        abs_out = self.nitems_written(0)
        # 突发第一个样本：打 tx_sob 与 tx_time
        if self._burst_idx == 0:
            self.add_item_tag(0, abs_out, TAG_TX_SOB, pmt.PMT_T)
            self.add_item_tag(0, abs_out, TAG_TX_TIME, self._tx_time_pmt())

        self._burst_idx += n
        # 突发最后一个样本：打 tx_eob，进入 idle 并预约下一突发
        if self._burst_idx >= self._burst_len:
            self.add_item_tag(0, abs_out + n - 1, TAG_TX_EOB, pmt.PMT_T)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
