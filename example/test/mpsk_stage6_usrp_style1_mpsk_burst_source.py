"""
GNU Radio 嵌入式 Python 块：Style 1 USRP QPSK 突发源

周期性输出差分 QPSK（RRC 成形）突发，并在流上附加 UHD stream tag：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

Style 1 约定：下游 USRP Sink 的 len_tag_name 必须留空，由 tag 而非包长 tag 界定突发边界。

载荷在参数变更时重新生成并缓存，突发期内按块重放；idle 期 general_work 返回 0。
调制参数对齐原 mpsk_stage6 连续版：sps=4、excess_bw=0.35、差分 QPSK。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from gnuradio import gr
from gnuradio.filter import firdes

try:
    _repo_root = Path(__file__).resolve().parents[2]
except NameError:
    _repo_root = Path.cwd().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from isac_imp.burst_pack import (
    TPP_DONT,
    add_style1_eob,
    add_style1_sob_time,
    schedule_idle_delay_s,
)


def _make_qpsk_burst(
    n_samples: int,
    sps: int,
    excess_bw: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成一段差分 QPSK + RRC 时域波形（未缩放），长度精确为 n_samples。"""
    sps = max(1, int(sps))
    n_syms = max(8, (n_samples + sps - 1) // sps + 16)
    # Gray 映射的差分相位步进：0, π/2, π, 3π/2
    dibits = rng.integers(0, 4, size=n_syms, dtype=np.int64)
    phase = np.cumsum(dibits.astype(np.float64) * (np.pi / 2.0))
    symbols = np.exp(1j * phase).astype(np.complex64)

    up = np.zeros(n_syms * sps, dtype=np.complex64)
    up[::sps] = symbols

    ntaps = 11 * sps
    taps = firdes.root_raised_cosine(
        1.0,  # gain
        float(sps),  # sampling rate (normalized to symbol rate = 1)
        1.0,  # symbol rate
        float(excess_bw),
        ntaps,
    )
    shaped = np.convolve(up, np.asarray(taps, dtype=np.complex64), mode="full")
    # 去掉滤波器群时延，取中段
    delay = (len(taps) - 1) // 2
    start = delay
    end = start + n_samples
    if end > len(shaped):
        pad = end - len(shaped)
        shaped = np.concatenate(
            [shaped, np.zeros(pad, dtype=np.complex64)]
        )
    burst = shaped[start:end].astype(np.complex64, copy=False)
    # 峰值归一化，便于后续 tx_amp 控制
    peak = float(np.max(np.abs(burst))) if burst.size else 1.0
    if peak > 0:
        burst = (burst / peak).astype(np.complex64, copy=False)
    return burst


class blk(gr.basic_block):
    """
    Style 1 QPSK 突发源：缓存差分 QPSK/RRC 波形，周期性重放并打 SOB/TIME/EOB。

    工作流程：突发期内从缓存输出样本并打 tag；突发结束后进入 idle 静默期，
    静默期不产出样本（general_work 返回 0），到期再启动下一突发。
    """

    def __init__(
        self,
        samp_rate=320000.0,
        burst_ms=100.0,
        idle_ms=400.0,
        tx_amp=0.3,
        sps=4,
        excess_bw=0.35,
        time_lead_s=0.3,
        startup_delay_s=0.2,
    ):
        """
        参数:
            samp_rate:       采样率 (Hz)
            burst_ms:        单次突发持续时间 (毫秒)
            idle_ms:         两次突发之间的静默间隔 (毫秒)，不含突发本身时长
            tx_amp:          输出幅度，建议 ≤ 1.0 以免 USRP 饱和
            sps:             每符号样本数（与 RX PFB 一致）
            excess_bw:       RRC 滚降系数
            time_lead_s:     tx_time 相对当前 wall-clock 的提前量 (秒)
            startup_delay_s: 首突发启动前的初始等待 (秒)
        """
        gr.basic_block.__init__(
            self,
            name="MPSK Burst Source",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._samp_rate = float(samp_rate)
        self._burst_ms = float(burst_ms)
        self._idle_ms = float(idle_ms)
        self._tx_amp = float(tx_amp)
        self._sps = max(1, int(sps))
        self._excess_bw = float(excess_bw)
        self._time_lead_s = float(time_lead_s)
        self._startup_delay_s = float(startup_delay_s)
        self._rng = np.random.default_rng(0)

        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self._burst_buf = _make_qpsk_burst(
            self._burst_len, self._sps, self._excess_bw, self._rng
        )

        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = time.monotonic() + self._startup_delay_s

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(self._burst_len * 2)

    def _rebuild_burst(self) -> None:
        """采样率/突发时长/成形参数变更后重建缓存与输出缓冲。"""
        self._burst_len = max(1, int(self._samp_rate * self._burst_ms / 1000.0))
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self._burst_buf = _make_qpsk_burst(
            self._burst_len, self._sps, self._excess_bw, self._rng
        )
        self.set_min_output_buffer(self._burst_len * 2)

    @property
    def samp_rate(self):
        return self._samp_rate

    @samp_rate.setter
    def samp_rate(self, value):
        self._samp_rate = float(value)
        self._rebuild_burst()

    @property
    def burst_ms(self):
        return self._burst_ms

    @burst_ms.setter
    def burst_ms(self, value):
        self._burst_ms = float(value)
        self._rebuild_burst()

    @property
    def idle_ms(self):
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        self._idle_ms = float(value)
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

    @property
    def tx_amp(self):
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        self._tx_amp = float(value)

    @property
    def sps(self):
        return self._sps

    @sps.setter
    def sps(self, value):
        self._sps = max(1, int(value))
        self._rebuild_burst()

    @property
    def excess_bw(self):
        return self._excess_bw

    @excess_bw.setter
    def excess_bw(self, value):
        self._excess_bw = float(value)
        self._rebuild_burst()

    @property
    def time_lead_s(self):
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        self._time_lead_s = float(value)

    @property
    def startup_delay_s(self):
        return self._startup_delay_s

    @startup_delay_s.setter
    def startup_delay_s(self, value):
        self._startup_delay_s = float(value)

    def forecast(self, noutput_items, ninput_items):
        """无输入端口，不向调度器声明输入需求。"""
        del noutput_items, ninput_items
        return []

    def _schedule_delay_s(self) -> float:
        """当前突发结束后到下一突发开始前的纯 idle 时长（秒）。"""
        return schedule_idle_delay_s(self._burst_len, self._samp_rate, self._idle_s)

    def general_work(self, input_items, output_items):
        """
        主工作函数：按块从缓存重放 QPSK 突发，在突发首尾附加 UHD tag。

        静默期内返回 0（不消耗输出缓冲）；突发可分多次 general_work 完成。
        """
        del input_items
        out = output_items[0]

        if not self._burst_active:
            if time.monotonic() < self._next_burst_at:
                return 0
            self._burst_active = True
            self._burst_idx = 0

        n_remain = self._burst_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        amp = self._tx_amp
        sl = self._burst_buf[self._burst_idx : self._burst_idx + n]
        out[:n] = (sl * amp).astype(np.complex64, copy=False)

        abs_out = self.nitems_written(0)
        if self._burst_idx == 0:
            epoch_s = time.time() + self._time_lead_s
            add_style1_sob_time(self, 0, abs_out, epoch_s)

        self._burst_idx += n
        if self._burst_idx >= self._burst_len:
            add_style1_eob(self, 0, abs_out + n - 1)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
