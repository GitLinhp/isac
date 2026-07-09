"""Style 1 UHD 突发打包：tag 约定、时域缓冲准备与 SOB/TIME/EOB 打 tag。

Style 1 约定：下游 USRP Sink 的 ``len_tag_name`` 须留空，由 stream tag 界定突发边界：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

同机感知 RX 另用消息口 ``tx_schedule`` 传递与 ``tx_time`` 同形的 epoch；
USRP Source 侧常见 ``rx_time``（同形 PMT）用于样点时间轴。

下文按功能分区：常量与时间 PMT → 缓冲准备 → 调度 → 打 tag。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pmt
from gnuradio import gr

# ---------------------------------------------------------------------------
# 1. Style1 / RX 时间常量与 PMT
# ---------------------------------------------------------------------------

TAG_SOB = pmt.intern("tx_sob")  # type: ignore[attr-defined]
TAG_EOB = pmt.intern("tx_eob")  # type: ignore[attr-defined]
TAG_TIME = pmt.intern("tx_time")  # type: ignore[attr-defined]
TAG_RX_TIME = pmt.intern("rx_time")  # type: ignore[attr-defined]
PORT_TX_SCHEDULE = "tx_schedule"  # TX→RX 消息口：每突发计划 epoch
TPP_DONT = gr.TPP_DONT  # type: ignore[attr-defined]


def make_tx_time_pmt(epoch_s: float):
    """构造 UHD tx_time / rx_time 同形 PMT：(整数秒, 小数秒)。"""
    sec = int(epoch_s)
    frac = epoch_s - sec
    return pmt.make_tuple(  # type: ignore[attr-defined]
        pmt.from_uint64(sec),  # type: ignore[attr-defined]
        pmt.from_double(frac),  # type: ignore[attr-defined]
    )


def parse_uhd_time_pmt(value) -> float:
    """解析 UHD 时间 PMT ``(uint64 秒, double 小数秒)`` → epoch 秒。"""
    sec = float(pmt.to_uint64(pmt.tuple_ref(value, 0)))  # type: ignore[attr-defined]
    frac = float(pmt.to_double(pmt.tuple_ref(value, 1)))  # type: ignore[attr-defined]
    return sec + frac


def make_tx_schedule_msg(epoch_s: float):
    """构造 ``tx_schedule`` 消息载荷（与 ``tx_time`` 同形）。"""
    return make_tx_time_pmt(epoch_s)


# ---------------------------------------------------------------------------
# 2. 突发缓冲准备
# ---------------------------------------------------------------------------


def load_burst_buffer(x_time_path: Path | str, tx_amp: float = 1.0) -> np.ndarray:
    """``np.load`` → squeeze → 可选 ``* tx_amp`` → ``complex64`` 一维缓冲。

    默认 ``tx_amp=1.0``（不缩放）。实时可调幅度时应按默认加载，在写出路径乘当前 amp。
    """
    path = Path(x_time_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"发射时域缓存不存在: {path}；请先离线运行 transmit() 生成"
        )
    x_time = np.asarray(np.load(path))
    burst = x_time.squeeze().astype(np.complex64, copy=False)
    amp = float(tx_amp)
    if amp == 1.0:
        return burst
    return (burst * amp).astype(np.complex64, copy=False)


# ---------------------------------------------------------------------------
# 3. 突发间 idle 调度
# ---------------------------------------------------------------------------


def schedule_idle_delay_s(burst_len: int, samp_rate: float, idle_s: float) -> float:
    """计算突发结束后到下一突发开始前的纯 idle 时长（秒）。

    周期 = max(idle + burst, burst)；返回值 = 周期 - burst。
    """
    idle = max(0.0, float(idle_s))
    if samp_rate <= 0:
        return idle
    burst_s = float(burst_len) / float(samp_rate)
    period_s = max(idle + burst_s, burst_s)
    return max(0.0, period_s - burst_s)


# ---------------------------------------------------------------------------
# 4. Style1 打 tag
# ---------------------------------------------------------------------------


def add_style1_sob_time(
    block: Any,
    port: int,
    abs_offset: int,
    epoch_s: float,
) -> None:
    """在 ``abs_offset`` 打 ``tx_sob`` + ``tx_time``。"""
    block.add_item_tag(port, abs_offset, TAG_SOB, pmt.PMT_T)
    block.add_item_tag(port, abs_offset, TAG_TIME, make_tx_time_pmt(epoch_s))


def add_style1_eob(block: Any, port: int, abs_offset: int) -> None:
    """在 ``abs_offset`` 打 ``tx_eob``。"""
    block.add_item_tag(port, abs_offset, TAG_EOB, pmt.PMT_T)
