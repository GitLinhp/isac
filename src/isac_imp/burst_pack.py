"""Style 1 UHD 突发打包：tag 约定、时域缓冲准备与 SOB/TIME/EOB 打 tag。

本模块只负责「打包与时间 PMT」，不生成 OFDM 波形。时域载荷通常来自
``isac.transmit_cache.TransmitCache`` 目录下的 ``x_time.npy``（经
``load_burst_buffer`` 读入）。

Style 1 约定：下游 USRP Sink 的 ``len_tag_name`` 须留空，由 stream tag 界定突发边界：

- ``tx_sob``  — 突发开始（Start Of Burst）
- ``tx_time`` — 计划发射时刻（绝对 Unix 时间，整数秒 + 小数秒）
- ``tx_eob``  — 突发结束（End Of Burst）

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

TAG_TX_SOB = pmt.intern("tx_sob")  # type: ignore[attr-defined]
"""UHD stream tag：突发开始（Start Of Burst）。"""

TAG_TX_EOB = pmt.intern("tx_eob")  # type: ignore[attr-defined]
"""UHD stream tag：突发结束（End Of Burst）。"""

TAG_TX_TIME = pmt.intern("tx_time")  # type: ignore[attr-defined]
"""UHD stream tag：计划发射绝对时刻；值为 ``make_tx_time_pmt`` 同形 PMT。"""

TAG_RX_TIME = pmt.intern("rx_time")  # type: ignore[attr-defined]
"""USRP Source stream tag：接收样点时间轴；PMT 形与 ``tx_time`` 相同。"""

TPP_DONT = gr.TPP_DONT  # type: ignore[attr-defined]
"""``gr.TPP_DONT`` 再导出；epy 块 ``__init__`` 中设 ``set_tag_propagation_policy``。"""


def make_tx_time_pmt(epoch_s: float):
    """构造 UHD ``tx_time`` / ``rx_time`` 同形 PMT：``(uint64 秒, double 小数秒)``。

    参数:
    -------
    - epoch_s : float
        Unix 绝对时间（秒，可含小数）

    返回:
    -------
    - pmt
        二元组 PMT，供 ``add_item_tag`` 或消息口使用
    """
    sec = int(epoch_s)
    frac = epoch_s - sec
    return pmt.make_tuple(  # type: ignore[attr-defined]
        pmt.from_uint64(sec),  # type: ignore[attr-defined]
        pmt.from_double(frac),  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# 2. 从 x_time.npy 准备时域突发缓冲
# ---------------------------------------------------------------------------
def load_burst_buffer(x_time_path: Path | str, tx_amp: float = 1.0) -> np.ndarray:
    """从 ``x_time.npy`` 加载时域突发：``np.load`` → squeeze → 可选缩放 → ``complex64`` 一维。

    期望文件内容 squeeze 后为 1D；默认 ``tx_amp=1.0``（不缩放）。实时可调幅度时应按
    默认加载，在写出路径乘当前 amp。

    参数:
    -------
    - x_time_path : Path | str
        时域缓存路径（通常为 ``TransmitCache`` 目录下的 ``x_time.npy``）
    - tx_amp : float
        幅度缩放；``1.0`` 时原样返回

    返回:
    -------
    - np.ndarray
        ``dtype=complex64`` 的一维缓冲

    异常:
    -------
    - FileNotFoundError
        路径不存在或不是文件
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

    周期 = ``max(idle + burst, burst)``；返回值 = 周期 - burst 时长。
    ``samp_rate <= 0`` 时无法换算突发时长，直接回退为钳位后的 ``idle``。

    参数:
    -------
    - burst_len : int
        突发样点数
    - samp_rate : float
        采样率（Hz）
    - idle_s : float
        期望突发间隙（秒）；负值按 ``0`` 处理

    返回:
    -------
    - float
        纯 idle 秒数（非负）
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


def add_style1_sob(block: Any, port: int, abs_offset: int) -> None:
    """在 ``abs_offset`` 仅打 ``tx_sob``（无 ``tx_time``，立即突发发射）。

    参数:
    -------
    - block
        具备 ``add_item_tag`` 的 GNU Radio 块（通常为 ``self``）
    - port : int
        输出口索引
    - abs_offset : int
        绝对样点索引（相对流起点）
    """
    block.add_item_tag(port, abs_offset, TAG_TX_SOB, pmt.PMT_T)


def add_style1_sob_time(
    block: Any,
    port: int,
    abs_offset: int,
    epoch_s: float,
) -> None:
    """在 ``abs_offset`` 打 ``tx_sob`` + ``tx_time``（定时突发）。

    参数:
    -------
    - block
        具备 ``add_item_tag`` 的 GNU Radio 块
    - port : int
        输出口索引
    - abs_offset : int
        绝对样点索引
    - epoch_s : float
        计划发射 Unix 绝对时间（秒），经 ``make_tx_time_pmt`` 写入 tag
    """
    block.add_item_tag(port, abs_offset, TAG_TX_SOB, pmt.PMT_T)
    block.add_item_tag(port, abs_offset, TAG_TX_TIME, make_tx_time_pmt(epoch_s))


def add_style1_eob(block: Any, port: int, abs_offset: int) -> None:
    """在 ``abs_offset`` 打 ``tx_eob``（通常为突发末样点）。

    参数:
    -------
    - block
        具备 ``add_item_tag`` 的 GNU Radio 块
    - port : int
        输出口索引
    - abs_offset : int
        绝对样点索引（一般为 ``abs_out + n - 1``）
    """
    block.add_item_tag(port, abs_offset, TAG_TX_EOB, pmt.PMT_T)
