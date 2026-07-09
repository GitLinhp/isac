"""
GNU Radio 嵌入式 Python 块：OFDM 突发源（SC 前导 + x_time.npy 时域重放）

周期性输出 OFDM 时域突发，并在流上附加 UHD 所需的 stream tag：
  tx_sob  — 突发开始（Start Of Burst）
  tx_time — 计划发射时刻（绝对 Unix 时间，秒 + 小数部分）
  tx_eob  — 突发结束（End Of Burst）

Style 1 约定：下游 USRP Sink 的 len_tag_name 必须留空，由 tag 而非包长 tag 界定突发边界。

发射链：由 config_file 解析 source.cache_file（缓存目录）与 OFDM 几何，
加载 x_time.npy（载荷，未缩放）→ 前缀拼接 Schmidl-Cox 时域前导 → 写出时乘
当前 tx_amp 周期性重放。前导运行时生成，无需重写缓存。
空口同步由 RX 对前导做相关完成；本块不再发 tx_schedule 消息。

注意：__init__ 形参默认值须与 GRC 变量保持同步。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from gnuradio import gr

from isac import PROJECT_ROOT
from isac_imp.burst_pack import (
    TPP_DONT,
    add_style1_eob,
    add_style1_sob_time,
    load_burst_buffer,
    schedule_idle_delay_s,
)
from isac_imp.gr_setup import resolve_ofdm_fft_cp, resolve_ofdm_samp_rate, resolve_source_cache_file
from isac_imp.ofdm_sc_preamble import preamble_time

# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------

# 默认 TOML（相对 PROJECT_ROOT/config）；须与 GRC 默认值一致
_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_LOG_PREFIX = "[OFDM Burst Source]"  # stderr 日志前缀


class blk(gr.basic_block):
    """
    Style 1 OFDM 突发源：SC 前导 + x_time.npy，周期性重放并打 UHD 定时 tag。

    工作流程：突发期内从缓存输出样本并打 tag；突发结束后进入 idle 静默期，
    静默期不产出样本（general_work 返回 0），到期再启动下一突发。
    """

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        idle_ms=900.0,
        tx_amp=0.3,
        time_lead_s=0.3,
        startup_delay_s=0.2,
    ):
        """
        参数:
            - config_file:      TOML 路径；从中解析 source.cache_file（目录）与 OFDM 几何
            - idle_ms:          两次突发之间的纯静默间隔 (毫秒)，不含突发本身时长
            - tx_amp:           输出幅度缩放（写出时相乘，可实时调节）
            - time_lead_s:      tx_time 相对当前 wall-clock 的提前量 (秒)
            - startup_delay_s:  首突发启动前的初始等待 (秒)
        """
        gr.basic_block.__init__(
            self,
            name="OFDM Burst Source",
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._config_file = str(config_file)
        self._cache_dir, self._samp_rate, self._fft_size, self._cp_len = (
            self._resolve_from_config(self._config_file)
        )
        self._idle_ms = float(idle_ms)
        self._tx_amp = float(tx_amp)
        self._time_lead_s = float(time_lead_s)
        self._startup_delay_s = float(startup_delay_s)

        # 突发状态机：是否正在输出突发、当前写出位置、下一突发允许开始的 monotonic 时刻
        self._burst_active = False
        self._burst_idx = 0
        self._next_burst_at = time.monotonic() + self._startup_delay_s

        # 未缩放缓存波形（前导+载荷）、样本数、静默时长（秒）；写出时再乘 _tx_amp
        self._burst_buffer: np.ndarray | None = None
        self._burst_len = 0
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

        # 源块无输入 tag；缓冲至少能装下一段突发，避免调度器切得过碎
        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(4096)

    # -----------------------------------------------------------------------
    # 配置解析与缓存加载
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_from_config(config_file: str) -> tuple[str, float, int, int]:
        """解析 cache 目录、samp_rate、fft_size、cp（不建 System）。"""
        cache_dir = resolve_source_cache_file(config_file)
        samp_rate = float(resolve_ofdm_samp_rate(config_file))
        fft_size, cp_len = resolve_ofdm_fft_cp(config_file)
        return cache_dir, samp_rate, fft_size, cp_len

    def start(self):
        """流图启动时预热加载波形，避免首包冷启动导致 Sink underflow。"""
        self._ensure_burst()
        self._next_burst_at = time.monotonic() + self._startup_delay_s
        return True

    def _log(self, msg: str) -> None:
        """向 stderr 输出带前缀的日志（flush，便于实时观察）。"""
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    def _resolve_cache_dir(self) -> Path:
        """将缓存目录解析为绝对路径；相对路径相对 PROJECT_ROOT。"""
        path = Path(self._cache_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _invalidate_burst(self) -> None:
        """GRC 参数变更时清空本地缓存，下次 general_work 触发重建。"""
        self._burst_buffer = None

    def _ensure_burst(self) -> np.ndarray:
        """懒加载入口：缓存未命中时 np.load + 拼前导，返回突发缓冲。"""
        if self._burst_buffer is None:
            self._load_burst()
        assert self._burst_buffer is not None
        return self._burst_buffer

    def _load_burst(self) -> None:
        """加载 x_time.npy，前缀 SC 前导，供写出时乘当前 tx_amp。"""
        path = self._resolve_cache_dir() / "x_time.npy"
        payload = load_burst_buffer(path, tx_amp=1.0)
        preamble = preamble_time(self._fft_size, self._cp_len)
        burst_buffer = np.concatenate([preamble, payload]).astype(
            np.complex64, copy=False
        )
        self._burst_buffer = burst_buffer
        self._burst_len = int(burst_buffer.size)
        self._idle_s = max(0.0, self._idle_ms / 1000.0)
        self.set_min_output_buffer(max(4096, self._burst_len * 2))
        self._log(
            f"load x_time.npy + SC preamble ok config={self._config_file!r} "
            f"path={path} preamble_len={preamble.size} payload_len={payload.size} "
            f"burst_len={self._burst_len} samp_rate={self._samp_rate}"
        )

    def _recompute_idle(self) -> None:
        """仅 idle_ms 变更时重算静默期，无需重建波形。"""
        self._idle_s = max(0.0, self._idle_ms / 1000.0)

    # -----------------------------------------------------------------------
    # GRC 可调属性
    # -----------------------------------------------------------------------

    @property
    def config_file(self):
        """TOML 配置路径（相对 config/ 或绝对路径）。"""
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        """更新配置路径，重新解析缓存目录与 OFDM 几何，并清空突发缓存。"""
        self._config_file = str(value)
        self._cache_dir, self._samp_rate, self._fft_size, self._cp_len = (
            self._resolve_from_config(self._config_file)
        )
        self._invalidate_burst()

    @property
    def cache_file(self):
        """只读：由 config_file 解析得到的缓存目录路径。"""
        return self._cache_dir

    @property
    def samp_rate(self):
        """只读：由 config_file 解析得到的 OFDM 采样率。"""
        return self._samp_rate

    @property
    def burst_len(self):
        """只读：当前突发缓冲的样本数（含前导；未加载前为 0）。"""
        return self._burst_len

    @property
    def idle_ms(self):
        """两次突发之间的纯静默间隔（毫秒）。"""
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        """更新静默间隔并重算 _idle_s，无需重建波形。"""
        self._idle_ms = float(value)
        self._recompute_idle()

    @property
    def tx_amp(self):
        """输出幅度缩放系数（写出时相乘，可实时调节）。"""
        return self._tx_amp

    @tx_amp.setter
    def tx_amp(self, value):
        """仅更新幅度系数；不重载波形，下一写出块立即生效。"""
        self._tx_amp = float(value)

    @property
    def time_lead_s(self):
        """tx_time 相对 wall-clock 的提前量（秒）。"""
        return self._time_lead_s

    @time_lead_s.setter
    def time_lead_s(self, value):
        """仅更新提前量；已发出的 tag 不受影响。"""
        self._time_lead_s = float(value)

    @property
    def startup_delay_s(self):
        """首突发启动前的初始等待（秒）。"""
        return self._startup_delay_s

    @startup_delay_s.setter
    def startup_delay_s(self, value):
        """仅更新参数；已调度的 _next_burst_at 需在 start() 中重设。"""
        self._startup_delay_s = float(value)

    # -----------------------------------------------------------------------
    # 调度与 general_work
    # -----------------------------------------------------------------------

    def forecast(self, noutput_items, ninputs):
        """无输入端口，不向调度器声明输入需求。"""
        del noutput_items, ninputs
        return []

    def _schedule_delay_s(self) -> float:
        """当前突发结束后到下一突发开始前的纯 idle 时长（秒）。"""
        return schedule_idle_delay_s(self._burst_len, self._samp_rate, self._idle_s)

    def general_work(self, input_items, output_items):
        """
        主工作函数：按块从缓存重放 OFDM 突发，在突发首尾附加 UHD tag。

        静默期内返回 0（不消耗输出缓冲）；突发可分多次 general_work 完成。
        返回值 n 表示本次写入输出端口的有效样本数。
        """
        del input_items
        burst_buffer = self._ensure_burst()
        out = output_items[0]

        # idle：未到下一突发时刻则不产出样本
        if not self._burst_active:
            if time.monotonic() < self._next_burst_at:
                return 0
            # 启动新突发
            self._burst_active = True
            self._burst_idx = 0

        # 本轮可写样本数：受输出缓冲与突发剩余长度限制
        n_remain = self._burst_len - self._burst_idx
        n = min(len(out), n_remain)
        if n <= 0:
            return 0

        # 从缓存拷贝一段并乘当前 tx_amp（实时可调）
        out[:n] = burst_buffer[self._burst_idx : self._burst_idx + n] * np.float32(
            self._tx_amp
        )

        abs_out = self.nitems_written(0)
        # 突发首样本：打 SOB + tx_time（仅服务 USRP 定时；无 tx_schedule）
        if self._burst_idx == 0:
            epoch = time.time() + self._time_lead_s
            add_style1_sob_time(self, 0, abs_out, epoch)

        self._burst_idx += n
        # 突发末样本：打 EOB，进入 idle，调度下一突发时刻
        if self._burst_idx >= self._burst_len:
            add_style1_eob(self, 0, abs_out + n - 1)
            self._burst_active = False
            self._next_burst_at = time.monotonic() + self._schedule_delay_s()

        return n
