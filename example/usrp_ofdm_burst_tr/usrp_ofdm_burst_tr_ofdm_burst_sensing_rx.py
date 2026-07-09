"""
GNU Radio 嵌入式 Python 块：OFDM 突发感知接收（tx_time + rx_time 切包 + LS/DD）

同机单站：TX 经消息口 ``tx_schedule`` 下发每突发计划 epoch（与 tx_time 同形）；
USRP Source 连续 IQ 上的 ``rx_time`` 建立样点时间轴。在
``t_target = tx_epoch + rx_delay_s`` 处切出 ``burst_len`` 样点后：
  y_rg = demodulator(y_time)
  h_freq = ls_channel_estimator(x_rg, y_rg)   # x_rg 来自离线 x_rg.npy
  h_dd = delay_doppler_spectrum(h_freq, sens_mode="monostatic")

输出：log10(|h_dd|²) 按多普勒行输出的 float32 向量流，首行打 packet_len tag，
供 DDSpectrogramPlot 显示。不打印距离/速度/RMSE。

System / burst_len / OFDM 几何一律来自 TOML（config_file），不做 GRC OFDM 覆盖。
不做相关同步；x_rg 从 system.cache_file 目录的 x_rg.npy 加载。

注意：__init__ 形参默认值须与 GRC 变量保持同步。
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import deque
from typing import Any

import numpy as np
import pmt
import torch
from gnuradio import gr

from isac.system import System
from isac.utils import set_random_seed
from isac_imp.burst_pack import (
    PORT_TX_SCHEDULE,
    TAG_RX_TIME,
    TPP_DONT,
    parse_uhd_time_pmt,
)
from isac_imp.gr_setup import (
    resolve_dd_output_vlen,
    resolve_ofdm_burst_len,
    resolve_ofdm_samp_rate,
)

# ---------------------------------------------------------------------------
# 模块常量与工具函数
# ---------------------------------------------------------------------------

# 默认 TOML（相对 PROJECT_ROOT/config）；须与 GRC 默认值一致
_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_TAG_PACKET_LEN = pmt.intern("packet_len")  # DD 帧首行：本帧多普勒行数
_LOG_EPS = 1e-20  # log10(|h|²+ε) 防零
_FORECAST_MAX = 16384  # 调度器单次输入上限；整帧靠内部 _buf 攒齐
_MAX_SCHEDULE = 32  # 待处理 tx_schedule 队列上限


def _dd_log_magnitude(h_dd: np.ndarray, log_eps: float = _LOG_EPS) -> np.ndarray:
    """|·|² → +ε → log10，形状保持 (n_doppler, n_delay)。"""
    mag2 = np.abs(h_dd.astype(np.complex128, copy=False)) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


class blk(gr.basic_block):
    """
    按 tx_schedule + rx_time 切包的 OFDM 感知接收。

    工作流程：IQ/rx_time 入缓冲 → 队头 schedule 映射样点 → LS/DD → 行队列写出。
    """

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        device="cpu",
        seed=42,
        idle_ms=900.0,
        rx_delay_s=0.0,
    ):
        """
        参数:
            - config_file / device / seed: TOML 路径与计算设备；OFDM 几何来自 TOML
            - idle_ms: 切包处理后抑制期 (ms)，避免同一突发重复解调
            - rx_delay_s: 期望回波相对 tx_time 的时延 (s)；单站默认 0

        输出向量长度 ``dd_vlen`` 由 TOML ``[dd_spectrum_roi]`` 经
        ``resolve_dd_output_vlen`` 解析，不作为构造参数传入。
        """
        self._config_file = str(config_file)
        self._device = str(device)
        self._seed = int(seed)
        self._idle_ms = float(idle_ms)
        self._rx_delay_s = float(rx_delay_s)
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._samp_rate = float(resolve_ofdm_samp_rate(self._config_file))
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)

        gr.basic_block.__init__(
            self,
            name="OFDM Burst Sensing RX",
            in_sig=[np.complex64],
            out_sig=[(np.float32, self._dd_vlen)],
        )

        # 运行时：本块独占 System、离线参考频域网格
        self._system: System | None = None
        self._x_rg: Any = None

        # IQ 缓冲：_buf_abs0 为缓冲首样点对应的绝对读入下标
        self._buf = np.zeros(0, dtype=np.complex64)
        self._buf_abs0 = 0
        self._suppress_until = 0.0

        # rx_time 锚点：(绝对样点下标, epoch 秒)；无锚点时无法按时间切包
        self._time_anchor_abs: int | None = None
        self._time_anchor_epoch: float | None = None

        # TX 下发的计划发射 epoch 队列
        self._tx_schedule: deque[float] = deque()

        # 待输出的 DD 行：每项 (row_f32[vlen], is_first_row, n_rows)
        self._pending_rows: deque[tuple[np.ndarray, bool, int]] = deque()

        self.set_tag_propagation_policy(TPP_DONT)
        self.message_port_register_in(pmt.intern(PORT_TX_SCHEDULE))
        self.set_msg_handler(pmt.intern(PORT_TX_SCHEDULE), self._handle_tx_schedule)

    def _handle_tx_schedule(self, msg) -> None:
        """消息口：入队 TX 计划 epoch；过长则丢弃最旧项。"""
        try:
            epoch = parse_uhd_time_pmt(msg)
        except Exception:
            return
        self._tx_schedule.append(float(epoch))
        while len(self._tx_schedule) > _MAX_SCHEDULE:
            self._tx_schedule.popleft()

    @property
    def burst_len(self) -> int:
        """只读：当前切包长度（样点数）。"""
        return self._burst_len

    @property
    def dd_vlen(self) -> int:
        """只读：输出向量长度（时延维列数）。"""
        return self._dd_vlen

    # -----------------------------------------------------------------------
    # System 与参考波形
    # -----------------------------------------------------------------------

    def _ensure_system(self) -> System:
        """懒构建本块独占 System；同步刷新 burst_len。"""
        if self._system is None:
            torch.set_num_threads(1)
            set_random_seed(self._seed)
            self._system = System(self._config_file, device=self._device)
            ofdm = self._system.params.ofdm
            if ofdm is not None:
                self._burst_len = int(
                    ofdm.num_symbols * (ofdm.fft_size + ofdm.cyclic_prefix_length)
                )
                self._samp_rate = float(ofdm.samp_rate)
        return self._system

    def _ensure_waveform(self) -> bool:
        """仅从缓存目录加载 x_rg.npy；成功返回 True。"""
        if self._x_rg is not None:
            return True
        system = self._ensure_system()
        try:
            self._x_rg = system.load_transmit_x_rg()
        except (FileNotFoundError, ValueError):
            return False
        return True

    def _invalidate(self) -> None:
        """GRC 参数变更时清空 System/波形/缓冲，并重算几何。"""
        self._system = None
        self._x_rg = None
        self._buf = np.zeros(0, dtype=np.complex64)
        self._buf_abs0 = 0
        self._time_anchor_abs = None
        self._time_anchor_epoch = None
        self._tx_schedule.clear()
        self._pending_rows.clear()
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._samp_rate = float(resolve_ofdm_samp_rate(self._config_file))
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)

    # -----------------------------------------------------------------------
    # GRC 可调属性
    # -----------------------------------------------------------------------

    @property
    def config_file(self):
        """TOML 配置路径。"""
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        """更新配置并清空运行时状态。"""
        self._config_file = str(value)
        self._invalidate()

    @property
    def device(self):
        """Sionna/Torch 计算设备。"""
        return self._device

    @device.setter
    def device(self, value):
        """更新设备并清空运行时状态（需重建 System）。"""
        self._device = str(value)
        self._invalidate()

    @property
    def seed(self):
        """随机种子（构建 System 前传给 set_random_seed）。"""
        return self._seed

    @seed.setter
    def seed(self, value):
        """更新种子并清空运行时状态。"""
        self._seed = int(value)
        self._invalidate()

    @property
    def idle_ms(self):
        """切包后抑制期（毫秒）。"""
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        """仅更新抑制时长；不影响已调度的 _suppress_until。"""
        self._idle_ms = float(value)

    @property
    def rx_delay_s(self):
        """期望回波相对 tx_time 的时延（秒）。"""
        return self._rx_delay_s

    @rx_delay_s.setter
    def rx_delay_s(self, value):
        """仅更新时延参数。"""
        self._rx_delay_s = float(value)

    # -----------------------------------------------------------------------
    # 时间轴与切包
    # -----------------------------------------------------------------------

    def forecast(self, noutput_items, ninputs):
        """向调度器声明希望的输入量（不超过 _FORECAST_MAX）。"""
        del noutput_items
        need = min(self.burst_len, _FORECAST_MAX)
        need = max(need, 4096)
        return [need] * ninputs

    def _ingest_rx_time_tags(self, n_in: int) -> None:
        """扫描本批输入上的 rx_time，更新时间锚点。"""
        if n_in <= 0:
            return
        base = self.nitems_read(0)
        for tag in self.get_tags_in_range(0, base, base + n_in):
            if tag.key != TAG_RX_TIME:
                continue
            try:
                epoch = parse_uhd_time_pmt(tag.value)
            except Exception:
                continue
            self._time_anchor_abs = int(tag.offset)
            self._time_anchor_epoch = float(epoch)

    def _epoch_to_abs_sample(self, epoch_s: float) -> int | None:
        """将绝对时刻映射为绝对样点下标；无锚点或 samp_rate 无效时返回 None。"""
        if self._time_anchor_abs is None or self._time_anchor_epoch is None:
            return None
        if self._samp_rate <= 0:
            return None
        dt = float(epoch_s) - float(self._time_anchor_epoch)
        return int(round(self._time_anchor_abs + dt * self._samp_rate))

    def _trim_buf(self, keep_from_rel: int = 0) -> None:
        """丢弃缓冲前缀，并同步 _buf_abs0。"""
        if keep_from_rel <= 0:
            return
        keep_from_rel = min(keep_from_rel, self._buf.size)
        self._buf = self._buf[keep_from_rel:]
        self._buf_abs0 += keep_from_rel

    def _queue_dd_frame(self, h_dd: torch.Tensor) -> None:
        """将 h_dd 转为 log 幅度行并入队。"""
        arr = h_dd.detach().cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        log_mag = _dd_log_magnitude(arr)
        n_rows, n_cols = int(log_mag.shape[0]), int(log_mag.shape[1])
        vlen = self._dd_vlen
        for i in range(n_rows):
            row = np.full(vlen, np.nan, dtype=np.float32)
            n = min(n_cols, vlen)
            row[:n] = log_mag[i, :n]
            self._pending_rows.append((row, i == 0, n_rows))

    def _process_burst(self, y_np: np.ndarray) -> None:
        """对一段 y_time 做解调 → LS → DD，并将谱行入队。"""
        if y_np.size < self.burst_len:
            return
        y_np = y_np[: self.burst_len]

        system = self._ensure_system()
        if not self._ensure_waveform():
            return
        x_rg = self._x_rg

        comps = system.components
        if comps.demodulator is None:
            return

        y_time = torch.from_numpy(np.ascontiguousarray(y_np)).to(
            device=system.device,
            dtype=torch.complex64,
        )
        while y_time.dim() < 3:
            y_time = y_time.unsqueeze(0)

        with torch.inference_mode():
            y_rg = comps.demodulator(y_time)
            h_freq = comps.ls_channel_estimator(x_rg, y_rg)
            h_dd = comps.delay_doppler_spectrum(h_freq, sens_mode="monostatic")
            self._queue_dd_frame(h_dd)

    def _try_slice_and_process(self) -> None:
        """按队头 tx_schedule + rx_time 切包；抑制期内只裁剪缓冲。"""
        burst_len = self.burst_len
        # 限制缓冲上限，避免长时间未切包时内存膨胀
        max_buf = burst_len * 4
        if self._buf.size > max_buf:
            self._trim_buf(self._buf.size - max_buf)

        if time.monotonic() < self._suppress_until:
            return

        if not self._tx_schedule:
            return

        tx_epoch = self._tx_schedule[0]
        t_target = tx_epoch + self._rx_delay_s
        start_abs = self._epoch_to_abs_sample(t_target)
        if start_abs is None:
            # 尚无 rx_time 锚点：等待，不丢 schedule
            return

        buf_end_abs = self._buf_abs0 + self._buf.size
        # 目标起点已落在缓冲之前：过期，丢弃该 schedule
        if start_abs < self._buf_abs0:
            self._tx_schedule.popleft()
            return

        # 目标尚未完全进入缓冲：等待更多 IQ
        if start_abs + burst_len > buf_end_abs:
            return

        rel = start_abs - self._buf_abs0
        y_burst = self._buf[rel : rel + burst_len].copy()
        # 丢弃已用前缀，保留突发之后的样点
        self._trim_buf(rel + burst_len)
        self._tx_schedule.popleft()
        self._suppress_until = time.monotonic() + max(0.0, self._idle_ms / 1000.0)

        try:
            self._process_burst(y_burst)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    # -----------------------------------------------------------------------
    # 输出与 general_work
    # -----------------------------------------------------------------------

    def _produce_pending(self, out: np.ndarray) -> int:
        """写出待发送 DD 行，返回写出行数。"""
        n_out = len(out)
        produced = 0
        while produced < n_out and self._pending_rows:
            row, is_first, n_rows = self._pending_rows.popleft()
            # 帧首行打 packet_len，供下游谱图块界定一帧
            if is_first:
                abs_out = self.nitems_written(0) + produced
                self.add_item_tag(
                    0, abs_out, _TAG_PACKET_LEN, pmt.from_long(int(n_rows))
                )
            out[produced][:] = row
            produced += 1
        return produced

    def general_work(self, input_items, output_items):
        """
        主工作函数：消费 IQ / rx_time → 按 schedule 切包感知 → 写出 DD 行。

        无 x_rg 缓存时仍 consume 输入以免堵死上游；返回值为本次写出的行数。
        """
        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)

        if n_in > 0:
            self._ingest_rx_time_tags(n_in)
            # 尚无参考网格：丢弃本批输入但仍 consume，避免 USRP Source 堵死
            if not self._ensure_waveform():
                # 仍推进绝对下标，使后续 rx_time 锚点与缓冲一致
                if self._buf.size == 0:
                    self._buf_abs0 = self.nitems_read(0) + n_in
                self.consume(0, n_in)
            else:
                if self._buf.size == 0:
                    self._buf_abs0 = self.nitems_read(0)
                self._buf = np.concatenate(
                    (self._buf, np.asarray(inp, dtype=np.complex64))
                )
                self.consume(0, n_in)
                self._try_slice_and_process()

        return self._produce_pending(out)
