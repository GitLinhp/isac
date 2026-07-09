"""
GNU Radio 嵌入式 Python 块：OFDM 突发感知接收（固定长度切包 + LS/DD）

从 USRP RX 连续 IQ 流中，攒满 burst_len 后直接切包解调：
  y_rg = demodulator(y_time)
  h_freq = ls_channel_estimator(x_rg, y_rg)   # x_rg 来自离线 x_rg.npy
  h_dd = delay_doppler_spectrum(h_freq, sens_mode="monostatic")

输出：log10(|h_dd|²) 按多普勒行输出的 float32 向量流，首行打 packet_len tag，
供 SionnaDDSpectrogramPlot 显示。不打印距离/速度/RMSE。

System / burst_len / OFDM 几何一律来自 TOML（config_file），不做 GRC OFDM 覆盖。
不做相关同步；x_rg 从 source.cache_file 目录的 x_rg.npy 加载。

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
from isac_imp.constants import TPP_DONT
from isac_imp.gr_setup import (
    create_system,
    make_cache_key,
    resolve_dd_output_vlen,
    resolve_ofdm_burst_len,
)

# ---------------------------------------------------------------------------
# 模块常量与工具函数
# ---------------------------------------------------------------------------

# 默认 TOML（相对 PROJECT_ROOT/config）；须与 GRC 默认值一致
_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_TAG_PACKET_LEN = pmt.intern("packet_len")  # DD 帧首行：本帧多普勒行数
_LOG_EPS = 1e-20  # log10(|h|²+ε) 防零
_LOG_PREFIX = "[OFDM Burst Sensing RX]"
_FORECAST_MAX = 16384  # 调度器单次输入上限；整帧靠内部 _buf 攒齐


def _dd_log_magnitude(h_dd: np.ndarray, log_eps: float = _LOG_EPS) -> np.ndarray:
    """|·|² → +ε → log10，形状保持 (n_doppler, n_delay)。"""
    mag2 = np.abs(h_dd.astype(np.complex128, copy=False)) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


def _as_bool(value) -> bool:
    """GRC 可能传入字符串；统一转为 bool。"""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class blk(gr.basic_block):
    """
    固定长度切包 OFDM 感知接收：攒满 burst_len 后解调、LS/DD，并输出 DD log 谱行。

    工作流程：IQ 入缓冲 → 切包 → LS/DD → 行队列 → 写出 float32 向量（首行打 packet_len）。
    """

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        device="cpu",
        seed=42,
        idle_ms=400.0,
        dd_vlen=0,
        debug=False,
    ):
        """
        参数:
            - config_file / device / seed: TOML 路径与计算设备；OFDM 几何来自 TOML
            - idle_ms: 切包处理后抑制期 (ms)，避免同一突发重复解调
            - dd_vlen: 输出向量长度；<=0 时按 TOML ROI 时延列数自动解析
            - debug: 是否向 stderr 打印切包/LS-DD 调试信息
        """
        self._config_file = str(config_file)
        self._device = str(device)
        self._seed = int(seed)
        self._idle_ms = float(idle_ms)
        self._debug = _as_bool(debug)
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        if int(dd_vlen) > 0:
            self._dd_vlen = int(dd_vlen)
        else:
            self._dd_vlen = resolve_dd_output_vlen(self._config_file)

        gr.basic_block.__init__(
            self,
            name="OFDM Burst Sensing RX",
            in_sig=[np.complex64],
            out_sig=[(np.float32, self._dd_vlen)],
        )

        # 运行时：共享 System、离线参考频域网格
        self._system: System | None = None
        self._x_rg: Any = None

        # 切包状态：IQ 环形缓冲、抑制截止时刻、已处理突发计数
        self._buf = np.zeros(0, dtype=np.complex64)
        self._suppress_until = 0.0
        self._processed_bursts = 0

        # 待输出的 DD 行：每项 (row_f32[vlen], is_first_row, n_rows)
        self._pending_rows: deque[tuple[np.ndarray, bool, int]] = deque()

        # 调试节流：等待缓存日志时间戳、是否已打印首帧写出
        self._last_waiting_cache_log_at = 0.0
        self._logged_first_produce = False

        self.set_tag_propagation_policy(TPP_DONT)
        self._log(
            f"init dd_vlen={self._dd_vlen} burst_len={self._burst_len} "
            f"idle_ms={self._idle_ms} debug={self._debug} (config-only, no OFDM override)"
        )

    def _log(self, msg: str) -> None:
        """debug 开启时向 stderr 输出带前缀日志。"""
        if not self._debug:
            return
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

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

    def _cache_key(self) -> tuple:
        """与 create_system 一致的 registry 键（无 OFDM 覆盖）。"""
        return make_cache_key(
            self._config_file,
            self._device,
            self._seed,
            None,
        )

    def _ensure_system(self) -> System:
        """懒构建/复用 System；同步刷新 burst_len。"""
        if self._system is None:
            torch.set_num_threads(1)
            cache_key = self._cache_key()
            self._system = create_system(
                self._config_file,
                device=self._device,
                seed=self._seed,
                ofdm_overrides=None,
            )
            ofdm = self._system.params.ofdm
            if ofdm is not None:
                self._burst_len = int(
                    ofdm.num_symbols * (ofdm.fft_size + ofdm.cyclic_prefix_length)
                )
            self._log(
                f"create_system config={self._config_file!r} device={self._device!r} "
                f"key={cache_key} system_id={id(self._system)} burst_len={self._burst_len}"
            )
        return self._system

    def _ensure_waveform(self) -> bool:
        """仅从缓存目录加载 x_rg.npy；成功返回 True。"""
        if self._x_rg is not None:
            return True
        system = self._ensure_system()
        try:
            self._x_rg = system.load_transmit_x_rg()
        except (FileNotFoundError, ValueError) as exc:
            now = time.monotonic()
            if now - self._last_waiting_cache_log_at >= 1.0:
                self._last_waiting_cache_log_at = now
                self._log(f"waiting for transmit x_rg.npy: {exc}")
            return False
        self._log(f"load_transmit_x_rg ok x_rg={tuple(self._x_rg.shape)}")
        return True

    def _invalidate(self) -> None:
        """GRC 参数变更时清空 System/波形/缓冲，并重算 burst_len。"""
        self._system = None
        self._x_rg = None
        self._buf = np.zeros(0, dtype=np.complex64)
        self._pending_rows.clear()
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))

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
        """随机种子（create_system registry 键的一部分）。"""
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
    def debug(self):
        """是否输出调试日志。"""
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = _as_bool(value)

    # -----------------------------------------------------------------------
    # 切包与感知
    # -----------------------------------------------------------------------

    def forecast(self, noutput_items, ninputs):
        """向调度器声明希望的输入量（不超过 _FORECAST_MAX）。"""
        del noutput_items
        need = min(self.burst_len, _FORECAST_MAX)
        need = max(need, 4096)
        return [need] * ninputs

    def _queue_dd_frame(self, h_dd: torch.Tensor) -> tuple[int, float, float]:
        """将 h_dd 转为 log 幅度行并入队；返回 (n_rows, finite_min, finite_max)。"""
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

        finite = log_mag[np.isfinite(log_mag)]
        if finite.size:
            return n_rows, float(finite.min()), float(finite.max())
        return n_rows, float("nan"), float("nan")

    def _process_burst(self, y_np: np.ndarray) -> None:
        """对一段 y_time 做解调 → LS → DD，并将谱行入队。"""
        if y_np.size < self.burst_len:
            self._log(f"burst too short: size={y_np.size} need={self.burst_len}")
            return
        y_np = y_np[: self.burst_len]

        system = self._ensure_system()
        if not self._ensure_waveform():
            self._log("skip sensing: transmit x_rg.npy not loaded")
            return
        x_rg = self._x_rg

        comps = system.components
        if comps.demodulator is None:
            self._log("skip sensing: demodulator is None")
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
            n_rows, lo, hi = self._queue_dd_frame(h_dd)

        self._processed_bursts += 1
        self._log(
            f"sensing ok #{self._processed_bursts} h_dd={tuple(h_dd.shape)} "
            f"rows={n_rows} log10_min={lo:.3f} log10_max={hi:.3f} "
            f"pending={len(self._pending_rows)}"
        )

    def _try_slice_and_process(self) -> None:
        """攒满 burst_len 后切包解调；idle 抑制期内只裁剪缓冲。"""
        burst_len = self.burst_len
        suppress = time.monotonic() < self._suppress_until

        # 限制缓冲上限，避免长时间未切包时内存膨胀
        max_buf = burst_len * 2
        if self._buf.size > max_buf:
            self._buf = self._buf[-max_buf:]

        if suppress:
            return

        if self._buf.size < burst_len:
            return

        # 切出一帧，进入抑制期，再跑感知
        y_burst = self._buf[:burst_len].copy()
        self._buf = self._buf[burst_len:]
        self._suppress_until = time.monotonic() + max(0.0, self._idle_ms / 1000.0)

        self._log(f"slice packet burst_len={burst_len} remain_buf={self._buf.size}")
        try:
            self._process_burst(y_burst)
        except Exception:
            self._log("sensing failed:")
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
                if not self._logged_first_produce:
                    self._logged_first_produce = True
                    self._log(
                        f"produce first frame packet_len={n_rows} vlen={self._dd_vlen}"
                    )
            out[produced][:] = row
            produced += 1
        return produced

    def general_work(self, input_items, output_items):
        """
        主工作函数：消费 IQ → 攒包切包感知 → 写出待发送 DD 行。

        无 x_rg 缓存时仍 consume 输入以免堵死上游；返回值为本次写出的行数。
        """
        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)

        if n_in > 0:
            # 尚无参考网格：丢弃本批输入但仍 consume，避免 USRP Source 堵死
            if not self._ensure_waveform():
                self.consume(0, n_in)
            else:
                self._buf = np.concatenate(
                    (self._buf, np.asarray(inp, dtype=np.complex64))
                )
                self.consume(0, n_in)
                self._try_slice_and_process()

        return self._produce_pending(out)
