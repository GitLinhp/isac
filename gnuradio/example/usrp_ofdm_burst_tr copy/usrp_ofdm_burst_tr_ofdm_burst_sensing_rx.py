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
corr_threshold / template_len 形参保留以兼容旧 GRC，内部忽略。
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

_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_TAG_PACKET_LEN = pmt.intern("packet_len")
_LOG_EPS = 1e-20
_LOG_PREFIX = "[OFDM Burst Sensing RX]"
_FORECAST_MAX = 16384  # 调度器单次输入上限；整帧靠内部 _buf 攒齐


def _dd_log_magnitude(h_dd: np.ndarray, log_eps: float = _LOG_EPS) -> np.ndarray:
    """|·|² → +ε → log10，形状保持 (n_doppler, n_delay)。"""
    mag2 = np.abs(h_dd.astype(np.complex128, copy=False)) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class blk(gr.basic_block):
    """
    固定长度切包 OFDM 感知接收：攒满 burst_len 后解调、LS/DD，并输出 DD log 谱行。
    """

    def __init__(
        self,
        config_file=_DEFAULT_CONFIG,
        device="cpu",
        seed=42,
        corr_threshold=0.3,
        template_len=0,
        idle_ms=400.0,
        dd_vlen=0,
        debug=False,
    ):
        """
        参数:
            - config_file / device / seed: TOML 路径与计算设备；OFDM 几何来自 TOML
            - corr_threshold / template_len: 兼容 GRC，已忽略（无相关同步）
            - idle_ms: 切包处理后抑制期 (ms)，避免同一突发重复解调
            - dd_vlen: 输出向量长度；<=0 时按 TOML ROI 时延列数自动解析
            - debug: 是否向 stderr 打印切包/LS-DD 调试信息
        """
        self._config_file = str(config_file)
        self._device = str(device)
        self._seed = int(seed)
        self._corr_threshold = float(corr_threshold)  # unused; GRC compat
        self._template_len = int(template_len)  # unused; GRC compat
        self._idle_ms = float(idle_ms)
        self._debug = _as_bool(debug)
        self._burst_len = int(
            resolve_ofdm_burst_len(self._config_file, self._device, self._seed)
        )
        if int(dd_vlen) > 0:
            self._dd_vlen = int(dd_vlen)
        else:
            self._dd_vlen = resolve_dd_output_vlen(
                self._config_file,
                self._device,
                self._seed,
            )

        gr.basic_block.__init__(
            self,
            name="OFDM Burst Sensing RX",
            in_sig=[np.complex64],
            out_sig=[(np.float32, self._dd_vlen)],
        )

        self._system: System | None = None
        self._x_rg: Any = None

        self._buf = np.zeros(0, dtype=np.complex64)
        self._suppress_until = 0.0
        self._processed_bursts = 0

        # 待输出的 DD 行：每项 (row_f32[vlen], is_first_row, n_rows)
        self._pending_rows: deque[tuple[np.ndarray, bool, int]] = deque()

        self._last_waiting_cache_log_at = 0.0
        self._logged_first_produce = False

        self.set_tag_propagation_policy(TPP_DONT)
        self._log(
            f"init dd_vlen={self._dd_vlen} burst_len={self._burst_len} "
            f"idle_ms={self._idle_ms} debug={self._debug} (config-only, no OFDM override)"
        )

    def _log(self, msg: str) -> None:
        if not self._debug:
            return
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    @property
    def burst_len(self) -> int:
        return self._burst_len

    @property
    def dd_vlen(self) -> int:
        return self._dd_vlen

    def _cache_key(self) -> tuple:
        return make_cache_key(
            self._config_file,
            self._device,
            self._seed,
            None,
        )

    def _ensure_system(self) -> System:
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
        self._system = None
        self._x_rg = None
        self._buf = np.zeros(0, dtype=np.complex64)
        self._pending_rows.clear()
        self._burst_len = int(
            resolve_ofdm_burst_len(self._config_file, self._device, self._seed)
        )

    # --- GRC 可调属性 ---
    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        self._config_file = str(value)
        self._invalidate()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = str(value)
        self._invalidate()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)
        self._invalidate()

    @property
    def corr_threshold(self):
        return self._corr_threshold

    @corr_threshold.setter
    def corr_threshold(self, value):
        self._corr_threshold = float(value)

    @property
    def template_len(self):
        return self._template_len

    @template_len.setter
    def template_len(self, value):
        self._template_len = int(value)

    @property
    def idle_ms(self):
        return self._idle_ms

    @idle_ms.setter
    def idle_ms(self, value):
        self._idle_ms = float(value)

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = _as_bool(value)

    def forecast(self, noutput_items, ninputs):
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

        max_buf = burst_len * 2
        if self._buf.size > max_buf:
            self._buf = self._buf[-max_buf:]

        if suppress:
            return

        if self._buf.size < burst_len:
            return

        y_burst = self._buf[:burst_len].copy()
        self._buf = self._buf[burst_len:]
        self._suppress_until = time.monotonic() + max(0.0, self._idle_ms / 1000.0)

        self._log(f"slice packet burst_len={burst_len} remain_buf={self._buf.size}")
        try:
            self._process_burst(y_burst)
        except Exception:
            self._log("sensing failed:")
            traceback.print_exc(file=sys.stderr)

    def _produce_pending(self, out: np.ndarray) -> int:
        """写出待发送 DD 行，返回写出行数。"""
        n_out = len(out)
        produced = 0
        while produced < n_out and self._pending_rows:
            row, is_first, n_rows = self._pending_rows.popleft()
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
        inp = input_items[0]
        out = output_items[0]
        n_in = len(inp)

        if n_in > 0:
            if not self._ensure_waveform():
                self.consume(0, n_in)
            else:
                self._buf = np.concatenate(
                    (self._buf, np.asarray(inp, dtype=np.complex64))
                )
                self.consume(0, n_in)
                self._try_slice_and_process()

        return self._produce_pending(out)
