"""
GNU Radio 嵌入式 Python 块：OFDM 突发感知接收（SC 前导相关同步 + LS/DD）

空口同步：仅在短滑动窗（约 8×前导长）上对已知 Schmidl-Cox 时域前导做
归一化相关找峰，估计粗 CFO；命中后再攒齐载荷 ``burst_len`` 样点：
  y_rg = demodulator(y_time)
  h_freq = ls_channel_estimator(x_rg, y_rg)   # x_rg 来自离线 x_rg.npy
  h_dd = delay_doppler_spectrum(h_freq, sens_mode="monostatic")

输出：log10(|h_dd|²) 按多普勒行输出的 float32 向量流，首行打 packet_len tag，
供 DDSpectrogramPlot 显示。不打印距离/速度/RMSE。

System / burst_len / OFDM 几何一律来自 TOML（config_file），不做 GRC OFDM 覆盖。
不再依赖同机 tx_schedule / rx_time 切包；x_rg 从 system.cache_file 目录加载。

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
from isac_imp.burst_pack import TPP_DONT
from isac_imp.gr_setup import (
    resolve_dd_output_vlen,
    resolve_ofdm_burst_len,
    resolve_ofdm_fft_cp,
    resolve_ofdm_samp_rate,
)
from isac_imp.ofdm_sc_preamble import (
    apply_cfo,
    default_search_span,
    detect_preamble,
    preamble_time,
    preamble_tpl_rev_conj,
)

# ---------------------------------------------------------------------------
# 模块常量与工具函数
# ---------------------------------------------------------------------------

# 默认 TOML（相对 PROJECT_ROOT/config）；须与 GRC 默认值一致
_DEFAULT_CONFIG = "simulation/sensing/sensing_monostatic.toml"
_TAG_PACKET_LEN = pmt.intern("packet_len")  # DD 帧首行：本帧多普勒行数
_LOG_EPS = 1e-20  # log10(|h|²+ε) 防零
_FORECAST_MAX = 16384  # 调度器单次输入上限；整帧靠内部 _buf 攒齐
_LOG_PREFIX = "[OFDM Burst Sensing RX]"


def _dd_log_magnitude(h_dd: np.ndarray, log_eps: float = _LOG_EPS) -> np.ndarray:
    """|·|² → +ε → log10，形状保持 (n_doppler, n_delay)。"""
    mag2 = np.abs(h_dd.astype(np.complex128, copy=False)) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


class blk(gr.basic_block):
    """
    按 SC 前导相关切包的 OFDM 感知接收。

    工作流程：IQ 入缓冲 → 短窗前导相关 → 等载荷 → CFO 补偿 → LS/DD → 写出。
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
        corr_threshold=0.6,
    ):
        """
        参数:
            - config_file / device / seed: TOML 路径与计算设备；OFDM 几何来自 TOML
            - idle_ms: 切包处理后抑制期 (ms)，避免同一突发重复解调
            - corr_threshold: 前导归一化相关峰门限（约 0~1）

        输出向量长度 ``dd_vlen`` 由 TOML ``[dd_spectrum_roi]`` 经
        ``resolve_dd_output_vlen`` 解析，不作为构造参数传入。
        """
        self._config_file = str(config_file)
        self._device = str(device)
        self._seed = int(seed)
        self._idle_ms = float(idle_ms)
        self._corr_threshold = float(corr_threshold)
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._samp_rate = float(resolve_ofdm_samp_rate(self._config_file))
        self._fft_size, self._cp_len = resolve_ofdm_fft_cp(self._config_file)
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)
        self._preamble = preamble_time(self._fft_size, self._cp_len)
        self._preamble_len = int(self._preamble.size)
        self._search_span = default_search_span(self._preamble_len)
        self._tpl_rev_conj = preamble_tpl_rev_conj(self._preamble)

        gr.basic_block.__init__(
            self,
            name="OFDM Burst Sensing RX",
            in_sig=[np.complex64],
            out_sig=[(np.float32, self._dd_vlen)],
        )

        # 运行时：本块独占 System、离线参考频域网格
        self._system: System | None = None
        self._x_rg: Any = None

        # IQ 缓冲；命中前导后挂起等载荷（cfo + 峰度量）
        self._buf = np.zeros(0, dtype=np.complex64)
        self._suppress_until = 0.0
        self._pending_cfo_hz: float | None = None
        self._pending_peak: float | None = None

        # 待输出的 DD 行：每项 (row_f32[vlen], is_first_row, n_rows)
        self._pending_rows: deque[tuple[np.ndarray, bool, int]] = deque()

        self.set_tag_propagation_policy(TPP_DONT)

    @property
    def burst_len(self) -> int:
        """只读：载荷切包长度（样点数，不含前导）。"""
        return self._burst_len

    @property
    def dd_vlen(self) -> int:
        """只读：输出向量长度（时延维列数）。"""
        return self._dd_vlen

    # -----------------------------------------------------------------------
    # System 与参考波形
    # -----------------------------------------------------------------------

    def _ensure_system(self) -> System:
        """懒构建本块独占 System；同步刷新 burst_len / 前导。"""
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
                self._fft_size = int(ofdm.fft_size)
                self._cp_len = int(ofdm.cyclic_prefix_length)
                self._preamble = preamble_time(self._fft_size, self._cp_len)
                self._preamble_len = int(self._preamble.size)
                self._search_span = default_search_span(self._preamble_len)
                self._tpl_rev_conj = preamble_tpl_rev_conj(self._preamble)
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
        self._pending_rows.clear()
        self._pending_cfo_hz = None
        self._pending_peak = None
        self._burst_len = int(resolve_ofdm_burst_len(self._config_file))
        self._samp_rate = float(resolve_ofdm_samp_rate(self._config_file))
        self._fft_size, self._cp_len = resolve_ofdm_fft_cp(self._config_file)
        self._dd_vlen = resolve_dd_output_vlen(self._config_file)
        self._preamble = preamble_time(self._fft_size, self._cp_len)
        self._preamble_len = int(self._preamble.size)
        self._search_span = default_search_span(self._preamble_len)
        self._tpl_rev_conj = preamble_tpl_rev_conj(self._preamble)

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
    def corr_threshold(self):
        """前导归一化相关峰门限。"""
        return self._corr_threshold

    @corr_threshold.setter
    def corr_threshold(self, value):
        """仅更新门限。"""
        self._corr_threshold = float(value)

    # -----------------------------------------------------------------------
    # 相关切包
    # -----------------------------------------------------------------------

    def forecast(self, noutput_items, ninputs):
        """向调度器声明希望的输入量（不超过 _FORECAST_MAX）。"""
        del noutput_items
        # 搜峰态只需短窗进样；等载荷时仍靠内部 _buf 攒齐
        need = min(max(self._search_span, 4096), _FORECAST_MAX)
        return [need] * ninputs

    def _trim_buf(self, keep_from_rel: int = 0) -> None:
        """丢弃缓冲前缀。"""
        if keep_from_rel <= 0:
            return
        keep_from_rel = min(keep_from_rel, self._buf.size)
        self._buf = self._buf[keep_from_rel:]

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
        """对一段 y_time（载荷）做解调 → LS → DD，并将谱行入队。"""
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

    def _finish_pending_burst(self) -> None:
        """等载荷态：缓冲已从 preamble 起点对齐，齐套后切包处理。"""
        need = self._preamble_len + self.burst_len
        if self._buf.size < need:
            return
        assert self._pending_cfo_hz is not None
        y_burst = self._buf[self._preamble_len : need].copy()
        y_burst = apply_cfo(y_burst, self._pending_cfo_hz, self._samp_rate)
        peak = float(self._pending_peak or 0.0)
        cfo = float(self._pending_cfo_hz)
        self._trim_buf(need)
        self._pending_cfo_hz = None
        self._pending_peak = None
        self._suppress_until = time.monotonic() + max(0.0, self._idle_ms / 1000.0)
        print(
            f"{_LOG_PREFIX} sync peak={peak:.3f} "
            f"cfo_hz={cfo:.1f} payload_len={y_burst.size}",
            file=sys.stderr,
            flush=True,
        )
        try:
            self._process_burst(y_burst)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _try_slice_and_process(self) -> None:
        """短窗搜峰 / 等载荷 / 抑制；禁止对整段载荷缓冲做相关。"""
        need = self._preamble_len + self.burst_len
        # 等载荷时允许攒满一帧；搜峰时缓冲上限为 search_span
        if self._pending_cfo_hz is not None:
            max_buf = need + self._search_span
        else:
            max_buf = self._search_span * 2
        if self._buf.size > max_buf:
            self._trim_buf(self._buf.size - max_buf)

        if time.monotonic() < self._suppress_until:
            # 抑制期：只裁剪，不做相关
            return

        # 等载荷：不再相关
        if self._pending_cfo_hz is not None:
            self._finish_pending_burst()
            return

        # 搜峰态
        if self._buf.size < self._preamble_len:
            return

        det = detect_preamble(
            self._buf,
            self._preamble,
            threshold=self._corr_threshold,
            fft_size=self._fft_size,
            cp_len=self._cp_len,
            samp_rate=self._samp_rate,
            max_search=self._search_span,
            tpl_rev_conj=self._tpl_rev_conj,
        )
        if det is None:
            # 未命中：只保留短窗尾部
            if self._buf.size > self._search_span:
                self._trim_buf(self._buf.size - self._search_span)
            return

        # 对齐到 preamble 起点，进入等载荷
        if det.start_idx > 0:
            self._trim_buf(det.start_idx)
        self._pending_cfo_hz = float(det.cfo_hz)
        self._pending_peak = float(det.peak_metric)
        self._finish_pending_burst()

    # -----------------------------------------------------------------------
    # 输出与 general_work
    # -----------------------------------------------------------------------

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
            out[produced][:] = row
            produced += 1
        return produced

    def general_work(self, input_items, output_items):
        """
        主工作函数：消费 IQ → 短窗前导相关切包感知 → 写出 DD 行。

        无 x_rg 缓存时仍 consume 输入以免堵死上游；返回值为本次写出的行数。
        """
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
