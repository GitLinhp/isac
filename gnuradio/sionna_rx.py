"""Sionna 接收链：OFDMDemodulator + LS 信道估计 + DelayDopplerSpectrum（GPU 专用线程）。"""
from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pmt
from gnuradio import gr

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gr_config import GrcOverrides, grc_overrides_from_grc_vars, merge_config, resolve_config_path
from sionna_tx import (
    TxPacket,
    get_tx_packet,
    log_gr_overrides,
)

_warmed_rx_keys: set[Tuple[str, int, int, int, int, float, float, int, str]] = set()


@dataclass(frozen=True)
class RxContext:
    """缓存的 Sionna 接收处理上下文（GPU）。"""

    x_rg: object
    demodulator: object
    delay_doppler: object
    packet_len_time: int
    packet_len_freq: int
    fft_len: int
    device: str


def _cache_key_rx(
    effective_key: Tuple[str, int, int, int, int, float, float, int, str],
) -> Tuple[str, int, int, int, int, float, float, int, str]:
    return effective_key


@lru_cache(maxsize=8)
def _build_rx_context_cached(
    cache_key: Tuple[str, int, int, int, int, float, float, int, str],
) -> RxContext:
    config_path = Path(cache_key[0])
    fft_len, n_sym, cp, scs_int, cf_int, seed = (
        cache_key[1],
        cache_key[2],
        cache_key[3],
        cache_key[4],
        cache_key[5],
        cache_key[6],
    )
    device = cache_key[7]
    overrides = GrcOverrides(
        fft_len=fft_len,
        ofdm_symbols=n_sym,
        cp_len=cp,
        subcarrier_spacing=float(scs_int),
        center_freq=float(cf_int),
        seed=seed,
        device=device,
    )
    effective = merge_config(str(config_path), overrides)
    pkt = get_tx_packet(str(config_path), seed=seed, device=device, overrides=overrides)
    return _build_rx_context_impl(effective, pkt)


def build_rx_context(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    tx_packet: Optional[TxPacket] = None,
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> RxContext:
    config_path = resolve_config_path(config_file)
    if overrides is None and grc_kw:
        overrides = grc_overrides_from_grc_vars(seed=int(seed), device=str(device), **grc_kw)
    if overrides is None:
        if tx_packet is not None:
            return _build_rx_context_impl_from_toml(config_path, seed, device, tx_packet)
        pkt = get_tx_packet(config_file, seed=seed, device=device)
        return _build_rx_context_impl_from_toml(config_path, seed, device, pkt)
    effective = merge_config(str(config_path), overrides)
    if tx_packet is not None:
        return _build_rx_context_impl(effective, tx_packet)
    return _build_rx_context_cached(effective.cache_key())


def prewarm_sionna_rx(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    tx_packet: Optional[TxPacket] = None,
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> RxContext:
    if overrides is None and grc_kw:
        overrides = grc_overrides_from_grc_vars(seed=int(seed), device=str(device), **grc_kw)
    config_path = resolve_config_path(config_file)
    if overrides is not None:
        key = merge_config(str(config_path), overrides).cache_key()
    else:
        pkt = tx_packet or get_tx_packet(config_file, seed=seed, device=device)
        raw = __import__("isac.utils", fromlist=["load_config"]).load_config(config_path)
        from isac.data_structures.params import SystemParams

        params = SystemParams.from_dict(raw)
        ofdm = params.ofdm
        key = merge_config(
            str(config_path),
            GrcOverrides(
                fft_len=ofdm.num_subcarriers,
                ofdm_symbols=ofdm.num_symbols,
                cp_len=ofdm.cyclic_prefix_length,
                subcarrier_spacing=ofdm.subcarrier_spacing,
                center_freq=params.carrier_frequency,
                seed=int(seed),
                device=str(device),
            ),
        ).cache_key()
    first = key not in _warmed_rx_keys
    if first:
        print(f"Sionna 接收链：GPU 初始化 (device={device}) ...")
        _warmed_rx_keys.add(key)
    else:
        print(f"Sionna 接收链：加载 GPU 上下文 (device={device})")
    ctx = build_rx_context(
        config_file,
        seed=seed,
        device=device,
        tx_packet=tx_packet,
        overrides=overrides,
        **grc_kw,
    )
    if first:
        print(
            f"  时域包 {ctx.packet_len_time} 样点 → "
            f"谱矩阵 {ctx.packet_len_freq}×{ctx.fft_len}"
        )
    return ctx


def _build_rx_context_impl_from_toml(
    config_path: Path,
    seed: int,
    device: str,
    tx_packet: TxPacket,
) -> RxContext:
    import sionna

    from isac.data_structures.components.ofdm_components import build_ofdm_components
    from isac.data_structures.components.sensing_components import build_sensing_components
    from isac.data_structures.params import SystemParams
    from isac.utils import load_config, set_random_seed

    config = load_config(config_path)
    params = SystemParams.from_dict(config)
    set_random_seed(seed)
    sionna.phy.config.device = device

    ofdm = build_ofdm_components(params, device=device)
    sens = build_sensing_components(params, ofdm.rg, device=device)

    import torch

    x_rg_t = torch.from_numpy(tx_packet.x_rg).to(device=device, dtype=torch.complex64)
    cp = params.ofdm.cyclic_prefix_length
    n_sym = params.ofdm.num_symbols
    fft_size = params.ofdm.num_subcarriers

    return RxContext(
        x_rg=x_rg_t,
        demodulator=ofdm.demodulator,
        delay_doppler=sens.delay_doppler_spectrum,
        packet_len_time=n_sym * (fft_size + cp),
        packet_len_freq=n_sym,
        fft_len=fft_size,
        device=device,
    )


def _build_rx_context_impl(effective, tx_packet: TxPacket) -> RxContext:
    import sionna
    import torch

    from isac.data_structures.components.ofdm_components import build_ofdm_components
    from isac.data_structures.components.sensing_components import build_sensing_components
    from isac.utils import set_random_seed

    params = effective.system_params
    device = effective.device
    set_random_seed(effective.seed)
    sionna.phy.config.device = device

    ofdm = build_ofdm_components(params, device=device)
    sens = build_sensing_components(params, ofdm.rg, device=device)

    x_rg_t = torch.from_numpy(tx_packet.x_rg).to(device=device, dtype=torch.complex64)
    cp = params.ofdm.cyclic_prefix_length
    n_sym = params.ofdm.num_symbols
    fft_size = params.ofdm.num_subcarriers

    return RxContext(
        x_rg=x_rg_t,
        demodulator=ofdm.demodulator,
        delay_doppler=sens.delay_doppler_spectrum,
        packet_len_time=n_sym * (fft_size + cp),
        packet_len_freq=n_sym,
        fft_len=fft_size,
        device=device,
    )


def dd_matrix_to_log_magnitude(
    h_dd: np.ndarray,
    log_eps: float = 1e-20,
) -> np.ndarray:
    """|·|² → +ε → log10，与 GRC mag²/add_const/nlog10 链一致。"""
    mag2 = np.abs(h_dd, dtype=np.float64) ** 2
    return np.log10(mag2 + float(log_eps)).astype(np.float32)


def prepare_dd_outputs(
    h_dd: np.ndarray,
    *,
    flip_doppler: bool = True,
    log_eps: float = 1e-20,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (complex IQ 矩阵, log10|·|² 矩阵)，形状均为 (n_sym, fft_len)。"""
    if flip_doppler:
        h_dd = np.flipud(h_dd)
    iq = np.ascontiguousarray(h_dd.astype(np.complex64, copy=False))
    log_mag = dd_matrix_to_log_magnitude(iq, log_eps=log_eps)
    return iq, log_mag


def compute_delay_doppler_matrix(
    y_time: np.ndarray,
    ctx: RxContext,
    device: str,
) -> np.ndarray:
    """时域 RX → 距离-多普勒复数矩阵 (num_symbols, fft_size)。"""
    import torch

    y = np.ascontiguousarray(y_time, dtype=np.complex64).reshape(-1)
    if y.size != ctx.packet_len_time:
        raise ValueError(f"RX 长度 {y.size} != 期望 {ctx.packet_len_time}")

    dev = device or ctx.device
    y_t = torch.from_numpy(y).to(device=dev, dtype=torch.complex64)
    y_batch = y_t.reshape(1, 1, 1, -1)
    ctx_mgr = (
        torch.cuda.device(dev) if str(dev).startswith("cuda") else _nullcontext()
    )
    with ctx_mgr:
        y_rg = ctx.demodulator(y_batch).squeeze()
        h = _estimate_channel_torch(ctx.x_rg, y_rg)
        h_dd = ctx.delay_doppler(h)
    return h_dd.detach().cpu().numpy().astype(np.complex64)


def _estimate_channel_torch(x_rg, y_rg, eps: float = 1e-12):
    denom = x_rg.abs().square() + eps
    return y_rg * x_rg.conj() / denom


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _CudaRxWorker:
    """专用 CUDA 线程，避免 GR 工作线程直接调 CUDA。"""

    _instances: dict[Tuple[int, str], "_CudaRxWorker"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, ctx: RxContext) -> "_CudaRxWorker":
        key = (id(ctx), ctx.device)
        with cls._instances_lock:
            worker = cls._instances.get(key)
            if worker is None:
                worker = cls(ctx)
                cls._instances[key] = worker
            return worker

    def __init__(self, ctx: RxContext) -> None:
        self._ctx = ctx
        self._device = ctx.device
        self._in_q: queue.Queue = queue.Queue(maxsize=4)
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="SionnaCudaRx", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=120):
            raise RuntimeError("Sionna CUDA 接收线程初始化超时")

    def _loop(self) -> None:
        import torch

        if str(self._device).startswith("cuda"):
            idx = int(str(self._device).split(":")[1]) if ":" in str(self._device) else 0
            torch.cuda.set_device(idx)

        dummy = np.zeros(self._ctx.packet_len_time, dtype=np.complex64)
        compute_delay_doppler_matrix(dummy, self._ctx, self._device)
        self._ready.set()

        while True:
            job = self._in_q.get()
            if job is None:
                break
            y_time, resp_q = job
            try:
                h_dd = compute_delay_doppler_matrix(y_time, self._ctx, self._device)
                resp_q.put(h_dd)
            except Exception as exc:
                resp_q.put(exc)

    def compute(self, y_time: np.ndarray) -> np.ndarray:
        resp_q: queue.Queue = queue.Queue(maxsize=1)
        self._in_q.put((y_time, resp_q))
        result = resp_q.get(timeout=120)
        if isinstance(result, Exception):
            raise result
        return result


class SionnaDelayDopplerRx(gr.basic_block):
    """Sionna 接收：GPU DD 谱；out0 复数 IQ（CFAR），out1 log10|·|²（谱图）。"""

    def __init__(
        self,
        config_file: str = "config/sensing_monostatic.toml",
        fft_len: int = 2048,
        ofdm_symbols: int = 512,
        cp_len: int = 512,
        subcarrier_spacing: float = 15000.0,
        center_freq: float = 6e9,
        length_tag_key: str = "packet_len",
        seed: int = 42,
        device: str = "cuda:0",
        log_eps: float = 1e-20,
        ctx: Optional[RxContext] = None,
        tx_packet: Optional[TxPacket] = None,
        burst_mode: bool = False,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna DD Rx",
            in_sig=[np.complex64],
            out_sig=[
                (np.complex64, int(fft_len)),
                (np.float32, int(fft_len)),
            ],
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

        cfg = str(resolve_config_path(config_file))
        gr_kw = dict(
            fft_len=int(fft_len),
            ofdm_symbols=int(ofdm_symbols),
            cp_len=int(cp_len),
            subcarrier_spacing=float(subcarrier_spacing),
            center_freq=float(center_freq),
        )
        log_gr_overrides(
            cfg,
            int(fft_len),
            int(ofdm_symbols),
            int(cp_len),
            float(subcarrier_spacing),
            float(center_freq),
            int(seed),
            str(device),
        )
        pkt = tx_packet or get_tx_packet(cfg, seed=int(seed), device=str(device), **gr_kw)
        self._ctx = ctx or build_rx_context(
            cfg, seed=int(seed), device=str(device), tx_packet=pkt, **gr_kw
        )
        self._worker = _CudaRxWorker.get(self._ctx)
        self._packet_len_time = self._ctx.packet_len_time
        self._packet_len_out = self._ctx.packet_len_freq
        self._fft_len = int(fft_len)
        self._log_eps = float(log_eps)
        self._tag_key = pmt.intern(length_tag_key)

        self._in_buf: List[np.complex64] = []
        self._pending_iq_rows: List[np.ndarray] = []
        self._pending_log_rows: List[np.ndarray] = []
        self._out_row_idx = 0
        self._burst_mode = bool(burst_mode)
        self._burst_armed = not self._burst_mode

        self.set_min_output_buffer(int(2 * int(ofdm_symbols)))

    def forecast(self, noutput_items, ninputs):
        return [1]

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out_iq = output_items[0]
        out_log = output_items[1]
        n_in = len(inp)
        n_out = min(len(out_iq), len(out_log))
        consumed = 0
        produced = 0

        while consumed < n_in:
            abs_idx = self.nitems_read(0) + consumed
            for tag in self.get_tags_in_window(0, abs_idx, 1):
                if tag.key == self._tag_key:
                    if len(self._in_buf) >= self._packet_len_time:
                        self._process_input_packet()
                        if self._burst_mode:
                            self._burst_armed = False
                    self._in_buf.clear()
                    if self._burst_mode:
                        self._burst_armed = True
            if self._burst_mode and not self._burst_armed:
                consumed += 1
                continue
            self._in_buf.append(inp[consumed])
            consumed += 1
            if len(self._in_buf) >= self._packet_len_time:
                self._process_input_packet()
                if self._burst_mode:
                    self._burst_armed = False

        while produced < n_out and self._pending_iq_rows:
            if self._out_row_idx == 0:
                tag = pmt.from_long(self._packet_len_out)
                wr = self.nitems_written(0) + produced
                self.add_item_tag(0, wr, self._tag_key, tag)
                self.add_item_tag(1, wr, self._tag_key, tag)
            out_iq[produced][:] = self._pending_iq_rows.pop(0)
            out_log[produced][:] = self._pending_log_rows.pop(0)
            self._out_row_idx += 1
            if self._out_row_idx >= self._packet_len_out:
                self._out_row_idx = 0
            produced += 1

        if consumed:
            self.consume(0, consumed)
        if produced:
            self.produce(0, produced)
            self.produce(1, produced)
        if consumed or produced:
            return gr.WORK_CALLED_PRODUCE
        return gr.WORK_DONE

    def _process_input_packet(self) -> None:
        y_time = np.array(self._in_buf[: self._packet_len_time], dtype=np.complex64)
        self._in_buf = self._in_buf[self._packet_len_time :]
        try:
            h_dd = self._worker.compute(y_time)
        except Exception as exc:
            print(f"Sionna DD Rx GPU 处理失败: {exc}")
            return
        if h_dd.shape != (self._packet_len_out, self._fft_len):
            print(
                f"Sionna DD Rx 形状异常 {h_dd.shape}, "
                f"期望 ({self._packet_len_out}, {self._fft_len})"
            )
            return
        iq, log_mag = prepare_dd_outputs(h_dd, log_eps=self._log_eps)
        self._pending_iq_rows.extend(row for row in iq)
        self._pending_log_rows.extend(row for row in log_mag)
        self._out_row_idx = 0
