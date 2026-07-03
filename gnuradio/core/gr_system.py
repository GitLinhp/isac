"""GRC 共享 System 上下文：发端 / RT 时域信道 / 接收感知与 run_sensing_baseline 对齐。"""
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from bootstrap import setup_gnuradio_paths_from

setup_gnuradio_paths_from(__file__)

import numpy as np

from gr_config import (  # noqa: E402
    EffectiveConfig,
    GrcOverrides,
    grc_overrides_from_grc_vars,
    merge_config,
    resolve_config_path,
)
from isac.data_structures import SystemComponents  # noqa: E402
from isac.system import System  # noqa: E402
from isac.utils import load_config, set_random_seed  # noqa: E402

_SNR_UNSET = object()


def ofdm_packet_len_time(num_symbols: int, fft_size: int, cp_len: int) -> int:
    """一帧 OFDM CPI 时域样点数（cp_len=0 时不加 CP）。"""
    return int(num_symbols) * (int(fft_size) + int(cp_len))


@dataclass
class GrSystemContext:
    """按 EffectiveConfig 缓存的 System 封装。"""

    effective: EffectiveConfig
    system: System
    snr_db: float
    packet_len_time: int
    packet_len_freq: int
    fft_len: int
    device: str
    _x_rg: object = None
    _x_time_shape: Tuple[int, ...] = ()
    _rx_packet_len: int = 0
    _rx_time_shape: Tuple[int, ...] = ()

    @classmethod
    def from_effective(cls, effective: EffectiveConfig) -> "GrSystemContext":
        import sionna

        set_random_seed(effective.seed)
        sionna.phy.config.device = effective.device

        raw = load_config(effective.config_path)
        system = System(
            config=raw,
            device=effective.device,
        )
        system.params = effective.system_params
        system.components = SystemComponents.build_from_params(
            effective.system_params, device=effective.device
        )

        params = effective.system_params
        cp = params.ofdm.cyclic_prefix_length
        n_sym = params.ofdm.num_symbols
        fft_size = params.ofdm.fft_size
        snr_db = float(params.channel.snr_db) if params.channel is not None else 0.0

        ctx = cls(
            effective=effective,
            system=system,
            snr_db=snr_db,
            packet_len_time=ofdm_packet_len_time(n_sym, fft_size, cp),
            packet_len_freq=int(n_sym),
            fft_len=int(fft_size),
            device=str(effective.device),
        )
        ctx._apply_baseline_scene()
        return ctx

    def _apply_baseline_scene(self) -> None:
        rt = self.system.components.rt_simulator
        if rt is None:
            return
        try:
            rt.get("reflector").velocity = [0, 0, -20]
            rt.get("bs1_tx").velocity = [30, 0, 0]
        except (KeyError, AttributeError, TypeError):
            pass

    def print_rt_paths(self) -> None:
        rt = self.system.components.rt_simulator
        if rt is None or rt.paths is None:
            return
        paths = rt.paths
        print("Delay - LoS Path (ns) :", paths.tau[0, 0, 0] / 1e-9)
        print("Doppler - LoS Path (Hz) :", paths.doppler[0, 0, 0])
        if paths.tau.shape[-1] > 1:
            print(
                "Delay - Reflected Path (ns) :",
                paths.tau[0, 0, 1].numpy() / 1e-9,
            )
            print(
                "Doppler - Reflected Path (Hz) :",
                paths.doppler[0, 0, 1],
            )

    def transmit_tensors(self):
        """调用 ``System.transmit()`` 并缓存 ``x_rg`` / 时域形状。"""
        _, x_rg, x_time = self.system.transmit()
        self._x_rg = x_rg
        self._x_time_shape = tuple(x_time.shape)
        return x_rg, x_time

    def transmit_packet(self):
        """``System.transmit()`` → ``TxPacket``（延迟 import 避免循环依赖）。"""
        from sionna_tx import TxPacket

        x_rg, x_time = self.transmit_tensors()

        x_rg_np = x_rg.squeeze().detach().cpu().numpy().astype(np.complex64)
        x_time_np = x_time.squeeze().detach().cpu().numpy().astype(np.complex64)
        if x_time_np.size != self.packet_len_time:
            raise RuntimeError(
                f"unexpected time length {x_time_np.size}, "
                f"expected {self.packet_len_time}"
            )

        return TxPacket(
            x_rg=x_rg_np,
            time=x_time_np,
            freq_ref=np.fft.fftshift(x_rg_np, axes=-1),
            packet_len_time=self.packet_len_time,
            packet_len_freq=self.packet_len_freq,
        )

    def _time_tensor(self, samples: np.ndarray, *, expect_tx_len: bool = True):
        import torch

        y = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
        expected = self.packet_len_time if expect_tx_len else self._rx_packet_len
        if expected and y.size != expected:
            raise ValueError(f"时域长度 {y.size} != 期望 {expected}")
        t = torch.from_numpy(y).to(device=self.device, dtype=torch.complex64)
        shape = self._x_time_shape if expect_tx_len else self._rx_time_shape
        if shape:
            t = t.reshape(shape)
        return t

    def apply_channel_time(
        self,
        tx_np: np.ndarray,
        snr_db: Union[float, None, object] = _SNR_UNSET,
    ) -> np.ndarray:
        """时域 RT 信道 + AWGN（等价 baseline ``--domain time`` 信道步）。

        ``snr_db`` 省略时用 TOML ``channel.snr_db``；显式 ``None`` 表示不加噪。
        """
        import torch

        if self._x_rg is None:
            self.transmit_tensors()

        if snr_db is _SNR_UNSET:
            snr: Optional[float] = self.snr_db
        else:
            snr = snr_db
        x_time = self._time_tensor(tx_np, expect_tx_len=True)
        ctx = (
            torch.cuda.device(self.device)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with ctx:
            y_time = self.system.components.channel(
                x_time, domain="time", snr_db=snr
            )
        self._rx_time_shape = tuple(y_time.shape)
        y_np = y_time.squeeze().detach().cpu().numpy().astype(np.complex64)
        self._rx_packet_len = int(y_np.size)
        return y_np

    def compute_delay_doppler(self, y_time_np: np.ndarray) -> np.ndarray:
        """解调 + ``System.compute_sensing_spectrum`` → DD 矩阵。"""
        import torch

        if self._x_rg is None:
            self.transmit_tensors()
        if not self._rx_packet_len:
            raise RuntimeError("须先调用 apply_channel_time 以确定 RX 包长")

        y_time = self._time_tensor(y_time_np, expect_tx_len=False)
        ctx = (
            torch.cuda.device(self.device)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with ctx:
            y_rg = self.system.components.demodulator(y_time).squeeze()
            h_dd = self.system.compute_sensing_spectrum(self._x_rg, y_rg)
        return h_dd.detach().cpu().numpy().astype(np.complex64)

    @property
    def rx_packet_len(self) -> int:
        """RT 时域信道输出样点数；未探测前回退为发端 CPI 长度。"""
        return self._rx_packet_len if self._rx_packet_len else self.packet_len_time

    def ensure_rx_geometry(self) -> int:
        """经一次无噪 RT 信道探测 RX 包长（bootstrap / DD Rx 预热前调用）。"""
        if self._rx_packet_len:
            return self._rx_packet_len
        pkt = self.transmit_packet()
        self.apply_channel_time(pkt.time, snr_db=None)
        return self._rx_packet_len


@lru_cache(maxsize=8)
def _get_context_cached(
    cache_key: Tuple[str, int, int, int, int, float, float, int, str],
) -> GrSystemContext:
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
    return GrSystemContext.from_effective(effective)


def get_gr_system_context(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> GrSystemContext:
    config_path = resolve_config_path(config_file)
    if overrides is None and grc_kw:
        overrides = grc_overrides_from_grc_vars(
            seed=int(seed), device=str(device), **grc_kw
        )
    if overrides is None:
        from isac.data_structures import SystemParams
        from isac.utils import load_config

        raw = load_config(config_path)
        params = SystemParams.from_dict(raw)
        overrides = GrcOverrides(
            fft_len=params.ofdm.fft_size,
            ofdm_symbols=params.ofdm.num_symbols,
            cp_len=params.ofdm.cyclic_prefix_length,
            subcarrier_spacing=params.ofdm.subcarrier_spacing,
            center_freq=params.carrier_frequency,
            seed=int(seed),
            device=str(device),
        )
    effective = merge_config(str(config_path), overrides)
    return _get_context_cached(effective.cache_key())


_warmed_keys: set[Tuple[str, int, int, int, int, float, float, int, str]] = set()


def prewarm_gr_system(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> GrSystemContext:
    """GPU 预热：transmit → channel → DD（与 bootstrap 一致）。"""
    ctx = get_gr_system_context(
        config_file, seed=seed, device=device, overrides=overrides, **grc_kw
    )
    key = ctx.effective.cache_key()
    first = key not in _warmed_keys
    if first:
        print(f"GrSystemContext：GPU 预热 (device={device}) ...")
        _warmed_keys.add(key)
    else:
        print(f"GrSystemContext：加载缓存 (device={device})")

    pkt = ctx.transmit_packet()
    y_clean = ctx.apply_channel_time(pkt.time, snr_db=None)
    _ = ctx.compute_delay_doppler(y_clean)

    if first:
        print(
            f"  TX 时域 {ctx.packet_len_time} 样点, "
            f"RX 时域 {ctx.rx_packet_len} 样点 → "
            f"谱矩阵 {ctx.packet_len_freq}×{ctx.fft_len}"
        )
        ctx.print_rt_paths()
    return ctx
