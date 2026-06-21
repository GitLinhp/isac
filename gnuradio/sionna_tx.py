"""Sionna ZC 发端：启动时 GPU 生成一次，运行时循环输出缓存时域波形。"""
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _ensure_gr_buffer(min_buffer: int = 8388608) -> None:
    """大 CPI buffer 须在 import gnuradio 之前配置。"""
    if os.environ.get("ISAC_GR_BUFFER_BOOTSTRAP"):
        return
    gr_home = Path(tempfile.mkdtemp(prefix="isac_gr_"))
    conf_dir = gr_home / ".gnuradio"
    conf_dir.mkdir()
    (conf_dir / "config.conf").write_text(
        f"[DEFAULT]\nbuffer_size = {min_buffer}\n",
        encoding="utf-8",
    )
    os.environ["HOME"] = str(gr_home)
    os.environ["ISAC_GR_BUFFER_BOOTSTRAP"] = "1"


_ensure_gr_buffer()

import numpy as np
import pmt
from gnuradio import gr

from isac.data_structures import SystemParams
from isac.utils import load_config, set_random_seed


@dataclass(frozen=True)
class TxPacket:
    """一帧 OFDM 发端数据。"""

    x_rg: np.ndarray
    time: np.ndarray
    freq_ref: np.ndarray
    packet_len_time: int
    packet_len_freq: int


from gr_config import (
    EffectiveConfig,
    GrcOverrides,
    grc_overrides_from_grc_vars,
    merge_config,
    resolve_config_path,
)


def print_sensing_performance(
    bandwidth: float,
    range_resolution: float,
    doppler_resolution: float,
    velocity_resolution: float,
    R_max: float,
    v_max: float,
    *,
    title: str = "感知性能参数 (GRC 变量)",
) -> None:
    delay_ns = 1 / bandwidth * 1e9 if bandwidth > 0 else 0.0
    print(title + ":")
    rows = [
        ("时间分辨率", f"{delay_ns:.2f}", "ns"),
        ("距离分辨率", f"{range_resolution:.2f}", "m"),
        ("多普勒分辨率", f"{doppler_resolution:.2f}", "Hz"),
        ("速度分辨率", f"{velocity_resolution:.2f}", "m/s"),
        ("最大探测距离", f"{R_max:.2f}", "m"),
        ("最大探测速度", f"{v_max:.2f}", "m/s"),
    ]
    for name, val, unit in rows:
        print(f"  {name}: {val} {unit}")


@lru_cache(maxsize=8)
def _build_tx_packet_cached(
    cache_key: Tuple[str, int, int, int, int, float, float, int, str],
) -> TxPacket:
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
    scs = float(scs_int)
    center_freq = float(cf_int)
    overrides = GrcOverrides(
        fft_len=fft_len,
        ofdm_symbols=n_sym,
        cp_len=cp,
        subcarrier_spacing=scs,
        center_freq=center_freq,
        seed=seed,
        device=device,
    )
    effective = merge_config(str(config_path), overrides)
    return _build_tx_packet_impl(effective)


def get_tx_packet(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    overrides: Optional[GrcOverrides] = None,
    *,
    fft_len: Optional[int] = None,
    ofdm_symbols: Optional[int] = None,
    cp_len: Optional[int] = None,
    subcarrier_spacing: Optional[float] = None,
    center_freq: Optional[float] = None,
) -> TxPacket:
    """获取已缓存的发端包；无 overrides 时按 TOML 原值构建 GrcOverrides。"""
    config_path = resolve_config_path(config_file)
    if overrides is None:
        raw = load_config(config_path)
        params = SystemParams.from_dict(raw)
        overrides = GrcOverrides(
            fft_len=int(fft_len if fft_len is not None else params.ofdm.fft_size),
            ofdm_symbols=int(
                ofdm_symbols if ofdm_symbols is not None else params.ofdm.num_symbols
            ),
            cp_len=int(cp_len if cp_len is not None else params.ofdm.cyclic_prefix_length),
            subcarrier_spacing=float(
                subcarrier_spacing
                if subcarrier_spacing is not None
                else params.ofdm.subcarrier_spacing
            ),
            center_freq=float(
                center_freq if center_freq is not None else params.carrier_frequency
            ),
            seed=int(seed),
            device=str(device),
        )
    effective = merge_config(str(config_path), overrides)
    return _build_tx_packet_cached(effective.cache_key())


_warmed_tx_keys: set[Tuple[str, int, int, int, int, float, float, int, str]] = set()


def bootstrap_sionna(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> TxPacket:
    """启动时一次性预热发端与收端 GPU 上下文。"""
    from sionna_rx import prewarm_sionna_rx

    print("=== Sionna 信号引导（仅执行一次）===")
    pkt = prewarm_sionna_tx(
        config_file, seed=seed, device=device, overrides=overrides, **grc_kw
    )
    prewarm_sionna_rx(
        config_file, seed=seed, device=device, tx_packet=pkt, overrides=overrides, **grc_kw
    )
    print("=== Sionna 引导完成，后续帧将复用缓存 ===")
    return pkt


def prewarm_sionna_tx(
    config_file: str,
    seed: int = 42,
    device: str = "cuda:0",
    overrides: Optional[GrcOverrides] = None,
    **grc_kw,
) -> TxPacket:
    """启动时在 GPU 上生成发端包（仅首次真正计算）。"""
    pkt = get_tx_packet(
        config_file,
        seed=seed,
        device=device,
        overrides=overrides,
        **{k: v for k, v in grc_kw.items() if k in (
            "fft_len", "ofdm_symbols", "cp_len", "subcarrier_spacing", "center_freq"
        )},
    )
    config_path = resolve_config_path(config_file)
    if overrides is None and grc_kw:
        overrides = grc_overrides_from_grc_vars(seed=int(seed), device=str(device), **grc_kw)
    elif overrides is None:
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
    key = effective.cache_key()
    first = key not in _warmed_tx_keys
    if first:
        print(f"Sionna 发端：GPU 生成 (device={device}) ...")
        _warmed_tx_keys.add(key)
    else:
        print(f"Sionna 发端：加载缓存 (device={device})")
    if first:
        print(
            f"  时域 {pkt.packet_len_time} 样点, "
            f"频域 {pkt.packet_len_freq}×{pkt.freq_ref.shape[-1]}"
        )
    return pkt


def _build_tx_packet_impl(effective: EffectiveConfig) -> TxPacket:
    import sionna

    from isac.data_structures import SystemComponents

    params = effective.system_params
    seed = effective.seed
    device = effective.device
    set_random_seed(seed)
    sionna.phy.config.device = device

    from isac.data_structures import SystemComponents
    rg = comps.rg

    if params.source.type != "zc":
        raise ValueError(
            f"sionna_tx only supports source.type='zc', "
            f"got {params.source.type!r}"
        )
    if comps.zc_source is None:
        raise RuntimeError("zc_source missing for ZC sensing source")

    x = comps.zc_source([1, 1, 1, rg.num_data_symbols])
    x_rg_t = comps.rg_mapper(x).squeeze()
    x_rg = x_rg_t.cpu().numpy().astype(np.complex64)

    x_batch = x_rg_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_time = comps.modulator(x_batch).squeeze().cpu().numpy().astype(np.complex64)

    cp = params.ofdm.cyclic_prefix_length
    n_sym = params.ofdm.num_symbols
    fft_size = params.ofdm.fft_size
    expected_time = n_sym * (fft_size + cp)
    if x_time.size != expected_time:
        raise RuntimeError(
            f"unexpected time length {x_time.size}, expected {expected_time}"
        )

    return TxPacket(
        x_rg=x_rg,
        time=x_time,
        freq_ref=np.fft.fftshift(x_rg, axes=-1),
        packet_len_time=expected_time,
        packet_len_freq=n_sym,
    )


_logged_override_keys: set[Tuple[str, int, int, int, int, float, float, int, str]] = set()


def log_gr_overrides(
    config_file: str,
    fft_len: int,
    ofdm_symbols: int,
    cp_len: int,
    subcarrier_spacing: float,
    center_freq: float,
    seed: int,
    device: str,
) -> EffectiveConfig:
    """GRC 与 TOML 不一致时打印 info 级覆盖日志（不告警）。"""
    overrides = grc_overrides_from_grc_vars(
        fft_len=fft_len,
        ofdm_symbols=ofdm_symbols,
        cp_len=cp_len,
        subcarrier_spacing=subcarrier_spacing,
        center_freq=center_freq,
        seed=seed,
        device=device,
    )
    effective = merge_config(config_file, overrides)
    key = effective.cache_key()
    if key not in _logged_override_keys:
        _logged_override_keys.add(key)
        if effective.override_log:
            for line in effective.override_log:
                print(f"  [config] {line}")
    return effective


class SionnaBootstrap(gr.basic_block):
    """GRC 启动块：GPU 预热 + 打印 EffectiveConfig 感知性能（无 IO）。"""

    def __init__(
        self,
        config_file: str = "config/sensing_monostatic.toml",
        seed: int = 42,
        device: str = "cuda:0",
        fft_len: int = 2048,
        ofdm_symbols: int = 512,
        cp_len: int = 512,
        subcarrier_spacing: float = 15000.0,
        center_freq: float = 6e9,
    ) -> None:
        gr.basic_block.__init__(self, name="Sionna Bootstrap", in_sig=None, out_sig=None)
        cfg = str(resolve_config_path(config_file))
        gr_kw = dict(
            fft_len=int(fft_len),
            ofdm_symbols=int(ofdm_symbols),
            cp_len=int(cp_len),
            subcarrier_spacing=float(subcarrier_spacing),
            center_freq=float(center_freq),
        )
        bootstrap_sionna(cfg, seed=int(seed), device=str(device), **gr_kw)
        effective = log_gr_overrides(
            cfg,
            int(fft_len),
            int(ofdm_symbols),
            int(cp_len),
            float(subcarrier_spacing),
            float(center_freq),
            int(seed),
            str(device),
        )
        print_sensing_performance(
            effective.bandwidth,
            effective.range_resolution,
            effective.doppler_resolution,
            effective.velocity_resolution,
            effective.R_max,
            effective.v_max,
            title="感知性能参数 (EffectiveConfig，Sionna 实际使用)",
        )

    def forecast(self, noutput_items, ninputs):
        return []

    def general_work(self, input_items, output_items):
        return gr.WORK_DONE


class SionnaOFDMTx(gr.basic_block):
    """输出缓存时域波形，带 ``packet_len`` tag；可选 PRI burst（CPI + 静默零填充）。"""

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
        packet: Optional[TxPacket] = None,
        burst_pri_sec: float = 0.0,
        samp_rate: int = 0,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Sionna OFDM Tx",
            in_sig=None,
            out_sig=[np.complex64],
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
        pkt = packet or get_tx_packet(
            cfg, seed=int(seed), device=str(device), **gr_kw
        )

        self._time = np.ascontiguousarray(pkt.time, dtype=np.complex64)
        self._packet_len_time = pkt.packet_len_time
        self._time_idx = 0
        self._tag_key = pmt.intern(length_tag_key)
        self._burst_mode = float(burst_pri_sec) > 0.0
        self._phase = "tx"
        self._idle_remaining = 0

        if self._burst_mode:
            if int(samp_rate) <= 0:
                raise ValueError("burst 模式须指定 samp_rate > 0")
            pri_samples = int(float(burst_pri_sec) * int(samp_rate))
            if pri_samples <= self._packet_len_time:
                raise ValueError(
                    f"PRI 样点数 {pri_samples} 须大于 CPI 样点数 "
                    f"{self._packet_len_time}（增大 burst_pri_sec 或检查 samp_rate）"
                )
            self._idle_samples = pri_samples - self._packet_len_time

        gr_cp = max(int(cp_len), 1)
        self.set_min_output_buffer(int(2 * int(ofdm_symbols) * (int(fft_len) + gr_cp)))

    def forecast(self, noutput_items, ninputs):
        return []

    def _emit_tx_sample(self, out, produced: int) -> None:
        if self._time_idx == 0:
            self.add_item_tag(
                0,
                self.nitems_written(0) + produced,
                self._tag_key,
                pmt.from_long(self._packet_len_time),
            )
        out[produced] = self._time[self._time_idx]
        self._time_idx += 1

    def general_work(self, input_items, output_items):
        out = output_items[0]
        max_out = len(out)
        produced = 0

        if not self._burst_mode:
            while produced < max_out:
                self._emit_tx_sample(out, produced)
                self._time_idx %= self._packet_len_time
                produced += 1
        else:
            while produced < max_out:
                if self._phase == "tx":
                    self._emit_tx_sample(out, produced)
                    produced += 1
                    if self._time_idx >= self._packet_len_time:
                        self._time_idx = 0
                        self._phase = "idle"
                        self._idle_remaining = self._idle_samples
                else:
                    n_idle = min(max_out - produced, self._idle_remaining)
                    if n_idle > 0:
                        out[produced : produced + n_idle] = 0
                        produced += n_idle
                        self._idle_remaining -= n_idle
                    if self._idle_remaining == 0:
                        self._phase = "tx"
                        self._time_idx = 0

        if produced:
            self.produce(0, produced)
            return gr.WORK_CALLED_PRODUCE
        return gr.WORK_DONE
