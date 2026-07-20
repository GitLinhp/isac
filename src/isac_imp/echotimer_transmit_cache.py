"""Echotimer 双设备流图：从 TransmitCache 重放 x_time / x_rg（packet_len tag）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pmt
from gnuradio import gr

import tomli

from isac import PROJECT_ROOT
from isac_imp.burst_pack import TPP_DONT, load_burst_buffer

_LOG_PREFIX = "[EchotimerTransmitCache]"
_DEFAULT_CONFIG = "implementaion/ofdm_echotimer_dd.toml"
_CONFIG_DIR = PROJECT_ROOT / "config"


def _load_config(config_file: str) -> dict:
    path = Path(config_file)
    if not path.is_absolute():
        for candidate in (_CONFIG_DIR / path, PROJECT_ROOT / path):
            if candidate.is_file():
                path = candidate
                break
        else:
            path = _CONFIG_DIR / path
    with open(path, "rb") as handle:
        return tomli.load(handle)


def _resolve_cache_dir(config_file: str) -> Path:
    raw = _load_config(config_file)
    src = raw.get("source") or {}
    cache_file = src.get("cache_file")
    if not cache_file:
        raise ValueError(f"TOML 未配置 source.cache_file: {config_file}")
    path = Path(str(cache_file))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _validate_geometry(
    config_file: str,
    *,
    transpose_len: int,
    fft_len: int,
    subcarrier_spacing: float,
    cp_len: int,
) -> None:
    raw = _load_config(config_file)
    ofdm = dict(raw.get("ofdm") or {})
    ofdm["num_symbols"] = int(transpose_len)
    ofdm["fft_size"] = int(fft_len)
    ofdm["subcarrier_spacing"] = float(subcarrier_spacing)
    ofdm["cyclic_prefix_length"] = int(cp_len)
    toml = raw.get("ofdm") or {}
    n_sym = int(toml.get("num_symbols", 0))
    fft_size = int(toml.get("fft_size", 0))
    cp = int(toml.get("cyclic_prefix_length", 0))
    scs = float(toml.get("subcarrier_spacing", 0.0))
    if (n_sym, fft_size, cp, scs) != (transpose_len, fft_len, cp_len, subcarrier_spacing):
        raise ValueError(
            f"GRC OFDM 参数与 TOML 不一致: "
            f"GRC=({transpose_len},{fft_len},{cp_len},{subcarrier_spacing}) "
            f"TOML=({n_sym},{fft_size},{cp},{scs})"
        )


class EchotimerTransmitCacheBlock(gr.basic_block):
    """双输出缓存发射源：out0 时域 x_time，out1 频域参考 x_rg（fftshift）。

    - out0：标量 complex64，CPI 首样点打 ``length_tag_key``，值为 ``burst_len_samples``
    - out1：vlen=fft_len 向量，CPI 首符号打 ``length_tag_key``，值为 ``transpose_len``
    """

    def __init__(
        self,
        config_file: str = _DEFAULT_CONFIG,
        length_tag_key: str = "packet_len",
        fft_len: int = 2048,
        transpose_len: int = 4,
        subcarrier_spacing: float = 60e3,
        cp_len: int = 512,
    ) -> None:
        gr.basic_block.__init__(
            self,
            name="Echotimer Transmit Cache",
            in_sig=None,
            out_sig=[np.complex64, (np.complex64, int(fft_len))],
        )
        self._config_file = str(config_file)
        self._length_tag_key = pmt.intern(length_tag_key)
        self._fft_len = int(fft_len)
        self._transpose_len = int(transpose_len)
        self._subcarrier_spacing = float(subcarrier_spacing)
        self._cp_len = int(cp_len)
        self._sym_samples = self._fft_len + self._cp_len
        self._burst_len_samples = self._transpose_len * self._sym_samples

        self._time_buf: np.ndarray | None = None
        self._freq_buf: np.ndarray | None = None
        self._time_idx = 0
        self._sym_idx = 0

        self.set_tag_propagation_policy(TPP_DONT)
        self.set_min_output_buffer(max(self._burst_len_samples * 2, self._transpose_len * 2))

    def _log(self, msg: str) -> None:
        print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)

    def _load_buffers(self) -> None:
        cache_dir = _resolve_cache_dir(self._config_file)
        x_time_path = cache_dir / "x_time.npy"
        x_rg_path = cache_dir / "x_rg.npy"
        if not x_time_path.is_file() or not x_rg_path.is_file():
            raise FileNotFoundError(
                f"发射缓存不完整: {cache_dir}；请先运行 "
                f"script/implementation/generate_transmit_cache.py "
                f"--config_file config/{self._config_file}"
            )

        _validate_geometry(
            self._config_file,
            transpose_len=self._transpose_len,
            fft_len=self._fft_len,
            subcarrier_spacing=self._subcarrier_spacing,
            cp_len=self._cp_len,
        )
        time_buf = load_burst_buffer(x_time_path, tx_amp=1.0)
        if time_buf.size != self._burst_len_samples:
            raise ValueError(
                f"x_time.npy 样点数 {time_buf.size} != 期望 {self._burst_len_samples} "
                f"(transpose_len*(fft_len+cp_len))"
            )

        x_rg = np.asarray(np.load(x_rg_path))
        freq = np.fft.fftshift(x_rg.squeeze(), axes=-1).astype(np.complex64, copy=False)
        if freq.ndim > 2:
            freq = freq.reshape(-1, freq.shape[-1])
        if freq.ndim == 1:
            freq = freq.reshape(1, -1)
        if freq.shape[-1] != self._fft_len:
            raise ValueError(
                f"x_rg.npy 末维 {freq.shape[-1]} != fft_len {self._fft_len}"
            )
        if freq.shape[0] != self._transpose_len:
            raise ValueError(
                f"x_rg.npy 符号数 {freq.shape[0]} != transpose_len {self._transpose_len}"
            )

        self._time_buf = time_buf
        self._freq_buf = freq
        self._time_idx = 0
        self._sym_idx = 0
        self._log(
            f"loaded cache_dir={cache_dir} burst_len={self._burst_len_samples} "
            f"symbols={self._transpose_len} fft_len={self._fft_len}"
        )

    def start(self) -> bool:
        self._load_buffers()
        return True

    def forecast(self, noutput_items: int, ninputs) -> list:
        del noutput_items, ninputs
        return []

    def general_work(self, input_items, output_items) -> int:
        del input_items
        if self._time_buf is None or self._freq_buf is None:
            return 0

        out_time = output_items[0]
        out_freq = output_items[1]
        max_time = len(out_time)
        max_freq = len(out_freq)

        n_time = 0
        n_freq = 0
        abs_time_base = self.nitems_written(0)
        abs_freq_base = self.nitems_written(1)

        while n_time < max_time:
            if self._time_idx == 0:
                self.add_item_tag(
                    0,
                    abs_time_base + n_time,
                    self._length_tag_key,
                    pmt.from_long(self._burst_len_samples),
                )
            out_time[n_time] = self._time_buf[self._time_idx]
            n_time += 1
            self._time_idx += 1
            if self._time_idx >= self._burst_len_samples:
                self._time_idx = 0

        while n_freq < max_freq:
            if self._sym_idx == 0:
                self.add_item_tag(
                    1,
                    abs_freq_base + n_freq,
                    self._length_tag_key,
                    pmt.from_long(self._transpose_len),
                )
            out_freq[n_freq][:] = self._freq_buf[self._sym_idx]
            n_freq += 1
            self._sym_idx += 1
            if self._sym_idx >= self._transpose_len:
                self._sym_idx = 0

        if n_freq > 0:
            self.produce(1, n_freq)
        if n_time > 0:
            return n_time
        if n_freq > 0:
            return gr.WORK_CALLED_PRODUCE
        return 0
