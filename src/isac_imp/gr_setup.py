"""GRC 共享 System 配置：对齐 run_sensing_monostatic，不复用 gnuradio/core/gr_system。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping

import sionna
import torch

from isac.data_structures import SystemComponents, SystemParams
from isac.system import System
from isac.utils import load_config, set_random_seed

_OfdmOverrides = Mapping[str, Any] | None


def _log_override(label: str, toml_val, grc_val) -> None:
    print(f"  [config] GRC 覆盖 TOML: {label} {toml_val}→{grc_val}")


def apply_grc_ofdm_overrides(
    raw: dict,
    *,
    num_symbols: int,
    fft_size: int,
    subcarrier_spacing: float,
    cp_len: int,
    log: bool = True,
) -> dict:
    """将 GRC 变量合并进 TOML 字典的 [ofdm] 段。"""
    merged = dict(raw)
    toml_ofdm = dict(merged.get("ofdm") or {})
    if log:
        if toml_ofdm.get("num_symbols") != num_symbols:
            _log_override("num_symbols", toml_ofdm.get("num_symbols"), num_symbols)
        if toml_ofdm.get("fft_size") != fft_size:
            _log_override("fft_size", toml_ofdm.get("fft_size"), fft_size)
        if float(toml_ofdm.get("subcarrier_spacing", 0)) != subcarrier_spacing:
            _log_override(
                "subcarrier_spacing",
                toml_ofdm.get("subcarrier_spacing"),
                subcarrier_spacing,
            )
        if int(toml_ofdm.get("cyclic_prefix_length", 0)) != cp_len:
            _log_override(
                "cp",
                toml_ofdm.get("cyclic_prefix_length", 0),
                cp_len,
            )
    ofdm = dict(toml_ofdm)
    ofdm["num_symbols"] = int(num_symbols)
    ofdm["fft_size"] = int(fft_size)
    ofdm["subcarrier_spacing"] = float(subcarrier_spacing)
    ofdm["cyclic_prefix_length"] = int(cp_len)
    merged["ofdm"] = ofdm
    return merged


def _normalize_ofdm_overrides(ofdm_overrides: _OfdmOverrides) -> tuple[int, int, float, int]:
    if ofdm_overrides is None:
        return (0, 0, 0.0, 0)
    return (
        int(ofdm_overrides["num_symbols"]),
        int(ofdm_overrides["fft_size"]),
        float(ofdm_overrides["subcarrier_spacing"]),
        int(ofdm_overrides["cyclic_prefix_length"]),
    )


def make_cache_key(
    config_file: str,
    device: str,
    seed: int,
    ofdm_overrides: _OfdmOverrides = None,
) -> tuple:
    """收发端共享 System 的 registry 键。"""
    n_sym, fft_size, scs, cp = _normalize_ofdm_overrides(ofdm_overrides)
    return (str(config_file), str(device), int(seed), n_sym, fft_size, scs, cp)


@dataclass
class SystemSession:
    """缓存的 System 及发端参考网格（供阶段二 sensing 使用）。"""

    system: System
    cache_key: tuple
    last_x_rg: Any = None


_SYSTEM_REGISTRY: dict[tuple, SystemSession] = {}


def invalidate_system_cache(cache_key: tuple | None = None) -> None:
    """按 key 清除 registry；省略 key 时清空全部。"""
    if cache_key is None:
        _SYSTEM_REGISTRY.clear()
        return
    _SYSTEM_REGISTRY.pop(cache_key, None)


def set_last_x_rg(system: System, x_rg) -> None:
    """TX transmit() 后写入参考频域网格。"""
    for session in _SYSTEM_REGISTRY.values():
        if session.system is system:
            session.last_x_rg = x_rg
            return


def get_last_x_rg(system: System):
    """读取与 system 对应 session 中的 last_x_rg。"""
    for session in _SYSTEM_REGISTRY.values():
        if session.system is system:
            return session.last_x_rg
    return None


def _build_system(
    config_file: str,
    *,
    device: str,
    seed: int,
    batch_size: int,
    ofdm_overrides: _OfdmOverrides,
    include_channel: bool,
) -> System:
    set_random_seed(seed)
    sionna.phy.config.device = device
    torch.set_num_threads(1)

    raw = load_config(config_file)
    if not include_channel:
        raw = dict(raw)
        raw.pop("channel", None)
    if ofdm_overrides is not None:
        raw = apply_grc_ofdm_overrides(
            raw,
            num_symbols=int(ofdm_overrides["num_symbols"]),
            fft_size=int(ofdm_overrides["fft_size"]),
            subcarrier_spacing=float(ofdm_overrides["subcarrier_spacing"]),
            cp_len=int(ofdm_overrides["cyclic_prefix_length"]),
        )

    args = argparse.Namespace(
        config_file=str(config_file),
        device=str(device),
        batch_size=int(batch_size),
    )
    # 合并 GRC 覆盖后直接绑定配置，避免 System.__init__ 用未合并 TOML 先构建组件
    system = System.__new__(System)
    system.args = args
    system.device = device
    system.config = raw
    system.params = SystemParams.from_dict(raw)
    system.components = SystemComponents.build_from_params(
        system.params, device=device
    )
    return system


def create_system(
    config_file: str,
    *,
    device: str = "cpu",
    seed: int = 42,
    batch_size: int = 1,
    ofdm_overrides: _OfdmOverrides = None,
    use_cache: bool = True,
    include_channel: bool = False,
) -> System:
    """
    构建或与 registry 共享 System 实例（对齐 run_sensing_monostatic 配置路径）。

    相同 cache key 下 TX/RX epy_block 得到同一 System 对象。
    GRC USRP OTA 默认 ``include_channel=False``（空口不经 RT 仿真信道）。
    """
    cache_key = make_cache_key(config_file, device, seed, ofdm_overrides)
    if use_cache and cache_key in _SYSTEM_REGISTRY:
        return _SYSTEM_REGISTRY[cache_key].system

    system = _build_system(
        config_file,
        device=device,
        seed=seed,
        batch_size=batch_size,
        ofdm_overrides=ofdm_overrides,
        include_channel=include_channel,
    )
    if use_cache:
        _SYSTEM_REGISTRY[cache_key] = SystemSession(system=system, cache_key=cache_key)
    return system
