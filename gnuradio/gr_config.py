"""GRC 优先 + TOML 基线：merge 后供 Sionna 块使用的有效配置。"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from isac.data_structures.params import SystemParams
from isac.sensing.sensing_performance import SensingPerformance
from isac.utils import load_config


_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_config_path(config_file: str) -> Path:
    path = Path(config_file)
    if path.is_absolute():
        return path
    candidate = (_REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    if path.parts and path.parts[0] == "..":
        return (_REPO_ROOT / Path(*path.parts[1:])).resolve()
    return candidate


@dataclass(frozen=True)
class GrcOverrides:
    fft_len: int
    ofdm_symbols: int
    cp_len: int
    subcarrier_spacing: float
    center_freq: float
    seed: int
    device: str


@dataclass(frozen=True)
class EffectiveConfig:
    config_path: Path
    system_params: SystemParams
    fft_len: int
    ofdm_symbols: int
    cp_len: int
    subcarrier_spacing: float
    center_freq: float
    seed: int
    device: str
    samp_rate: int
    bandwidth: float
    symbol_duration: float
    range_resolution: float
    doppler_resolution: float
    velocity_resolution: float
    R_max: float
    v_max: float
    transpose_len: int
    override_log: Tuple[str, ...]

    def cache_key(self) -> Tuple[str, int, int, int, int, float, float, int, str]:
        return (
            str(self.config_path),
            self.fft_len,
            self.ofdm_symbols,
            self.cp_len,
            int(self.subcarrier_spacing),
            int(self.center_freq),
            int(self.seed),
            str(self.device),
        )


def _toml_path_solver_seed(config: dict) -> Optional[int]:
    rt = config.get("rt_scene") or {}
    ps = rt.get("path_solver") if isinstance(rt, dict) else None
    if isinstance(ps, dict) and "seed" in ps:
        return int(ps["seed"])
    ps2 = config.get("rt_scene.path_solver")
    if isinstance(ps2, dict) and "seed" in ps2:
        return int(ps2["seed"])
    return None


def _compare_override(
    messages: list[str],
    label: str,
    toml_val,
    grc_val,
    fmt=str,
) -> None:
    if toml_val is None:
        return
    if fmt(toml_val) != fmt(grc_val):
        messages.append(f"GRC 覆盖 TOML: {label} {fmt(toml_val)}→{fmt(grc_val)}")


def merge_config(
    config_file: str,
    overrides: GrcOverrides,
) -> EffectiveConfig:
    """TOML 为底，GRC overrides 覆盖同语义字段；派生量由 SensingPerformance 重算。"""
    config_path = resolve_config_path(config_file)
    raw = load_config(config_path)
    params = SystemParams.from_dict(raw)
    ofdm = params.ofdm
    messages: list[str] = []

    _compare_override(
        messages, "num_subcarriers", ofdm.num_subcarriers, overrides.fft_len, int
    )
    _compare_override(
        messages, "num_symbols", ofdm.num_symbols, overrides.ofdm_symbols, int
    )
    cp_toml = ofdm.cyclic_prefix_length
    _compare_override(messages, "cp", cp_toml, overrides.cp_len, int)
    _compare_override(
        messages,
        "subcarrier_spacing",
        ofdm.subcarrier_spacing,
        overrides.subcarrier_spacing,
        float,
    )
    _compare_override(
        messages,
        "carrier_frequency",
        params.ofdm.carrier_frequency,
        overrides.center_freq,
        float,
    )
    toml_seed = _toml_path_solver_seed(raw)
    _compare_override(messages, "seed", toml_seed, overrides.seed, int)

    valid_sc = min(ofdm.num_valid_subcarriers, overrides.fft_len)
    if overrides.fft_len != ofdm.num_subcarriers:
        valid_sc = overrides.fft_len

    effective_ofdm = replace(
        ofdm,
        num_subcarriers=overrides.fft_len,
        num_symbols=overrides.ofdm_symbols,
        cyclic_prefix_length=overrides.cp_len,
        subcarrier_spacing=overrides.subcarrier_spacing,
        num_valid_subcarriers=valid_sc,
        carrier_frequency=overrides.center_freq,
    )
    effective_params = replace(
        params,
        ofdm=effective_ofdm,
    )

    from isac.data_structures.components.ofdm import OFDMComponents

    rg = OFDMComponents.build_from_params(effective_ofdm, device=overrides.device).rg
    sp = SensingPerformance(rg, carrier_frequency=overrides.center_freq)
    samp_rate = int(overrides.fft_len * overrides.subcarrier_spacing)
    sym_dur = (overrides.fft_len + overrides.cp_len) / samp_rate if samp_rate else 0.0

    return EffectiveConfig(
        config_path=config_path,
        system_params=effective_params,
        fft_len=overrides.fft_len,
        ofdm_symbols=overrides.ofdm_symbols,
        cp_len=overrides.cp_len,
        subcarrier_spacing=overrides.subcarrier_spacing,
        center_freq=overrides.center_freq,
        seed=overrides.seed,
        device=overrides.device,
        samp_rate=samp_rate,
        bandwidth=float(sp.rg.bandwidth),
        symbol_duration=sym_dur,
        range_resolution=float(sp.range_resolution),
        doppler_resolution=float(sp.doppler_resolution),
        velocity_resolution=float(sp.velocity_resolution),
        R_max=float(sp.max_range),
        v_max=float(sp.max_velocity),
        transpose_len=overrides.ofdm_symbols,
        override_log=tuple(messages),
    )


def log_overrides(effective: EffectiveConfig) -> None:
    for line in effective.override_log:
        print(f"  [config] {line}")


def grc_overrides_from_grc_vars(
    *,
    fft_len: int,
    ofdm_symbols: int,
    cp_len: int,
    subcarrier_spacing: float,
    center_freq: float,
    seed: int,
    device: str,
) -> GrcOverrides:
    return GrcOverrides(
        fft_len=int(fft_len),
        ofdm_symbols=int(ofdm_symbols),
        cp_len=int(cp_len),
        subcarrier_spacing=float(subcarrier_spacing),
        center_freq=float(center_freq),
        seed=int(seed),
        device=str(device),
    )
