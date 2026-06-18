"""静态点目标散射参数（与 ``[static_target]`` TOML 对应）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


def _as_float_vector(values: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        return (float(values),)
    seq = tuple(float(v) for v in values)
    if not seq:
        raise ValueError(f"{name} 不能为空")
    return seq


@dataclass
class StaticTargetConfig:
    """TOML ``[static_target]`` 配置（不含运行时 ``samp_rate`` / ``center_freq``）。"""

    range_m: float | Sequence[float] = 100.0
    velocity_mps: float | Sequence[float] = 0.0
    rcs: float | Sequence[float] = 1e25
    azimuth_deg: float | Sequence[float] = 0.0
    position_rx_m: Sequence[float] = (0.0,)
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StaticTargetConfig":
        if not isinstance(config_dict, dict):
            config_dict = {}
        return cls(
            range_m=config_dict.get("range_m", 100.0),
            velocity_mps=config_dict.get("velocity_mps", 0.0),
            rcs=config_dict.get("rcs", 1e25),
            azimuth_deg=config_dict.get("azimuth_deg", 0.0),
            position_rx_m=config_dict.get("position_rx_m", [0.0]),
            self_coupling_db=float(config_dict.get("self_coupling_db", -10.0)),
            rndm_phaseshift=bool(config_dict.get("rndm_phaseshift", True)),
            self_coupling=bool(config_dict.get("self_coupling", True)),
        )

    def to_params(self, samp_rate: int, center_freq: float) -> "StaticTargetParams":
        return StaticTargetParams(
            range_m=self.range_m,
            velocity_mps=self.velocity_mps,
            rcs=self.rcs,
            azimuth_deg=self.azimuth_deg,
            position_rx_m=self.position_rx_m,
            samp_rate=samp_rate,
            center_freq=center_freq,
            self_coupling_db=self.self_coupling_db,
            rndm_phaseshift=self.rndm_phaseshift,
            self_coupling=self.self_coupling,
        )


@dataclass(frozen=True)
class StaticTargetParams:
    """与 gr-radar static_target_simulator_cc 块参数对应。

    ``samp_rate`` / ``center_freq`` 由组件构建时从 OFDM 资源网格注入。
    """

    range_m: Sequence[float]
    velocity_mps: Sequence[float]
    rcs: Sequence[float]
    azimuth_deg: Sequence[float]
    position_rx_m: Sequence[float]
    samp_rate: int
    center_freq: float
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True

    def __post_init__(self) -> None:
        ranges = _as_float_vector(self.range_m, "range_m")
        velocities = _as_float_vector(self.velocity_mps, "velocity_mps")
        rcs_vals = _as_float_vector(self.rcs, "rcs")
        azimuths = _as_float_vector(self.azimuth_deg, "azimuth_deg")
        rx_positions = _as_float_vector(self.position_rx_m, "position_rx_m")
        n = len(ranges)
        if not (len(velocities) == len(rcs_vals) == len(azimuths) == n):
            raise ValueError("range_m / velocity_mps / rcs / azimuth_deg 长度须一致")
        if not rx_positions:
            raise ValueError("position_rx_m 不能为空")
        if self.samp_rate <= 0:
            raise ValueError("samp_rate 须为正")
        if self.center_freq <= 0:
            raise ValueError("center_freq 须为正")

    @property
    def num_targets(self) -> int:
        return len(_as_float_vector(self.range_m, "range_m"))

    @property
    def num_rx(self) -> int:
        return len(_as_float_vector(self.position_rx_m, "position_rx_m"))

def static_target_params_from_grc(
    *,
    range_m: float | Sequence[float] = 100.0,
    velocity_mps: float | Sequence[float] = 5.0,
    rcs: float | Sequence[float] = 1e25,
    azimuth_deg: float | Sequence[float] = 0.0,
    position_rx_m: Sequence[float] = (0.0,),
    center_freq: float = 6e9,
    samp_rate: int = 30_720_000,
    self_coupling_db: float = -10.0,
    rndm_phaseshift: bool = True,
    self_coupling: bool = True,
) -> StaticTargetParams:
    """与 simulator_ofdm.grc 中 radar_static_target_simulator_cc_0 默认参数对齐。"""
    return StaticTargetParams(
        range_m=range_m,
        velocity_mps=velocity_mps,
        rcs=rcs,
        azimuth_deg=azimuth_deg,
        position_rx_m=position_rx_m,
        samp_rate=samp_rate,
        center_freq=center_freq,
        self_coupling_db=self_coupling_db,
        rndm_phaseshift=rndm_phaseshift,
        self_coupling=self_coupling,
    )
