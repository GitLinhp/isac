"""RCS 点目标参数（TOML [rcs_scene]）。"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RCSTargetParams:
    """单 RCS 点目标：距离、径向速度、RCS 与方位几何。"""

    range_m: float = 100.0
    velocity_mps: float = 0.0
    rcs: float = 1e25
    azimuth_deg: float = 0.0
    position_rx_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "range_m",
            "velocity_mps",
            "rcs",
            "azimuth_deg",
            "position_rx_m",
        ):
            val = getattr(self, name)
            if isinstance(val, (list, tuple)):
                raise ValueError(f"{name} 仅支持标量输入")
            setattr(self, name, float(val))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RCSTargetParams":
        def _scalar_float(key: str, default: float) -> float:
            raw = config_dict.get(key, default)
            if isinstance(raw, (list, tuple)):
                raise ValueError(f"{key} 仅支持标量输入")
            return float(raw)

        return cls(
            range_m=_scalar_float("range_m", 100.0),
            velocity_mps=_scalar_float("velocity_mps", 0.0),
            rcs=_scalar_float("rcs", 1e25),
            azimuth_deg=_scalar_float("azimuth_deg", 0.0),
            position_rx_m=_scalar_float("position_rx_m", 0.0),
        )


@dataclass
class RCSSceneParams:
    """RCS 点目标场景参数（单目标）。"""

    target: RCSTargetParams = field(default_factory=RCSTargetParams)
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RCSSceneParams":
        target_raw = config_dict.get("target")
        if not isinstance(target_raw, dict) or not target_raw:
            raise ValueError("[rcs_scene] 须配置非空 [rcs_scene.target]")
        return cls(
            target=RCSTargetParams.from_dict(target_raw),
            self_coupling_db=float(config_dict.get("self_coupling_db", -10.0)),
            rndm_phaseshift=bool(config_dict.get("rndm_phaseshift", True)),
            self_coupling=bool(config_dict.get("self_coupling", True)),
        )
