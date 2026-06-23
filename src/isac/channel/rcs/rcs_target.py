"""RCS 点目标几何/散射状态：距离、径向速度、RCS 与方位几何。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...data_structures.params.channel_params.rcs_scene_params import RCSTargetParams

_TARGET_FIELDS = (
    "range_m",
    "velocity_mps",
    "rcs",
    "azimuth_deg",
    "position_rx_m",
)


def _coerce_target_field(name: str, val: Any) -> float:
    if isinstance(val, (list, tuple)):
        raise ValueError(f"{name} 仅支持标量输入")
    return float(val)


@dataclass
class RCSTarget:
    """RCS 点目标运行时状态（与 ``RCSTargetParams`` 字段一致）。"""

    range_m: float = 100.0
    velocity_mps: float = 0.0
    rcs: float = 1e25
    azimuth_deg: float = 0.0
    position_rx_m: float = 0.0

    def __post_init__(self) -> None:
        for name in _TARGET_FIELDS:
            setattr(self, name, _coerce_target_field(name, getattr(self, name)))

    def update(
        self,
        *,
        range_m: Optional[float] = None,
        velocity_mps: Optional[float] = None,
        rcs: Optional[float] = None,
        azimuth_deg: Optional[float] = None,
        position_rx_m: Optional[float] = None,
    ) -> None:
        """运行时更新点目标状态（类似 ``RTTarget.update``）。"""
        updates = {
            "range_m": range_m,
            "velocity_mps": velocity_mps,
            "rcs": rcs,
            "azimuth_deg": azimuth_deg,
            "position_rx_m": position_rx_m,
        }
        for name, val in updates.items():
            if val is not None:
                setattr(self, name, _coerce_target_field(name, val))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> RCSTarget:
        def _scalar_float(key: str, default: float) -> float:
            raw = config.get(key, default)
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

    @classmethod
    def from_params(cls, params: RCSTargetParams) -> RCSTarget:
        return cls(
            range_m=params.range_m,
            velocity_mps=params.velocity_mps,
            rcs=params.rcs,
            azimuth_deg=params.azimuth_deg,
            position_rx_m=params.position_rx_m,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "range_m": self.range_m,
            "velocity_mps": self.velocity_mps,
            "rcs": self.rcs,
            "azimuth_deg": self.azimuth_deg,
            "position_rx_m": self.position_rx_m,
        }
