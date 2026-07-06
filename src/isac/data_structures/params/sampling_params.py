"""蒙特卡洛运动学采样配置：TOML ``[monte_carlo_sampling]``（不纳入 SystemParams）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isac.collection.roi_sampling import (
    SamplingMode,
    parse_roi_xy,
    parse_speed_range,
)

_VALID_MODES = frozenset({"uniform", "gaussian"})


def _parse_sampling_mode(raw: Any, *, field: str) -> SamplingMode:
    mode = str(raw).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"monte_carlo_sampling.{field} 仅支持 'uniform' 或 'gaussian'，收到 {raw!r}"
        )
    return mode  # type: ignore[return-value]


@dataclass(frozen=True)
class CollectionSamplingParams:
    """平面 ROI 蒙特卡洛采样参数（``[monte_carlo_sampling]``）。"""

    roi: tuple[float, float, float, float]
    position_sampling_mode: SamplingMode
    speed_range: tuple[float, float]
    speed_sampling_mode: SamplingMode

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CollectionSamplingParams:
        """从 ``load_config`` 返回的 dict 解析 ``[monte_carlo_sampling]``。"""
        raw = config.get("monte_carlo_sampling")
        if raw is None:
            raise ValueError("配置缺少 [monte_carlo_sampling]")
        if not isinstance(raw, dict):
            raise TypeError(
                f"monte_carlo_sampling 须为表(dict)，收到 {type(raw)!r}"
            )
        if not raw:
            raise ValueError("[monte_carlo_sampling] 不能为空")

        if "roi" not in raw:
            raise ValueError("monte_carlo_sampling.roi 为必填项")
        if "speed_range" not in raw:
            raise ValueError("monte_carlo_sampling.speed_range 为必填项")

        roi = parse_roi_xy(raw["roi"])
        speed_range = parse_speed_range(raw["speed_range"])
        position_sampling_mode = _parse_sampling_mode(
            raw.get("position_sampling_mode", "uniform"),
            field="position_sampling_mode",
        )
        speed_sampling_mode = _parse_sampling_mode(
            raw.get("speed_sampling_mode", "uniform"),
            field="speed_sampling_mode",
        )

        return cls(
            roi=roi,
            position_sampling_mode=position_sampling_mode,
            speed_range=speed_range,
            speed_sampling_mode=speed_sampling_mode,
        )
