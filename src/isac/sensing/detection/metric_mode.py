"""MUSIC / DD 谱 ``metric_mode`` 与感知 ``sens_mode`` 类型及别名解析。"""

from typing import Literal

metric_mode = Literal["delay_doppler", "range_velocity", "dd", "rv"]
SensMode = Literal["monostatic", "bistatic"]

MODE_CANONICAL: dict[str, metric_mode] = {
    "delay_doppler": "delay_doppler",
    "dd": "delay_doppler",
    "range_velocity": "range_velocity",
    "rv": "range_velocity",
}


def canonical_metric_mode(metric_mode: str) -> metric_mode:
    """将 ``metric_mode`` 别名解析为 ``delay_doppler`` / ``range_velocity``。"""
    key = metric_mode.strip().lower()
    try:
        return MODE_CANONICAL[key]
    except KeyError as exc:
        raise ValueError(
            "metric_mode 须为 'delay_doppler'、'dd'、'range_velocity' 或 'rv'，"
            f"当前为: {metric_mode!r}"
        ) from exc
