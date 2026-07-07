"""采集辅助工具：输出文件名 slug。"""

from __future__ import annotations

from ..channel.rt.rt_simulator import RTSimulator


def scene_slug_from_rt_simulator(rt_simulator: RTSimulator) -> str:
    """输出文件名用：取 ``rt_simulator_params.filename``；未配置或为空时用 ``\"None\"``。"""
    raw = getattr(rt_simulator.rt_simulator_params, "filename", None)
    if raw is None:
        return "None"
    s = str(raw).strip()
    if not s or s.lower() == "none":
        return "None"
    return s
