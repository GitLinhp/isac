from .rt_scene_params import (
    AntennaArrayParams,
    CameraParams,
    PathSolverParams,
    RtSceneParams,
    TargetMaterialParams,
    TargetParams,
    TransceiverParams,
)
from .system_params import (
    CFARParams,
    ChannelParams,
    MTDParams,
    MTIParams,
    MusicParams,
    OFDMParams,
    QAMParams,
    SensingPerformanceParams,
    SourceParams,
    StaticTargetParams,
    StreamManagementParams,
    SystemParams,
    WindowParams,
)

__all__ = [
    "AntennaArrayParams",
    "CameraParams",
    "CFARParams",
    "ChannelParams",
    "MTDParams",
    "MTIParams",
    "MusicParams",
    "OFDMParams",
    "PathSolverParams",
    "QAMParams",
    "RtSceneParams",
    "SensingPerformanceParams",
    "SourceParams",
    "StaticTargetParams",
    "StreamManagementParams",
    "SystemComponents",
    "SystemParams",
    "TargetMaterialParams",
    "TargetParams",
    "TransceiverParams",
    "WindowParams",
]


def __getattr__(name: str):
    if name == "SystemComponents":
        from .system_components import SystemComponents

        return SystemComponents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
