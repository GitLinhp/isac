from .system_params import (
    AntennaArrayParams,
    CameraParams,
    CFARParams,
    ChannelParams,
    MTDParams,
    MTIParams,
    MusicParams,
    OFDMParams,
    PathSolverParams,
    QAMParams,
    RtSceneParams,
    SensingPerformanceParams,
    SourceParams,
    StaticTargetParams,
    StreamManagementParams,
    SystemParams,
    TargetMaterialParams,
    TargetParams,
    TransceiverParams,
    WindowParams,
    _as_float_vector,
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
    "_as_float_vector",
]


def __getattr__(name: str):
    if name == "SystemComponents":
        from .system_components import SystemComponents

        return SystemComponents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
