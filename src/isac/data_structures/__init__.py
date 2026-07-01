from .params import (
    AntennaArrayParams,
    CameraParams,
    CFARParams,
    ChannelParams,
    MTDParams,
    MTIParams,
    MusicParams,
    OFDMParams,
    PathSolverParams,
    RCSSceneParams,
    RCSTargetParams,
    RTSimulatorParams,
    SourceParams,
    StreamManagementParams,
    SystemParams,
    TargetMaterialParams,
    TargetParams,
    TransceiverParams,
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
    "RTSimulatorParams",
    "SourceParams",
    "RCSSceneParams",
    "RCSTargetParams",
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
