"""
RT 仿真器参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraParams:
    """相机配置"""

    position: list[float]
    orientation: list[float] | None = None
    look_at: list[float] | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CameraParams":
        if "position" not in config_dict:
            raise ValueError("camera.position 是必选配置项")
        return cls(
            position=config_dict["position"],
            orientation=config_dict.get("orientation", [0, 0, 0]),
            look_at=config_dict.get("look_at"),
        )


@dataclass
class AntennaArrayParams:
    """天线阵列配置"""

    num_rows: int
    num_cols: int
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern: str = "iso"
    polarization: str = "V"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AntennaArrayParams":
        return cls(
            num_rows=config_dict["num_rows"],
            num_cols=config_dict["num_cols"],
            vertical_spacing=config_dict.get("vertical_spacing", 0.5),
            horizontal_spacing=config_dict.get("horizontal_spacing", 0.5),
            pattern=config_dict.get("pattern", "iso"),
            polarization=config_dict.get("polarization", "V"),
        )


@dataclass
class TransceiverParams:
    """收发器配置"""

    position: list[float] = field(default_factory=lambda: [0, 100, 50])
    look_at: list[float] = field(default_factory=lambda: [0, 0, 30])
    type: str = "tx"
    power_dbm: float | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TransceiverParams":
        raw_power = config_dict.get("power_dbm")
        power_dbm = float(raw_power) if raw_power is not None else None
        return cls(
            position=config_dict.get("position", [0, 100, 50]),
            look_at=config_dict.get("look_at", [0, 0, 30]),
            type=config_dict.get("type", "tx"),
            power_dbm=power_dbm,
        )


@dataclass
class TargetMaterialParams:
    """目标材料配置"""

    type: str = "metal"
    thickness: float = 0.01
    color: list[float] = field(default_factory=lambda: [0, 0.2, 0.6])

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TargetMaterialParams":
        return cls(
            type=config_dict.get("type", "metal"),
            thickness=config_dict.get("thickness", 0.01),
            color=config_dict.get("color", [0, 0.2, 0.6]),
        )


@dataclass
class TargetParams:
    """目标配置"""

    # mesh 逻辑名；优先解析 isac/channel/rt/scenes/{fname}.ply，否则回退 sionna.rt.scene
    fname: str = "low_poly_car"
    material: str = "car_material"
    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    velocity: list[float] = field(default_factory=lambda: [0, 0, 0])

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TargetParams":
        if "trajectory" in config_dict:
            raise ValueError(
                "trajectory 已移除，请使用 position/velocity 配置目标初始位姿"
            )
        return cls(
            fname=config_dict.get("fname", "low_poly_car"),
            material=config_dict.get("material", "car_material"),
            position=config_dict.get("position", [0, 0, 0]),
            velocity=config_dict.get("velocity", [0, 0, 0]),
        )


@dataclass
class PathSolverParams:
    """路径求解器配置"""

    max_depth: int = 3
    max_num_paths_per_src: int = int(1e6)
    samples_per_src: int = int(1e7)
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    synthetic_array: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PathSolverParams":
        return cls(
            max_depth=config_dict.get("max_depth", 1),
            max_num_paths_per_src=int(
                config_dict.get("max_num_paths_per_src", int(1e6))
            ),
            samples_per_src=int(config_dict.get("samples_per_src", int(1e7))),
            los=config_dict.get("los", True),
            specular_reflection=config_dict.get("specular_reflection", True),
            diffuse_reflection=config_dict.get("diffuse_reflection", False),
            refraction=config_dict.get("refraction", False),
            diffraction=config_dict.get("diffraction", False),
            edge_diffraction=config_dict.get("edge_diffraction", False),
            synthetic_array=config_dict.get("synthetic_array", True),
            seed=config_dict.get("seed", 42),
        )


@dataclass
class RTSimulatorParams:
    """RT 仿真器配置"""

    filename: str | None = None
    """场景文件名"""
    merge_shapes: bool = False
    """是否合并相同材质的形状"""
    camera: CameraParams | None = None
    """相机配置"""
    antenna_arrays: dict[str, AntennaArrayParams] | None = None
    """天线阵列配置"""
    transceivers: dict[str, TransceiverParams] | None = None
    """收发器配置"""
    target_materials: dict[str, TargetMaterialParams] | None = None
    """目标材质配置"""
    targets: dict[str, TargetParams] | None = None
    """目标配置"""
    path_solver: PathSolverParams | None = None
    """路径解析器配置"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RTSimulatorParams":
        antenna_arrays_cfg = config_dict.get("antenna_arrays", {})
        transceivers_cfg = config_dict.get("transceivers", {})
        target_materials_cfg = config_dict.get("target_materials", {})
        targets_cfg = config_dict.get("targets", {})

        return cls(
            filename=config_dict.get("filename", None),
            merge_shapes=config_dict.get("merge_shapes", False),
            camera=(
                CameraParams.from_dict(config_dict["camera"])
                if isinstance(config_dict.get("camera"), dict)
                else None
            ),
            antenna_arrays=(
                {
                    name: AntennaArrayParams.from_dict(antenna_array)
                    for name, antenna_array in antenna_arrays_cfg.items()
                }
                if antenna_arrays_cfg
                else None
            ),
            transceivers=(
                {
                    name: TransceiverParams.from_dict(transceiver)
                    for name, transceiver in transceivers_cfg.items()
                }
                if transceivers_cfg
                else None
            ),
            target_materials=(
                {
                    name: TargetMaterialParams.from_dict(target_material)
                    for name, target_material in target_materials_cfg.items()
                }
                if target_materials_cfg
                else None
            ),
            targets=(
                {
                    name: TargetParams.from_dict(target)
                    for name, target in targets_cfg.items()
                }
                if targets_cfg
                else None
            ),
            path_solver=(
                PathSolverParams.from_dict(config_dict["path_solver"])
                if isinstance(config_dict.get("path_solver"), dict)
                else None
            ),
        )
