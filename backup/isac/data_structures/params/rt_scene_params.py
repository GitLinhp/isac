from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraParams:
    """相机配置"""

    position: list[float]
    """相机位置"""
    orientation: list[float] | None = None
    """相机朝向"""
    look_at: list[float] | None = None
    """相机朝向点"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CameraParams":
        """从配置字典创建相机配置对象"""
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

    type: str
    """天线阵列类型"""
    num_rows: int
    """天线阵列行数"""
    num_cols: int
    """天线阵列列数"""
    vertical_spacing: float = 0.5
    """天线阵列垂直间距"""
    horizontal_spacing: float = 0.5
    """天线阵列水平间距"""
    pattern: str = "iso"
    """天线阵列模式"""
    polarization: str = "V"
    """天线阵列极化"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AntennaArrayParams":
        """从配置字典创建天线阵列配置对象"""
        return cls(
            type=config_dict["type"],
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
    """收发器位置"""
    look_at: list[float] = field(default_factory=lambda: [0, 0, 30])
    """收发器朝向点"""
    type: str = "tx"
    """收发器类型"""
    power_dbm: float | None = None
    """发射功率 (dBm)，仅当 ``type`` 含 ``tx`` 时生效；``None`` 时使用 Sionna ``Transmitter`` 默认值"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TransceiverParams":
        """从配置字典创建收发器配置对象"""
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
    """目标材料名称"""
    thickness: float = 0.01
    """目标材料厚度"""
    color: list[float] = field(default_factory=lambda: [0, 0.2, 0.6])
    """目标材料颜色"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TargetMaterialParams":
        """从配置字典创建目标材料配置对象"""
        return cls(
            type=config_dict.get("type", "metal"),
            thickness=config_dict.get("thickness", 0.01),
            color=config_dict.get("color", [0, 0.2, 0.6]),
        )


@dataclass
class TrajectoryParams:
    """目标运动配置（trajectory-only）。"""

    points: list[list[float]]
    """轨迹控制点，至少 1 个 3D 点。"""
    velocity: float
    """沿轨迹移动速度，单位 m/s，必须 > 0。"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrajectoryParams":
        """从配置字典创建运动参数"""
        if "looping_mode" in config_dict:
            raise ValueError("trajectory.looping_mode 已废弃，不再支持循环轨迹，请删除该字段")

        points_raw = config_dict.get("points")
        if points_raw is None:
            raise ValueError("trajectory.points 是必选配置项，且至少包含 1 个 3D 点")
        if not isinstance(points_raw, list) or len(points_raw) == 0:
            raise ValueError("trajectory.points 必须为非空列表，元素为长度为 3 的坐标")

        points: list[list[float]] = []
        for idx, point in enumerate(points_raw):
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"trajectory.points[{idx}] 必须是长度为 3 的列表")
            points.append([float(point[0]), float(point[1]), float(point[2])])

        velocity = float(config_dict.get("velocity", 0.0))
        if velocity <= 0.0:
            raise ValueError("trajectory.velocity 必须大于 0")

        return cls(
            points=points,
            velocity=velocity,
        )


@dataclass
class TargetParams:
    """目标配置"""

    fname: str = "low_poly_car"
    """目标文件名"""
    material: str = "car_material"
    """目标材料"""
    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    """目标位置"""
    velocity: list[float] = field(default_factory=lambda: [0, 0, 0])
    """目标速度"""
    trajectory: TrajectoryParams | None = None
    """可选轨迹参数；未配置则不启用路径生成与步进"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TargetParams":
        """从配置字典创建目标配置对象"""
        return cls(
            fname=config_dict.get("fname", "low_poly_car"),
            material=config_dict.get("material", "car_material"),
            position=config_dict.get("position", [0, 0, 0]),
            velocity=config_dict.get("velocity", [0, 0, 0]),
            trajectory=config_dict.get("trajectory", None),
        )


@dataclass
class PathSolverParams:
    """路径求解器配置"""

    max_depth: int = 3
    """最大深度"""
    max_num_paths_per_src: int = int(1e6)
    """每个信号源允许的最大路径数"""
    samples_per_src: int = int(1e7)
    """每个信号源的射线采样数"""
    los: bool = True
    """是否考虑LOS路径"""
    specular_reflection: bool = True
    """是否考虑镜面反射"""
    diffuse_reflection: bool = True
    """是否考虑漫反射"""
    refraction: bool = False
    """是否考虑折射"""
    diffraction: bool = False
    """是否考虑衍射"""
    edge_diffraction: bool = False
    """是否考虑边缘衍射"""
    synthetic_array: bool = True
    """是否使用合成阵列"""
    seed: int = 42
    """随机种子"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PathSolverParams":
        """从配置字典创建路径求解器配置对象"""
        return cls(
            max_depth=config_dict.get("max_depth", 1),
            max_num_paths_per_src=int(config_dict.get("max_num_paths_per_src", int(1e6))),
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
class RtSceneParams:
    """射线追踪场景配置"""

    filename: str | None = None
    """场景文件名"""
    merge_shapes: bool = False
    """是否合并形状"""
    camera: CameraParams | None = None
    """相机配置"""
    antenna_arrays: dict[str, AntennaArrayParams] | None = None
    """天线阵列配置列表"""
    transceivers: dict[str, TransceiverParams] | None = None
    """收发器配置"""
    target_materials: dict[str, TargetMaterialParams] | None = None
    """目标材料配置"""
    targets: dict[str, TargetParams] | None = None
    """目标配置"""
    path_solver: PathSolverParams | None = None
    """路径求解器配置"""

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RtSceneParams":
        """从配置字典创建射线追踪场景配置对象"""
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
                {name: TargetParams.from_dict(target) for name, target in targets_cfg.items()}
                if targets_cfg
                else None
            ),
            path_solver=(
                PathSolverParams.from_dict(config_dict["path_solver"])
                if isinstance(config_dict.get("path_solver"), dict)
                else None
            ),
        )
