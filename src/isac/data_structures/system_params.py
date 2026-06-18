"""
系统参数数据结构和配置类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Union


def _as_float_vector(values: float | Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (int, float)):
        return (float(values),)
    seq = tuple(float(v) for v in values)
    if not seq:
        raise ValueError(f"{name} 不能为空")
    return seq


@dataclass
class QAMParams:
    """QAM 配置"""

    num_bits_per_symbol: int = 2
    """QAM 每符号比特数"""
    order: int = field(init=False)
    """QAM 阶数，由 num_bits_per_symbol 计算得出"""

    def __post_init__(self) -> None:
        self.order = self.num_bits_per_symbol ** 2

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QAMParams":
        return cls(
            num_bits_per_symbol=int(config_dict.get("num_bits_per_symbol", 2)),
        )


@dataclass
class SourceParams:
    """OFDM 信源配置（binary / ZC）"""

    type: Literal["binary", "zc"] = "binary"
    root_index: int = 1
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.type not in ("binary", "zc"):
            raise ValueError("ofdm.source.type must be 'binary' or 'zc'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SourceParams":
        raw_type = config_dict.get("type", "binary")
        if not isinstance(raw_type, str):
            raise ValueError(f"ofdm.source.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            root_index=int(config_dict.get("root_index", 1)),
            normalize=bool(config_dict.get("normalize", True)),
        )


@dataclass
class OFDMParams:
    """OFDM 网格与调制参数"""

    num_symbols: int = 512
    num_subcarriers: int = 2048
    num_valid_subcarriers: int = 2048
    subcarrier_spacing: float = 30000.0
    cyclic_prefix_length: int = 0
    l_min: int = -6
    dc_null: bool = False

    @property
    def samp_rate(self) -> int:
        return int(self.subcarrier_spacing * self.num_subcarriers)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        raw_l_min = config_dict.get("l_min", -6)
        l_min = int(-6 if raw_l_min is None else raw_l_min)
        cp = config_dict.get(
            "cyclic_prefix_length", config_dict.get("num_cyclic_prefix", 0)
        )
        return cls(
            num_symbols=int(config_dict.get("num_symbols", 1024)),
            num_subcarriers=int(config_dict.get("num_subcarriers", 1024)),
            num_valid_subcarriers=int(
                config_dict.get("num_valid_subcarriers", 1024)
            ),
            subcarrier_spacing=float(config_dict.get("subcarrier_spacing", 30000.0)),
            cyclic_prefix_length=int(cp),
            l_min=l_min,
            dc_null=bool(config_dict.get("dc_null", False)),
        )


@dataclass
class StreamManagementParams:
    """资源网格解映射流管理配置"""

    rx_tx_association: list = field(default_factory=lambda: [[1]])
    num_streams: int = 1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StreamManagementParams":
        assoc = config_dict.get("rx_tx_association", [[1]])
        return cls(
            rx_tx_association=assoc,
            num_streams=int(config_dict.get("num_streams", 1)),
        )


@dataclass
class ChannelParams:
    """信道配置"""

    type: Literal["rt", "rcs"] = "rt"
    snr_db: float = 10.0

    def __post_init__(self) -> None:
        if self.type not in ("rt", "rcs"):
            raise ValueError("channel.type must be 'rt' or 'rcs'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChannelParams":
        raw_type = config_dict.get("type", "rt")
        if not isinstance(raw_type, str):
            raise ValueError(f"channel.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            snr_db=float(config_dict.get("snr_db", 10.0)),
        )


@dataclass
class SensingPerformanceParams:
    """感知性能占位；载波频率用 SystemParams.carrier_frequency，rg 在 build 时注入。"""


@dataclass
class WindowParams:
    """时延 / 多普勒窗配置"""

    delay_window: Optional[Union[str, Dict[str, Any]]] = None
    doppler_window: Optional[Union[str, Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WindowParams":
        return cls(
            delay_window=config_dict.get("delay_window"),
            doppler_window=config_dict.get("doppler_window"),
        )


@dataclass
class CFARParams:
    """CFAR 检测配置"""

    type: str = "ca"
    k: Optional[int] = None
    guard: Union[int, list[int]] = 2
    trailing: Union[int, list[int]] = 20
    pfa: float = 1e-4
    detector: str = "linear"
    offset: Optional[float] = None

    def __post_init__(self) -> None:
        t = self.type.strip().lower()
        if t not in ("ca", "os"):
            raise ValueError("sensing.cfar.type must be 'ca' or 'os'")
        self.type = t
        if t == "os" and self.k is None:
            raise ValueError("sensing.cfar: type 'os' requires integer 'k' in config")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CFARParams":
        raw_type = config_dict.get("type", "ca")
        if not isinstance(raw_type, str):
            raise ValueError(
                f"sensing.cfar.type must be a string, got {type(raw_type)!r}"
            )
        k_raw = config_dict.get("k", None)
        cfar_k: Optional[int] = int(k_raw) if k_raw is not None else None
        return cls(
            type=raw_type.strip().lower(),
            k=cfar_k,
            guard=config_dict.get("guard", 2),
            trailing=config_dict.get("trailing", 20),
            pfa=config_dict.get("pfa", 1e-4),
            detector=config_dict.get("detector", "linear"),
            offset=config_dict.get("offset", None),
        )


@dataclass
class MusicParams:
    """MUSIC 估计器默认调用参数"""

    threshold: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicParams":
        return cls(
            threshold=float(config_dict.get("threshold", 0.1)),
        )


@dataclass
class MTIParams:
    """动目标显示（MTI）配置"""

    filter_order: int = 1
    prf: Optional[float] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MTIParams":
        prf_raw = config_dict.get("prf", None)
        return cls(
            filter_order=int(config_dict.get("filter_order", 1)),
            prf=float(prf_raw) if prf_raw is not None else None,
        )


@dataclass
class MTDParams:
    """动目标检测（MTD）配置"""

    num_filters: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MTDParams":
        nf = config_dict.get("num_filters", None)
        return cls(
            num_filters=int(nf) if nf is not None else None,
        )


@dataclass
class StaticTargetParams:
    """静态点目标信道参数（TOML + OFDM 派生物理量）。"""

    range_m: float | Sequence[float] = 100.0
    velocity_mps: float | Sequence[float] = 0.0
    rcs: float | Sequence[float] = 1e25
    azimuth_deg: float | Sequence[float] = 0.0
    position_rx_m: float | Sequence[float] = 0.0
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True
    samp_rate: Optional[int] = None
    center_freq: Optional[float] = None

    def __post_init__(self) -> None:
        ranges = _as_float_vector(self.range_m, "range_m")
        velocities = _as_float_vector(self.velocity_mps, "velocity_mps")
        rcs_vals = _as_float_vector(self.rcs, "rcs")
        azimuths = _as_float_vector(self.azimuth_deg, "azimuth_deg")
        rx_positions = _as_float_vector(self.position_rx_m, "position_rx_m")
        n = len(ranges)
        if not (len(velocities) == len(rcs_vals) == len(azimuths) == n):
            raise ValueError("range_m / velocity_mps / rcs / azimuth_deg 长度须一致")
        if not rx_positions:
            raise ValueError("position_rx_m 不能为空")
        if self.samp_rate is not None and self.samp_rate <= 0:
            raise ValueError("samp_rate 须为正")
        if self.center_freq is not None and self.center_freq <= 0:
            raise ValueError("center_freq 须为正")

    def apply_phy(self, carrier_frequency: float, ofdm: OFDMParams) -> None:
        self.center_freq = float(carrier_frequency)
        self.samp_rate = ofdm.samp_rate

    def ensure_phy(self) -> None:
        if self.samp_rate is None or self.center_freq is None:
            raise ValueError(
                "StaticTargetParams 须先调用 apply_phy 设置 samp_rate / center_freq"
            )

    @property
    def num_targets(self) -> int:
        return len(_as_float_vector(self.range_m, "range_m"))

    @property
    def num_rx(self) -> int:
        return len(_as_float_vector(self.position_rx_m, "position_rx_m"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StaticTargetParams":
        return cls(
            range_m=config_dict.get("range_m", 100.0),
            velocity_mps=config_dict.get("velocity_mps", 0.0),
            rcs=config_dict.get("rcs", 1e25),
            azimuth_deg=config_dict.get("azimuth_deg", 0.0),
            position_rx_m=config_dict.get("position_rx_m", [0.0]),
            self_coupling_db=float(config_dict.get("self_coupling_db", -10.0)),
            rndm_phaseshift=bool(config_dict.get("rndm_phaseshift", True)),
            self_coupling=bool(config_dict.get("self_coupling", True)),
        )


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

    type: str
    num_rows: int
    num_cols: int
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern: str = "iso"
    polarization: str = "V"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AntennaArrayParams":
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
    merge_shapes: bool = False
    camera: CameraParams | None = None
    antenna_arrays: dict[str, AntennaArrayParams] | None = None
    transceivers: dict[str, TransceiverParams] | None = None
    target_materials: dict[str, TargetMaterialParams] | None = None
    targets: dict[str, TargetParams] | None = None
    path_solver: PathSolverParams | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RtSceneParams":
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


@dataclass
class SystemParams:
    """系统配置（嵌套 Params，顺序对齐 system_components）。"""

    carrier_frequency: float = 2.6e9

    qam: QAMParams = field(default_factory=QAMParams)
    source: SourceParams = field(default_factory=SourceParams)
    ofdm: OFDMParams = field(default_factory=OFDMParams)
    stream_management: StreamManagementParams = field(
        default_factory=StreamManagementParams
    )

    channel: ChannelParams = field(default_factory=ChannelParams)

    sensing_performance: SensingPerformanceParams = field(
        default_factory=SensingPerformanceParams
    )
    windows: WindowParams = field(default_factory=WindowParams)
    music: MusicParams = field(default_factory=MusicParams)
    cfar: CFARParams = field(default_factory=CFARParams)
    mti: MTIParams = field(default_factory=MTIParams)
    mtd: MTDParams = field(default_factory=MTDParams)

    rt_scene: Optional[RtSceneParams] = None
    static_target: Optional[StaticTargetParams] = None

    @property
    def qam_order(self) -> int:
        return self.qam.order

    @property
    def samp_rate(self) -> int:
        return self.ofdm.samp_rate

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        ofdm_raw = config_dict.get("ofdm", {})
        if not isinstance(ofdm_raw, dict):
            ofdm_raw = {}
        channel_raw = config_dict.get("channel", {})
        if not isinstance(channel_raw, dict):
            channel_raw = {}
        sensing = config_dict.get("sensing") or {}
        if not isinstance(sensing, dict):
            sensing = {}
        static_target_cfg = config_dict.get("static_target")
        rt_scene_cfg = config_dict.get("rt_scene")

        src_dict = ofdm_raw.get("source")
        if not isinstance(src_dict, dict):
            src_dict = {}

        w = sensing.get("windows")
        if not isinstance(w, dict):
            w = {}
        cfar_dict = sensing.get("cfar")
        if not isinstance(cfar_dict, dict):
            cfar_dict = {}
        music_dict = sensing.get("music")
        if not isinstance(music_dict, dict):
            music_dict = {}
        mti_dict = sensing.get("mti")
        if not isinstance(mti_dict, dict):
            mti_dict = {}
        mtd_dict = sensing.get("mtd")
        if not isinstance(mtd_dict, dict):
            mtd_dict = {}

        stream_dict = ofdm_raw.get("stream_management")
        if not isinstance(stream_dict, dict):
            stream_dict = {}

        static_target: Optional[StaticTargetParams] = None
        if isinstance(static_target_cfg, dict) and static_target_cfg:
            static_target = StaticTargetParams.from_dict(static_target_cfg)

        rt_scene: Optional[RtSceneParams] = None
        if isinstance(rt_scene_cfg, dict) and rt_scene_cfg:
            rt_scene = RtSceneParams.from_dict(rt_scene_cfg)

        ofdm = OFDMParams.from_dict(ofdm_raw)

        if "carrier_frequency" in config_dict:
            carrier_frequency = float(config_dict["carrier_frequency"])
        else:
            carrier_frequency = float(ofdm_raw.get("carrier_frequency", 2.6e9))

        qam_raw = config_dict.get("qam", {})
        if not isinstance(qam_raw, dict):
            qam_raw = {}
        if "num_bits_per_symbol" not in qam_raw and "num_bits_per_symbol" in ofdm_raw:
            qam_raw = {
                **qam_raw,
                "num_bits_per_symbol": ofdm_raw["num_bits_per_symbol"],
            }

        params = cls(
            carrier_frequency=carrier_frequency,
            qam=QAMParams.from_dict(qam_raw),
            source=SourceParams.from_dict(src_dict),
            ofdm=ofdm,
            stream_management=StreamManagementParams.from_dict(stream_dict),
            channel=ChannelParams.from_dict(channel_raw),
            windows=WindowParams.from_dict(w),
            music=MusicParams.from_dict(music_dict),
            cfar=CFARParams.from_dict(cfar_dict),
            mti=MTIParams.from_dict(mti_dict),
            mtd=MTDParams.from_dict(mtd_dict),
            rt_scene=rt_scene,
            static_target=static_target,
        )
        if params.static_target is not None:
            params.static_target.apply_phy(params.carrier_frequency, params.ofdm)
        params._validate_channel_dependencies()
        return params

    def _validate_channel_dependencies(self) -> None:
        if self.channel.type == "rt" and self.rt_scene is None:
            raise ValueError("channel.type='rt' 要求配置 [rt_scene]")
        if self.channel.type == "rcs" and self.static_target is None:
            raise ValueError("channel.type='rcs' 要求配置 [static_target]")
