"""
系统参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union
from .rt_scene_params import RtSceneParams


@dataclass
class SourceParams:
    """OFDM 信源配置（binary / ZC）"""

    type: Literal["binary", "zc"] = "binary"
    root_index: int = 1
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.type not in ("binary", "zc"):
            raise ValueError("source.type must be 'binary' or 'zc'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SourceParams":
        raw_type = config_dict.get("type", "binary")
        if not isinstance(raw_type, str):
            raise ValueError(f"source.type must be a string, got {type(raw_type)!r}")
        return cls(
            type=raw_type.strip().lower(),
            root_index=int(config_dict.get("root_index", 1)),
            normalize=bool(config_dict.get("normalize", True)),
        )


@dataclass
class OFDMParams:
    """OFDM 网格与调制参数"""

    num_symbols: int = 512
    fft_size: int = 2048
    subcarrier_spacing: float = 30000.0
    cyclic_prefix_length: int = 0
    l_min: int = -6
    dc_null: bool = False

    @property
    def samp_rate(self) -> int:
        return int(self.subcarrier_spacing * self.fft_size)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        raw_l_min = config_dict.get("l_min", -6)
        l_min = int(-6 if raw_l_min is None else raw_l_min)
        cp = config_dict.get(
            "cyclic_prefix_length", config_dict.get("num_cyclic_prefix", 0)
        )
        fft_size_raw = config_dict.get(
            "fft_size", config_dict.get("num_subcarriers", 1024)
        )
        return cls(
            num_symbols=int(config_dict.get("num_symbols", 1024)),
            fft_size=int(fft_size_raw),
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
    near_range_guard_m: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicParams":
        return cls(
            threshold=float(config_dict.get("threshold", 0.1)),
            near_range_guard_m=float(config_dict.get("near_range_guard_m", 1.0)),
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
    """静态点目标信道参数（TOML + OFDM 派生物理量，单目标）。"""

    range_m: float = 100.0
    velocity_mps: float = 0.0
    rcs: float = 1e25
    azimuth_deg: float = 0.0
    position_rx_m: float = 0.0
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True
    samp_rate: Optional[int] = None
    center_freq: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "range_m",
            "velocity_mps",
            "rcs",
            "azimuth_deg",
            "position_rx_m",
        ):
            val = getattr(self, name)
            if isinstance(val, (list, tuple)):
                raise ValueError(f"{name} 仅支持标量输入")
            setattr(self, name, float(val))
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StaticTargetParams":
        def _scalar_float(key: str, default: float) -> float:
            raw = config_dict.get(key, default)
            if isinstance(raw, (list, tuple)):
                raise ValueError(f"{key} 仅支持标量输入")
            return float(raw)

        return cls(
            range_m=_scalar_float("range_m", 100.0),
            velocity_mps=_scalar_float("velocity_mps", 0.0),
            rcs=_scalar_float("rcs", 1e25),
            azimuth_deg=_scalar_float("azimuth_deg", 0.0),
            position_rx_m=_scalar_float("position_rx_m", 0.0),
            self_coupling_db=float(config_dict.get("self_coupling_db", -10.0)),
            rndm_phaseshift=bool(config_dict.get("rndm_phaseshift", True)),
            self_coupling=bool(config_dict.get("self_coupling", True)),
        )


@dataclass
class SystemParams:
    """系统配置（嵌套 Params，顺序对齐 system_components）。"""

    carrier_frequency: float = 2.6e9
    num_bits_per_symbol: int = 2

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
    def samp_rate(self) -> int:
        return self.ofdm.samp_rate

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        # 载波频率
        carrier_frequency = float(config_dict.get("carrier_frequency", 2.6e9))

        # QAM 每符号比特数
        num_bits_per_symbol = int(config_dict.get("num_bits_per_symbol", 2))

        # 信源
        source = SourceParams.from_dict(config_dict.get("source", {}))

        # ofdm
        ofdm = OFDMParams.from_dict(config_dict.get("ofdm", {}))

        # 资源网格解映射流管理
        stream_management = StreamManagementParams.from_dict(
            config_dict.get("stream_management", {})
        )

        # 信道
        channel = ChannelParams.from_dict(config_dict.get("channel", {}))

        # windows / music / cfar / mti / mtd
        sensing = config_dict.get("sensing", {})
        windows = WindowParams.from_dict(sensing.get("windows", {}))
        music = MusicParams.from_dict(sensing.get("music", {}))
        cfar = CFARParams.from_dict(sensing.get("cfar", {}))
        mti = MTIParams.from_dict(sensing.get("mti", {}))
        mtd = MTDParams.from_dict(sensing.get("mtd", {}))

        # rt_scene
        rt_scene = RtSceneParams.from_dict(config_dict.get("rt_scene", {}))

        # static_target
        static_target = StaticTargetParams.from_dict(
            config_dict.get("static_target", {})
        )

        params = cls(
            carrier_frequency=carrier_frequency,
            num_bits_per_symbol=num_bits_per_symbol,
            source=source,
            ofdm=ofdm,
            stream_management=stream_management,
            channel=channel,
            windows=windows,
            music=music,
            cfar=cfar,
            mti=mti,
            mtd=mtd,
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
