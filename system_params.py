"""
系统参数数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict


@dataclass
class SourceParams:
    """信源配置参数类"""

    dataset_name: str = "cifar10"
    """数据集名称"""
    split: str = "all"
    """数据集分割"""
    compress_type: str = "jpeg"
    """压缩类型，如 'jpeg' 或 'bpg'"""
    num_images: int = 50
    """图片数量"""
    quality: int = 80
    """压缩质量，JPEG: 1-100, BPG: 1-51"""
    max_size: Optional[Tuple[int, int]] = None
    """最大图像尺寸 (宽度, 高度)，None表示不限制"""
    psnr_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    """PSNR筛选范围 (min_psnr, max_psnr)，None表示不限制"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SourceParams":
        """从字典创建配置对象"""
        max_size = config_dict.get("max_size")
        if max_size is not None and isinstance(max_size, list):
            max_size = tuple(max_size)

        psnr_range = config_dict.get("psnr_range")
        if psnr_range is not None and isinstance(psnr_range, list):
            psnr_range = tuple(psnr_range) if len(psnr_range) == 2 else None

        return cls(
            dataset_name=config_dict.get("dataset_name", "cifar10"),
            split=config_dict.get("split", "all"),
            compress_type=config_dict.get("compress_type", "jpeg"),
            num_images=config_dict.get("num_images", 50),
            quality=config_dict.get("quality", 80),
            max_size=max_size,
            psnr_range=psnr_range,
        )


@dataclass
class LDPCParams:
    """LDPC配置"""

    codelength: int = 256
    """码长"""
    coderate: float = 0.5
    """码率"""
    infolength: int = field(init=False)
    """信息长度，由 codelength * coderate 计算得出"""

    def __post_init__(self):
        """计算信息长度"""
        self.infolength = int(self.codelength * self.coderate)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LDPCParams":
        """从字典创建配置对象"""
        return cls(
            codelength=config_dict.get("codelength", 256),
            coderate=config_dict.get("coderate", 0.5),
        )


@dataclass
class QAMParams:
    """QAM配置"""

    num_bits_per_symbol: int = 2
    """QAM每符号比特数，必须为2的幂次"""
    order: int = field(init=False)
    """QAM阶数，由 num_bits_per_symbol 计算得出"""

    def __post_init__(self):
        """计算QAM阶数"""
        self.order = self.num_bits_per_symbol**2

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QAMParams":
        """从字典创建配置对象"""
        return cls(
            num_bits_per_symbol=config_dict.get("num_bits_per_symbol", 2),
        )


@dataclass
class OFDMParams:
    """OFDM配置"""

    num_symbols: int = 100
    """符号数"""
    num_subcarriers: int = 4096
    """子载波数"""
    num_valid_subcarriers: int = 4096
    """有效子载波数"""
    subcarrier_spacing: float = 30000.0
    """子载波间隔(Hz)"""
    cyclic_prefix_length: int = 0
    """循环前缀长度"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OFDMParams":
        """从字典创建配置对象"""
        return cls(
            num_symbols=config_dict.get("num_symbols", 100),
            num_subcarriers=config_dict.get("num_subcarriers", 4096),
            num_valid_subcarriers=config_dict.get("num_valid_subcarriers", 4096),
            subcarrier_spacing=config_dict.get("subcarrier_spacing", 30000.0),
            cyclic_prefix_length=config_dict.get("cyclic_prefix_length", 0),
        )


@dataclass
class ToneReservationParams:
    """音调保留配置"""

    num_tone_reservation: int = 128
    """音调保留数量"""
    tone_selection_mode: str = "equal_spacing"
    """音调选择模式"""
    random_seed: int = 42
    """随机种子"""
    target_papr_db: float = 6.0
    """目标峰值比(dB)"""
    max_iterations: int = 100
    """最大迭代次数"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ToneReservationParams":
        """从字典创建配置对象"""
        return cls(
            num_tone_reservation=config_dict.get("num_tone_reservation", 128),
            tone_selection_mode=config_dict.get("tone_selection_mode", "equal_spacing"),
            random_seed=config_dict.get("random_seed", 42),
            target_papr_db=config_dict.get("target_papr_db", 6.0),
            max_iterations=config_dict.get("max_iterations", 100),
        )


@dataclass
class PulseShapingParams:
    """脉冲成形配置"""

    beta: float = 0.22
    """滚降因子"""
    span_in_symbols: int = 32
    """滤波器span长度"""
    samples_per_symbol: int = 4
    """过采样率"""
    window: str = "blackman"
    """窗函数"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PulseShapingParams":
        """从字典创建配置对象"""
        return cls(
            beta=config_dict.get("beta", 0.22),
            span_in_symbols=config_dict.get("span_in_symbols", 32),
            samples_per_symbol=config_dict.get("samples_per_symbol", 4),
            window=config_dict.get("window", "blackman"),
        )


@dataclass
class PowerAmplifierParams:
    """功率放大器配置"""

    ibo_db: float = 6.0
    """输入回退(dB)"""
    p_param: float = 5.0
    """Rapp模型参数"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PowerAmplifierParams":
        """从字典创建配置对象"""
        return cls(
            ibo_db=config_dict.get("ibo_db", 6.0),
            p_param=config_dict.get("p_param", 5.0),
        )


@dataclass
class AclrParams:
    """Welch 子载波 ACLR 测量参数。"""

    bandwidth: float = 20e6
    """保留配置项（如 TS 38.104 信道带宽），不参与 Welch ACLR 计算。"""
    measurement_bandwidth: float = 19.08e6
    """有效占用带宽 (Hz)；由 num_valid_subcarriers * subcarrier_spacing 自动计算。"""
    adjacent_channel_offset: Tuple[float, ...] = (-125e6, 125e6)
    """保留配置项，不参与 Welch ACLR 计算。"""
    sampling_rate: float = 122.88e6
    """功放后时域采样率 (Hz)；默认 N_fft * subcarrier_spacing * OSR"""
    num_valid_subcarriers: int = 4096
    """有效子载波数，用于确定主带半宽。"""
    subcarrier_spacing: float = 30_000.0
    """子载波间隔 (Hz)。"""
    adjacent_subcarrier_count: int = 10
    """每侧邻道积分的子载波数 n。"""

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        ofdm_dict: Dict[str, Any],
        pulse_dict: Dict[str, Any],
    ) -> "AclrParams":
        n_fft = int(ofdm_dict.get("num_subcarriers", 4096))
        n_valid = int(ofdm_dict.get("num_valid_subcarriers", n_fft))
        scs = float(ofdm_dict.get("subcarrier_spacing", 30_000.0))
        osr = int(pulse_dict.get("samples_per_symbol", 4))

        fs = float(n_fft) * scs * float(osr)
        meas = float(n_valid) * scs

        bw_cfg = config_dict.get("bandwidth", None)
        if bw_cfg is None:
            raise ValueError(
                "须在 [aclr] 中设置 bandwidth (Hz)。MeasurementBandwidth 由 "
                "num_valid_subcarriers * subcarrier_spacing 自动计算。"
            )
        B = float(bw_cfg)
        default_offsets: Tuple[float, ...] = (-B, B)

        raw_off = config_dict.get("adjacent_channel_offset", None)
        if raw_off is None:
            offsets = default_offsets
        elif isinstance(raw_off, (list, tuple)):
            offsets = tuple(float(x) for x in raw_off)
        else:
            raise ValueError(
                "[aclr].adjacent_channel_offset 须为列表或省略；"
                "省略时按 ±bandwidth 自动计算一对一邻道中心。"
            )

        n_adj = int(config_dict.get("adjacent_subcarrier_count", 10))
        if n_adj <= 0:
            raise ValueError("[aclr].adjacent_subcarrier_count 须为正整数")

        inst = cls(
            bandwidth=B,
            measurement_bandwidth=meas,
            adjacent_channel_offset=offsets,
            sampling_rate=float(config_dict.get("sampling_rate", fs)),
            num_valid_subcarriers=n_valid,
            subcarrier_spacing=scs,
            adjacent_subcarrier_count=n_adj,
        )
        inst._verify_adjacent_bands()
        return inst

    def _verify_adjacent_bands(self) -> None:
        """校验左右邻道积分带落在奈奎斯特内。"""
        fs = float(self.sampling_rate)
        nyq = fs * 0.5
        f_edge = float(self.num_valid_subcarriers) * 0.5 * float(self.subcarrier_spacing)
        band_w = float(self.adjacent_subcarrier_count) * float(self.subcarrier_spacing)
        left_lo = -f_edge - band_w
        right_hi = f_edge + band_w
        if abs(left_lo) > nyq or abs(right_hi) > nyq:
            raise ValueError(
                f"邻道积分带超出奈奎斯特 ±{nyq/1e6:.4g} MHz；"
                "请减小 num_valid_subcarriers 或 adjacent_subcarrier_count。"
            )


@dataclass
class SystemParams:
    """系统配置"""

    carrier_frequency: float = 4.0e9
    """载波频率(Hz)"""
    snr_db: float = 10.0
    """信噪比(dB)，默认值为10dB"""
    quantization_bits: int = 16
    """量化比特数，用于量化压缩特征"""
    source: SourceParams = field(default_factory=SourceParams)
    """信源配置"""
    ldpc: LDPCParams = field(default_factory=LDPCParams)
    """LDPC配置"""
    qam: QAMParams = field(default_factory=QAMParams)
    """QAM配置"""
    tone_reservation: ToneReservationParams = field(default_factory=ToneReservationParams)
    """音调保留配置"""
    ofdm: OFDMParams = field(default_factory=OFDMParams)
    """OFDM配置"""
    pulse_shaping: PulseShapingParams = field(default_factory=PulseShapingParams)
    """脉冲成形配置"""
    power_amplifier: PowerAmplifierParams = field(default_factory=PowerAmplifierParams)
    """功率放大器配置"""
    aclr: AclrParams = field(default_factory=AclrParams)
    """NR ACLR 测量带宽、邻道偏移与期望采样率"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemParams":
        """从配置字典创建系统配置对象"""
        ofdm_raw: Dict[str, Any] = dict(config_dict.get("ofdm", {}) or {})
        pulse_raw: Dict[str, Any] = dict(config_dict.get("pulse_shaping", {}) or {})
        aclr_raw: Dict[str, Any] = dict(config_dict.get("aclr", {}) or {})

        return cls(
            carrier_frequency=config_dict.get("carrier_frequency", 4.0e9),
            snr_db=config_dict.get("snr_db", 10.0),
            quantization_bits=config_dict.get("quantization_bits", 16),
            source=SourceParams.from_dict(config_dict.get("source", {})),
            ldpc=LDPCParams.from_dict(config_dict.get("ldpc", {})),
            qam=QAMParams.from_dict(config_dict.get("qam", {})),
            tone_reservation=ToneReservationParams.from_dict(
                config_dict.get("tone_reservation", {})
            ),
            ofdm=OFDMParams.from_dict(ofdm_raw),
            pulse_shaping=PulseShapingParams.from_dict(pulse_raw),
            power_amplifier=PowerAmplifierParams.from_dict(config_dict.get("power_amplifier", {})),
            aclr=AclrParams.from_dict(aclr_raw, ofdm_raw, pulse_raw),
        )

    def waveform_sampling_rate_hz(self) -> float:
        """功放后时域复包络采样率：N_fft * subcarrier_spacing * OSR（Hz）。"""
        return (
            float(self.ofdm.num_subcarriers)
            * float(self.ofdm.subcarrier_spacing)
            * float(self.pulse_shaping.samples_per_symbol)
        )

    def verify_aclr_sampling_rate(self) -> None:
        """校验 OFDM 等效采样率与 [aclr].sampling_rate 一致。"""
        fs = self.waveform_sampling_rate_hz()
        tgt = float(self.aclr.sampling_rate)
        tol = max(1.0, 1e-9 * abs(tgt))
        if abs(fs - tgt) > tol:
            raise ValueError(
                f"OFDM 等效采样率 {fs:.6f} Hz 与 [aclr].sampling_rate={tgt:.6f} Hz 不一致；"
                "须满足 num_subcarriers * subcarrier_spacing * samples_per_symbol == sampling_rate。"
            )

    def format_aclr_parameters_table(self, *, tablefmt: str = "simple_grid") -> str:
        """将 ACLR 相关量格式化为 tabulate 表格（频率列单位为 MHz）。"""
        from tabulate import tabulate

        ac = self.aclr
        ofdm_sr = float(self.ofdm.num_subcarriers) * float(self.ofdm.subcarrier_spacing)
        wf_fs = self.waveform_sampling_rate_hz()
        mhz = 1e-6
        rows = [
            ("Bandwidth / MHz", f"{ac.bandwidth * mhz:.12g}"),
            ("MeasurementBandwidth / MHz", f"{ac.measurement_bandwidth * mhz:.12g}"),
            ("NumValidSubcarriers", str(int(ac.num_valid_subcarriers))),
            ("SubcarrierSpacing / MHz", f"{ac.subcarrier_spacing * mhz:.12g}"),
            ("AdjacentSubcarrierCount", str(int(ac.adjacent_subcarrier_count))),
            (
                "AdjacentChannelOffset / MHz",
                ", ".join(f"{x * mhz:.12g}" for x in ac.adjacent_channel_offset),
            ),
            ("SamplingRate / MHz", f"{ac.sampling_rate * mhz:.12g}"),
            ("OFDM sample rate (N_fft·Δf) / MHz", f"{ofdm_sr * mhz:.12g}"),
            ("Waveform Fs (N_fft·Δf·OSR) / MHz", f"{wf_fs * mhz:.12g}"),
            ("OSR (samples_per_symbol)", str(int(self.pulse_shaping.samples_per_symbol))),
        ]
        return tabulate(rows, headers=["参数", "数值"], tablefmt=tablefmt)
