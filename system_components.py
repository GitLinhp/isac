"""
系统组件数据结构和配置类
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper, SymbolDemapper
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.signal import RootRaisedCosineFilter, Upsampling, Downsampling
from sionna.phy.channel import AWGN

from ..image_source import ImageSource
from ..ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator,
)
from .system_params import SystemParams

from ..tone_reservation import ToneReservation
from ..power_amplifier import Rapp
from .batch_metrics import BatchStatistics
from ..utils.datasets import CIFAR10Dataset, DATASET_DIR, DEFAULT_NUM_WORKERS

if TYPE_CHECKING:
    from ..models.importance_analyzer import ImportanceAnalyzer


class LearnableConstellationMapping(nn.Module):
    """可学习 QAM 星座：Mapper / Demapper / SymbolDemapper 共享 Constellation。"""

    def __init__(self, num_bits: int, device: str) -> None:
        super().__init__()
        qam_points = Constellation("qam", num_bits, device=device).points
        self.points_r = nn.Parameter(qam_points.real.clone())
        self.points_i = nn.Parameter(qam_points.imag.clone())
        self.constellation = Constellation(
            "custom",
            num_bits,
            points=torch.complex(self.points_r, self.points_i),
            normalize=True,
            center=True,
            device=device,
        )
        self.mapper = Mapper(constellation=self.constellation, device=device)
        self.demapper = Demapper("app", constellation=self.constellation, device=device)
        self.symbol_demapper = SymbolDemapper(
            constellation=self.constellation, hard_out=True, device=device
        )
        self.set_trainable(False)

    def bind_points(self) -> None:
        """前向/加载时将可训练 r/i 写入 ``constellation.points``。"""
        self.constellation.points = torch.complex(self.points_r, self.points_i)

    def set_trainable(self, trainable: bool) -> None:
        """是否训练可学习星座（仅 ``points_r`` / ``points_i``）。"""
        self.points_r.requires_grad_(trainable)
        self.points_i.requires_grad_(trainable)


@dataclass
class SystemComponents:
    """系统组件"""

    binary_source: BinarySource = field(default_factory=BinarySource)
    """二进制源"""
    image_source: ImageSource = field(default_factory=ImageSource)
    """图像源"""
    image_dataset: Optional[CIFAR10Dataset] = field(default=None)
    """CIFAR-10 仿真数据集（与 E2E 训练共用子集抽样方式）"""
    image_loader: Optional[DataLoader] = field(default=None)
    """CIFAR-10 仿真 DataLoader（shuffle=False）"""
    constellation_mapping: Optional[LearnableConstellationMapping] = None
    """可学习星座映射模块（Mapper/Demapper/SymbolDemapper 共享）"""
    mapper: Mapper = field(default_factory=Mapper)
    """映射器"""
    demapper: Demapper = field(default_factory=Demapper)
    """解映射器"""
    symbol_demapper: SymbolDemapper = field(default_factory=SymbolDemapper)
    """符号解映射器（与 Mapper/Demapper 共享星座；hard_out=True 用于 SER）"""
    encoder: LDPC5GEncoder = field(default_factory=LDPC5GEncoder)
    """LDPC编码器"""
    decoder: LDPC5GDecoder = field(default_factory=LDPC5GDecoder)
    """LDPC解码器"""
    rg: ResourceGrid = field(default_factory=ResourceGrid)
    """资源网格"""
    rg_mapper: ResourceGridMapper = field(default_factory=ResourceGridMapper)
    """资源网格映射器"""
    rg_demapper: ResourceGridDemapper = field(default_factory=ResourceGridDemapper)
    """资源网格解映射器"""
    modulator: OFDMModulator = field(default_factory=OFDMModulator)
    """OFDM调制器"""
    demodulator: OFDMDemodulator = field(default_factory=OFDMDemodulator)
    """OFDM解调器"""
    us: Upsampling = field(default_factory=Upsampling)
    """上采样器"""
    ds: Downsampling = field(default_factory=Downsampling)
    """下采样器"""
    rrcf: RootRaisedCosineFilter = field(default_factory=RootRaisedCosineFilter)
    """脉冲成形滤波器"""
    awgn: AWGN = field(default_factory=AWGN)
    """AWGN信道"""
    rapp: Optional[Rapp] = field(default=None)
    """Rapp 功率放大器（与 E2E / Traditional 仿真共用）"""
    tone_reservation: Optional[ToneReservation] = field(default_factory=lambda: None)
    """音调保留器；当 num_tone_reservation=0 时为 None（表示不进行音调保留）"""

    importance_analyzer: Optional["ImportanceAnalyzer"] = field(default=None)
    """GAN 特征 importance 分析器（需加载 GAN checkpoint 后通过 init_importance_analyzer 设置）"""
    importance_mask_e2e: Optional[torch.Tensor] = field(default=None)
    """E2E 分段传输 importance 掩码（qam_importance 复制拼接）"""

    batch_statistics: BatchStatistics = field(default_factory=BatchStatistics)
    """批量性能指标"""

    def init_importance_analyzer(
        self,
        gan_model: nn.Module,
        system_params: SystemParams,
        device: str,
    ) -> torch.Tensor:
        """从 GAN 信源模型构建 ImportanceAnalyzer 与 E2E 分段 importance 掩码。"""
        from ..models.importance_analyzer import ImportanceAnalyzer

        self.importance_analyzer = ImportanceAnalyzer(
            model=gan_model,
            normalize=True,
            quantize_bits=system_params.quantization_bits,
            ldpc_encoder=self.encoder,
            num_bits_per_symbol=system_params.qam.num_bits_per_symbol,
        )
        mask = self.importance_analyzer.qam_importance.to(device)
        self.importance_mask_e2e = torch.cat([mask, mask])
        return self.importance_mask_e2e

    @classmethod
    def from_system_params(
        cls,
        system_params: SystemParams,
        device: str,
        *,
        batch_size: int = 128,
        seed: int = 42,
    ) -> "SystemComponents":
        """从系统参数创建系统组件"""
        binary_source = BinarySource(device=device)
        image_source = ImageSource(
            num_images=system_params.source.num_images,
            dataset_name=system_params.source.dataset_name,
            split=system_params.source.split,
            compress_type=system_params.source.compress_type,
            quality=system_params.source.quality,
        )
        constellation_mapping = LearnableConstellationMapping(
            system_params.qam.num_bits_per_symbol, device
        )
        mapper = constellation_mapping.mapper
        demapper = constellation_mapping.demapper
        symbol_demapper = constellation_mapping.symbol_demapper
        # 初始化LDPC编码器和解码器
        encoder = LDPC5GEncoder(
            system_params.ldpc.infolength, system_params.ldpc.codelength, device=device
        )
        decoder = LDPC5GDecoder(
            encoder,
            hard_out=False,
            return_infobits=True,
            device=device,
        )

        tone_reservation: Optional[ToneReservation]
        tone_mask: torch.Tensor
        if (
            system_params.tone_reservation.num_tone_reservation
            and system_params.tone_reservation.num_tone_reservation > 0
        ):
            tone_reservation = ToneReservation(
                num_subcarriers=system_params.ofdm.num_subcarriers,
                num_tone_reservation=system_params.tone_reservation.num_tone_reservation,
                device=device,
                tone_selection_mode=system_params.tone_reservation.tone_selection_mode,
                random_seed=system_params.tone_reservation.random_seed,
            )
            tone_mask = tone_reservation.tone_mask
        else:
            # 不保留音调：数据子载波 = 全部子载波
            tone_reservation = None
            tone_mask = torch.ones(
                system_params.ofdm.num_subcarriers,
                dtype=torch.float32,
                device=torch.device(device),
            )

        # 初始化资源网格及映射器和解映射器
        rg = ResourceGrid(
            system_params.ofdm.num_symbols,
            system_params.ofdm.num_subcarriers,
            system_params.ofdm.num_valid_subcarriers,
            tone_mask,
            system_params.ofdm.subcarrier_spacing,
            device=device,
        )
        rg_mapper = ResourceGridMapper(rg, device=device)
        rg_demapper = ResourceGridDemapper(rg, device=device)
        # 初始化OFDM调制器和解调器
        ofdm_modulator = OFDMModulator(
            cyclic_prefix_length=system_params.ofdm.cyclic_prefix_length,
            device=device,
        )
        ofdm_demodulator = OFDMDemodulator(
            fft_size=system_params.ofdm.num_subcarriers,
            cyclic_prefix_length=system_params.ofdm.cyclic_prefix_length,
            device=device,
        )
        # 初始化脉冲成形滤波器和上采样器
        us = Upsampling(
            system_params.pulse_shaping.samples_per_symbol, normalize=True, device=device
        )
        rrcf = RootRaisedCosineFilter(
            system_params.pulse_shaping.span_in_symbols,
            system_params.pulse_shaping.samples_per_symbol,
            system_params.pulse_shaping.beta,
            window=system_params.pulse_shaping.window,
            normalize=True,
            device=device,
        )
        ds = Downsampling(
            system_params.pulse_shaping.samples_per_symbol,
            rrcf.length - 1,
            num_symbols=system_params.ofdm.num_subcarriers
            * system_params.pulse_shaping.samples_per_symbol,
            normalize=True,
            device=device,
        )
        # 初始化AWGN信道
        awgn = AWGN(device=device)
        rapp = Rapp(
            ibo_db=system_params.power_amplifier.ibo_db,
            p=system_params.power_amplifier.p_param,
            device=device,
        )
        # CIFAR-10 仿真数据集与 DataLoader
        image_dataset: Optional[CIFAR10Dataset] = None
        image_loader: Optional[DataLoader] = None
        if system_params.source.dataset_name.lower() == "cifar10":
            image_dataset = CIFAR10Dataset(
                root=DATASET_DIR,
                split=system_params.source.split,
                transform=ToTensor(),
                download=True,
                num_samples=system_params.source.num_images,
                subset_seed=seed,
            )
            image_loader = DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=DEFAULT_NUM_WORKERS,
                pin_memory=True,
            )
        # 初始化批次性能指标
        batch_statistics: BatchStatistics = BatchStatistics()

        return cls(
            binary_source=binary_source,
            image_source=image_source,
            image_dataset=image_dataset,
            image_loader=image_loader,
            constellation_mapping=constellation_mapping,
            mapper=mapper,
            demapper=demapper,
            symbol_demapper=symbol_demapper,
            encoder=encoder,
            decoder=decoder,
            tone_reservation=tone_reservation,
            rg=rg,
            rg_mapper=rg_mapper,
            rg_demapper=rg_demapper,
            modulator=ofdm_modulator,
            demodulator=ofdm_demodulator,
            rrcf=rrcf,
            us=us,
            ds=ds,
            awgn=awgn,
            rapp=rapp,
            batch_statistics=batch_statistics,
        )
