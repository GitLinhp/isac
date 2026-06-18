"""
系统组件数据结构和配置类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator,
)

from .system_params import SystemParams
from ..channel.awgn import AWGN
from ..channel.channel import Channel
from ..channel.rt.rt_scene import RTScene
from ..channel.static_target_simulator import StaticTargetSimulator
from ..sensing.sensing_performance import SensingPerformance
from ..sensing.music_estimator import MUSICEstimator
from ..sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ..sensing.cfar import CFARDetector
from ..sensing.clutter_suppression import MovingTargetIndication, MovingTargetDetection
from ..zc_source import ZCSource


@dataclass
class SystemComponents:
    """系统组件（最小调用单元扁平字段）。"""

    # 基本组件
    binary_source: BinarySource = field(default_factory=BinarySource)
    """二进制源"""
    mapper: Mapper = field(default_factory=Mapper)
    """映射器"""
    demapper: Demapper = field(default_factory=Demapper)
    """解映射器"""
    zc_source: Optional[ZCSource] = None
    """Zadoff-Chu 源"""
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

    # 信道组件
    channel_type: Literal["rt", "rcs"] = "rt"
    """信道类型"""
    rt_channel: Optional[Channel] = None
    """RT信道"""
    awgn: AWGN = field(default_factory=AWGN)
    """AWGN信道"""

    # 感知组件
    sensing_performance: SensingPerformance = field(default_factory=SensingPerformance)
    """感知性能"""
    delay_doppler_spectrum: DelayDopplerSpectrum = field(
        default_factory=DelayDopplerSpectrum
    )
    """延迟多普勒谱"""
    music_estimator: MUSICEstimator = field(default_factory=MUSICEstimator)
    """MUSIC估计器"""
    cfar: CFARDetector = field(default_factory=CFARDetector)
    """CFAR检测器"""
    moving_target_indication: MovingTargetIndication = field(
        default_factory=MovingTargetIndication
    )
    """移动目标指示"""
    moving_target_detection: MovingTargetDetection = field(
        default_factory=MovingTargetDetection
    )
    """移动目标检测"""
    rt_scene: Optional[RTScene] = None
    """RT场景"""
    static_target_sim: Optional[StaticTargetSimulator] = None
    """静态目标模拟器"""

    def apply_channel(
        self,
        inputs: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        if self.channel_type == "rcs":
            if domain != "time":
                raise ValueError("channel.type='rcs' 仅支持 domain='time'")
            y_clean = self.static_target_sim(inputs)
        else:
            y_clean = self.rt_channel(inputs, domain=domain, snr_db=None)
        if snr_db is None:
            return y_clean
        return self.awgn(y_clean, snr_db)

    def cfr_per_tx(
        self,
        rt_scene: RTScene,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64,
    ) -> dict[str, torch.Tensor]:
        """按发射机分离的 OFDM 频域信道（仅 ``channel_type='rt'``）。"""
        if self.channel_type != "rt" or self.rt_channel is None:
            raise ValueError("cfr_per_tx 仅适用于 channel.type='rt'")
        return self.rt_channel.cfr_per_tx(
            rt_scene,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str = "cuda:0",
    ) -> "SystemComponents":
        qam = system_params.qam
        source_p = system_params.source
        ofdm = system_params.ofdm
        stream_p = system_params.stream_management
        channel_p = system_params.channel
        windows = system_params.windows
        cfar_p = system_params.cfar
        mti_p = system_params.mti
        mtd_p = system_params.mtd
        carrier_frequency = system_params.carrier_frequency

        sm = StreamManagement(
            np.array(stream_p.rx_tx_association),
            stream_p.num_streams,
        )

        # 基本组件
        binary_source = BinarySource(device=device)
        mapper = Mapper("qam", qam.num_bits_per_symbol, device=device)
        demapper = Demapper(
            "app",
            "qam",
            qam.num_bits_per_symbol,
            hard_out=True,
            device=device,
        )
        zc_source: Optional[ZCSource] = None
        if source_p.type == "zc":
            zc_source = ZCSource(
                root_index=source_p.root_index,
                normalize=source_p.normalize,
                device=device,
            )
        rg = ResourceGrid(
            num_ofdm_symbols=ofdm.num_symbols,
            fft_size=ofdm.num_subcarriers,
            subcarrier_spacing=ofdm.subcarrier_spacing,
            cyclic_prefix_length=ofdm.cyclic_prefix_length,
            dc_null=ofdm.dc_null,
            device=device,
        )
        rg_mapper = ResourceGridMapper(rg, device=device)
        rg_demapper = ResourceGridDemapper(rg, sm, device=device)
        modulator = OFDMModulator(
            cyclic_prefix_length=ofdm.cyclic_prefix_length,
            device=device,
        )
        demodulator = OFDMDemodulator(
            fft_size=ofdm.num_subcarriers,
            l_min=ofdm.l_min,
            cyclic_prefix_length=ofdm.cyclic_prefix_length,
            device=device,
        )

        # 信道组件（rt_scene / static_target_sim 须先于 rt_channel 构建）
        channel_type = channel_p.type
        rt_scene: Optional[RTScene] = None
        if system_params.rt_scene is not None:
            rt_scene = RTScene(scene_params=system_params.rt_scene)
            rt_scene.frequency = float(carrier_frequency)
            rt_scene.bandwidth = float(rg.bandwidth)
        static_target_sim: Optional[StaticTargetSimulator] = None
        if system_params.static_target is not None:
            st_params = system_params.static_target
            st_params.ensure_phy()
            static_target_sim = StaticTargetSimulator(st_params)
        rt_channel: Optional[Channel] = None
        if channel_type == "rt":
            if rt_scene is None:
                raise ValueError("channel.type='rt' 要求已构建 rt_scene")
            rt_channel = Channel(rg=rg, paths=lambda: rt_scene.paths)
        elif static_target_sim is None:
            raise ValueError("channel.type='rcs' 要求已构建 static_target")
        awgn = AWGN(device=device)

        # 感知组件
        sensing_performance = SensingPerformance(
            resource_grid=rg,
            carrier_frequency=carrier_frequency,
        )
        delay_doppler_spectrum = DelayDopplerSpectrum(
            sensing_performance=sensing_performance,
            delay_window=windows.delay_window,
            doppler_window=windows.doppler_window,
        )
        music_estimator = MUSICEstimator(
            device=device, sensing_performance=sensing_performance
        )
        cfar = CFARDetector(
            cfar_type=cfar_p.type,
            k=cfar_p.k,
            guard=cfar_p.guard,
            trailing=cfar_p.trailing,
            pfa=cfar_p.pfa,
            detector=cfar_p.detector,
            offset=cfar_p.offset,
        )
        moving_target_indication = MovingTargetIndication(
            sensing_performance,
            filter_order=mti_p.filter_order,
            prf=mti_p.prf,
        )
        moving_target_detection = MovingTargetDetection(
            sensing_performance,
            carrier_frequency,
            num_filters=mtd_p.num_filters,
        )

        return cls(
            binary_source=binary_source,
            mapper=mapper,
            demapper=demapper,
            zc_source=zc_source,
            rg=rg,
            rg_mapper=rg_mapper,
            rg_demapper=rg_demapper,
            modulator=modulator,
            demodulator=demodulator,
            channel_type=channel_type,
            rt_channel=rt_channel,
            awgn=awgn,
            sensing_performance=sensing_performance,
            delay_doppler_spectrum=delay_doppler_spectrum,
            music_estimator=music_estimator,
            cfar=cfar,
            moving_target_indication=moving_target_indication,
            moving_target_detection=moving_target_detection,
            rt_scene=rt_scene,
            static_target_sim=static_target_sim,
        )
