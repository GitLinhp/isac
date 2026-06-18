"""
系统组件聚合（与 ``system_params`` 对应）
"""

from dataclasses import dataclass, field
from typing import Optional

from sionna.phy.mapping import BinarySource, Mapper, Demapper
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    OFDMModulator,
    OFDMDemodulator,
)

from ..params import SystemParams
from ...channel.rt.rt_scene import RTScene
from ...sensing.sensing_performance import SensingPerformance
from ...sensing.music_estimator import MUSICEstimator
from ...sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ...sensing.cfar import CFARDetector, default_cfar_detector
from ...sensing.clutter_suppression import MovingTargetIndication, MovingTargetDetection
from ...channel.channel import Channel
from ...zc_source import ZCSource

from .channel_components import ChannelComponents
from .ofdm_components import OFDMComponents
from .rt_scene_components import RTSceneComponents
from .sensing_components import SensingComponents


@dataclass
class SystemComponents:
    """系统组件"""

    binary_source: BinarySource = field(default_factory=BinarySource)
    """二进制源"""
    mapper: Mapper = field(default_factory=Mapper)
    """映射器"""
    demapper: Demapper = field(default_factory=Demapper)
    """解映射器"""
    rg: ResourceGrid = field(default_factory=ResourceGrid)
    """资源网格"""
    rg_mapper: ResourceGridMapper = field(default_factory=ResourceGridMapper)
    """资源网格映射器"""
    rg_demapper: ResourceGridDemapper = field(default_factory=ResourceGridDemapper)
    """资源网格解映射器"""
    modulator: OFDMModulator = field(default_factory=OFDMModulator)
    """OFDM调制器"""
    demodulator: OFDMDemodulator = field(default_factory=OFDMDemodulator)
    """OFDM 解调器（由 ``OFDMComponents.build_from_params`` 按 ``SystemParams.ofdm`` 构造，含 ``l_min`` 等）。"""
    rt_scene: Optional[RTScene] = None
    """射线追踪场景"""
    channel: Channel = field(default_factory=Channel)
    """信道"""
    sensing_performance: SensingPerformance = field(default_factory=SensingPerformance)
    """感知性能"""
    delay_doppler_spectrum: DelayDopplerSpectrum = field(default_factory=DelayDopplerSpectrum)
    """时延多普勒谱"""
    cfar_detector: CFARDetector = field(default_factory=default_cfar_detector)
    """CFAR 检测器实例（由 ``SystemComponents.build_from_params`` 按 ``SystemParams.sensing.cfar`` 构造）。"""
    music_estimator: MUSICEstimator = field(default_factory=MUSICEstimator)
    """MUSIC估计器"""
    moving_target_indication: MovingTargetIndication = field(
        default_factory=lambda: MovingTargetIndication(SensingPerformance(), filter_order=1)
    )
    """动目标显示（MTI），由 ``SensingComponents.build_from_params`` 构造。"""
    moving_target_detection: MovingTargetDetection = field(
        default_factory=lambda: MovingTargetDetection(SensingPerformance(), 0.0)
    )
    """动目标检测（MTD），由 ``SensingComponents.build_from_params`` 构造。"""
    zc_source: Optional[ZCSource] = None
    """Zadoff-Chu 源（仅 ``sensing.source.type == 'zc'`` 时由 ``OFDMComponents.build_from_params`` 构造）。"""

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str = "cuda:0",
    ) -> "SystemComponents":
        """从系统参数创建系统组件"""
        ofdm = OFDMComponents.build_from_params(system_params, device=device)
        rt = RTSceneComponents.build_from_params(system_params, resource_grid=ofdm.rg)
        chan = ChannelComponents.build_from_params(ofdm.rg, rt.rt_scene)
        sens = SensingComponents.build_from_params(system_params, ofdm.rg, device=device)

        return cls(
            binary_source=ofdm.binary_source,  # 二进制源
            mapper=ofdm.mapper,  # 映射器
            demapper=ofdm.demapper,  # 解映射器
            rg=ofdm.rg,  # 资源网格
            rg_mapper=ofdm.rg_mapper,  # 资源网格映射器
            rg_demapper=ofdm.rg_demapper,  # 资源网格解映射器
            modulator=ofdm.modulator,  # OFDM调制器
            demodulator=ofdm.demodulator,  # OFDM解调器
            rt_scene=rt.rt_scene,  # 射线追踪场景
            sensing_performance=sens.sensing_performance,  # 感知性能
            channel=chan.channel,  # 信道
            delay_doppler_spectrum=sens.delay_doppler_spectrum,  # 时延多普勒谱
            cfar_detector=sens.cfar,  # CFAR 检测器（构建于 sensing_components）
            music_estimator=sens.music_estimator,  # MUSIC估计器
            moving_target_indication=sens.moving_target_indication,
            moving_target_detection=sens.moving_target_detection,
            zc_source=ofdm.zc_source,
        )
