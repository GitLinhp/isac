"""
系统组件数据结构和配置类
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
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
from ..channel.isac_channel import IsacChannel
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
    binary_source: Optional[BinarySource] = None
    """二进制源"""
    mapper: Optional[Mapper] = None
    """映射器"""
    demapper: Optional[Demapper] = None
    """解映射器"""
    zc_source: Optional[ZCSource] = None
    """Zadoff-Chu 源"""
    rg: Optional[ResourceGrid] = None
    """资源网格"""
    rg_mapper: Optional[ResourceGridMapper] = None
    """资源网格映射器"""
    rg_demapper: Optional[ResourceGridDemapper] = None
    """资源网格解映射器"""
    modulator: Optional[OFDMModulator] = None
    """OFDM调制器"""
    demodulator: Optional[OFDMDemodulator] = None
    """OFDM解调器"""

    # 信道组件
    channel: Optional[IsacChannel] = None
    """统一信道（RT / RCS + AWGN）"""
    rt_scene: Optional[RTScene] = None
    """RT场景"""
    static_target_sim: Optional[StaticTargetSimulator] = None
    """静态目标模拟器"""

    # 感知组件
    sensing_performance: Optional[SensingPerformance] = None
    """感知性能"""
    moving_target_indication: Optional[MovingTargetIndication] = None
    """移动目标指示"""
    moving_target_detection: Optional[MovingTargetDetection] = None
    """移动目标检测"""
    delay_doppler_spectrum: Optional[DelayDopplerSpectrum] = None
    """延迟多普勒谱"""
    cfar: Optional[CFARDetector] = None
    """CFAR检测器"""
    music_estimator: Optional[MUSICEstimator] = None
    """MUSIC估计器"""

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str = "cuda:0",
    ) -> "SystemComponents":
        kwargs: dict = {}
        carrier_frequency = system_params.carrier_frequency
        rg: Optional[ResourceGrid] = None
        sensing_performance: Optional[SensingPerformance] = None

        # 信源（互斥）
        if system_params.source is not None:
            source = system_params.source
            if source.type == "binary":
                n_bps = source.num_bits_per_symbol
                if n_bps is None:
                    raise ValueError(
                        "source.type='binary' 要求在 [source] 中配置 num_bits_per_symbol"
                    )
                kwargs["binary_source"] = BinarySource(device=device)
                kwargs["mapper"] = Mapper("qam", n_bps, device=device)
                kwargs["demapper"] = Demapper(
                    "app",
                    "qam",
                    n_bps,
                    hard_out=True,
                    device=device,
                )
            elif source.type == "zc":
                kwargs["zc_source"] = ZCSource(
                    root_index=source.root_index,
                    normalize=source.normalize,
                    device=device,
                )
            else:
                raise ValueError(f"unsupported source.type: {source.type!r}")

        # OFDM
        if system_params.ofdm is not None:
            ofdm = system_params.ofdm
            rg = ResourceGrid(
                num_ofdm_symbols=ofdm.num_symbols,
                fft_size=ofdm.fft_size,
                subcarrier_spacing=ofdm.subcarrier_spacing,
                cyclic_prefix_length=ofdm.cyclic_prefix_length,
                dc_null=ofdm.dc_null,
                device=device,
            )
            kwargs["rg"] = rg
            kwargs["rg_mapper"] = ResourceGridMapper(rg, device=device)
            kwargs["modulator"] = OFDMModulator(
                cyclic_prefix_length=ofdm.cyclic_prefix_length,
                device=device,
            )
            kwargs["demodulator"] = OFDMDemodulator(
                fft_size=ofdm.fft_size,
                l_min=ofdm.l_min,
                cyclic_prefix_length=ofdm.cyclic_prefix_length,
                device=device,
            )

            if system_params.stream_management is not None:
                sm = StreamManagement(
                    np.array(system_params.stream_management.rx_tx_association),
                    system_params.stream_management.num_streams,
                )
                kwargs["rg_demapper"] = ResourceGridDemapper(rg, sm, device=device)

        # 信道（rt_scene / static_target_sim 须先于 RT 后端构建）
        if system_params.channel is not None and rg is not None:
            channel = system_params.channel
            rt_scene: Optional[RTScene] = None
            if system_params.rt_scene is not None:
                rt_scene = RTScene(scene_params=system_params.rt_scene)
                if carrier_frequency is not None:
                    rt_scene.frequency = float(carrier_frequency)
                rt_scene.bandwidth = float(rg.bandwidth)
                kwargs["rt_scene"] = rt_scene

            static_target_sim: Optional[StaticTargetSimulator] = None
            if system_params.static_target is not None:
                st_params = system_params.static_target
                st_params.ensure_phy()
                static_target_sim = StaticTargetSimulator(st_params)
                kwargs["static_target_sim"] = static_target_sim

            rt_channel: Optional[Channel] = None
            if channel.type == "rt":
                if rt_scene is None:
                    raise ValueError("channel.type='rt' 要求已构建 rt_scene")
                rt_channel = Channel(rg=rg, paths=lambda: rt_scene.paths)
            elif static_target_sim is None:
                raise ValueError("channel.type='rcs' 要求已构建 static_target")

            kwargs["channel"] = IsacChannel(
                channel_type=channel.type,
                default_snr_db=channel.snr_db,
                rt=rt_channel,
                static_target_sim=static_target_sim,
                awgn=AWGN(device=device),
            )

        # 感知
        if system_params.ofdm is not None and carrier_frequency is not None:
            sensing_performance = SensingPerformance(
                resource_grid=rg,
                carrier_frequency=carrier_frequency,
            )
            kwargs["sensing_performance"] = sensing_performance

            if system_params.mti is not None:
                mti = system_params.mti
                kwargs["moving_target_indication"] = MovingTargetIndication(
                    sensing_performance,
                    filter_order=mti.filter_order,
                    prf=mti.prf,
                )

            if system_params.mtd is not None:
                mtd = system_params.mtd
                kwargs["moving_target_detection"] = MovingTargetDetection(
                    sensing_performance,
                    carrier_frequency,
                    num_filters=mtd.num_filters,
                )

            if system_params.windows is not None:
                windows = system_params.windows
                kwargs["delay_doppler_spectrum"] = DelayDopplerSpectrum(
                    sensing_performance=sensing_performance,
                    delay_window=windows.delay_window,
                    doppler_window=windows.doppler_window,
                )

            if system_params.cfar is not None:
                cfar = system_params.cfar
                kwargs["cfar"] = CFARDetector(
                    cfar_type=cfar.type,
                    k=cfar.k,
                    guard=cfar.guard,
                    trailing=cfar.trailing,
                    pfa=cfar.pfa,
                    detector=cfar.detector,
                    offset=cfar.offset,
                )

            if system_params.music is not None:
                music = system_params.music
                kwargs["music_estimator"] = MUSICEstimator(
                    device=device,
                    sensing_performance=sensing_performance,
                    near_range_guard_m=music.near_range_guard_m,
                )

        return cls(**kwargs)
