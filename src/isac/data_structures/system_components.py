"""
系统组件数据结构和配置类

将 ``SystemParams``（TOML 解析结果）实例化为可调用的 Sionna / ISAC 运行时对象。
构建入口：``SystemComponents.build_from_params``。
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

from .params.system_params import SystemParams
from ..channel import Channel, RCSChannel, RCSScene, RTChannel, RTSimulator
from ..sensing.sensing_performance import SensingPerformance
from ..sensing.ls_channel_estimator import LSChannelEstimator
from ..sensing.music_estimator import MUSICEstimator
from ..sensing.delay_doppler_spectrum import DelayDopplerSpectrum
from ..sensing.cfar import CFARDetector
from ..sensing.clutter_suppression import MovingTargetIndication, MovingTargetDetection
from ..zc_source import ZCSource


@dataclass
class SystemComponents:
    """系统运行时组件（扁平 Optional 字段，顺序对齐 ``SystemParams``）。

    由 ``build_from_params`` 分三阶段组装：基本（信源 + OFDM）→ 信道 → 感知。
    未配置的 TOML 段对应字段保持 ``None``。
    """

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
    channel: Optional[Channel] = None
    """统一信道入口（``RTChannel`` 频域 / ``RCSChannel`` 时域，可选 AWGN）"""
    rt_simulator: Optional[RTSimulator] = None
    """RT 仿真器；``channel.type='rt'`` 时构建"""
    rcs_scene: Optional[RCSScene] = None
    """RCS 点目标场景；``channel.type='rcs'`` 时构建"""

    # 感知组件
    sensing_performance: Optional[SensingPerformance] = None
    """感知性能（距离/速度分辨率等）；感知链子组件的公共依赖"""
    ls_channel_estimator: Optional[LSChannelEstimator] = None
    """频域 LS 信道估计"""
    moving_target_indication: Optional[MovingTargetIndication] = None
    """移动目标指示"""
    moving_target_detection: Optional[MovingTargetDetection] = None
    """移动目标检测"""
    delay_doppler_spectrum: Optional[DelayDopplerSpectrum] = None
    """延迟多普勒谱"""
    cfar_detector: Optional[CFARDetector] = None
    """CFAR检测器"""
    music_estimator: Optional[MUSICEstimator] = None
    """MUSIC估计器"""

    @classmethod
    def build_from_params(
        cls,
        system_params: SystemParams,
        device: str = "cuda:0",
    ) -> "SystemComponents":
        """由 ``SystemParams`` 构建完整组件集。

        Args:
            system_params: 已解析且通过校验的系统参数。
            device: Sionna / Torch 计算设备字符串。

        Returns:
            填充了对应 Optional 字段的 ``SystemComponents`` 实例。
        """
        kwargs, rg = cls._build_basic(system_params, device)
        # 信道依赖 ResourceGrid；无 [ofdm] 或 [channel] 时跳过
        if system_params.channel is not None and rg is not None:
            kwargs.update(cls._build_channel(system_params, rg, device))
        kwargs.update(cls._build_sensing(system_params, rg, device))
        return cls(**kwargs)

    @staticmethod
    def _build_basic(
        system_params: SystemParams,
        device: str,
    ) -> tuple[dict, Optional[ResourceGrid]]:
        """构建基本组件：信源分支 + OFDM 链。

        ``source.type`` 为 ``binary`` / ``zc`` 互斥；``rg_demapper`` 仅在有
        ``[stream_management]`` 时创建（通信解调用）。

        Returns:
            (kwargs, rg): 待合并的字段字典与 ``ResourceGrid``（无 ``[ofdm]`` 时为 ``None``）。
        """
        kwargs: dict = {}
        rg: Optional[ResourceGrid] = None

        # --- 信源：binary → binary_source + mapper + demapper；zc → zc_source ---
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

        # --- OFDM：rg → rg_mapper / modulator / demodulator；可选 rg_demapper ---
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
            if system_params.stream_management is not None:
                sm = StreamManagement(
                    np.array(system_params.stream_management.rx_tx_association),
                    system_params.stream_management.num_streams,
                )
                kwargs["rg_demapper"] = ResourceGridDemapper(rg, sm, device=device)
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

        return kwargs, rg

    @staticmethod
    def _build_channel(
        system_params: SystemParams,
        rg: ResourceGrid,
        device: str,
    ) -> dict:
        """构建信道组件，按 ``channel.type`` 分发。

        - ``rt``：由 ``[rt_simulator]`` 构建 ``RTSimulator``，``RTChannel`` 经 ``paths`` 回调取路径。
        - ``rcs``：由 ``RCSScene`` + ``Callable`` 构建 ``RCSChannel``。

        前置校验由 ``SystemParams._validate_channel_dependencies`` 完成。
        """
        channel_cfg = system_params.channel
        carrier_frequency = system_params.carrier_frequency

        match channel_cfg.type:
            case "rt":
                rt_simulator = RTSimulator(
                    rt_simulator_params=system_params.rt_simulator,
                    frequency=carrier_frequency,
                    bandwidth=float(rg.bandwidth),
                )
                return {
                    "rt_simulator": rt_simulator,
                    "channel": RTChannel(
                        rg=rg,
                        # 延迟绑定：paths 在每次信道调用时按当前场景状态求解
                        paths=lambda: rt_simulator.paths,
                        device=device,
                    ),
                }
            case "rcs":
                rcs_scene = RCSScene.from_params(system_params.rcs_scene)
                return {
                    "rcs_scene": rcs_scene,
                    "channel": RCSChannel(
                        rcs_scene=lambda: rcs_scene,
                        center_freq=float(carrier_frequency),
                        samp_rate=float(system_params.ofdm.samp_rate),
                        device=device,
                    ),
                }
            case _:
                raise ValueError(f"unsupported channel.type: {channel_cfg.type!r}")

    @staticmethod
    def _build_sensing(
        system_params: SystemParams,
        rg: Optional[ResourceGrid],
        device: str,
    ) -> dict:
        """构建感知链组件。

        ``sensing_performance`` 需 ``[ofdm]`` 与 ``carrier_frequency``；MTI/MTD/DD/CFAR/MUSIC
        各自依赖对应 TOML 段且以 ``sensing_performance`` 为公共输入。
        ``ls_channel_estimator`` 仅需 ``rg``，可独立于载频存在。
        """
        kwargs: dict = {}
        carrier_frequency = system_params.carrier_frequency
        sensing_performance: Optional[SensingPerformance] = None

        if system_params.ofdm is not None and carrier_frequency is not None:
            sensing_performance = SensingPerformance(
                resource_grid=rg,
                carrier_frequency=carrier_frequency,
            )
            kwargs["sensing_performance"] = sensing_performance

        if rg is not None:
            kwargs["ls_channel_estimator"] = LSChannelEstimator(rg)

        # 以下子组件共享 sensing_performance，按 params 段独立开关
        if sensing_performance is not None:
            if system_params.mti is not None:
                kwargs["moving_target_indication"] = MovingTargetIndication(
                    sensing_performance,
                    filter_order=system_params.mti.filter_order,
                    prf=system_params.mti.prf,
                )

            if system_params.mtd is not None:
                kwargs["moving_target_detection"] = MovingTargetDetection(
                    sensing_performance,
                    carrier_frequency,
                    num_filters=system_params.mtd.num_filters,
                )

            if system_params.windows is not None:
                kwargs["delay_doppler_spectrum"] = DelayDopplerSpectrum(
                    sensing_performance=sensing_performance,
                    delay_window=system_params.windows.delay_window,
                    doppler_window=system_params.windows.doppler_window,
                )

            if system_params.cfar is not None:
                kwargs["cfar_detector"] = CFARDetector(
                    cfar_type=system_params.cfar.type,
                    k=system_params.cfar.k,
                    guard=system_params.cfar.guard,
                    trailing=system_params.cfar.trailing,
                    pfa=system_params.cfar.pfa,
                    detector=system_params.cfar.detector,
                    offset=system_params.cfar.offset,
                )

            if system_params.music is not None:
                kwargs["music_estimator"] = MUSICEstimator(
                    device=device,
                    sensing_performance=sensing_performance,
                    near_range_guard_m=system_params.music.near_range_guard_m,
                )

        return kwargs
