"""ISAC 端到端仿真编排：发射、接收与感知流水线 API。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from sionna.phy import config as sn_config
import torch

from .data_structures import SystemComponents, SystemParams
from .utils import match_peaks_and_compute_radial_rmse

# 单基地（colocated 往返）或双基地（单程几何路径）感知模式
SensMode = Literal["monostatic", "bistatic"]
# 谱图/MUSIC 日志展示单位；``dd``/``rv`` 为别名
MetricMode = Literal["delay_doppler", "range_velocity", "dd", "rv"]


@dataclass
class MusicEstimate:
    """``System.estimate_sensing_music`` 的返回结构。"""

    est_ranges: torch.Tensor  # 估计距离 (m)，一维
    est_velocities: torch.Tensor  # 估计径向/几何速度 (m/s)，一维


@dataclass
class SensingRmse:
    """``System.evaluate_sensing_rmse`` 的返回结构。"""

    rmse_range_m: torch.Tensor  # 距离 RMSE (m)，标量
    rmse_velocity_mps: torch.Tensor  # 速度 RMSE (m/s)，标量


class System:
    """ISAC 仿真顶层编排：配置加载、组件构建与标准链路 API。

    持有 ``params``（结构化配置）与 ``components``（OFDM/信道/感知子模块）。

    典型通信链::

        transmit() → channel(...) → receive(y_time)

    典型感知链::

        compute_sensing_spectrum(x_rg, y_rg)
        → components.sensing_performance() / display_sensing_geometry / visualize_sensing_spectrum
        → estimate_sensing_music(h_dd)
        → evaluate_sensing_rmse(music)
    """

    def __init__(
        self,
        config: dict,
        *,
        device: str = "cuda:0",
    ) -> None:
        """初始化系统。

        参数:
        -------
        - config : dict
            已解析的配置字典（通常由 ``load_config`` 在外部加载）
        - device : str
            Sionna / Torch 计算设备
        """
        self.device = device
        self.config: dict = config

        sn_config.device = self.device
        self.params = SystemParams.from_dict(self.config)
        self.components = SystemComponents.build_from_params(
            self.params, device=self.device
        )

    # ==================== 发射 ====================
    def transmit(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """生成发射波形。

        按 ``params.source.type`` 分支：

        - ``binary``：随机比特 → QAM 映射
        - ``zc``：Zadoff-Chu 序列（无比特 ``b``）

        返回:
        -------
        - b : torch.Tensor | None
            发射比特；``zc`` 源时为 ``None``
        - x_rg : torch.Tensor
            频域 OFDM 资源网格
        - x_time : torch.Tensor
            时域 OFDM 波形
        """
        src_type = self.params.source.type
        comps = self.components
        rg = comps.rg

        if src_type == "binary":
            b = comps.binary_source(
                [
                    1,
                    1,
                    1,
                    rg.num_data_symbols * int(self.params.source.num_bits_per_symbol),
                ]
            )
            x = comps.mapper(b)

        elif src_type == "zc":
            b = None
            x = comps.zc_source([1, 1, 1, rg.num_data_symbols])

        else:
            raise ValueError(f"unsupported source.type: {src_type!r}")

        x_rg = comps.rg_mapper(x)
        x_time = comps.modulator(x_rg)
        return b, x_rg, x_time

    # ==================== 接收 ====================
    def receive(
        self,
        y_time: torch.Tensor,
        no: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """时域接收与译码。

        参数:
        -------
        - y_time : torch.Tensor
            时域接收信号
        - no : torch.Tensor | float
            AWGN 噪声方差，供 QAM 软解映射；默认 ``0.0`` 表示无噪

        返回:
        -------
        - b_hat : torch.Tensor
            译码比特
        """
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=self.device, dtype=torch.float32)

        comps = self.components
        y_rg = comps.demodulator(y_time)
        y = comps.rg_demapper(y_rg)
        return comps.demapper(y, no=no)

    # ==================== 感知 ====================
    def compute_sensing_spectrum(
        self,
        x_rg: torch.Tensor,
        y_rg: torch.Tensor,
        *,
        apply_mti: bool = False,
        mti_axis: int = -2,
    ) -> torch.Tensor:
        """LS 信道估计 → 可选 MTI → 时延–多普勒谱。

        参数:
        -------
        - x_rg, y_rg : torch.Tensor
            发射/接收频域资源网格
        - apply_mti : bool
            是否在 DD 变换前施加动目标指示（MTI）
        - mti_axis : int
            MTI 沿哪一维滤波，默认 ``-2``（OFDM 符号维）

        返回:
        -------
        - h_delay_doppler : torch.Tensor
            二维谱 ``(num_ofdm_symbols, num_subcarriers)``，复数

        异常:
        ------
        ValueError
            未构建 ``ls_channel_estimator`` 组件时抛出。
        """
        comps = self.components
        if comps.ls_channel_estimator is None:
            raise ValueError(
                "compute_sensing_spectrum 要求已构建 ls_channel_estimator 组件"
            )
        h = comps.ls_channel_estimator(x_rg, y_rg)
        if apply_mti:
            h = comps.moving_target_indication(h, axis=mti_axis)
        return comps.delay_doppler_spectrum(h)

    def display_sensing_geometry(self) -> None:
        """打印 RT 场景各 (RX, 目标, TX) 三元组的路径类型、路径长度与径向速度。

        异常:
        ------
        ValueError
            未配置 ``rt_simulator`` 时抛出。
        """
        scene = self.components.rt_simulator
        if scene is None:
            raise ValueError("display_sensing_geometry 要求已配置 rt_simulator")
        scene.rx_target_tx_geometric.display()

    def visualize_sensing_spectrum(
        self,
        h_delay_doppler: torch.Tensor,
        *,
        file: Union[Path, str],
        to_db: bool = False,
        metric_mode: MetricMode = "range_velocity",
        backend: str = "matplotlib",
        announce_save: bool = True,
    ) -> None:
        """保存或绘制时延–多普勒（或距离–速度）三维谱图。

        参数:
        -------
        - h_delay_doppler : torch.Tensor
            由 ``compute_sensing_spectrum`` 得到的 DD 谱
        - file : Path | str
            输出图像路径
        - to_db : bool
            是否以 dB 显示幅度
        - metric_mode : MetricMode
            ``delay_doppler`` 用时延 (ns)/多普勒 (Hz) 轴；``range_velocity`` 用距离 (m)/速度 (m/s)
        - backend : str
            ``matplotlib`` 或 ``plotly``
        - announce_save : bool
            保存后是否在 stdout 打印路径

        异常:
        ------
        ValueError
            未构建 ``delay_doppler_spectrum`` 组件，或未配置 ``[dd_spectrum_roi]`` 时抛出。
        """
        dd = self.components.delay_doppler_spectrum
        if dd is None:
            raise ValueError(
                "visualize_sensing_spectrum 要求已构建 delay_doppler_spectrum 组件"
            )
        if not dd.has_roi:
            raise ValueError("visualize_sensing_spectrum 要求配置 [dd_spectrum_roi]")
        dd.h_delay_doppler = h_delay_doppler
        dd.visualize(
            file_name=file,
            to_db=to_db,
            metric_mode=metric_mode,
            backend=backend,
            announce_save=announce_save,
        )

    def estimate_sensing_music(
        self,
        h_delay_doppler: torch.Tensor,
        *,
        sens_mode: SensMode = "monostatic",
        metric_mode: MetricMode = "range_velocity",
        music_num_sources: Optional[int] = None,
        log_peaks: bool = True,
    ) -> MusicEstimate:
        """在 DD 谱上运行 2D-MUSIC，得到距离与速度估计。

        ``sens_mode`` 同时作用于时延→距离、多普勒→速度的物理换算：

        - ``monostatic``：往返尺度（``τ·c/2``、``v∝f_d/(2f_c)``）
        - ``bistatic``：单程几何路径尺度（``τ·c``、``v∝f_d/f_c``）

        参数:
        -------
        - h_delay_doppler : torch.Tensor
            时延–多普勒谱
        - sens_mode : SensMode
            单基地或双基地速度/距离语义
        - metric_mode : MetricMode
            仅影响谱峰日志列名与单位，不改变返回值
        - music_num_sources : int | None
            信号源个数；``None`` 时由 MUSIC 内部自动估计
        - log_peaks : bool
            是否在 stdout 打印检峰表格

        返回:
        -------
        MusicEstimate
            各谱峰的距离 (m) 与速度 (m/s) 一维张量

        异常:
        ------
        ValueError
            未构建 ``music_estimator`` 组件时抛出。
        """
        comps = self.components
        if comps.music_estimator is None:
            raise ValueError("estimate_sensing_music 要求已构建 music_estimator 组件")

        music_kwargs: dict = {
            "spectrum_tensor": h_delay_doppler,
            "metric_mode": metric_mode,
            "sens_mode": sens_mode,
            "log_peaks": log_peaks,
        }
        dd = comps.delay_doppler_spectrum
        if dd is not None and dd._roi_slices is not None:
            dop_start, _, delay_start, _ = dd._roi_slices
            music_kwargs["bin_origin"] = (dop_start, delay_start)
        if music_num_sources is not None:
            music_kwargs["num_sources"] = music_num_sources
        est_ranges, est_velocities, _ = comps.music_estimator(**music_kwargs)
        return MusicEstimate(
            est_ranges=est_ranges,
            est_velocities=est_velocities,
        )

    def evaluate_sensing_rmse(
        self,
        estimate: MusicEstimate,
        *,
        true_ranges: Optional[torch.Tensor] = None,
        true_velocities: Optional[torch.Tensor] = None,
        label: str = "感知",
        distance_axis_label: str = "径向距离",
        velocity_axis_label: str = "径向速度",
        verbose: bool = True,
    ) -> SensingRmse:
        """匈牙利算法匹配 MUSIC 峰与真值格点，计算径向 RMSE。

        真值来源（按优先级）：

        1. 显式传入 ``true_ranges`` / ``true_velocities``
        2. ``rt_simulator.rx_target_tx_geometric`` 的 ``range_tensor`` / ``vel_tensor``

        参数:
        -------
        - estimate : MusicEstimate
            ``estimate_sensing_music`` 的输出
        - true_ranges, true_velocities : torch.Tensor | None
            真值距离 (m) 与速度 (m/s)；缺省时从 RT 几何读取
        - label : str
            RMSE 日志前缀
        - distance_axis_label, velocity_axis_label : str
            日志中距离/速度轴名称（如双基地可用「LoS路径长度」）
        - verbose : bool
            是否打印匹配过程与 RMSE 行

        返回:
        -------
        SensingRmse
            距离与速度的 RMSE 标量张量

        异常:
        ------
        ValueError
            无显式真值且未配置 ``rt_simulator`` 时抛出。
        """
        tr = true_ranges
        tv = true_velocities
        if tr is None or tv is None:
            scene = self.components.rt_simulator
            if scene is None:
                raise ValueError(
                    "evaluate_sensing_rmse 须传入 true_ranges/true_velocities，"
                    "或配置 rt_simulator 以从 rx_target_tx_geometric 读取真值"
                )
            geom = scene.rx_target_tx_geometric
            tr = geom.range_tensor
            tv = geom.vel_tensor
        (
            rmse_range_m,
            rmse_velocity_mps,
            _,
            _,
            _,
        ) = match_peaks_and_compute_radial_rmse(
            est_ranges=estimate.est_ranges,
            est_velocities=estimate.est_velocities,
            true_ranges=tr,
            true_velocities=tv,
            label=label,
            distance_axis_label=distance_axis_label,
            velocity_axis_label=velocity_axis_label,
            verbose=verbose,
        )
        return SensingRmse(
            rmse_range_m=rmse_range_m,
            rmse_velocity_mps=rmse_velocity_mps,
        )
