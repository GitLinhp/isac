# 标准库
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import torch
import sionna

# 自定义模块
from .utils import (
    load_config,
    match_peaks_and_compute_radial_rmse,
)
from .data_structures import SystemParams, SystemComponents

SensMode = Literal["monostatic", "bistatic"]
MetricMode = Literal["delay_doppler", "range_velocity", "dd", "rv"]


@dataclass
class SensingResult:
    """``System.sensing(..., evaluate=True)`` 的返回结构。"""

    h_delay_doppler: torch.Tensor
    est_ranges: Optional[torch.Tensor] = None
    est_velocities: Optional[torch.Tensor] = None
    rmse_range_m: Optional[torch.Tensor] = None
    rmse_velocity_mps: Optional[torch.Tensor] = None


class System:

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        """
        初始化系统

        参数:
        -------
        - config_file : str | Path, 可选
            配置文件路径，默认为 "base.toml"
        - device : torch.device, 可选
            计算设备，如果为 None 则自动选择GPU或CPU
        """
        self.args: argparse.Namespace = args
        self.config: dict = load_config(args.config_file)
        self.device: str = args.device

        # 设置 Sionna 全局设备
        sionna.phy.config.device = self.device

        # 加载系统参数
        self.params = SystemParams.from_dict(self.config)

        # 构建系统组件
        self.components = SystemComponents.build_from_params(
            self.params, device=self.device
        )

    # 发射
    def transmit(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """生成发射比特 ``b``（仅 binary 源）、频域资源网格 ``x_rg`` 与时域 ``x_time``。"""
        src_type = self.params.source.type
        batch_size = self.args.batch_size
        comps = self.components
        rg = comps.rg

        if src_type == "binary":  # 二进制源
            b = comps.binary_source(
                [
                    batch_size,
                    1,
                    1,
                    rg.num_data_symbols * self.params.source.num_bits_per_symbol,
                ]
            )
            x = comps.mapper(b)

        elif src_type == "zc":  # Zadoff-Chu 源
            b = None
            x = comps.zc_source([batch_size, 1, 1, rg.num_data_symbols])

        else:  # 不支持的源类型
            raise ValueError(f"unsupported source.type: {src_type!r}")

        x_rg = comps.rg_mapper(x)  # 频域资源网格映射
        x_time = comps.modulator(x_rg)  # OFDM调制
        return b, x_rg, x_time

    # 接收
    def receive(
        self,
        y_time: torch.Tensor,
        no: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """时域接收：``demodulator`` → ``rg_demapper`` → ``demapper``，返回译码比特 ``b_hat``。"""
        if no is None:
            no = torch.tensor(0.0, device=self.device)
        elif not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=self.device, dtype=torch.float32)

        comps = self.components
        # OFDM 解调
        y_rg = comps.demodulator(y_time)

        # 资源网格解映射
        y = comps.rg_demapper(y_rg)

        # QAM 解映射
        b_hat = comps.demapper(y, no=no)

        return b_hat

    def sensing(
        self,
        x_rg: torch.Tensor,
        y_rg: torch.Tensor,
        *,
        apply_mti: bool = False,
        mti_axis: int = -2,
        evaluate: bool = False,
        metric_mode: MetricMode = "range_velocity",
        sens_mode: SensMode = "monostatic",
        label: str = "感知",
        spectrum_file: Union[Path, str, None] = None,
        visualize_offset: int = 50,
        to_db: bool = False,
        backend: str = "matplotlib",
        display_performance: bool = True,
        display_geometry: bool = True,
        run_music: bool = True,
        compute_rmse: bool = True,
        true_ranges: Optional[torch.Tensor] = None,
        true_velocities: Optional[torch.Tensor] = None,
        distance_axis_label: str = "径向距离",
        velocity_axis_label: str = "径向速度",
        music_num_sources: Optional[int] = None,
    ) -> Union[torch.Tensor, SensingResult]:
        """接收端感知：LS 信道估计 → 可选 MTI → 时延–多普勒谱。

        ``evaluate=False`` 时仅返回 ``h_delay_doppler``；``evaluate=True`` 时可选执行
        性能表、谱图、几何真值展示、MUSIC 与 RMSE，并返回 ``SensingResult``。
        """
        comps = self.components

        if comps.ls_channel_estimator is None:
            raise ValueError("sensing 要求已构建 ls_channel_estimator 组件")
        h = comps.ls_channel_estimator(x_rg, y_rg)
        if apply_mti:
            h = comps.moving_target_indication(h, axis=mti_axis)
        h_delay_doppler = comps.delay_doppler_spectrum(h)

        if not evaluate:
            return h_delay_doppler

        if display_performance:
            if comps.sensing_performance is None:
                raise ValueError("evaluate 要求已构建 sensing_performance 组件")
            comps.sensing_performance.display_performance()

        if spectrum_file is not None:
            if comps.delay_doppler_spectrum is None:
                raise ValueError("evaluate 要求已构建 delay_doppler_spectrum 组件")
            comps.delay_doppler_spectrum.visualize(
                offset=visualize_offset,
                file_name=spectrum_file,
                to_db=to_db,
                metric_mode=metric_mode,
                backend=backend,
            )

        if display_geometry:
            scene = comps.rt_scene
            if scene is None:
                raise ValueError(
                    "display_geometry=True 要求已构建 rt_scene；"
                    "或设 display_geometry=False 并传入 true_ranges/true_velocities"
                )
            scene.rx_target_tx_geometric.display()

        est_ranges: Optional[torch.Tensor] = None
        est_velocities: Optional[torch.Tensor] = None
        rmse_range_m: Optional[torch.Tensor] = None
        rmse_velocity_mps: Optional[torch.Tensor] = None

        if run_music:
            if comps.music_estimator is None:
                raise ValueError("run_music=True 要求已构建 music_estimator 组件")
            music_kwargs: dict = {
                "spectrum_tensor": h_delay_doppler,
                "metric_mode": metric_mode,
                "sens_mode": sens_mode,
            }
            if music_num_sources is not None:
                music_kwargs["num_sources"] = music_num_sources
            est_ranges, est_velocities, _ = comps.music_estimator(**music_kwargs)

        if compute_rmse:
            if est_ranges is None or est_velocities is None:
                raise ValueError("compute_rmse=True 要求 run_music=True")
            tr = true_ranges
            tv = true_velocities
            if tr is None or tv is None:
                scene = comps.rt_scene
                if scene is None:
                    raise ValueError(
                        "compute_rmse=True 须传入 true_ranges/true_velocities，"
                        "或配置 rt_scene 以从 rx_target_tx_geometric 读取真值"
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
                est_ranges=est_ranges,
                est_velocities=est_velocities,
                true_ranges=tr,
                true_velocities=tv,
                label=label,
                distance_axis_label=distance_axis_label,
                velocity_axis_label=velocity_axis_label,
            )

        return SensingResult(
            h_delay_doppler=h_delay_doppler,
            est_ranges=est_ranges,
            est_velocities=est_velocities,
            rmse_range_m=rmse_range_m,
            rmse_velocity_mps=rmse_velocity_mps,
        )
