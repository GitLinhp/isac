# 标准库
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import torch
import numpy as np
import sionna

# 自定义模块
from .utils import (
    cartesian_direction_to_yaw_pitch_roll,
    load_config,
    match_peaks_and_compute_radial_rmse,
)
from .data_structures import SystemParams, SystemComponents
from .utils.rt_doppler import align_rt_monostatic_doppler_phase
from . import PROJECT_ROOT


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

    # 信道估计
    def estimate_channel(
        self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        """估计频域信道（LS）：``h = y * conj(x) / (|x|^2 + ``eps``)``。

        对 ``x``/``y`` 做 ``squeeze`` 后，``x`` 为 ``(num_ofdm_symbols, fft_size)``；
        ``y`` 为 ``(rx_num, num_ofdm_symbols, fft_size)``，``rx_num=1`` 时退化为 2D。
        假定 ``batch_size=1``；squeeze 后 ``y.ndim > 3`` 将报错。
        """
        rg = self.components.rg
        s, f = rg.num_ofdm_symbols, rg.fft_size

        x = x.squeeze()
        y = y.squeeze()

        if x.shape[-2:] != (s, f):
            raise ValueError(f"x 末两维须为 ({s}, {f})，收到 {tuple(x.shape)}")
        x = x.reshape(s, f)

        if y.shape[-2:] != (s, f):
            raise ValueError(f"y 末两维须为 ({s}, {f})，收到 {tuple(y.shape)}")
        if y.ndim == 2:
            y = y.reshape(s, f)
        elif y.ndim == 3:
            y = y.reshape(y.shape[0], s, f)
            if y.shape[0] == 1:
                y = y.squeeze(0)
        else:
            raise ValueError(
                f"y squeeze 后须为 2D (S,F) 或 3D (rx_num,S,F)，收到 ndim={y.ndim}"
            )

        denom = torch.abs(x) ** 2 + eps
        h = y * torch.conj(x) / denom

        return h

    def _align_rt_monostatic_doppler_if_needed(
        self,
        h: torch.Tensor,
        sens_mode: SensMode,
        *,
        axis: int,
    ) -> torch.Tensor:
        channel = self.params.channel
        if channel is None or channel.type != "rt" or sens_mode != "monostatic":
            return h
        return align_rt_monostatic_doppler_phase(h, axis=axis)

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

        h = self.estimate_channel(x_rg, y_rg)
        h = self._align_rt_monostatic_doppler_if_needed(h, sens_mode, axis=mti_axis)
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

    def _update_rt_target_pose_from_velocity(
        self,
        target: object,
        pos: np.ndarray | list[float],
        vel: np.ndarray | list[float],
    ) -> None:
        """更新目标位置与速度"""
        pos_a = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_a = np.asarray(vel, dtype=np.float64).reshape(-1)
        if pos_a.size != 3 or vel_a.size != 3:
            raise ValueError("位置与速度须为三维向量")
        v_eps = 1e-9
        speed = float(np.linalg.norm(vel_a))
        if speed > v_eps:
            direction = (vel_a / speed).astype(np.float64)
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        orientation = cartesian_direction_to_yaw_pitch_roll(direction)
        target.update(
            position=pos_a,
            velocity=vel_a,
            orientation=orientation,
        )

    def save_episodes_csv(
        self,
        *,
        scene_slug: str,
        rows: list[dict[str, str | int]],
        run_sensing: bool,
        csv_mode: Literal["unified", "legacy"] = "unified",
        output_root: Path | None = None,
    ) -> None:
        """写入 Episode CSV：统一表或 legacy 分裂文件名。"""
        if not rows:
            print("无 CSV 行，跳过写入")
            return
        out_dir = output_root if output_root is not None else PROJECT_ROOT / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        if csv_mode == "unified":
            path = out_dir / f"{scene_slug}_mc_dataset_episodes.csv"
            keys_set: set[str] = set()
            for r in rows:
                keys_set.update(r.keys())
            keys = sorted(keys_set)
            with path.open("w", newline="", encoding="utf-8") as csv_f:
                writer = csv.DictWriter(csv_f, fieldnames=keys, restval="")
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in keys})
            print(f"统一 Episode CSV 已写入: {path}")
            return

        if run_sensing:
            path = out_dir / f"{scene_slug}_mc_dataset_sensing_metrics.csv"
            fieldnames = [
                "sample_idx",
                "pos_x_m",
                "pos_y_m",
                "pos_z_m",
                "vel_x_mps",
                "vel_y_mps",
                "vel_z_mps",
                "true_range_m",
                "est_range_m",
                "rmse_range_m",
                "true_radial_velocity_mps",
                "est_radial_velocity_mps",
                "rmse_radial_velocity_mps",
            ]
        else:
            path = out_dir / f"{scene_slug}_mc_dataset_kinematics.csv"
            fieldnames = [
                "sample_idx",
                "pos_x_m",
                "pos_y_m",
                "pos_z_m",
                "vel_x_mps",
                "vel_y_mps",
                "vel_z_mps",
                "true_range_m",
                "true_radial_velocity_mps",
            ]
        slim = [{k: r[k] for k in fieldnames if k in r} for r in rows]
        with path.open("w", newline="", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slim)
        print(f"CSV 已写入: {path}")
