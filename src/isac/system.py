# 标准库
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import torch
import numpy as np
import sionna

# 自定义模块
from .utils import load_config, cartesian_direction_to_yaw_pitch_roll
from .data_structures.params import SystemParams
from .data_structures.components.system_components import SystemComponents
from . import PROJECT_ROOT


@dataclass
class SensingResult:
    """``System.sensing`` 输出：LS 信道估计与时延–多普勒谱。"""

    h: torch.Tensor
    h_delay_doppler: torch.Tensor
    h_clean: Optional[torch.Tensor] = None
    h_delay_doppler_clean: Optional[torch.Tensor] = None


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

        # 设置全局设备
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
        src_type = self.params.ofdm.source.type
        batch_size = self.args.batch_size
        ofdm = self.components.ofdm
        rg = ofdm.rg

        if src_type == "binary":  # 二进制源
            b = ofdm.binary_source(
                [
                    batch_size,
                    1,
                    1,
                    rg.num_data_symbols * self.params.ofdm.num_bits_per_symbol,
                ]
            )
            x = ofdm.mapper(b)

        elif src_type == "zc":  # Zadoff-Chu 源
            b = None
            x = ofdm.zc_source([batch_size, 1, 1, rg.num_data_symbols])

        else:  # 不支持的源类型
            raise ValueError(f"unsupported ofdm.source.type: {src_type!r}")

        x_rg = ofdm.rg_mapper(x)  # 频域资源网格映射
        x_time = ofdm.modulator(x_rg)  # OFDM调制
        return b, x_rg, x_time

    # 应用信道
    def apply_channel(
        self,
        inputs: torch.Tensor,
        domain: str = "frequency",
    ) -> torch.Tensor:
        """经信道并加 AWGN；TOML ``snr_db`` 为接收端 SNR (dB)，按 ``E[|y_clean|^2]`` 定标 ``no``。"""
        return self.components.channel(
            inputs,
            domain=domain,
            snr_db=self.params.channel.snr_db,
        )

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

        ofdm = self.components.ofdm
        # OFDM 解调
        y_rg = ofdm.demodulator(y_time)

        # 资源网格解映射
        y = ofdm.rg_demapper(y_rg)

        # QAM 解映射
        b_hat = ofdm.demapper(y, no=no)

        return b_hat

    def demodulate(self, y_time: torch.Tensor) -> torch.Tensor:
        """时域 IQ → 频域资源网格（squeeze，供 ``estimate_channel`` / ``sensing`` 使用）。"""
        return self.components.ofdm.demodulator(y_time).squeeze()

    # 信道估计
    def estimate_channel(
        self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
    ) -> torch.Tensor:
        """估计频域信道（LS）：``h = y * conj(x) / (|x|^2 + ``eps``)``。

        对 ``x``/``y`` 做 ``squeeze`` 后，``x`` 为 ``(num_ofdm_symbols, fft_size)``；
        ``y`` 为 ``(rx_num, num_ofdm_symbols, fft_size)``，``rx_num=1`` 时退化为 2D。
        假定 ``batch_size=1``；squeeze 后 ``y.ndim > 3`` 将报错。
        """
        rg = self.components.ofdm.rg
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

    def sensing(
        self,
        x_rg: torch.Tensor,
        y_rg: Optional[torch.Tensor] = None,
        *,
        y_time: Optional[torch.Tensor] = None,
        apply_mti: bool = False,
        mti_axis: int = -2,
        y_rg_clean: Optional[torch.Tensor] = None,
        y_time_clean: Optional[torch.Tensor] = None,
    ) -> SensingResult:
        """接收端感知：LS 信道估计 → 可选 MTI → 时延–多普勒谱。

        传入 ``y_time`` 时内部 ``demodulate`` 为 ``y_rg``；与 ``y_rg`` 二选一。
        ``y_rg_clean`` / ``y_time_clean`` 用于无噪参考谱（SNR 诊断）。
        """
        if y_rg is not None and y_time is not None:
            raise ValueError("y_rg 与 y_time 不可同时传入")
        if y_rg is None and y_time is None:
            raise ValueError("须传入 y_rg 或 y_time")

        if y_time is not None:
            y_rg = self.demodulate(y_time)

        sensing = self.components.sensing

        h = self.estimate_channel(x_rg, y_rg)
        if apply_mti:
            h = sensing.moving_target_indication(h, axis=mti_axis)
        h_delay_doppler = sensing.delay_doppler_spectrum(h)

        h_clean: Optional[torch.Tensor] = None
        h_dd_clean: Optional[torch.Tensor] = None
        y_rg_clean_resolved: Optional[torch.Tensor] = y_rg_clean
        if y_time_clean is not None:
            y_rg_clean_resolved = self.demodulate(y_time_clean)
        if y_rg_clean_resolved is not None:
            h_clean = self.estimate_channel(x_rg, y_rg_clean_resolved)
            if apply_mti:
                h_clean = sensing.moving_target_indication(h_clean, axis=mti_axis)
            h_dd_clean = sensing.delay_doppler_spectrum(h_clean)

        return SensingResult(
            h=h,
            h_delay_doppler=h_delay_doppler,
            h_clean=h_clean,
            h_delay_doppler_clean=h_dd_clean,
        )

    def _reference_tx_power_dbm(self) -> float | None:
        """首个含发射机的收发机在 TOML 中配置的 ``power_dbm``；用于将归一化 ``mean(|x|^2)`` 映射为 dBm。"""
        scene = self.components.rt_scene
        if scene is None:
            return None
        params_map = scene.scene_params.transceivers
        if not params_map:
            return None
        for name, tc in scene.transceivers.items():
            if tc.tx is None:
                continue
            p = params_map.get(name)
            if p is not None and p.power_dbm is not None:
                return float(p.power_dbm)
        return None

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
        source: Literal["monte_carlo", "trajectory"],
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
            path = (
                out_dir / f"{scene_slug}_mc_dataset_episodes.csv"
                if source == "monte_carlo"
                else out_dir / f"{scene_slug}_dataset_episodes.csv"
            )
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

        # legacy：文件名与列集与旧版一致
        if source == "monte_carlo":
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
        else:
            if run_sensing:
                path = out_dir / f"{scene_slug}_dataset_sensing_metrics.csv"
                fieldnames = [
                    "step",
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
                path = out_dir / f"{scene_slug}_dataset_kinematics.csv"
                fieldnames = [
                    "step",
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
