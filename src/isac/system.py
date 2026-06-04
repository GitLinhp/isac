# 标准库
import argparse
import csv
from pathlib import Path
from typing import Literal, Optional, Union
import torch
import numpy as np
import sionna

# 自定义模块
from .utils import get_logger, load_config
from .data_structures import SystemParams, SystemComponents
from .sensing.clutter_suppression import MovingTargetDetection, MovingTargetIndication
from .utils import cartesian_direction_to_yaw_pitch_roll
from . import PROJECT_ROOT

logger = get_logger(__name__)


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

        self.params = SystemParams.from_dict(self.config)

        sionna.phy.config.device = self.device

        self.components = SystemComponents.from_system_params(
            self.params, device=self.device
        )

        self.moving_target_indication = MovingTargetIndication(
            self.components.sensing_performance,
            filter_order=1,
        )
        self.moving_target_detection = MovingTargetDetection(
            self.components.sensing_performance,
            self.params.carrier_frequency,
        )

    def apply_channel(
        self,
        inputs: torch.Tensor,
        domain: str = "frequency",
        *,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """经信道并加 AWGN；TOML ``snr_db`` (Es/N0) 换算 Eb/N0 后 ``ebnodb2no``（同 OFDM notebook）。"""
        snr = snr_db if snr_db is not None else self.params.channel.snr_db
        return self.components.channel(
            inputs,
            domain=domain,
            snr_db=snr,
            num_bits_per_symbol=self.params.qam.num_bits_per_symbol,
            coderate=self.params.channel.coderate,
        )

    def tx_symbols_to_resource_grid(self) -> torch.Tensor:
        """按 ``params.sensing.source`` 生成发射侧频域资源网格 ``x_rg``（``ResourceGridMapper`` 输出）。"""
        src_type = self.params.sensing.source.type
        batch = self.args.batch_size
        rg = self.components.rg
        if src_type == "binary":
            b = self.components.binary_source(
                [
                    batch,
                    1,
                    1,
                    rg.num_data_symbols * self.params.qam.num_bits_per_symbol,
                ]
            )
            x = self.components.mapper(b)
            return self.components.rg_mapper(x)
        if src_type == "zc":
            zc = self.components.zc_source
            if zc is None:
                raise RuntimeError(
                    "sensing.source.type is 'zc' but components.zc_source is missing; "
                    "check SystemComponents wiring"
                )
            x = zc([batch, 1, 1, rg.num_data_symbols])
            return self.components.rg_mapper(x)
        raise ValueError(f"unsupported sensing.source.type: {src_type!r}")

    def estimate_channel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """估计频域信道（LS）：``h = y * conj(x) / (|x|^2 + eps)``。

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

        eps = 1e-12
        if y.ndim == 2:
            denom = torch.abs(x) ** 2 + eps
            h = y * torch.conj(x) / denom
        else:
            x_bc = x.unsqueeze(0)
            denom = torch.abs(x_bc) ** 2 + eps
            h = y * torch.conj(x_bc) / denom

        return h

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
        """写入 Episode CSV：统一表或 legacy 分裂文件名（与历史脚本兼容）。"""
        if not rows:
            logger.warning("无 CSV 行，跳过写入")
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
            logger.info("统一 Episode CSV 已写入: %s", path)
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
        logger.info("CSV 已写入: %s", path)
