"""ISAC 数据集采集入口：蒙特卡洛 ROI 采样生成目标位姿序列，可选每步感知，并写出 CSV / HDF5 / GIF。

流程概要
--------
1. 解析 CLI，设置随机种子，解析 ROI。
2. 构建 ``System``，输出目录固定为 ``out/dataset_collection/``。
3. 蒙特卡洛生成 episode 序列后进入主循环：更新 RT 目标位姿 → 记录几何真值 →（可选）感知评估 → 累计 I/O 缓冲。
4. 循环结束后写出 CSV / HDF5 / GIF；HDF5 默认仅 CFR + kinematics，加 ``--save-cir`` 才写入 CIR。

约定
----
- 几何真值与感知评估默认取 ``RxTargetTxGeometric`` 的 ``[0, 0, 0]`` 切片（单 RX × 单目标 × 单 TX）。
- ``--run_sensing`` 时谱图固定覆盖写入 ``sensing_monostatic_delay_doppler_spectrum.png``，便于查看最新一步。
- CSV 中逐步 RMSE 为「本 episode 估计 vs 真值」；``match_peaks_and_compute_radial_rmse`` 打印的是匈牙利跨峰 RMSE。
- 蒙特卡洛 ROI：CLI 为 ``--roi XMIN XMAX YMIN YMAX`` 四元组，``z`` 固定为 ``0``（即 ``(0, 0)``）。
- HDF5 根属性 ``has_cir`` 标记是否包含 ``channel_impulse_response_*`` 数据集。
- HDF5 另含 ``collection_*`` 根属性（ROI、seed、source、采样参数等），见 ``CollectionMetadata``。

与 ``run_sensing_monostatic.py`` 的差异：感知嵌在数据采集循环内，并承担批量 episode 的 I/O。
"""

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.datasets import CollectionMetadata, Dataset
from isac.channel.rt.rx_target_tx_geometric import RxTargetTxGeometric
from isac.sensing.sample_quality import (
    QualityFilterStats,
    SampleQualityConfig,
    evaluate_sample_quality,
)
from isac.sensing.utils import doppler_to_velocity
from isac.system import System
from isac.channel import RTScene
from isac.utils import csv_float2_scalar
from isac.utils import (
    cartesian_direction_to_yaw_pitch_roll,
    compute_rmse,
    images_to_gif,
    match_peaks_and_compute_radial_rmse,
    set_random_seed,
)
from isac.utils import target_generation as tg

# Sionna/DrJit 射线追踪在大量 episode 时会触发 AST 装饰器次数告警，不影响数值结果
warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_MC_ROI_XY = ((0.0, 80.0), (-40.0, 40.0))
DEFAULT_MC_ROI_Z = (0.0, 0.0)
SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "dataset_collection"
SENSING_SPECTRUM_FILENAME = "sensing_monostatic_delay_doppler_spectrum.png"

# 单 RX / 单目标 / 单 TX 场景下，几何张量 ``(n_rx, n_target, n_tx)`` 的默认索引
RX_IDX = TARGET_IDX = TX_IDX = 0
TRIPLE_SLICE = (RX_IDX, TARGET_IDX, TX_IDX)

CsvMode = Literal["unified", "legacy"]
SensingLayout = Literal["monostatic", "bistatic"]


@dataclass(frozen=True)
class CollectionConfig:
    """从 CLI 解包后的采集选项，避免 ``main`` 内散落大量局部变量。"""

    run_sensing: bool
    save_h5: bool
    save_cir: bool
    save_csv: bool
    save_gif: bool
    csv_mode: CsvMode
    sensing_domain: str
    sensing_layout: SensingLayout
    log_per_step_sensing: bool
    quality_filter: bool
    require_los: bool
    min_los_ratio: float
    min_peak_prominence_db: float
    max_bin_offset: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CollectionConfig":
        csv_mode: CsvMode = args.csv_mode
        if (
            args.run_sensing
            and args.sensing_layout == "bistatic"
            and csv_mode == "legacy"
        ):
            print("双基地感知 CSV 列与 legacy 固定表头不一致，已改用 csv_mode=unified")
            csv_mode = "unified"
        return cls(
            run_sensing=args.run_sensing,
            save_h5=args.save_h5,
            save_cir=args.save_cir,
            save_csv=args.save_csv,
            save_gif=args.save_gif,
            csv_mode=csv_mode,
            sensing_domain=args.sensing_domain,
            sensing_layout=args.sensing_layout,
            log_per_step_sensing=args.log_per_step_sensing,
            quality_filter=args.quality_filter,
            require_los=args.require_los,
            min_los_ratio=float(args.min_los_ratio),
            min_peak_prominence_db=float(args.min_peak_prominence_db),
            max_bin_offset=int(args.max_bin_offset),
        )

    def sample_quality_config(self) -> SampleQualityConfig:
        return SampleQualityConfig(
            require_los=self.require_los,
            min_los_ratio=self.min_los_ratio,
            min_peak_prominence_db=self.min_peak_prominence_db,
            max_bin_offset=self.max_bin_offset,
            rx_idx=RX_IDX,
            tx_idx=TX_IDX,
        )


# ---------------------------------------------------------------------------
# 路径 / 几何 / 信道 辅助函数
# ---------------------------------------------------------------------------


def _paths_doppler_hz_los(
    rt_scene: object,
    tau_true_s: float | torch.Tensor,
    *,
    rx_idx: int = RX_IDX,
    tx_idx: int = TX_IDX,
    device: torch.device | None = None,
) -> torch.Tensor:
    """从 Sionna ``Paths`` 中取与几何 LoS 总时延最接近的路径的多普勒（Hz）。

    用于双基地真值：几何给出路径长 → ``τ``，再在 ``paths.tau`` 中找最近路径读 ``paths.doppler``。
    """
    paths = rt_scene.paths
    tau_np = np.asarray(paths.tau, dtype=np.float64)
    valid_np = np.asarray(paths.valid, dtype=bool)
    doppler_np = np.asarray(paths.doppler, dtype=np.float64)
    if tau_np.ndim != 3:
        raise ValueError(
            "_paths_doppler_hz_los 当前仅支持形状 [num_rx,num_tx,max_paths]（与 path_solver.synthetic_array=true 一致）；"
            f"当前 ndim={tau_np.ndim}, shape={tau_np.shape}。"
        )
    if rx_idx >= tau_np.shape[0] or tx_idx >= tau_np.shape[1]:
        raise IndexError(
            f"_paths_doppler_hz_los: rx_idx={rx_idx} 或 tx_idx={tx_idx} 越界，形状 {tau_np.shape}"
        )

    tau_slice = tau_np[rx_idx, tx_idx, :]
    valid_slice = np.asarray(valid_np[rx_idx, tx_idx, :], dtype=bool)
    dop_slice = doppler_np[rx_idx, tx_idx, :]

    if isinstance(tau_true_s, torch.Tensor):
        t0 = float(tau_true_s.detach().cpu().to(dtype=torch.float64).reshape(()).item())
    else:
        t0 = float(tau_true_s)

    # 优先 valid 且 τ≥0；逐步放宽，避免 paths 稀疏时无候选
    candidates = np.flatnonzero(valid_slice & (tau_slice >= 0.0))
    if candidates.size == 0:
        candidates = np.flatnonzero(valid_slice)
    if candidates.size == 0:
        candidates = np.arange(tau_slice.size, dtype=np.int64)

    err = np.abs(tau_slice[candidates] - t0)
    k = int(candidates[int(np.argmin(err))])
    fd = float(dop_slice[k])

    dev = device if device is not None else torch.device("cpu")
    return torch.tensor(fd, dtype=torch.float64, device=dev)


def _roi_xy_to_box3d(
    roi_xy: tuple[tuple[float, float], tuple[float, float]],
    *,
    z_bounds: tuple[float, float] = DEFAULT_MC_ROI_Z,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """平面 ROI 四元组 ``((xmin,xmax),(ymin,ymax))`` → 三维 ``RoiBox3D``，``z`` 默认固定为 0。"""
    return roi_xy[0], roi_xy[1], z_bounds


def _resolve_roi(args: argparse.Namespace) -> tuple:
    """解析蒙特卡洛 ROI（xy 四元组 + z=0）。"""
    if args.roi is not None:
        r = args.roi
        return _roi_xy_to_box3d(((r[0], r[1]), (r[2], r[3])))
    return _roi_xy_to_box3d(DEFAULT_MC_ROI_XY)


def _los_truth_at_first_triple(
    scene: object,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回默认三元组 ``(range, radial_velocity)``，形状均为标量张量。"""
    geom = RxTargetTxGeometric.from_states(
        scene.targets_states,
        scene.rx_states,
        scene.tx_states,
        device=device,
    )
    rx_i, tgt_i, tx_i = TRIPLE_SLICE
    return geom.range_tensor[rx_i, tgt_i, tx_i], geom.vel_tensor[rx_i, tgt_i, tx_i]


def _estimate_delay_doppler_spectrum(system: System, domain: str) -> torch.Tensor:
    """OFDM 参考网格 → 信道施加 → LS 信道估计 → 时延–多普勒谱。"""
    _, x_rg, x_time = system.transmit()

    if domain == "frequency":
        y_rg = system.components.channel(x_rg, domain=domain, snr_db=system.params.channel.snr_db)
    elif domain == "time":
        y_time = system.components.channel(x_time, domain=domain, snr_db=system.params.channel.snr_db)
        y_rg = system.components.demodulator(y_time)
    else:
        raise ValueError(f"不支持的域: {domain}")

    h = system.components.ls_channel_estimator(x_rg, y_rg)
    return system.components.delay_doppler_spectrum(h)


def _save_sensing_spectrum_preview(system: System, out_dir: Path) -> None:
    """覆盖写入固定文件名，循环内每步刷新，便于查看最新一步谱图。"""
    dd = system.components.delay_doppler_spectrum
    dd.visualize(
        offset=200,
        file_name=out_dir / SENSING_SPECTRUM_FILENAME,
        to_db=False,
        metric_mode="delay_doppler",
        backend="matplotlib",
    )


# ---------------------------------------------------------------------------
# 感知评估（与 run_sensing_*.py 管线对齐）
# ---------------------------------------------------------------------------


def monostatic_sensing_eval(
    system: System,
    out_dir: Path,
    domain: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对**当前** RT 场景跑单基地感知链，写谱图并打印匈牙利 RMSE。

    返回 ``(est_range, est_velocity, est_power_db)``，供 CSV 与本步诊断。
    """
    h_delay_doppler = _estimate_delay_doppler_spectrum(system, domain)
    _save_sensing_spectrum_preview(system, out_dir)

    est_ranges, est_velocities, _ = system.components.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode="delay_doppler",
        sens_mode="monostatic",
    )
    if est_ranges.numel() == 0:
        raise RuntimeError("单基地感知评估：MUSIC 未检测到谱峰，无法估计距离/速度")

    scene = system.components.rt_scene
    true_range, true_velocity = _los_truth_at_first_triple(scene, system.device)

    _, _, est_range, est_velocity, est_power_db = match_peaks_and_compute_radial_rmse(
        est_ranges=est_ranges,
        est_velocities=est_velocities,
        true_ranges=true_range,
        true_velocities=true_velocity,
        label="单基地感知",
    )
    return est_range, est_velocity, est_power_db


def bistatic_sensing_eval(
    system: System,
    out_dir: Path,
    domain: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """双基地感知：几何 LoS 路径长 + ``paths.doppler`` 速度真值，MUSIC ``sens_mode=bistatic``。

    返回 ``(est_path, est_velocity, est_power_db, true_path, true_velocity)``。
    """
    h_delay_doppler = _estimate_delay_doppler_spectrum(system, domain)
    _save_sensing_spectrum_preview(system, out_dir)

    est_paths, est_velocities, _ = system.components.music_estimator(
        spectrum_tensor=h_delay_doppler,
        metric_mode="delay_doppler",
        sens_mode="bistatic",
    )
    if est_paths.numel() == 0:
        raise RuntimeError("双基地感知评估：MUSIC 未检测到谱峰")

    scene = system.components.rt_scene
    true_path_m, _ = _los_truth_at_first_triple(scene, system.device)
    tau_true_s = true_path_m / c
    fd_true = _paths_doppler_hz_los(
        scene, tau_true_s, rx_idx=RX_IDX, tx_idx=TX_IDX, device=system.device
    )
    true_velocity = doppler_to_velocity(
        fd_true,
        float(system.params.carrier_frequency),
        "bistatic",
    )

    _, _, est_path, est_vel, est_power_db = match_peaks_and_compute_radial_rmse(
        est_ranges=est_paths,
        est_velocities=est_velocities,
        true_ranges=true_path_m,
        true_velocities=true_velocity,
        label="双基地数据集感知（LoS路径+paths.doppler）",
        distance_axis_label="LoS路径长度",
        velocity_axis_label="标量速度",
    )
    return est_path, est_vel, est_power_db, true_path_m, true_velocity


# ---------------------------------------------------------------------------
# CSV 行构建
# ---------------------------------------------------------------------------


def _kinematics_row(
    episode_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
) -> dict[str, str | int]:
    """构造每条 episode 共有的 kinematics + 几何真值列。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
    return {
        "sample_idx": episode_idx,
        "pos_x_m": csv_float2_scalar(pos_row[0]),
        "pos_y_m": csv_float2_scalar(pos_row[1]),
        "pos_z_m": csv_float2_scalar(pos_row[2]),
        "vel_x_mps": csv_float2_scalar(vel_row[0]),
        "vel_y_mps": csv_float2_scalar(vel_row[1]),
        "vel_z_mps": csv_float2_scalar(vel_row[2]),
        "true_range_m": csv_float2_scalar(true_range),
        "true_radial_velocity_mps": csv_float2_scalar(true_velocity),
    }


def _append_monostatic_sensing_columns(
    row: dict[str, str | int],
    *,
    est_range: torch.Tensor,
    est_velocity: torch.Tensor,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
) -> None:
    """追加单基地感知列；RMSE 为单 episode 估计 vs 真值（非跨 episode 统计）。"""
    rmse_range = compute_rmse(est_range.reshape(1), true_range.reshape(1))
    rmse_velocity = compute_rmse(est_velocity.reshape(1), true_velocity.reshape(1))
    row["est_range_m"] = csv_float2_scalar(est_range)
    row["rmse_range_m"] = csv_float2_scalar(rmse_range)
    row["est_radial_velocity_mps"] = csv_float2_scalar(est_velocity)
    row["rmse_radial_velocity_mps"] = csv_float2_scalar(rmse_velocity)


def _append_bistatic_sensing_columns(
    row: dict[str, str | int],
    *,
    est_path: torch.Tensor,
    est_velocity: torch.Tensor,
    true_path: torch.Tensor,
    true_velocity: torch.Tensor,
) -> None:
    """追加双基地感知列。"""
    rmse_path = compute_rmse(est_path.reshape(1), true_path.reshape(1))
    rmse_velocity = compute_rmse(est_velocity.reshape(1), true_velocity.reshape(1))
    row["true_los_path_length_m"] = csv_float2_scalar(true_path)
    row["true_velocity_paths_doppler_mps"] = csv_float2_scalar(true_velocity)
    row["est_los_path_length_m"] = csv_float2_scalar(est_path)
    row["est_velocity_paths_doppler_mps"] = csv_float2_scalar(est_velocity)
    row["rmse_los_path_m"] = csv_float2_scalar(rmse_path)
    row["rmse_velocity_paths_doppler_mps"] = csv_float2_scalar(rmse_velocity)


def _log_sensing_step(
    episode_idx: int,
    *,
    layout: SensingLayout,
    est_range_or_path: torch.Tensor,
    est_velocity: torch.Tensor,
    est_power_db: torch.Tensor,
) -> None:
    if layout == "bistatic":
        print(
            f"sample_idx={episode_idx:03d} 双基地感知: "
            f"LoS_path={float(est_range_or_path.item()):.3f} m, "
            f"velocity={float(est_velocity.item()):.3f} m/s, "
            f"MUSIC_peak={float(est_power_db.item()):.3f} dB"
        )
    else:
        print(
            f"sample_idx={episode_idx:03d} 感知: "
            f"range={float(est_range_or_path.item()):.3f} m, "
            f"velocity={float(est_velocity.item()):.3f} m/s, "
            f"MUSIC_peak={float(est_power_db.item()):.3f} dB"
        )
    print()


# ---------------------------------------------------------------------------
# 后处理导出
# ---------------------------------------------------------------------------


def _resolve_h5_output(
    *,
    run_sensing: bool,
    scene_slug: str,
    n_episodes: int,
    out_dir: Path,
) -> tuple[Path, str | None, str]:
    """按 ``run_sensing`` 决定 HDF5 路径、描述与 ``scene_name`` 元数据。"""
    mc_slug = f"{scene_slug}_mc"
    if run_sensing:
        return (
            out_dir / f"{scene_slug}_mc_monostatic_sensing.h5",
            f"Monte Carlo + monostatic sensing ({n_episodes} samples) in {scene_slug}",
            mc_slug,
        )
    return (
        out_dir / f"{scene_slug}_mc_sionna_dataset.h5",
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) in {scene_slug}",
        mc_slug,
    )


def _resolve_gif_path(out_dir: Path) -> Path:
    return out_dir / "scene_image_mc.gif"


def _capture_scene_frame(scene: object) -> np.ndarray:
    """``render()`` 可能返回 Matplotlib Figure 或 ndarray，统一为 RGB 帧。"""
    scene_image = scene.render()
    if hasattr(scene_image, "canvas"):
        scene_image.canvas.draw()
        frame = np.asarray(scene_image.canvas.buffer_rgba())[..., :3].copy()
        plt.close(scene_image)
        return frame
    return scene_image


def _build_collection_metadata(
    args: argparse.Namespace,
    cfg: CollectionConfig,
    scene_slug: str,
    roi_box3d: tuple | None,
    *,
    quality_stats: QualityFilterStats | None = None,
) -> CollectionMetadata:
    """汇总本次采集 CLI/配置，写入 HDF5 ``collection_*`` 根属性。"""
    roi_xmin = roi_xmax = roi_ymin = roi_ymax = None
    roi_z = DEFAULT_MC_ROI_Z[0]
    if roi_box3d is not None:
        (roi_xmin, roi_xmax), (roi_ymin, roi_ymax), (z_lo, _z_hi) = roi_box3d
        roi_xmin, roi_xmax = float(roi_xmin), float(roi_xmax)
        roi_ymin, roi_ymax = float(roi_ymin), float(roi_ymax)
        roi_z = float(z_lo)

    return CollectionMetadata(
        seed=int(args.seed),
        config_file=str(args.config_file),
        scene_slug=scene_slug,
        num_samples=int(args.num_samples),
        run_sensing=cfg.run_sensing,
        save_cir=cfg.save_cir,
        roi_xmin=roi_xmin,
        roi_xmax=roi_xmax,
        roi_ymin=roi_ymin,
        roi_ymax=roi_ymax,
        roi_z=roi_z,
        sampling_mode=args.sampling_mode,
        velocity_sampling=args.velocity_sampling,
        safe_margin=float(args.safe_margin),
        max_trials_factor=int(args.max_trials_factor),
        speed_min=float(args.speed_range[0]),
        speed_max=float(args.speed_range[1]),
        quality_filter=cfg.quality_filter,
        quality_accepted=quality_stats.accepted if quality_stats else None,
        quality_rejected=quality_stats.rejected if quality_stats else None,
        quality_reject_no_valid_paths=(
            quality_stats.reject_counts.get("no_valid_paths") if quality_stats else None
        ),
        quality_reject_weak_los=(
            quality_stats.reject_counts.get("weak_los") if quality_stats else None
        ),
        quality_reject_low_peak_prominence=(
            quality_stats.reject_counts.get("low_peak_prominence")
            if quality_stats
            else None
        ),
        quality_reject_peak_misaligned=(
            quality_stats.reject_counts.get("peak_misaligned")
            if quality_stats
            else None
        ),
        require_los=cfg.require_los if cfg.quality_filter else None,
        min_los_ratio=cfg.min_los_ratio if cfg.quality_filter else None,
        min_peak_prominence_db=(
            cfg.min_peak_prominence_db if cfg.quality_filter else None
        ),
        max_bin_offset=cfg.max_bin_offset if cfg.quality_filter else None,
    )


def _update_rt_target_pose_from_velocity(
    target: object,
    pos: np.ndarray | list[float],
    vel: np.ndarray | list[float],
) -> None:
    """更新 RT 目标位置、速度与朝向（速度方向 → yaw/pitch/roll）。"""
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


def _export_h5(
    system: System,
    scene: object,
    *,
    cfg: CollectionConfig,
    scene_slug: str,
    n_episodes: int,
    h_freq_list: list[np.ndarray],
    cir_a_list: list[np.ndarray],
    cir_tau_list: list[np.ndarray],
    target_pos_list: list[np.ndarray],
    target_vel_list: list[np.ndarray],
    collection_meta: CollectionMetadata,
    out_dir: Path,
) -> None:
    if not cfg.save_h5:
        return
    if not h_freq_list:
        print("未采集 CFR，跳过 HDF5")
        return

    cir_a_arr: np.ndarray | None = None
    cir_tau_arr: np.ndarray | None = None
    if cfg.save_cir:
        if not cir_a_list:
            raise RuntimeError("save_cir 已启用但主循环未采集 CIR")
        cir_a_arr, cir_tau_arr = RTScene.stack_ragged_cir_samples(cir_a_list, cir_tau_list)

    h5_path, desc_h5, scene_name = _resolve_h5_output(
        run_sensing=cfg.run_sensing,
        scene_slug=scene_slug,
        n_episodes=n_episodes,
        out_dir=out_dir,
    )
    # 发射机位置取 bs1（与 data_collection 默认场景约定一致）
    Dataset.from_export_arrays(
        np.array(h_freq_list),
        np.array(target_pos_list),
        np.array(target_vel_list),
        np.array(scene.transceivers["bs1"].position),
        system.params.carrier_frequency,
        system.params.ofdm.subcarrier_spacing,
        system.params.ofdm.fft_size,
        len(h_freq_list),
        scene_name,
        dataset_cir_a=cir_a_arr,
        dataset_cir_tau=cir_tau_arr,
        description=desc_h5,
        collection_meta=collection_meta,
    ).save(h5_path)


def _export_gif(
    *,
    cfg: CollectionConfig,
    scene_frames: list[np.ndarray],
    out_dir: Path,
) -> None:
    if not cfg.save_gif:
        return
    if not scene_frames:
        print("无场景帧，跳过 GIF 导出")
        return
    images_to_gif(
        filepath=_resolve_gif_path(out_dir),
        images=scene_frames,
        time_slot=1,
        speed=5,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（蒙特卡洛、感知、导出格式）。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    # --- 系统与随机性 ---
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_monostatic_canyon.toml",
        help="配置文件路径（须含非空 [rt_scene]）",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="计算设备类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（蒙特卡洛位置/速度采样）",
    )

    # --- Episode 来源与导出开关 ---
    parser.add_argument(
        "--run_sensing", action="store_true", help="每步/每样本执行单站感知"
    )
    parser.add_argument(
        "--no-save-h5", dest="save_h5", action="store_false", help="不写 HDF5"
    )
    parser.add_argument(
        "--no-save-csv", dest="save_csv", action="store_false", help="不写 CSV"
    )
    parser.add_argument(
        "--save-cir",
        action="store_true",
        help="HDF5 中额外写入 CIR（channel_impulse_response_*）；默认仅 CFR + kinematics",
    )
    parser.add_argument("--save_gif", action="store_true", help="导出场景 GIF")
    parser.set_defaults(save_h5=True, save_csv=True)

    # --- 感知与 CSV 模式 ---
    parser.add_argument(
        "--csv_mode",
        type=str,
        default="legacy",
        choices=["unified", "legacy"],
        help="CSV 模式",
    )
    parser.add_argument(
        "--sensing_domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="感知链路 domain（与 run_sensing 配合）",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic", "bistatic_rx_radial"],
        help="保留 CLI 兼容；双基地真值速度经 paths.doppler + bistatic 换算",
    )
    parser.add_argument(
        "--sensing_layout",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="run_sensing 时：monostatic=RX 径向；bistatic=LoS 总路径 + paths.doppler（建议 csv_mode=unified）",
    )
    parser.add_argument(
        "--log_per_step_sensing",
        action="store_true",
        help="每步打印感知一行日志",
    )

    # --- 蒙特卡洛采样参数 ---
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=None,
        help="蒙特卡洛：平面 ROI 四元组，z 固定为 0",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10e3, help="蒙特卡洛：样本数"
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="蒙特卡洛：目标位置在 ROI 内的采样分布（均匀或高斯）",
    )
    parser.add_argument(
        "--safe_margin",
        type=float,
        default=1.0,
        help="蒙特卡洛：障碍物包围盒额外安全距离（米），位置合法性校验用",
    )
    parser.add_argument(
        "--max_trials_factor",
        type=int,
        default=20,
        help="蒙特卡洛：拒绝采样最大尝试次数 = num_samples × 该因子（防死循环）",
    )
    parser.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        default=[0.1, 10.0],
        metavar=("MIN", "MAX"),
        help="蒙特卡洛速度幅值范围（未提供 velocities 数组时）",
    )
    parser.add_argument(
        "--velocity_sampling",
        type=str,
        default="sphere_uniform",
        choices=["sphere_uniform", "axis_box"],
        help="蒙特卡洛：速度方向/分量采样方式（球面均匀或各轴独立盒式）",
    )

    # --- 样本质量门控（CNN 训练推荐开启）---
    parser.add_argument(
        "--quality_filter",
        action="store_true",
        default=True,
        help="启用 LoS + DD 谱峰质量过滤（默认开启）",
    )
    parser.add_argument(
        "--no-quality-filter",
        dest="quality_filter",
        action="store_false",
        help="关闭质量过滤，保留所有几何合法样本",
    )
    parser.add_argument(
        "--require_los",
        action="store_true",
        default=True,
        help="质量过滤：要求 RT 存在与几何时延一致的有效路径",
    )
    parser.add_argument(
        "--no-require_los",
        dest="require_los",
        action="store_false",
        help="质量过滤：跳过 LoS 路径检查",
    )
    parser.add_argument(
        "--min_los_ratio",
        type=float,
        default=0.3,
        help="质量过滤：几何最近路径幅度 / 最强路径幅度 下限",
    )
    parser.add_argument(
        "--min_peak_prominence_db",
        type=float,
        default=6.0,
        help="质量过滤：DD 谱峰相对全局均值的突出度下限 (dB)",
    )
    parser.add_argument(
        "--max_bin_offset",
        type=int,
        default=3,
        help="质量过滤：谱峰与几何 bin 的最大允许偏差 (bin)",
    )
    parser.add_argument(
        "--quality_max_trials_factor",
        type=int,
        default=50,
        help="蒙特卡洛+质量过滤：最大尝试次数 = num_samples × 该因子",
    )

    return parser.parse_args()


def _process_episode(
    *,
    system: System,
    scene: object,
    cfg: CollectionConfig,
    episode_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    h_freq_list: list[np.ndarray],
    cir_a_list: list[np.ndarray],
    cir_tau_list: list[np.ndarray],
    target_pos_list: list[np.ndarray],
    target_vel_list: list[np.ndarray],
    scene_frames: list[np.ndarray],
    csv_rows: list[dict[str, str | int]],
    out_dir: Path,
) -> None:
    """单条 episode：真值/感知/CFR/CSV/GIF 缓冲写入。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
    if cfg.save_h5:
        target_pos_list.append(pos_row.copy())
        target_vel_list.append(vel_row.copy())

    true_range, true_velocity = _los_truth_at_first_triple(scene, system.device)
    row = _kinematics_row(episode_idx, pos_row, vel_row, true_range, true_velocity)

    if cfg.run_sensing:
        if cfg.sensing_layout == "bistatic":
            est_path, est_vel, est_db, true_path, true_vp = bistatic_sensing_eval(
                system, out_dir, cfg.sensing_domain
            )
            if cfg.log_per_step_sensing:
                _log_sensing_step(
                    episode_idx,
                    layout="bistatic",
                    est_range_or_path=est_path,
                    est_velocity=est_vel,
                    est_power_db=est_db,
                )
            _append_bistatic_sensing_columns(
                row,
                est_path=est_path,
                est_velocity=est_vel,
                true_path=true_path,
                true_velocity=true_vp,
            )
        else:
            est_range, est_vel, est_db = monostatic_sensing_eval(
                system, out_dir, cfg.sensing_domain
            )
            if cfg.log_per_step_sensing:
                _log_sensing_step(
                    episode_idx,
                    layout="monostatic",
                    est_range_or_path=est_range,
                    est_velocity=est_vel,
                    est_power_db=est_db,
                )
            _append_monostatic_sensing_columns(
                row,
                est_range=est_range,
                est_velocity=est_vel,
                true_range=true_range,
                true_velocity=true_velocity,
            )

    if cfg.save_csv:
        csv_rows.append(row)

    if cfg.save_h5:
        h_freq_list.append(scene.cfr_numpy(system.components.rg))
        if cfg.save_cir:
            ca, ct = scene.cir_numpy(system.components.rg)
            cir_a_list.append(ca)
            cir_tau_list.append(ct)

    if cfg.save_gif:
        scene_frames.append(_capture_scene_frame(scene))


def _run_monte_carlo_with_quality_filter(
    *,
    system: System,
    scene: object,
    target: object,
    cfg: CollectionConfig,
    args: argparse.Namespace,
    roi_box3d: tuple,
    h_freq_list: list[np.ndarray],
    cir_a_list: list[np.ndarray],
    cir_tau_list: list[np.ndarray],
    target_pos_list: list[np.ndarray],
    target_vel_list: list[np.ndarray],
    scene_frames: list[np.ndarray],
    csv_rows: list[dict[str, str | int]],
    out_dir: Path,
) -> QualityFilterStats:
    """拒绝采样直至凑满 ``num_samples`` 个可检测样本。"""
    num_target = int(args.num_samples)
    max_trials = num_target * int(args.quality_max_trials_factor)
    rng = np.random.default_rng(int(args.seed))
    quality_stats = QualityFilterStats()
    quality_cfg = cfg.sample_quality_config()
    sensing_perf = system.components.sensing_performance

    accepted = 0
    trials = 0
    pbar = tqdm(total=num_target, desc="MC 数据集(质量过滤)", unit="sample")

    while accepted < num_target and trials < max_trials:
        trials += 1
        pos_batch = tg.generate_monte_carlo_points(
            scene,
            roi_box3d,
            1,
            sampling_mode=args.sampling_mode,
            safe_margin=args.safe_margin,
            max_trials_factor=args.max_trials_factor,
            rng=rng,
        )
        vel_batch = tg.sample_monte_carlo_velocities(
            1,
            rng,
            None,
            (float(args.speed_range[0]), float(args.speed_range[1])),
            args.velocity_sampling,
            None,
            None,
            None,
        )
        pos = pos_batch[0]
        vel = vel_batch[0]

        _update_rt_target_pose_from_velocity(target, pos, vel)
        true_range, true_velocity = _los_truth_at_first_triple(scene, system.device)
        cfr = scene.cfr_numpy(system.components.rg)

        result = evaluate_sample_quality(
            scene,
            cfr,
            float(true_range.item()),
            float(true_velocity.item()),
            sensing_perf,
            cfg=quality_cfg,
            device=torch.device(system.device),
        )
        if not result.passed:
            quality_stats.record_reject(result.reason or "low_peak_prominence")
            continue

        quality_stats.record_accept()
        _process_episode(
            system=system,
            scene=scene,
            cfg=cfg,
            episode_idx=accepted,
            pos=pos,
            vel=vel,
            h_freq_list=h_freq_list,
            cir_a_list=cir_a_list,
            cir_tau_list=cir_tau_list,
            target_pos_list=target_pos_list,
            target_vel_list=target_vel_list,
            scene_frames=scene_frames,
            csv_rows=csv_rows,
            out_dir=out_dir,
        )
        accepted += 1
        pbar.update(1)

    pbar.close()
    if accepted < num_target:
        raise RuntimeError(
            f"质量过滤后仅采集 {accepted}/{num_target} 个样本，"
            f"尝试 {trials}/{max_trials} 次。请放宽 ROI/阈值或增大 --quality_max_trials_factor。"
        )
    print(quality_stats.summary_line())
    return quality_stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    """生成 episode → 可选感知 → 写出 CSV / HDF5 / GIF。"""
    # 1. 解析 CLI、固定随机种子、整理导出/感知选项
    args = argument_parser()
    set_random_seed(args.seed)
    cfg = CollectionConfig.from_args(args)

    # 2. 构建仿真系统，准备输出目录
    system = System(args)
    SCRIPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 取 RT 场景与待驱动的目标（本脚本假定至少一个 rt_target）
    scene = system.components.rt_scene
    if not scene.rt_targets:
        raise RuntimeError("当前场景中没有可用的 RT 目标（scene.rt_targets 为空）")
    _, target = next(iter(scene.rt_targets.items()))
    scene_slug = scene.output_slug
    roi_box3d = _resolve_roi(args)

    h_freq_list: list[np.ndarray] = []
    cir_a_list: list[np.ndarray] = []
    cir_tau_list: list[np.ndarray] = []
    target_pos_list: list[np.ndarray] = []
    target_vel_list: list[np.ndarray] = []
    scene_frames: list[np.ndarray] = []
    csv_rows: list[dict[str, str | int]] = []
    quality_stats: QualityFilterStats | None = None

    if cfg.quality_filter:
        quality_stats = _run_monte_carlo_with_quality_filter(
            system=system,
            scene=scene,
            target=target,
            cfg=cfg,
            args=args,
            roi_box3d=roi_box3d,
            h_freq_list=h_freq_list,
            cir_a_list=cir_a_list,
            cir_tau_list=cir_tau_list,
            target_pos_list=target_pos_list,
            target_vel_list=target_vel_list,
            scene_frames=scene_frames,
            csv_rows=csv_rows,
            out_dir=SCRIPT_OUT_DIR,
        )
        n_ep = quality_stats.accepted
    else:
        pos_arr, vel_arr = tg.generate_targets_monte_carlo(
            scene,
            roi=roi_box3d,
            num_samples=int(args.num_samples),
            sampling_mode=args.sampling_mode,
            safe_margin=args.safe_margin,
            max_trials_factor=args.max_trials_factor,
            velocities=None,
            speed_range=(float(args.speed_range[0]), float(args.speed_range[1])),
            velocity_sampling=args.velocity_sampling,
            velocity_roi_vx=None,
            velocity_roi_vy=None,
            velocity_roi_vz=None,
            seed=args.seed,
            rng=None,
        )
        n_ep = int(pos_arr.shape[0])
        if n_ep == 0:
            print("无有效 Episode，结束")
            return

        quality_cfg = cfg.sample_quality_config() if cfg.quality_filter else None
        sensing_perf = system.components.sensing_performance
        if cfg.quality_filter:
            quality_stats = QualityFilterStats()

        accepted_idx = 0
        for i in tqdm(range(n_ep), desc="MC 数据集", unit="sample"):
            pos = pos_arr[i]
            vel = vel_arr[i]
            _update_rt_target_pose_from_velocity(target, pos, vel)

            if cfg.quality_filter:
                true_range, true_velocity = _los_truth_at_first_triple(
                    scene, system.device
                )
                cfr = scene.cfr_numpy(system.components.rg)
                result = evaluate_sample_quality(
                    scene,
                    cfr,
                    float(true_range.item()),
                    float(true_velocity.item()),
                    sensing_perf,
                    cfg=quality_cfg,
                    device=torch.device(system.device),
                )
                if not result.passed:
                    quality_stats.record_reject(result.reason or "low_peak_prominence")
                    continue
                quality_stats.record_accept()

            _process_episode(
                system=system,
                scene=scene,
                cfg=cfg,
                episode_idx=accepted_idx if cfg.quality_filter else i,
                pos=pos,
                vel=vel,
                h_freq_list=h_freq_list,
                cir_a_list=cir_a_list,
                cir_tau_list=cir_tau_list,
                target_pos_list=target_pos_list,
                target_vel_list=target_vel_list,
                scene_frames=scene_frames,
                csv_rows=csv_rows,
                out_dir=SCRIPT_OUT_DIR,
            )
            accepted_idx += 1

        if cfg.quality_filter:
            print(quality_stats.summary_line())
            n_ep = accepted_idx
            if n_ep == 0:
                print("质量过滤后无有效样本，结束")
                return

    collection_meta = _build_collection_metadata(
        args, cfg, scene_slug, roi_box3d, quality_stats=quality_stats
    )

    # 8. 落盘：Episode CSV、HDF5（CFR [+ 可选 CIR] + kinematics）、场景 GIF
    if cfg.save_csv:
        save_episodes_csv(
            scene_slug=scene_slug,
            rows=csv_rows,
            run_sensing=cfg.run_sensing,
            csv_mode=cfg.csv_mode,
            output_root=SCRIPT_OUT_DIR,
        )

    _export_h5(
        system,
        scene,
        cfg=cfg,
        scene_slug=scene_slug,
        n_episodes=n_ep,
        h_freq_list=h_freq_list,
        cir_a_list=cir_a_list,
        cir_tau_list=cir_tau_list,
        target_pos_list=target_pos_list,
        target_vel_list=target_vel_list,
        collection_meta=collection_meta,
        out_dir=SCRIPT_OUT_DIR,
    )
    _export_gif(cfg=cfg, scene_frames=scene_frames, out_dir=SCRIPT_OUT_DIR)


if __name__ == "__main__":
    main()
