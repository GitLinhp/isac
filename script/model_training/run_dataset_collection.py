"""ISAC 数据集采集入口：蒙特卡洛 ROI 采样生成目标位姿序列，并写出 CSV / HDF5。

流程概要
--------
1. 解析 CLI，设置随机种子，解析 ROI。
2. 构建 ``System``，输出目录固定为 ``out/dataset_collection/``。
3. 蒙特卡洛批量生成 episode 序列：更新 RT 目标位姿 → 记录几何真值 → 采集 CFR → 累计 I/O 缓冲。
4. 循环结束后始终写出 CSV 与 HDF5；HDF5 默认仅 CFR + kinematics，加 ``--save-cir`` 才写入 CIR。

约定
----
- CSV 固定输出 ``{scene_slug}_mc_dataset_episodes.csv``（各行列名的并集，缺失列填空）。
- HDF5 与 CSV **始终写入**，无 CLI 关闭开关。
- 几何真值默认取 ``RxTargetTxGeometric`` 的 ``[0, 0, 0]`` 切片（单 RX × 单目标 × 单 TX）。
- 蒙特卡洛 ROI：CLI 为 ``--roi XMIN XMAX YMIN YMAX`` 四元组，``z`` 固定为 ``0``（即 ``(0, 0)``）。
- HDF5 根属性 ``has_cir`` 标记是否包含 ``channel_impulse_response_*`` 数据集。
- HDF5 另含 ``collection_*`` 根属性（ROI、seed、source、采样参数等），见 ``CollectionMetadata``。
"""

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.datasets import CollectionMetadata, Dataset
from isac.channel.rt.rx_target_tx_geometric import RxTargetTxGeometric
from isac.system import System
from isac.channel import RTScene
from isac.utils import csv_float2_scalar
from isac.utils import cartesian_direction_to_yaw_pitch_roll, set_random_seed
from isac.utils import target_generation as tg

# Sionna/DrJit 射线追踪在大量 episode 时会触发 AST 装饰器次数告警，不影响数值结果
warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)

SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "dataset_collection"


# ---------------------------------------------------------------------------
# Episode 缓冲
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBuffers:
    """主循环共享的 episode 级写出缓冲（由 ``main`` 创建，``_process_episode`` 追加）。"""

    h_freq_list: list[np.ndarray]
    cir_a_list: list[np.ndarray]
    cir_tau_list: list[np.ndarray]
    target_pos_list: list[np.ndarray]
    target_vel_list: list[np.ndarray]
    csv_rows: list[dict[str, str | int]]


# ---------------------------------------------------------------------------
# ROI 解析
# ---------------------------------------------------------------------------


def _roi_xy_to_box3d(
    roi_xy: tuple[tuple[float, float], tuple[float, float]],
    *,
    z_bounds: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """平面 ROI 四元组 ``((xmin,xmax),(ymin,ymax))`` → 三维 ``RoiBox3D``，``z`` 默认固定为 0。"""
    return roi_xy[0], roi_xy[1], z_bounds


def _resolve_roi(args: argparse.Namespace) -> tuple:
    """解析蒙特卡洛 ROI（xy 四元组 + z=0）。"""
    if args.roi is not None:
        r = args.roi
        return _roi_xy_to_box3d(((r[0], r[1]), (r[2], r[3])))
    return _roi_xy_to_box3d(((0.0, 80.0), (-40.0, 40.0)))


# ---------------------------------------------------------------------------
# Episode 循环（位姿更新 / 单条处理）
# ---------------------------------------------------------------------------


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
    rx_i, tgt_i, tx_i = 0, 0, 0
    return geom.range_tensor[rx_i, tgt_i, tx_i], geom.vel_tensor[rx_i, tgt_i, tx_i]


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


def _process_episode(
    *,
    system: System,
    scene: object,
    save_cir: bool,
    episode_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    buffers: EpisodeBuffers,
) -> None:
    """单条 episode：几何真值 / CFR / CSV 缓冲写入。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)

    buffers.target_pos_list.append(pos_row.copy())
    buffers.target_vel_list.append(vel_row.copy())
    true_range, true_velocity = _los_truth_at_first_triple(scene, system.device)
    row = _kinematics_row(episode_idx, pos_row, vel_row, true_range, true_velocity)

    buffers.csv_rows.append(row)
    buffers.h_freq_list.append(scene.cfr_numpy(system.components.rg))
    if save_cir:
        ca, ct = scene.cir_numpy(system.components.rg)
        buffers.cir_a_list.append(ca)
        buffers.cir_tau_list.append(ct)


# ---------------------------------------------------------------------------
# 落盘（元数据 / CSV / HDF5）
# ---------------------------------------------------------------------------


def _build_collection_metadata(
    args: argparse.Namespace,
    scene_slug: str,
    roi_box3d: tuple | None,
) -> CollectionMetadata:
    """汇总本次采集 CLI/配置，写入 HDF5 ``collection_*`` 根属性。"""
    roi_xmin = roi_xmax = roi_ymin = roi_ymax = None
    roi_z = 0.0
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
        run_sensing=False,
        save_cir=args.save_cir,
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
        quality_filter=False,
        quality_accepted=None,
        quality_rejected=None,
        quality_reject_no_valid_paths=None,
        quality_reject_weak_los=None,
        quality_reject_low_peak_prominence=None,
        quality_reject_peak_misaligned=None,
        require_los=None,
        min_los_ratio=None,
        min_peak_prominence_db=None,
        max_bin_offset=None,
    )


def save_episodes_csv(
    *,
    scene_slug: str,
    rows: list[dict[str, str | int]],
    output_root: Path | None = None,
) -> None:
    """写入 Episode CSV（动态列并集）。"""
    if not rows:
        print("无 CSV 行，跳过写入")
        return
    out_dir = output_root if output_root is not None else PROJECT_ROOT / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Episode CSV 已写入: {path}")


def _resolve_h5_output(
    *,
    scene_slug: str,
    n_episodes: int,
    out_dir: Path,
) -> tuple[Path, str | None, str]:
    """HDF5 路径、描述与 ``scene_name`` 元数据。"""
    mc_slug = f"{scene_slug}_mc"
    return (
        out_dir / f"{scene_slug}_mc_sionna_dataset.h5",
        f"Sionna generated ISAC Monte Carlo dataset ({n_episodes} samples) in {scene_slug}",
        mc_slug,
    )


def _export_h5(
    system: System,
    scene: object,
    *,
    save_cir: bool,
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
    """将主循环缓冲的 CFR/kinematics 封装为 ``Dataset`` 并落盘 HDF5。

    ``save_cir`` 启用时对 ragged CIR 样本做 stack 后一并写入。
    """
    if not h_freq_list:
        print("未采集 CFR，跳过 HDF5")
        return

    cir_a_arr: np.ndarray | None = None
    cir_tau_arr: np.ndarray | None = None
    if save_cir:
        if not cir_a_list:
            raise RuntimeError("save_cir 已启用但主循环未采集 CIR")
        cir_a_arr, cir_tau_arr = RTScene.stack_ragged_cir_samples(
            cir_a_list, cir_tau_list
        )

    h5_path, desc_h5, scene_name = _resolve_h5_output(
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（蒙特卡洛、导出格式）。"""
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

    # --- 导出开关 ---
    parser.add_argument(
        "--save-cir",
        action="store_true",
        help="HDF5 中额外写入 CIR（channel_impulse_response_*）；默认仅 CFR + kinematics",
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
        "--num_samples", type=int, default=10000, help="蒙特卡洛：样本数"
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

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """蒙特卡洛采集 episode → 写出 CSV / HDF5。"""
    # 1. 解析 CLI、固定随机种子
    args = argument_parser()
    set_random_seed(args.seed)

    # 2. 构建仿真系统，准备输出目录
    system = System(args)
    SCRIPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 取 RT 场景与待驱动的目标，解析 ROI，初始化 episode 缓冲
    scene = system.components.rt_scene
    if not scene.rt_targets:
        raise RuntimeError("当前场景中没有可用的 RT 目标（scene.rt_targets 为空）")
    _, target = next(iter(scene.rt_targets.items()))
    scene_slug = scene.output_slug
    roi_box3d = _resolve_roi(args)
    buffers = EpisodeBuffers(
        h_freq_list=[],
        cir_a_list=[],
        cir_tau_list=[],
        target_pos_list=[],
        target_vel_list=[],
        csv_rows=[],
    )

    # 4. 蒙特卡洛批量生成 (pos, vel) 并逐条处理
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

    for i in tqdm(range(n_ep), desc="MC 数据集", unit="sample"):
        pos = pos_arr[i]
        vel = vel_arr[i]
        _update_rt_target_pose_from_velocity(target, pos, vel)
        _process_episode(
            system=system,
            scene=scene,
            save_cir=args.save_cir,
            episode_idx=i,
            pos=pos,
            vel=vel,
            buffers=buffers,
        )

    # 5. 汇总采集元数据（写入 HDF5 collection_* 根属性）
    collection_meta = _build_collection_metadata(args, scene_slug, roi_box3d)

    # 6. 落盘：Episode CSV、HDF5（CFR [+ 可选 CIR] + kinematics）
    save_episodes_csv(
        scene_slug=scene_slug,
        rows=buffers.csv_rows,
        output_root=SCRIPT_OUT_DIR,
    )

    _export_h5(
        system,
        scene,
        save_cir=args.save_cir,
        scene_slug=scene_slug,
        n_episodes=n_ep,
        h_freq_list=buffers.h_freq_list,
        cir_a_list=buffers.cir_a_list,
        cir_tau_list=buffers.cir_tau_list,
        target_pos_list=buffers.target_pos_list,
        target_vel_list=buffers.target_vel_list,
        collection_meta=collection_meta,
        out_dir=SCRIPT_OUT_DIR,
    )


if __name__ == "__main__":
    main()
