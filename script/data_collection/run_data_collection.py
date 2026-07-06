"""ISAC 数据集采集入口：平面 ROI 蒙特卡洛采样 → RT 目标位姿驱动 → TOML / CSV / HDF5。

流程概要
--------
1. 解析 CLI，设置随机种子，批量采样 ROI 内位置与速度。
2. 构建 ``System``，循环更新 RT 目标位姿并采集 CFR / 几何真值。
3. 写出 ``data/`` 下的 TOML、CSV 与 HDF5。
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from isac.datasets import (
    CollectionMetadata,
    DEFAULT_COLLECTION_OUT_DIR,
    EpisodeBuffers,
    RTDataset,
    SensingMetadata,
    collection_h5_path,
    save_collection_artifacts,
)
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.data_collection.channel_export import (
    los_truth_from_kinematics,
    paths_intersect_target,
    scene_slug_from_rt_simulator,
)
from isac.utils.data_collection.episode_filter import accept_episode_kinematics
from isac.utils.data_collection.roi_sampling import RoiKinematicsSampler
from isac.data_structures import CollectionSamplingParams
from isac.utils.misc import csv_float2_scalar, csv_vec3

warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（蒙特卡洛、导出格式）。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/data_collection/data_collection.toml",
        help="配置文件路径",
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

    # 采集参数
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="采样条数",
    )
    parser.add_argument(
        "--sampler_pool_factor",
        type=int,
        default=5,
        help="采样池倍数：预采样 num_samples * factor 条，循环中过滤至 num_samples",
    )
    parser.add_argument(
        "--h5_compression",
        type=str,
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 h_dd 压缩算法（流式写入，默认 lzf）",
    )

    return parser.parse_args()


def _print_sampling_config(
    *,
    sampling: CollectionSamplingParams,
    num_samples: int,
    sampler_pool_factor: int,
    seed: int,
) -> None:
    """以 tabulate 打印采样配置参数。"""
    x_lo, x_hi, y_lo, y_hi = sampling.roi
    config_rows = [
        ["num_samples", num_samples],
        ["sampler_pool_factor", sampler_pool_factor],
        ["roi", f"x=[{x_lo}, {x_hi}], y=[{y_lo}, {y_hi}], z=0"],
        ["position_sampling_mode", sampling.position_sampling_mode],
        ["speed_range", f"[{sampling.speed_range[0]}, {sampling.speed_range[1]}] m/s"],
        ["speed_sampling_mode", sampling.speed_sampling_mode],
        ["seed", seed],
    ]
    print(tabulate(config_rows, headers=["参数", "值"], tablefmt="simple_grid"))


def main() -> None:
    """蒙特卡洛采集 episode → 写出 TOML / CSV / HDF5。"""
    args = argument_parser()
    if args.sampler_pool_factor < 1:
        raise ValueError("sampler_pool_factor 须 >= 1")

    config = load_config(args.config_file)
    sampling = CollectionSamplingParams.from_dict(config)

    set_random_seed(args.seed)

    pool_size = args.num_samples * args.sampler_pool_factor
    sampler = RoiKinematicsSampler(
        roi=sampling.roi,
        position_sampling_mode=sampling.position_sampling_mode,
        speed_range=sampling.speed_range,
        speed_sampling_mode=sampling.speed_sampling_mode,
        num_samples=pool_size,
    )
    _print_sampling_config(
        sampling=sampling,
        num_samples=args.num_samples,
        sampler_pool_factor=args.sampler_pool_factor,
        seed=args.seed,
    )

    # 构建 System
    system = System(
        config=config,
        device=args.device,
    )
    system.components.sensing_performance
    sensing_meta = SensingMetadata.from_system(system)

    # 获取 RT 模拟器与目标
    rt_simulator = system.components.rt_simulator
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}")

    # 初始化 CSV 缓冲与流式 HDF5 写入
    buffers = EpisodeBuffers()
    bs_pos = np.asarray(rt_simulator.transceivers["bs1"].position, dtype=np.float64)
    h5_path = collection_h5_path(scene_slug, DEFAULT_COLLECTION_OUT_DIR)
    collection_meta = CollectionMetadata.from_sampling_params(args.seed, sampling)

    with RTDataset.open_for_collection(
        h5_path,
        bs_pos,
        compression=args.h5_compression,
    ) as dataset:
        # 采集 episode
        accepted = 0
        attempts = 0
        pbar = tqdm(total=args.num_samples, desc="数据采集", unit="ep")
        while accepted < args.num_samples:
            if len(sampler) == 0:
                raise RuntimeError(
                    f"采样池已耗尽：已采纳 {accepted}/{args.num_samples} 条。"
                    "请增大 --sampler_pool_factor 或调整 [monte_carlo_sampling] / 过滤条件（scene_filter / 目标路径交互）。"
                )
            pos, vel, ori = sampler.pop()
            attempts += 1

            # 过滤不符合条件的 episode
            if not accept_episode_kinematics(
                pos=pos,
                vel=vel,
                ori=ori,
                scene_filter=rt_simulator.scene_filter,
            ):
                continue

            # 更新 RT 目标位姿
            target(
                position=pos,
                velocity=vel,
                orientation=ori,
            )
            # 判断是否与目标路径交互
            if not paths_intersect_target(rt_simulator, target):
                continue

            true_range, true_velocity = los_truth_from_kinematics(
                pos, vel, rt_simulator, system.device
            )
            buffers.csv_rows.append(
                {
                    "sample_idx": accepted,
                    "position": csv_vec3(pos),
                    "velocity": csv_vec3(vel),
                    "true_range_m": csv_float2_scalar(true_range),
                    "true_radial_velocity_mps": csv_float2_scalar(true_velocity),
                }
            )

            # 信号传输
            _, x_rg, x_time = system.transmit()
            y_rg = system.components.channel(x_rg, x_time, domain="frequency")
            h_dd = system.compute_sensing_spectrum(x_rg, y_rg)

            # 写入数据集
            dataset.append_episode(
                h_dd.detach().cpu().numpy().astype(np.complex64),
                pos,
                vel,
            )

            # 更新计数器
            accepted += 1
            pbar.update(1)
        pbar.close()

        # 计算接受率
        acceptance_rate = accepted / attempts if attempts else 0.0
        print(f"接受率: {acceptance_rate:.1%} ({accepted}/{attempts})")

        # 写出数据集
        dataset.finalize(
            collection_meta=collection_meta,
            scene_slug=scene_slug,
            sensing_meta=sensing_meta,
        )

    # 写出其余采集产物（TOML / CSV / 场景 PNG）
    save_collection_artifacts(
        scene_slug=scene_slug,
        config_file=args.config_file,
        buffers=buffers,
        rt_simulator=rt_simulator,
        out_dir=DEFAULT_COLLECTION_OUT_DIR,
    )


if __name__ == "__main__":
    main()
