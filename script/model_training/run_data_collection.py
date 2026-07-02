"""ISAC 数据集采集入口：平面 ROI 蒙特卡洛采样 → RT 目标位姿驱动 → TOML / CSV / HDF5。

流程概要
--------
1. 解析 CLI，设置随机种子，批量采样 ROI 内位置与速度。
2. 构建 ``System``，循环更新 RT 目标位姿并采集 CFR / 几何真值。
3. 写出 ``out/dataset_collection/`` 下的 TOML、CSV 与 HDF5。
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from isac.datasets import (
    DEFAULT_COLLECTION_OUT_DIR,
    EpisodeBuffers,
    save_collection_artifacts,
)
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.data_collection.channel_export import (
    paths_intersect_target,
    scene_slug_from_rt_simulator,
)
from isac.utils.data_collection.episode_filter import accept_episode_kinematics
from isac.utils.data_collection.episode import (
    process_episode,
)
from isac.utils.data_collection.roi_sampling import RoiKinematicsSampler

warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（蒙特卡洛、导出格式）。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="采样条数",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[-2.8, 2.8, -4.8, 4.8],
        help="平面 ROI 四元组",
    )
    parser.add_argument(
        "--position_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="位置采样分布（均匀或高斯）",
    )
    parser.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[0.1, 3.0],
        help="速度模值范围 (m/s)",
    )
    parser.add_argument(
        "--speed_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="速度模值采样分布（均匀或高斯）",
    )
    parser.add_argument(
        "--sampler_pool_factor",
        type=int,
        default=5,
        help="采样池倍数：预采样 num_samples * factor 条，循环中过滤至 num_samples",
    )

    return parser.parse_args()


def _print_sampling_config(args: argparse.Namespace, *, pool_size: int) -> None:
    """以 tabulate 打印采样配置参数。"""
    x_lo, x_hi, y_lo, y_hi = args.roi
    config_rows = [
        ["num_samples", args.num_samples],
        ["sampler_pool_size", pool_size],
        ["sampler_pool_factor", args.sampler_pool_factor],
        ["roi", f"x=[{x_lo}, {x_hi}], y=[{y_lo}, {y_hi}], z=0"],
        ["position_sampling_mode", args.position_sampling_mode],
        ["speed_range", f"[{args.speed_range[0]}, {args.speed_range[1]}] m/s"],
        ["speed_sampling_mode", args.speed_sampling_mode],
        ["seed", args.seed],
    ]
    print(tabulate(config_rows, headers=["参数", "值"], tablefmt="simple_grid"))


def main() -> None:
    """蒙特卡洛采集 episode → 写出 TOML / CSV / HDF5。"""
    args = argument_parser()
    if args.sampler_pool_factor < 1:
        raise ValueError("sampler_pool_factor 须 >= 1")
    set_random_seed(args.seed)

    pool_size = args.num_samples * args.sampler_pool_factor
    sampler = RoiKinematicsSampler(
        roi=args.roi,
        position_sampling_mode=args.position_sampling_mode,
        speed_range=args.speed_range,
        speed_sampling_mode=args.speed_sampling_mode,
        num_samples=pool_size,
    )
    _print_sampling_config(args, pool_size=pool_size)

    # 加载配置，构建 System
    config = load_config(args.config_file)
    system = System(
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )
    # 获取 RT 模拟器与目标
    rt_simulator = system.components.rt_simulator
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}")

    # 初始化缓冲区
    buffers = EpisodeBuffers()

    # 采集 episode
    accepted = 0
    attempts = 0
    pbar = tqdm(total=args.num_samples, desc="数据采集", unit="ep")
    while accepted < args.num_samples:
        if len(sampler) == 0:
            raise RuntimeError(
                f"采样池已耗尽：已采纳 {accepted}/{args.num_samples} 条。"
                "请增大 --sampler_pool_factor 或放宽过滤条件（scene_filter / 目标路径交互）。"
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

        # 采集 episode
        process_episode(
            system=system,
            rt_simulator=rt_simulator,
            episode_idx=accepted,
            pos=pos,
            vel=vel,
            buffers=buffers,
        )
        accepted += 1
        pbar.update(1)
    pbar.close()

    acceptance_rate = accepted / attempts if attempts else 0.0
    print(f"接受率: {acceptance_rate:.1%} ({accepted}/{attempts})")

    # 写出数据集
    save_collection_artifacts(
        scene_slug=scene_slug,
        config_file=args.config_file,
        buffers=buffers,
        bs_pos=np.asarray(rt_simulator.transceivers["bs1"].position, dtype=np.float64),
        args=args,
        rt_simulator=rt_simulator,
        out_dir=DEFAULT_COLLECTION_OUT_DIR,
    )


if __name__ == "__main__":
    main()
