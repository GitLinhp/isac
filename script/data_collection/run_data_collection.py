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
from tqdm import tqdm

from isac import DEFAULT_COLLECTION_OUT_DIR
from isac.collection import (
    CollectionMetadata,
    RTDatasetWriter,
    collection_h5_path,
    save_collection_artifacts,
)
from isac.system import System
from isac.utils import load_config
from isac.collection.utils import scene_slug_from_rt_simulator
from isac.utils.misc import csv_float2_scalar, csv_vec3

# 忽略 Sionna 警告
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

    # 采集参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（蒙特卡洛位置/速度采样）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="目标采纳 episode 数",
    )
    parser.add_argument(
        "--sampler_pool_factor",
        type=int,
        default=5,
        help="预采样池倍数（池大小 = num_samples × 本参数）",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[-2.5, 2.5, -4.5, 4.5],
        help="平面 ROI 四元组（m），z 固定为 0",
    )
    parser.add_argument(
        "--position_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="位置采样分布",
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
        help="速度模值采样分布",
    )
    parser.add_argument(
        "--h5_compression",
        type=str,
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 h_dd 压缩算法（流式写入，默认 lzf）",
    )

    return parser.parse_args()


def main() -> None:
    """蒙特卡洛采集 episode → 写出 TOML / CSV / HDF5。"""
    args = argument_parser()

    collection_meta = CollectionMetadata.from_args(args)
    sampler = collection_meta.build_sampler()

    system = System(
        config=load_config(args.config_file),
        device=args.device,
    )
    comps = system.components

    # 获取 RT 模拟器与目标
    rt_simulator = comps.rt_simulator
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}")

    # 初始化 CSV 缓冲与流式 HDF5 写入
    csv_rows: list[dict[str, str | int]] = []
    bs_pos = np.asarray(rt_simulator.transceivers["bs1"].position, dtype=np.float64)
    h5_path = collection_h5_path(scene_slug, DEFAULT_COLLECTION_OUT_DIR)
    with RTDatasetWriter.open(
        h5_path,
        bs_pos,
        compression=args.h5_compression,
    ) as dataset:
        # 采集 episode
        accepted = 0
        attempts = 0
        pbar = tqdm(total=collection_meta.num_samples, desc="数据采集", unit="ep")
        while accepted < collection_meta.num_samples:
            if len(sampler) == 0:
                raise RuntimeError(
                    f"采样池已耗尽：已采纳 {accepted}/{collection_meta.num_samples} 条。"
                    "请增大 --sampler_pool_factor 或调整过滤条件（scene_filter / 目标路径交互）。"
                )
            pos, vel, ori = sampler.pop()
            attempts += 1

            # 更新 RT 目标位姿
            target(
                position=pos,
                velocity=vel,
                orientation=ori,
            )

            # 更新路径
            rt_simulator.paths(update=True)
            if not rt_simulator.paths_intersect_target(target):
                continue

            geom = rt_simulator.rx_target_tx_geometric
            true_range = geom.range_tensor[0, 0, 0]
            true_velocity = geom.vel_tensor[0, 0, 0]
            csv_rows.append(
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
            snr_db = system.params.channel.snr_db
            y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)
            h_freq = comps.ls_channel_estimator(x_rg, y_rg)
            h_dd = comps.delay_doppler_spectrum(h_freq)

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
        )

    # 写出其余采集产物（TOML / CSV / 场景 PNG）
    save_collection_artifacts(
        scene_slug=scene_slug,
        config_file=args.config_file,
        csv_rows=csv_rows,
        rt_simulator=rt_simulator,
        out_dir=DEFAULT_COLLECTION_OUT_DIR,
    )


if __name__ == "__main__":
    main()
