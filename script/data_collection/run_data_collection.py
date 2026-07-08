"""ISAC 数据集采集入口：平面 ROI 蒙特卡洛采样 → RT 目标位姿驱动 → TOML / CSV / HDF5。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/data_collection/run_data_collection.py

流程概要
--------
1. 解析 CLI，设置随机种子，批量采样 ROI 内位置与速度。
2. 构建 ``System``，循环更新 RT 目标位姿并采集 h_dd / 几何真值。
3. 流式写出 HDF5，事后写出 TOML、CSV 与场景 PNG。

输出目录
--------
``data/``（``DEFAULT_COLLECTION_OUT_DIR``），按 ``scene_slug`` 命名：
``{scene_slug}_mc_sionna_dataset.h5``、``{scene_slug}_mc_dataset_episodes.csv``、
配置 TOML 副本与 ``{scene_slug}_scene.png``。

采纳条件
--------
从预采样池 ``pop`` 位姿后，仅当 ``paths_intersect_target`` 为真时计入 episode；
池耗尽则 ``RuntimeError``（提示增大 ``--sampler_pool_factor``）。

单 episode 感知链（无 MTI）
---------------------------
transmit → channel → ls_channel_estimator → delay_doppler_spectrum → append_episode

详见 ``docs/run_data_collection.md``。
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
from isac.utils.misc import csv_float2_scalar, csv_vec3

# Sionna/drjit 在大量 RT 路径更新时会重复触发 AST decorator 警告，不影响仿真结果，
# 在采集长循环前过滤以免刷屏。
warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数。

    蒙特卡洛相关参数经 ``CollectionMetadata.from_args`` 解析，
    并在 ``RTDatasetWriter.finalize`` 时序列化到 HDF5 根属性。
    """
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    sys_group = parser.add_argument_group("系统配置")
    sys_group.add_argument(
        "--config_file",
        type=str,
        default="config/data_collection/data_collection.toml",
        help="仿真与感知链 TOML 配置路径",
    )
    sys_group.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="PyTorch / Sionna 计算设备",
    )

    mc_group = parser.add_argument_group(
        "蒙特卡洛采样",
        "平面 ROI 内位置与速度采样；预采样池大小 = num_samples × sampler_pool_factor",
    )
    mc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（蒙特卡洛位置/速度采样）",
    )
    mc_group.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="目标采纳 episode 数",
    )
    mc_group.add_argument(
        "--sampler_pool_factor",
        type=int,
        default=5,
        help="预采样池倍数（池大小 = num_samples × 本参数）",
    )
    mc_group.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[-2.5, 2.5, -4.5, 4.5],
        help="平面 ROI 四元组（m），xy 平面；z 由 --roi_z 指定",
    )
    mc_group.add_argument(
        "--roi_z",
        type=float,
        default=0.0,
        help="ROI 内目标位置的固定 z 高度（m）",
    )
    mc_group.add_argument(
        "--position_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="位置采样分布",
    )
    mc_group.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=[0.1, 3.0],
        help="速度模值范围 (m/s)",
    )
    mc_group.add_argument(
        "--speed_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="速度模值采样分布",
    )

    h5_group = parser.add_argument_group(
        "HDF5 导出",
        "h_dd 流式写入压缩算法，仅影响 ``delay_doppler_spectrum`` dataset",
    )
    h5_group.add_argument(
        "--h5_compression",
        type=str,
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 h_dd 压缩算法（流式写入，默认 lzf）",
    )

    return parser.parse_args()


def main() -> None:
    """蒙特卡洛采集 episode → 流式写出 HDF5 → 写出 TOML / CSV / PNG。"""
    # --- 初始化：CLI → 采集元数据 → System ---
    args = argument_parser()
    collection_meta = CollectionMetadata.from_args(args)  # 设种子并校验 num_samples 等
    sampler = collection_meta.build_sampler()  # 预生成 pool_size 条候选位姿

    system = System(
        config=load_config(args.config_file),
        device=args.device,
    )
    comps = system.components

    # --- 场景绑定：RT 目标、scene_slug、参考基站位置 ---
    rt_simulator = comps.rt_simulator
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = getattr(rt_simulator.rt_simulator_params, "filename", "None")
    print(f"目标: {target_name}, 场景: {scene_slug}")

    csv_rows: list[dict[str, str | int]] = []
    bs_pos = np.asarray(rt_simulator.transceivers["bs1"].position, dtype=np.float64)
    h5_path = collection_h5_path(scene_slug, DEFAULT_COLLECTION_OUT_DIR)

    # --- 流式采集循环：pop → 位姿/路径 → 过滤 → 感知链 → append ---
    with RTDatasetWriter.open(
        h5_path,
        bs_pos,
        compression=args.h5_compression,
    ) as dataset:
        accepted = 0  # 已采纳 episode 数
        attempts = 0  # pop 次数（含被 paths_intersect_target 拒绝的候选）
        pbar = tqdm(total=collection_meta.num_samples, desc="数据采集", unit="ep")
        while accepted < collection_meta.num_samples:
            if len(sampler) == 0:
                raise RuntimeError(
                    f"采样池已耗尽：已采纳 {accepted}/{collection_meta.num_samples} 条。"
                    "请增大 --sampler_pool_factor 或调整过滤条件（scene_filter / 目标路径交互）。"
                )
            pos, vel, ori = sampler.pop()
            attempts += 1

            # 更新目标位姿并重算 RT 路径（结果缓存供后续 channel 复用）
            target(
                position=pos,
                velocity=vel,
                orientation=ori,
            )
            rt_simulator.paths(update=True)

            # 路径未与目标 mesh 交互则跳过（不计入 accepted，但计入 attempts）
            if not rt_simulator.paths_intersect_target(target):
                continue

            # CSV 几何真值（单基地距离与径向速度）
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

            # 感知链：OFDM 发射 → 加噪信道 → LS 估计 → DD 谱（含 ROI 裁剪，无 MTI）
            _, x_rg, x_time = system.transmit()
            snr_db = system.params.channel.snr_db
            y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)
            h_freq = comps.ls_channel_estimator(x_rg, y_rg)
            h_dd = comps.delay_doppler_spectrum(h_freq)

            dataset.append_episode(
                h_dd.detach().cpu().numpy().astype(np.complex64),
                pos,
                vel,
            )

            accepted += 1
            pbar.update(1)
        pbar.close()

        # accepted / attempts：采纳率（拒绝候选仍计入 attempts）
        acceptance_rate = accepted / attempts if attempts else 0.0
        print(f"接受率: {acceptance_rate:.1%} ({accepted}/{attempts})")

        # --- 落盘：写入 HDF5 根属性并关闭文件 ---
        dataset.finalize(
            collection_meta=collection_meta,
            scene_slug=scene_slug,
        )

    # TOML / CSV / 场景 PNG 须在 HDF5 finalize 关闭后写出
    save_collection_artifacts(
        scene_slug=scene_slug,
        config_file=args.config_file,
        csv_rows=csv_rows,
        rt_simulator=rt_simulator,
        out_dir=DEFAULT_COLLECTION_OUT_DIR,
    )


if __name__ == "__main__":
    main()
