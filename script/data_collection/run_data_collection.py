"""ISAC 数据集采集入口：平面 ROI 蒙特卡洛采样 → RT 目标位姿驱动 → TOML / CSV / HDF5。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/data_collection/run_data_collection.py

双基地示例::

    python script/data_collection/run_data_collection.py \\
        --config_file config/data_collection/data_collection_bistatic.toml \\
        --sens_mode bistatic

流程概要
--------
1. 解析 CLI，设置随机种子，批量采样 ROI 内位置与速度。
2. 构建 ``System``，循环更新 RT 目标位姿并过滤镜面反射路径与速度分辨率门槛。
3. 流式写出 HDF5，事后写出 TOML、CSV 与场景 PNG。

输出目录
--------
``data/{scene_slug}_{sens_mode}_{scs}/``（``scs`` 为子载波间隔 slug，如 ``30kHz``），其下：
``{scene_slug}_{sens_mode}_mc_sionna_dataset.h5``、
``{scene_slug}_{sens_mode}_mc_dataset_episodes.csv``、
配置 TOML 副本与 ``{scene_slug}_{sens_mode}_scene.png``。

相对旧版 ``data/{scene_slug}_{scs}/``，单基地现含 ``_monostatic`` 后缀（破坏性变更）。

``--sens_mode``
---------------
- ``monostatic``（默认）：距离真值为斜距；典型 TOML 为 ``data_collection.toml``
- ``bistatic``：距离真值为折叠路径长 ``||T-X||+||R-T||``；须分离收发 TOML（如 ``data_collection_bistatic.toml``，bs1=RX）

采纳条件
--------
从预采样池 ``pop`` 位姿后，须同时满足：

- 存在与目标的 **镜面反射** 路径（``paths_intersect_target_with_interaction(..., "specular")``）
- ``|true_velocity| > velocity_resolution_{sens_mode}``（单基地为径向速度，双基地为路径变化率）

池耗尽则 ``RuntimeError``（提示增大 ``--sampler_pool_factor``）。

单 episode 感知链（默认 MTI）
-----------------------------
transmit → channel → ls_channel_estimator →（默认 MTI）→ delay_doppler_spectrum(sens_mode) → append_episode

HDF5 ``bs_pos`` 取 ``transceivers["bs1"].position``；双基地 TOML 下 bs1 为 RX。

详见 ``docs/run_data_collection.md``。
"""

from __future__ import annotations

import argparse
import warnings
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from isac import DEFAULT_COLLECTION_OUT_DIR
from isac.collection import (
    CollectionMetadata,
    RTDatasetWriter,
    collection_dataset_dir,
    collection_h5_path,
    save_collection_artifacts,
)
from isac.system import System
from isac.utils import load_config
from isac.utils.misc import csv_float2_scalar, csv_vec3

if TYPE_CHECKING:
    from isac.channel.rt.rt_simulator import RTSimulator

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
        default=50,
        help="预采样池倍数（池大小 = num_samples × 本参数）",
    )
    mc_group.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[-4.5, 4.5, -2.5, 2.5],
        help="平面 ROI 四元组（m），xy 平面；z 由 --roi_z 指定",
    )
    mc_group.add_argument(
        "--roi_z",
        type=float,
        default=0.5,
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
        default=[0.5, 3.0],
        help="速度模值范围 (m/s)",
    )
    mc_group.add_argument(
        "--speed_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="速度模值采样分布",
    )

    proc_group = parser.add_argument_group("感知链")
    proc_group.add_argument(
        "--sens_mode",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="感知模式；影响 DD 谱 ROI 裁切、速度分辨率筛选与输出目录",
    )
    proc_group.add_argument(
        "--apply_mti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 LS 估计与 DD 谱之间施加 MTI（须 TOML 含 [mti] 段）；默认开，可用 --no-apply_mti 关闭",
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


def _preflight_checks(system: System, *, apply_mti: bool) -> None:
    """校验 RT 链路与速度分辨率筛选、MTI 依赖已就绪。"""
    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")
    if system.components.sensing_performance is None:
        raise ValueError(
            "速度分辨率筛选需要 sensing_performance（[ofdm] + carrier_frequency）"
        )
    if apply_mti and system.components.moving_target_indication is None:
        raise ValueError("--apply_mti 需要 TOML [mti] 段以构建 MovingTargetIndication")


def _dataset_slug(scene_slug: str, sens_mode: str) -> str:
    """构造数据集文件名前缀 ``{scene_slug}_{sens_mode}``。"""
    return f"{scene_slug}_{sens_mode}"


def _assert_bistatic_topology(rt_simulator: RTSimulator) -> None:
    """校验 TOML 为分离收发拓扑，且首条链路为双基地。"""
    geom = rt_simulator.rx_target_tx_geometric
    if len(geom.tx_names) < 1 or len(geom.rx_names) < 1:
        raise ValueError(
            f"双基地采集需要至少 1 个 TX 与 1 个 RX，"
            f"收到 tx={geom.tx_names!r}, rx={geom.rx_names!r}"
        )
    if not bool(geom.type_tensor[0, 0, 0].item()):
        raise ValueError(
            "首条链路 type_tensor[0,0,0] 为单基地；"
            "请使用分离收发 TOML（如 data_collection_bistatic.toml）"
        )


def main() -> None:
    """蒙特卡洛采集 episode → 流式写出 HDF5 → 写出 TOML / CSV / PNG。"""
    # --- 初始化：CLI → 采集元数据 → System ---
    args = argument_parser()
    collection_meta = CollectionMetadata.from_args(args)  # 设种子并校验 num_samples 等
    sampler = collection_meta.build_sampler()  # 预生成 pool_size 条候选位姿

    system = System(args.config_file, device=args.device)
    comps = system.components
    _preflight_checks(system, apply_mti=args.apply_mti)

    # --- 场景绑定：RT 目标、scene_slug、参考基站位置 ---
    rt_simulator = comps.rt_simulator
    assert rt_simulator is not None
    target_name, target = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = getattr(rt_simulator.rt_simulator_params, "filename", "None")
    sens_mode = args.sens_mode
    if sens_mode == "bistatic":
        _assert_bistatic_topology(rt_simulator)
    dataset_slug = _dataset_slug(scene_slug, sens_mode)
    scs_hz = float(comps.rg.subcarrier_spacing)
    out_dir = collection_dataset_dir(dataset_slug, scs_hz, DEFAULT_COLLECTION_OUT_DIR)
    mti_label = "开" if args.apply_mti else "关"
    print(
        f"目标: {target_name}, 场景: {scene_slug}, 配置: {args.config_file}\n"
        f"采集 | sens_mode={sens_mode} | MTI={mti_label} | "
        f"目标采纳数={collection_meta.num_samples}\n"
        f"采集输出: {out_dir}/"
    )

    csv_rows: list[dict[str, str | int]] = []
    bs_pos = np.asarray(rt_simulator.transceivers["bs1"].position, dtype=np.float64)
    h5_path = collection_h5_path(dataset_slug, out_dir)
    v_res = float(
        getattr(comps.sensing_performance, f"velocity_resolution_{sens_mode}")
    )

    # --- 流式采集循环：pop → 位姿/路径 → 过滤 → 感知链 → append ---
    with RTDatasetWriter.open(
        h5_path,
        bs_pos,
        compression=args.h5_compression,
    ) as dataset:
        accepted = 0  # 已采纳 episode 数
        attempts = 0  # pop 次数（含被过滤拒绝的候选）
        pbar = tqdm(total=collection_meta.num_samples, desc="数据采集", unit="ep")
        while accepted < collection_meta.num_samples:
            if len(sampler) == 0:
                raise RuntimeError(
                    f"采样池已耗尽：已采纳 {accepted}/{collection_meta.num_samples} 条。"
                    "请增大 --sampler_pool_factor"
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

            geom = rt_simulator.rx_target_tx_geometric
            true_range = geom.range_tensor[0, 0, 0]
            true_velocity = geom.vel_tensor[0, 0, 0]
            if abs(float(true_velocity.item())) <= v_res / 2:
                continue

            if not rt_simulator.paths_intersect_target_with_interaction(
                target, "specular"
            ):
                continue

            csv_rows.append(
                {
                    "sample_idx": accepted,
                    "position": csv_vec3(pos),
                    "velocity": csv_vec3(vel),
                    "true_range_m": csv_float2_scalar(true_range),
                    "true_radial_velocity_mps": csv_float2_scalar(true_velocity),
                }
            )

            # 感知链：OFDM 发射 → 加噪信道 → LS 估计 →（可选 MTI）→ DD 谱（含 ROI 裁剪）
            _, x_rg, x_time = system.transmit()
            snr_db = system.params.channel.snr_db
            y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)
            h_freq = comps.ls_channel_estimator(x_rg, y_rg)
            if args.apply_mti:
                h_freq = comps.moving_target_indication(h_freq)
            h_dd = comps.delay_doppler_spectrum(h_freq, sens_mode=sens_mode)

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
            scene_slug=dataset_slug,
            apply_mti=args.apply_mti,
        )

    # TOML / CSV / 场景 PNG 须在 HDF5 finalize 关闭后写出
    save_collection_artifacts(
        scene_slug=dataset_slug,
        config_file=args.config_file,
        csv_rows=csv_rows,
        rt_simulator=rt_simulator,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
