"""从 HDF5 数据集回放单基地感知前序流程（不含谱计算 / MUSIC）。

须在 **ISAC conda 环境**中运行（Sionna RT 与 OFDM 信道依赖完整 CUDA/Sionna 栈）::

    /opt/miniconda3/envs/ISAC/bin/python script/data_loading/run_sensing_from_dataset.py
    # 或
    conda run -n ISAC python script/data_loading/run_sensing_from_dataset.py

流程：加载 ``Dataset`` → 构建 ``System`` → 逐 episode 经 ``dataset[i]`` 读取
``(cfr, label)`` → 注入 ``RTChannel.cfr`` → ``transmit`` → ``channel`` →
``compute_sensing_spectrum``（不重新跑 RT path_solver）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.datasets import Dataset
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.data_collection.channel_export import (
    cfr_numpy_to_h_freq,
    scene_slug_from_rt_simulator,
)

DEFAULT_DATASET_H5 = (
    PROJECT_ROOT / "out" / "dataset_collection" / "empty_room_mc_sionna_dataset.h5"
)


def _default_config_for_h5(h5_path: Path) -> Path:
    """与 HDF5 同目录的 ``data_collection.toml`` 副本。"""
    sibling = h5_path.parent / "data_collection.toml"
    if sibling.is_file():
        return sibling
    return PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ISAC — 从 HDF5 数据集回放感知前序（transmit / channel / geometry）"
    )

    parser.add_argument(
        "--dataset_h5",
        type=Path,
        default=DEFAULT_DATASET_H5,
        help="HDF5 数据集路径",
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=None,
        help="TOML 配置；默认取与 --dataset_h5 同目录的 data_collection.toml",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
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
        help="随机种子",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="信道施加域；存储 CFR 回放仅支持 frequency",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最多回放 episode 数；默认全部",
    )

    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    h5_path = args.dataset_h5.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {h5_path}")

    config_path = args.config_file or _default_config_for_h5(h5_path)
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    set_random_seed(args.seed)
    dataset = Dataset.load(h5_path)
    print(f"已加载数据集: {dataset} (CFR shape={dataset.cfr.shape})")

    config = load_config(config_path)
    system = System(
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )

    rt_simulator = system.components.rt_simulator
    if rt_simulator is None:
        raise ValueError("此脚本要求 channel.type='rt' 且已配置 [rt_simulator]")

    target_name, _ = next(iter(rt_simulator.rt_targets.items()))
    scene_slug = scene_slug_from_rt_simulator(rt_simulator)
    print(f"目标: {target_name}, 场景: {scene_slug}, 配置: {config_path}")

    script_out_dir = PROJECT_ROOT / "out" / "data_loading"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    rt_simulator.render_to_file(
        filename=f"{scene_slug}_scene.png",
        output_dir=script_out_dir,
    )

    system.display_sensing_performance()

    n_episodes = len(dataset)
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)

    domain = args.domain
    if domain != "frequency":
        raise ValueError(
            "存储 CFR 回放仅支持 --domain frequency；"
            "时域需 live RT 信道，与本脚本用途不符。"
        )
    snr_db = system.params.channel.snr_db
    channel = system.components.channel

    for i in tqdm(range(n_episodes), desc="感知前序回放", unit="ep"):
        cfr, _label = dataset[i]
        _, x_rg, _ = system.transmit()
        channel.cfr = cfr_numpy_to_h_freq(cfr, device=x_rg.device)
        y_rg = channel(x_rg, domain=domain, snr_db=snr_db)
        system.compute_sensing_spectrum(x_rg, y_rg)

    print(f"回放完成: {n_episodes}/{len(dataset)} episodes")


if __name__ == "__main__":
    main()
