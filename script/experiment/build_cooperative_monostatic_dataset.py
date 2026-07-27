#!/usr/bin/env python3
"""将 cooperative monostatic 实验目录转为 HDF5 数据集。

默认无压缩（``--compression none``）以加速构建；需要更小文件时可选用 ``lzf``。

示例::

    python script/experiment/build_cooperative_monostatic_dataset.py \\
        --input-dir data/experiment/cooperative_monostatic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DEFAULT_COOPERATIVE_VLEN,
    CooperativeMonostaticDataset,
    build_cooperative_monostatic_h5,
    summarize_cooperative_monostatic_h5,
)


def _default_input_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "experiment" / "cooperative_monostatic"


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cooperative monostatic HDF5 dataset from experiment directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help="directory with target_positions.csv and dev0/dev1/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output HDF5 path (default: {input-dir}/cooperative_monostatic_dataset.h5)",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=DEFAULT_COOPERATIVE_VLEN,
        help=f"complex profile length per CPI frame (default: {DEFAULT_COOPERATIVE_VLEN})",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=("lzf", "gzip", "none"),
        help="HDF5 compression for profile datasets (default: none, fastest build)",
    )
    parser.add_argument(
        "--target-z-m",
        type=float,
        default=0.0,
        help="target z coordinate in meters when CSV only has x/y (cm)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="load full dataset into memory after build (slow, for validation)",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    input_dir = args.input_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "cooperative_monostatic_dataset.h5"
    )
    compression = None if args.compression == "none" else args.compression

    h5_path = build_cooperative_monostatic_h5(
        input_dir,
        output_path,
        vlen=args.vlen,
        compression=compression,
        target_z_m=args.target_z_m,
        show_progress=not args.no_progress,
    )

    if args.verify_load:
        ds = CooperativeMonostaticDataset.load(h5_path)
        print(f"output: {h5_path}")
        print(f"  file size: {h5_path.stat().st_size / (1024 * 1024):.1f} MiB")
        print(f"  sessions: {int(ds.attrs.get('num_sessions', 0))}")
        print(f"  frames per session: {int(ds.attrs.get('frames_per_session', 0))}")
        print(f"  total frames: {len(ds)}")
        print(f"  profiles_dev0 shape: {ds.profiles_dev0.shape}")
        print(f"  profiles_dev1 shape: {ds.profiles_dev1.shape}")
        print(f"  target_position shape: {ds.target_position.shape}")
        return

    summary = summarize_cooperative_monostatic_h5(h5_path)
    file_size_mb = summary["file_size_bytes"] / (1024 * 1024)
    print(f"output: {summary['path']}")
    print(f"  file size: {file_size_mb:.1f} MiB")
    print(f"  sessions: {summary['num_sessions']}")
    print(f"  frames per session: {summary['frames_per_session']}")
    print(f"  total frames: {summary['total_frames']}")
    print(f"  profiles_dev0 shape: {summary['profiles_dev0_shape']}")
    print(f"  profiles_dev1 shape: {summary['profiles_dev1_shape']}")
    print(f"  target_position shape: {summary['target_position_shape']}")


if __name__ == "__main__":
    main()
