"""平面 ROI 内位置与速度采样：给定条数、四元组 ROI 与分布，打印 3D 位姿。

用法示例::

    python script/model_training/run_sample_roi_positions.py \\
        --num_samples 5 \\
        --roi 10 60 -20 20 \\
        --position_sampling_mode uniform \\
        --speed_range 1 5 \\
        --speed_sampling_mode gaussian \\
        --seed 0

平面 ROI 为 ``XMIN XMAX YMIN YMAX``，位置 z 固定为 0；速度方向为 xy 平面均匀随机，
速度模值在 ``--speed_range`` 内按 ``--speed_sampling_mode`` 采样。不做 RT 场景或障碍物过滤。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tabulate import tabulate

from isac import PROJECT_ROOT
from isac.utils.data_collection.roi_sampling import sample_roi_kinematics
from isac.utils import csv_float2_scalar, set_random_seed

CSV_FIELDNAMES = ["idx", "position", "velocity", "orientation"]


def _csv_vec3(vec: np.ndarray) -> str:
    """将三维向量格式化为 CSV 单元格字符串，如 ``[1.00, 2.00, 3.00]``。"""
    row = np.asarray(vec, dtype=np.float64).reshape(-1)
    parts = ", ".join(csv_float2_scalar(row[i]) for i in range(3))
    return f"[{parts}]"


def _build_sample_row(
    idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    orientation: np.ndarray,
) -> dict[str, str | int]:
    """构造单条采样记录的 CSV 行。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
    ori_row = np.asarray(orientation, dtype=np.float64).reshape(-1)
    return {
        "idx": idx,
        "position": _csv_vec3(pos_row),
        "velocity": _csv_vec3(vel_row),
        "orientation": _csv_vec3(ori_row),
    }


def save_samples_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    """写入采样结果 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 已写入: {path}")


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="平面 ROI 内采样位置与速度并打印（位置 z=0，速度方向在 xy 平面）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="采样条数",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=[0.0, 80.0, -40.0, 40.0],
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
        default=[0.1, 10.0],
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
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_roi_positions.csv",
        help="CSV 输出路径（默认 data/sample_roi_positions.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    set_random_seed(args.seed)

    positions, velocities, orientations = sample_roi_kinematics(
        roi=args.roi,
        position_sampling_mode=args.position_sampling_mode,
        speed_range=args.speed_range,
        speed_sampling_mode=args.speed_sampling_mode,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    csv_rows = [
        _build_sample_row(i, pos, vel, ori)
        for i, (pos, vel, ori) in enumerate(zip(positions, velocities, orientations))
    ]

    table_rows = [
        [row["idx"], row["position"], row["velocity"], row["orientation"]]
        for row in csv_rows
    ]
    print(tabulate(table_rows, headers=CSV_FIELDNAMES, tablefmt="simple_grid"))

    save_samples_csv(args.output, csv_rows)


if __name__ == "__main__":
    main()
