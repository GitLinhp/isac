"""平面 ROI 内位置与速度采样：给定条数、四元组 ROI 与分布，打印 3D 位姿。

用法示例::

    python script/model_training/run_sample_roi_positions.py \\
        --num_samples 5 \\
        --roi 10 60 -20 20 \\
        --sampling_mode uniform \\
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
from typing import Literal

import numpy as np
from tabulate import tabulate

from isac import PROJECT_ROOT
from isac.utils import (
    cartesian_direction_to_yaw_pitch_roll,
    csv_float2_scalar,
    set_random_seed,
)

SamplingMode = Literal["uniform", "gaussian"]

CSV_FIELDNAMES = ["idx", "position", "velocity", "orientation"]


def parse_roi_xy(
    roi4: list[float] | tuple[float, ...],
) -> tuple[float, float, float, float]:
    """解析平面 ROI 四元组，返回 ``(x_lo, x_hi, y_lo, y_hi)``。"""
    if len(roi4) != 4:
        raise ValueError("平面 ROI 须为四元组：XMIN XMAX YMIN YMAX")
    x_lo, x_hi, y_lo, y_hi = (float(v) for v in roi4)
    for name, lo, hi in (("x", x_lo, x_hi), ("y", y_lo, y_hi)):
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"ROI 维度 `{name}` 非法：须为有限值")
        if lo > hi:
            raise ValueError(f"ROI 维度 `{name}` 非法：须满足 min <= max")
    return x_lo, x_hi, y_lo, y_hi


def parse_speed_range(
    pair: list[float] | tuple[float, ...],
) -> tuple[float, float]:
    """解析速度模值范围，返回 ``(smin, smax)``。"""
    if len(pair) != 2:
        raise ValueError("speed_range 须为二元组：MIN MAX")
    smin, smax = float(pair[0]), float(pair[1])
    if not np.isfinite(smin) or not np.isfinite(smax):
        raise ValueError("speed_range 须为有限值")
    if smin < 0 or smax <= smin:
        raise ValueError("speed_range 须满足 0 <= min < max")
    return smin, smax


# 位置采样
def sample_positions(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    num_samples: int,
    sampling_mode: SamplingMode,
) -> np.ndarray:
    """在平面 ROI 内采样位置，返回形状 ``(num_samples, 3)``。"""
    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0")

    n = int(num_samples)
    if sampling_mode == "uniform":
        x = np.random.uniform(x_lo, x_hi, size=n)
        y = np.random.uniform(y_lo, y_hi, size=n)
        z = np.zeros(n, dtype=np.float64)
        return np.column_stack((x, y, z)).astype(np.float64)
    elif sampling_mode == "gaussian":
        center = np.array(
            [(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, 0.0], dtype=np.float64
        )
        std = np.array(
            [(x_hi - x_lo) / 6.0, (y_hi - y_lo) / 6.0, 0.0],
            dtype=np.float64,
        )
        pts = np.random.normal(loc=center, scale=std, size=(n, 3)).astype(np.float64)
        pts = np.clip(pts, [x_lo, y_lo, 0.0], [x_hi, y_hi, 0.0])
        return pts
    else:
        raise ValueError("sampling_mode 仅支持 'uniform' 或 'gaussian'")


# 速度采样
def sample_speeds(
    smin: float,
    smax: float,
    sampling_mode: SamplingMode,
    num_samples: int,
) -> np.ndarray:
    """在 ``[smin, smax]`` 内采样速度模值，返回形状 ``(num_samples,)``。"""
    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0")

    n = int(num_samples)
    if sampling_mode == "uniform":
        return np.random.uniform(smin, smax, size=n).astype(np.float64)
    elif sampling_mode == "gaussian":
        center = (smin + smax) / 2.0
        std = (smax - smin) / 6.0
        speeds = np.random.normal(loc=center, scale=std, size=n).astype(np.float64)
        return np.clip(speeds, smin, smax)
    else:
        raise ValueError("speed_sampling_mode 仅支持 'uniform' 或 'gaussian'")


def sample_planar_directions(
    num_samples: int,
) -> np.ndarray:
    """xy 平面均匀随机单位方向，返回形状 ``(num_samples, 3)``，``vz=0``。"""
    n = int(num_samples)
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    dirs = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(n))).astype(
        np.float64
    )
    return dirs


def sample_velocities(
    smin: float,
    smax: float,
    num_samples: int,
    sampling_mode: SamplingMode,
) -> np.ndarray:
    """在 ``[smin, smax]`` 内采样速度模值，返回形状 ``(num_samples, 3)``。"""
    speeds = sample_speeds(smin, smax, sampling_mode, num_samples)
    dirs = sample_planar_directions(num_samples)
    orientations = cartesian_direction_to_yaw_pitch_roll(dirs)
    velocities = (speeds[:, None] * dirs).astype(np.float64)
    return velocities, speeds, dirs, orientations


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


# 参数解析
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


# 主函数
def main() -> None:
    args = argument_parser()
    set_random_seed(args.seed)
    x_lo, x_hi, y_lo, y_hi = parse_roi_xy(args.roi)
    smin, smax = parse_speed_range(args.speed_range)
    n = int(args.num_samples)
    positions = sample_positions(
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        n,
        args.position_sampling_mode,
    )
    velocities, speeds, dirs, orientations = sample_velocities(
        smin, smax, n, args.speed_sampling_mode
    )

    print(
        f"n={n}, roi=({x_lo}, {x_hi}) x ({y_lo}, {y_hi}), z=0, "
        f"pos_mode={args.position_sampling_mode}, speed_range=[{smin}, {smax}], "
        f"speed_mode={args.speed_sampling_mode}, seed={args.seed}"
    )
    csv_rows = [
        _build_sample_row(i, pos, vel, ori)
        for i, (pos, vel, _, ori) in enumerate(
            zip(positions, velocities, speeds, orientations)
        )
    ]
    table_rows = [
        [row["idx"], row["position"], row["velocity"], row["orientation"]]
        for row in csv_rows
    ]
    print(tabulate(table_rows, headers=CSV_FIELDNAMES, tablefmt="simple_grid"))

    csv_path = args.output
    save_samples_csv(csv_path, csv_rows)


if __name__ == "__main__":
    main()
