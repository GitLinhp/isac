#!/usr/bin/env python3
"""方案 C 采集路书：Run2 = MasterGrid 217 复采 + 外环 10 cm 细网格额外 20 格。

示例::

    python script/experiment/generate_plan_c_collection_manifest.py \\
        --output data/experiment/cooperative_monostatic_plan_c/collection_run2_targets.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isac import PROJECT_ROOT
from isac_imp.record_target_metadata import (
    INNER_RADIUS_CM,
    is_inner_target_xy_m,
    target_region_index_xy_m,
    target_region_name,
)

DEFAULT_GRID_MIN_M = -1.0
DEFAULT_GRID_MAX_M = 1.0
OUTER_GRID_STEP_M = 0.2
INNER_GRID_STEP_M = 0.1
FINE_GRID_STEP_M = 0.1
INNER_RADIUS_M = INNER_RADIUS_CM / 100.0
GRID_COORD_DECIMALS = 1

DEFAULT_EXTRA_COUNT = 20
DEFAULT_SEED = 42

MANIFEST_COLUMNS = (
    "session_id",
    "target_x_cm",
    "target_y_cm",
    "region",
    "role",
)

ROLE_RESAMPLE_RUN1 = "resample_run1"
ROLE_EXTRA_30PCT = "extra_30pct"


@dataclass(frozen=True)
class ManifestRow:
    session_id: int
    target_x_cm: int
    target_y_cm: int
    region: str
    role: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "session_id": self.session_id,
            "target_x_cm": self.target_x_cm,
            "target_y_cm": self.target_y_cm,
            "region": self.region,
            "role": self.role,
        }


def _rounded_linspace_axis(min_m: float, max_m: float, step_m: float) -> np.ndarray:
    count = int(round((max_m - min_m) / step_m)) + 1
    axis = np.linspace(min_m, max_m, count, dtype=np.float64)
    return np.round(axis, GRID_COORD_DECIMALS)


def build_master_grid_xy_m(
    *,
    min_m: float = DEFAULT_GRID_MIN_M,
    max_m: float = DEFAULT_GRID_MAX_M,
    outer_step_m: float = OUTER_GRID_STEP_M,
    inner_step_m: float = INNER_GRID_STEP_M,
    inner_radius_m: float = INNER_RADIUS_M,
) -> set[tuple[float, float]]:
    """MasterGrid：外环 20 cm + 内环 10 cm，共 217 格。"""
    outer_axis = _rounded_linspace_axis(min_m, max_m, outer_step_m)
    inner_axis = _rounded_linspace_axis(-inner_radius_m, inner_radius_m, inner_step_m)

    master: set[tuple[float, float]] = set()
    for x in outer_axis:
        for y in outer_axis:
            x_m, y_m = float(x), float(y)
            if is_inner_target_xy_m(x_m, y_m, radius_m=inner_radius_m):
                continue
            master.add((x_m, y_m))
    for x in inner_axis:
        for y in inner_axis:
            master.add((float(x), float(y)))
    return master


def _hamilton_allocate(total: int, weights: dict[int, int]) -> dict[int, int]:
    """Hamilton 最大余数法，将 ``total`` 按 ``weights`` 比例分配到各区。"""
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    alloc = {rid: int(total * weights[rid] / weight_sum) for rid in weights}
    while sum(alloc.values()) < total:
        remainders = {
            rid: total * weights[rid] / weight_sum - alloc[rid] for rid in weights
        }
        rid = max(remainders, key=remainders.get)
        alloc[rid] += 1
    return alloc


def build_extra_grid_xy_m(
    master_grid: set[tuple[float, float]],
    *,
    extra_count: int = DEFAULT_EXTRA_COUNT,
    seed: int = DEFAULT_SEED,
    min_m: float = DEFAULT_GRID_MIN_M,
    max_m: float = DEFAULT_GRID_MAX_M,
    fine_step_m: float = FINE_GRID_STEP_M,
    inner_radius_m: float = INNER_RADIUS_M,
) -> set[tuple[float, float]]:
    """外环 8 区从 10 cm 细网格（不在 MasterGrid）中抽取额外格点。"""
    fine_axis = _rounded_linspace_axis(min_m, max_m, fine_step_m)
    by_region: dict[int, list[tuple[float, float]]] = {i: [] for i in range(9)}

    for x in fine_axis:
        for y in fine_axis:
            x_m, y_m = float(x), float(y)
            if (x_m, y_m) in master_grid:
                continue
            region_id = target_region_index_xy_m(x_m, y_m, radius_m=inner_radius_m)
            if region_id == 4:
                continue
            by_region[region_id].append((x_m, y_m))

    pool_sizes = {rid: len(points) for rid, points in by_region.items() if rid != 4}
    alloc = _hamilton_allocate(extra_count, pool_sizes)

    extra: set[tuple[float, float]] = set()
    for region_id, pick_count in alloc.items():
        pool = list(by_region[region_id])
        rng = np.random.default_rng(seed + region_id)
        rng.shuffle(pool)
        extra.update(pool[:pick_count])
    return extra


def _xy_m_to_cm(x_m: float, y_m: float) -> tuple[int, int]:
    return int(round(x_m * 100.0)), int(round(y_m * 100.0))


def build_run2_manifest_rows(
    *,
    extra_count: int = DEFAULT_EXTRA_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[ManifestRow]:
    """生成 Run2 路书：217 复采 + ``extra_count`` 额外格。"""
    master = build_master_grid_xy_m()
    extra = build_extra_grid_xy_m(master, extra_count=extra_count, seed=seed)

    rows: list[ManifestRow] = []
    session_id = 1

    for x_m, y_m in sorted(master, key=lambda p: (p[1], p[0])):
        x_cm, y_cm = _xy_m_to_cm(x_m, y_m)
        region_id = target_region_index_xy_m(x_m, y_m)
        rows.append(
            ManifestRow(
                session_id=session_id,
                target_x_cm=x_cm,
                target_y_cm=y_cm,
                region=target_region_name(region_id),
                role=ROLE_RESAMPLE_RUN1,
            )
        )
        session_id += 1

    for x_m, y_m in sorted(extra, key=lambda p: (p[1], p[0])):
        x_cm, y_cm = _xy_m_to_cm(x_m, y_m)
        region_id = target_region_index_xy_m(x_m, y_m)
        rows.append(
            ManifestRow(
                session_id=session_id,
                target_x_cm=x_cm,
                target_y_cm=y_cm,
                region=target_region_name(region_id),
                role=ROLE_EXTRA_30PCT,
            )
        )
        session_id += 1

    return rows


def write_run2_manifest_csv(output_path: Path, rows: list[ManifestRow]) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return output_path


def _default_output_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "experiment"
        / "cooperative_monostatic_plan_c"
        / "collection_run2_targets.csv"
    )


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Plan C Run2 collection manifest CSV (217 resample + 20 extra)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="output CSV path",
    )
    parser.add_argument(
        "--extra-count",
        type=int,
        default=DEFAULT_EXTRA_COUNT,
        help=f"number of extra fine-grid points (default: {DEFAULT_EXTRA_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for extra point selection (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    rows = build_run2_manifest_rows(extra_count=args.extra_count, seed=args.seed)
    output_path = write_run2_manifest_csv(args.output, rows)

    resample_count = sum(1 for row in rows if row.role == ROLE_RESAMPLE_RUN1)
    extra_count = sum(1 for row in rows if row.role == ROLE_EXTRA_30PCT)
    print(
        f"Wrote {len(rows)} rows → {output_path} "
        f"(resample={resample_count}, extra={extra_count})"
    )


if __name__ == "__main__":
    main()
