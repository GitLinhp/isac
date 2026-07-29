#!/usr/bin/env python3
"""分站距离域标定：用各站 MUSIC 距离相对真值路径和的 MAE 分别搜 d0、d1。

约定：
  - 站内 TX=+d、RX=-d
  - dev0: TX=(+d0,-2), RX=(-d0,-2)
  - dev1: TX=(-2,+d1), RX=(-2,-d1)
  - 指标：mean |r_i - path_sum(TX,RX,T)/2|

示例::

    python script/experiment/run_cooperative_monostatic_tx_offset_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.sensing.localization import path_sum_xy, position_rmse_xy
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_DEV0_XY,
    DEFAULT_DEV1_XY,
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
    localize_xy_from_two_ranges,
)
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_PROFILES_DEV0,
    DATASET_KEY_PROFILES_DEV1,
    DATASET_KEY_TARGET_POSITION,
)
from isac_imp.record_target_metadata import is_inner_target_xy_m

DEFAULT_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out/cooperative_monostatic/tx_offset_sweep"


def _load_music_mod():
    path = (
        PROJECT_ROOT
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_music_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("music_rmse_sweep", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def antenna_xy_for_dev(dev: int, offset_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """返回 (tx_xy, rx_xy)。"""
    d = float(offset_m)
    if int(dev) == 0:
        return (d, -2.0), (-d, -2.0)
    if int(dev) == 1:
        return (-2.0, d), (-2.0, -d)
    raise ValueError(f"dev must be 0 or 1, got {dev}")


def antenna_xy_pair(
    offset_dev0_m: float,
    offset_dev1_m: float,
) -> dict[str, tuple[float, float]]:
    tx0, rx0 = antenna_xy_for_dev(0, offset_dev0_m)
    tx1, rx1 = antenna_xy_for_dev(1, offset_dev1_m)
    return {"tx0_xy": tx0, "rx0_xy": rx0, "tx1_xy": tx1, "rx1_xy": rx1}


def argument_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-device 1D antenna-offset sweep by range MAE vs true path sum"
    )
    p.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--offset-min-cm", type=float, default=0.0)
    p.add_argument("--offset-max-cm", type=float, default=10.0)
    p.add_argument("--offset-step-cm", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _offset_grid_cm(min_cm: float, max_cm: float, step_cm: float) -> np.ndarray:
    if step_cm <= 0:
        raise ValueError("--offset-step-cm must be > 0")
    if max_cm < min_cm:
        raise ValueError("--offset-max-cm must be >= --offset-min-cm")
    n = int(round((max_cm - min_cm) / step_cm)) + 1
    return np.linspace(min_cm, max_cm, n)


def cache_music_ranges(
    h5_path: Path,
    *,
    max_samples: int | None,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    music_mod = _load_music_mod()
    proc_params = grc_cooperative_processing_params()
    range_roi = DEFAULT_RANGE_ROI

    with h5py.File(h5_path, "r") as f:
        n_total = int(f[DATASET_KEY_PROFILES_DEV0].shape[0])
        n = n_total if max_samples is None else min(int(max_samples), n_total)
        true_xy = np.asarray(f[DATASET_KEY_TARGET_POSITION][:n, :2], dtype=np.float64)
        r0 = np.full(n, np.nan, dtype=np.float64)
        r1 = np.full(n, np.nan, dtype=np.float64)
        for i in tqdm(
            range(n),
            desc="MUSIC ranges cache",
            unit="frame",
            disable=not show_progress,
        ):
            r0[i] = music_mod._music_range_from_divide_cpi(
                f[DATASET_KEY_PROFILES_DEV0][i],
                proc_params=proc_params,
                range_roi=range_roi,
                cfar_detector=None,
            )
            r1[i] = music_mod._music_range_from_divide_cpi(
                f[DATASET_KEY_PROFILES_DEV1][i],
                proc_params=proc_params,
                range_roi=range_roi,
                cfar_detector=None,
            )
    return true_xy[:, 0], true_xy[:, 1], r0, r1


def eval_range_mae(
    *,
    true_x: np.ndarray,
    true_y: np.ndarray,
    r_est: np.ndarray,
    dev: int,
    offset_m: float,
) -> dict[str, float | int]:
    """mean |r_est - path_sum(TX,RX,T)/2| over finite positive ranges."""
    tx, rx = antenna_xy_for_dev(dev, offset_m)
    errs: list[float] = []
    for i in range(true_x.size):
        ri = float(r_est[i])
        if not (np.isfinite(ri) and ri > 0.0):
            continue
        s = path_sum_xy((float(true_x[i]), float(true_y[i])), tx, rx)
        errs.append(abs(ri - 0.5 * s))
    mae = float(np.mean(errs)) if errs else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return {
        "offset_cm": float(offset_m * 100.0),
        "offset_m": float(offset_m),
        "n_valid": int(len(errs)),
        "range_mae_m": mae,
        "range_rmse_m": rmse,
    }


def eval_localize_global(
    *,
    true_x: np.ndarray,
    true_y: np.ndarray,
    r0: np.ndarray,
    r1: np.ndarray,
    offset_dev0_m: float,
    offset_dev1_m: float,
) -> dict[str, float | int]:
    ant = antenna_xy_pair(offset_dev0_m, offset_dev1_m)
    mid0, mid1 = DEFAULT_DEV0_XY, DEFAULT_DEV1_XY
    rmse = np.full(true_x.size, np.nan, dtype=np.float64)
    for i in range(true_x.size):
        if not (
            np.isfinite(r0[i])
            and np.isfinite(r1[i])
            and r0[i] > 0
            and r1[i] > 0
        ):
            continue
        try:
            est = localize_xy_from_two_ranges(
                mid0,
                float(r0[i]),
                mid1,
                float(r1[i]),
                tx0_xy=ant["tx0_xy"],
                rx0_xy=ant["rx0_xy"],
                tx1_xy=ant["tx1_xy"],
                rx1_xy=ant["rx1_xy"],
            )
        except ValueError:
            continue
        rmse[i] = position_rmse_xy(est, (float(true_x[i]), float(true_y[i])))
    valid = np.isfinite(rmse)
    inner = np.array(
        [is_inner_target_xy_m(float(x), float(y)) for x, y in zip(true_x, true_y)],
        dtype=bool,
    )
    return {
        "n_valid": int(valid.sum()),
        "global_rmse_m": float(np.mean(rmse[valid])) if valid.any() else float("nan"),
        "inner_rmse_m": (
            float(np.mean(rmse[valid & inner])) if (valid & inner).any() else float("nan")
        ),
        "outer_rmse_m": (
            float(np.mean(rmse[valid & ~inner]))
            if (valid & ~inner).any()
            else float("nan")
        ),
    }


def _write_dev_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fields = ["offset_cm", "offset_m", "n_valid", "range_mae_m", "range_rmse_m"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = argument_parser()
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    offsets_cm = _offset_grid_cm(
        float(args.offset_min_cm),
        float(args.offset_max_cm),
        float(args.offset_step_cm),
    )
    print(
        f"Per-device range-MAE offset sweep: "
        f"{offsets_cm[0]:.1f}–{offsets_cm[-1]:.1f} cm step {args.offset_step_cm} cm "
        f"({len(offsets_cm)} points × 2 devices)",
        flush=True,
    )

    true_x, true_y, r0, r1 = cache_music_ranges(
        h5_path,
        max_samples=args.max_samples,
        show_progress=not args.no_progress,
    )
    print(f"Cached MUSIC ranges for {true_x.size} frames", flush=True)

    rows0: list[dict[str, float | int]] = []
    rows1: list[dict[str, float | int]] = []
    for cm in tqdm(offsets_cm, desc="dev0 range MAE", disable=args.no_progress):
        rows0.append(
            eval_range_mae(
                true_x=true_x,
                true_y=true_y,
                r_est=r0,
                dev=0,
                offset_m=float(cm) / 100.0,
            )
        )
    for cm in tqdm(offsets_cm, desc="dev1 range MAE", disable=args.no_progress):
        rows1.append(
            eval_range_mae(
                true_x=true_x,
                true_y=true_y,
                r_est=r1,
                dev=1,
                offset_m=float(cm) / 100.0,
            )
        )

    best0 = min(rows0, key=lambda r: float(r["range_mae_m"]))
    best1 = min(rows1, key=lambda r: float(r["range_mae_m"]))
    d0 = float(best0["offset_m"])
    d1 = float(best1["offset_m"])

    loc_stats = eval_localize_global(
        true_x=true_x,
        true_y=true_y,
        r0=r0,
        r1=r1,
        offset_dev0_m=d0,
        offset_dev1_m=d1,
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv0 = out_dir / "sweep_dev0.csv"
    csv1 = out_dir / "sweep_dev1.csv"
    _write_dev_csv(csv0, rows0)
    _write_dev_csv(csv1, rows1)

    best_txt = out_dir / "best.txt"
    best_txt.write_text(
        (
            f"best_offset_dev0_cm={best0['offset_cm']}\n"
            f"best_offset_dev1_cm={best1['offset_cm']}\n"
            f"best_offset_dev0_m={d0}\n"
            f"best_offset_dev1_m={d1}\n"
            f"dev0_range_mae_m={best0['range_mae_m']}\n"
            f"dev1_range_mae_m={best1['range_mae_m']}\n"
            f"dev0_range_rmse_m={best0['range_rmse_m']}\n"
            f"dev1_range_rmse_m={best1['range_rmse_m']}\n"
            f"loc_global_rmse_m={loc_stats['global_rmse_m']}\n"
            f"loc_inner_rmse_m={loc_stats['inner_rmse_m']}\n"
            f"loc_outer_rmse_m={loc_stats['outer_rmse_m']}\n"
            f"loc_n_valid={loc_stats['n_valid']}\n"
            f"geometry_dev0_tx=({d0}, -2.0)\n"
            f"geometry_dev0_rx=(-{d0}, -2.0)\n"
            f"geometry_dev1_tx=(-2.0, {d1})\n"
            f"geometry_dev1_rx=(-2.0, -{d1})\n"
        ),
        encoding="utf-8",
    )

    def _table(rows: list[dict[str, float | int]], title: str) -> None:
        print(f"\n=== {title} ===")
        print(
            tabulate(
                [
                    [r["offset_cm"], r["range_mae_m"], r["range_rmse_m"], r["n_valid"]]
                    for r in rows
                ],
                headers=["offset_cm", "mae", "rmse", "n"],
                tablefmt="simple_grid",
                floatfmt=".4f",
            )
        )

    _table(rows0, "dev0 range-MAE vs offset")
    _table(rows1, "dev1 range-MAE vs offset")
    print(
        f"\nBest: d0={best0['offset_cm']} cm (MAE={best0['range_mae_m']:.4f} m), "
        f"d1={best1['offset_cm']} cm (MAE={best1['range_mae_m']:.4f} m)"
    )
    print(
        f"Localize with best (report only): Global={loc_stats['global_rmse_m']:.4f} m "
        f"(Inner={loc_stats['inner_rmse_m']:.4f}, Outer={loc_stats['outer_rmse_m']:.4f})"
    )
    print(f"Wrote {csv0}")
    print(f"Wrote {csv1}")
    print(f"Wrote {best_txt}")


if __name__ == "__main__":
    main()
