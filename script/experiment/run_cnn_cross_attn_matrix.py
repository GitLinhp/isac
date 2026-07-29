#!/usr/bin/env python3
"""晚融合交叉注意力（S2-xattn）对照：叠加几何残差。

在新默认超参下训练两组：geom 端到端 / geom+cross-attn，
逐项 Run2 全帧评测，汇总 Global/Inner/Outer RMSE 到 CSV。

示例::

    python script/experiment/run_cnn_cross_attn_matrix.py
    python script/experiment/run_cnn_cross_attn_matrix.py --only geom_cross_attn
    python script/experiment/run_cnn_cross_attn_matrix.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from isac import PROJECT_ROOT
from isac_imp.record_target_metadata import is_inner_target_xy_m

PYTHON = Path(sys.executable)
RUN1_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
)
RUN2_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
TRAIN_SCRIPT = PROJECT_ROOT / "script/model_training/run_train_cooperative_monostatic_cnn.py"
EVAL_SCRIPT = PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_cnn_rmse.py"
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_cross_attn_summary.csv"
MATRIX_ROOT = PROJECT_ROOT / "models/cnn_xattn_ab"
EVAL_ROOT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_xattn_ab"

COMMON_TRAIN = [
    "--epochs",
    "100",
    "--batch-size",
    "128",
    "--label-jitter-m",
    "0.02",
    "--weight-decay",
    "1e-4",
    "--feature-noise-std",
    "0.02",
    "--spec-augment-prob",
    "0.3",
    "--early-stop-patience",
    "15",
    "--lr-scheduler-patience",
    "5",
    "--feature-mode",
    "real_imag",
    "--fusion-mode",
    "late",
    "--pool-mode",
    "attention",
    "--num-layers",
    "3",
    "--dropout",
    "0.3",
    "--lr",
    "0.0003",
    "--base-channels",
    "32",
    "--outer-ring-weight",
    "3.0",
    "--session-aggregated-loss",
    "--no-filter-outliers",
    "--no-aux-range",
    "--no-eval-after-train",
    "--geom-residual",
    "--no-stopgrad-geom",
]

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "geom_residual",
    "cross_attn",
    "checkpoint",
    "global_rmse_m",
    "inner_rmse_m",
    "outer_rmse_m",
    "output_csv",
]


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    description: str
    cross_attn: bool = False
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_dir(self) -> Path:
        return MATRIX_ROOT / self.exp_id

    def resolved_checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_cli_args(self) -> list[str]:
        args = list(COMMON_TRAIN)
        if self.cross_attn:
            args.append("--cross-attn")
        else:
            args.append("--no-cross-attn")
        args.extend(self.extra_train_args)
        return args

    def meta_row(self) -> dict[str, str | float | int]:
        return {
            "exp_id": self.exp_id,
            "description": self.description,
            "geom_residual": 1,
            "cross_attn": int(self.cross_attn),
        }


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        exp_id="geom_only",
        description="geom residual (no cross-attn)",
        cross_attn=False,
    ),
    Experiment(
        exp_id="geom_cross_attn",
        description="geom residual + late bidirectional cross-attn",
        cross_attn=True,
    ),
)


def _run(cmd: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _rmse_from_csv(csv_path: Path) -> dict[str, float]:
    rmses: list[float] = []
    inner: list[float] = []
    outer: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rmse = float(row["rmse_xy_m"])
            rmses.append(rmse)
            tx = float(row["true_x_m"])
            ty = float(row["true_y_m"])
            if is_inner_target_xy_m(tx, ty):
                inner.append(rmse)
            else:
                outer.append(rmse)
    return {
        "global_rmse_m": float(np.mean(rmses)) if rmses else float("nan"),
        "inner_rmse_m": float(np.mean(inner)) if inner else float("nan"),
        "outer_rmse_m": float(np.mean(outer)) if outer else float("nan"),
    }


def _train_experiment(exp: Experiment) -> None:
    ckpt = exp.resolved_checkpoint()
    if ckpt.is_file():
        print(f"[skip train] checkpoint exists: {ckpt}")
        return
    exp.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--h5-path",
        str(RUN1_H5),
        "--output-dir",
        str(exp.output_dir),
        *exp.train_cli_args(),
    ]
    _run(cmd)


def _eval_experiment(exp: Experiment) -> dict[str, str | float | int]:
    checkpoint = exp.resolved_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing for {exp.exp_id}: {checkpoint}")

    out_dir = EVAL_ROOT / exp.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_rmse.csv"
    cmd = [
        str(PYTHON),
        str(EVAL_SCRIPT),
        "--h5-path",
        str(RUN2_H5),
        "--checkpoint",
        str(checkpoint),
        "--range-roi",
        "0.0",
        "4.0",
        "--output-csv",
        str(csv_path),
        "--output-heatmap",
        str(out_dir / "cnn_rmse_heatmap.png"),
        "--output-cdf",
        str(out_dir / "cnn_rmse_cdf.png"),
        "--device",
        "cuda:0",
        "--no-filter-outliers",
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    metrics = _rmse_from_csv(csv_path)
    row: dict[str, str | float | int] = dict(exp.meta_row())
    row.update(
        {
            "checkpoint": str(checkpoint),
            "output_csv": str(csv_path),
            **metrics,
        }
    )
    return row


def _write_summary(rows: list[dict[str, str | float | int]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    print(f"\nSummary written ({len(rows)} rows): {SUMMARY_CSV}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="S2-xattn + geom residual A/B matrix")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="skip training; evaluate existing checkpoints only",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated exp_id filter",
    )
    args = parser.parse_args()

    only_ids = None if args.only is None else {s.strip() for s in args.only.split(",")}

    if not RUN1_H5.is_file():
        raise FileNotFoundError(f"train HDF5 missing: {RUN1_H5}")
    if not RUN2_H5.is_file():
        raise FileNotFoundError(f"eval HDF5 missing: {RUN2_H5}")

    rows: list[dict[str, str | float | int]] = []
    for exp in EXPERIMENTS:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== {exp.exp_id}: {exp.description} ===", flush=True)
        if not args.skip_train:
            _train_experiment(exp)
        rows.append(_eval_experiment(exp))
        _write_summary(rows)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
