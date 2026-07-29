#!/usr/bin/env python3
"""Cooperative Monostatic CNN 性能改善实验矩阵。

按 plan 依次训练/评估各 ablation，汇总 Run2 Global/Inner/Outer RMSE 到 CSV。

示例::

    python script/experiment/run_cnn_improvement_matrix.py
    python script/experiment/run_cnn_improvement_matrix.py --skip-train --only infer_aggregate
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isac import PROJECT_ROOT
from isac_imp.record_target_metadata import is_inner_target_xy_m

PYTHON = Path("/home/caict/radioconda/envs/ISAC/bin/python")
RUN1_H5 = PROJECT_ROOT / "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
RUN2_H5 = PROJECT_ROOT / "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
TRAIN_SCRIPT = PROJECT_ROOT / "script/model_training/run_train_cooperative_monostatic_cnn.py"
EVAL_SCRIPT = PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_cnn_rmse.py"
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_improvement_summary.csv"

DEPLOY_STRICT_BASELINE = PROJECT_ROOT / "models/cnn_deploy_strict/best_model.pth"

COMMON_TRAIN = [
    "--epochs",
    "50",
    "--batch-size",
    "128",
    "--label-jitter-m",
    "0.05",
    "--weight-decay",
    "1e-4",
    "--feature-noise-std",
    "0.02",
    "--spec-augment-prob",
    "0.3",
    "--early-stop-patience",
    "10",
]


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    description: str
    output_dir: Path
    train_args: list[str]
    eval_aggregate_session: bool = False
    eval_val_only: bool = False
    train: bool = True
    checkpoint: Path | None = None


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        exp_id="baseline_real_imag_bc64",
        description="deploy_strict real_imag base_channels=64 (reference)",
        output_dir=PROJECT_ROOT / "models/cnn_deploy_strict",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "real_imag",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "64",
        ],
        train=False,
        checkpoint=DEPLOY_STRICT_BASELINE,
    ),
    Experiment(
        exp_id="feat_logmag_fixed_norm",
        description="feature ablation: logmag_fixed_norm",
        output_dir=PROJECT_ROOT / "models/cnn_ablation_logmag_fixed_norm",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "logmag_fixed_norm",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "32",
        ],
    ),
    Experiment(
        exp_id="feat_complex_roi",
        description="feature ablation: complex_roi",
        output_dir=PROJECT_ROOT / "models/cnn_ablation_complex_roi",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "complex_roi",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "32",
        ],
    ),
    Experiment(
        exp_id="loss_session_w3",
        description="side=corner=3.0 + session aggregated loss",
        output_dir=PROJECT_ROOT / "models/cnn_ablation_session_loss_w3",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "real_imag",
            "--center-weight",
            "1.0",
            "--side-weight",
            "3.0",
            "--corner-weight",
            "3.0",
            "--session-aggregated-loss",
            "--base-channels",
            "32",
        ],
    ),
    Experiment(
        exp_id="model_layers4_bc64",
        description="model capacity: num_layers=4, base_channels=64",
        output_dir=PROJECT_ROOT / "models/cnn_ablation_layers4_bc64",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "real_imag",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "64",
            "--num-layers",
            "4",
        ],
    ),
    Experiment(
        exp_id="range_slowtime_2d",
        description="range×slow-time 2D Conv2d input",
        output_dir=PROJECT_ROOT / "models/cnn_ablation_range_slowtime_2d",
        train_args=[
            *COMMON_TRAIN,
            "--feature-mode",
            "range_slowtime_2d",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "32",
            "--num-layers",
            "2",
        ],
    ),
    Experiment(
        exp_id="run2_finetune",
        description="Run2 finetune from deploy_strict (adaptation line, val-only eval)",
        output_dir=PROJECT_ROOT / "models/cnn_run2_finetune",
        train_args=[
            *COMMON_TRAIN,
            "--h5-path",
            str(RUN2_H5),
            "--finetune",
            "--resume",
            str(DEPLOY_STRICT_BASELINE),
            "--finetune-lr",
            "1e-4",
            "--feature-mode",
            "real_imag",
            "--center-weight",
            "1.0",
            "--side-weight",
            "2.0",
            "--corner-weight",
            "2.0",
            "--base-channels",
            "64",
            "--feature-noise-std",
            "0.01",
            "--spec-augment-prob",
            "0.15",
        ],
        eval_val_only=True,
    ),
    Experiment(
        exp_id="infer_aggregate",
        description="inference --aggregate-session on deploy_strict (no retrain)",
        output_dir=PROJECT_ROOT / "models/cnn_deploy_strict",
        train_args=[],
        train=False,
        eval_aggregate_session=True,
        checkpoint=DEPLOY_STRICT_BASELINE,
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
    if not exp.train:
        return
    ckpt = exp.output_dir / "best_model.pth"
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
        *exp.train_args,
    ]
    _run(cmd)


def _eval_experiment(exp: Experiment) -> dict[str, str | float]:
    checkpoint = exp.checkpoint or (exp.output_dir / "best_model.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing for {exp.exp_id}: {checkpoint}")

    out_dir = PROJECT_ROOT / "out/cooperative_monostatic" / exp.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_rmse.csv"
    cmd = [
        str(PYTHON),
        str(EVAL_SCRIPT),
        "--h5-path",
        str(RUN2_H5),
        "--checkpoint",
        str(checkpoint),
        "--output-csv",
        str(csv_path),
        "--output-heatmap",
        str(out_dir / "cnn_rmse_heatmap.png"),
        "--output-cdf",
        str(out_dir / "cnn_rmse_cdf.png"),
    ]
    if exp.eval_aggregate_session:
        cmd.append("--aggregate-session")
    if exp.eval_val_only:
        cmd.extend(["--val-only", "--val-ratio", "0.2", "--seed", "42"])
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
    row: dict[str, str | float] = {
        "exp_id": exp.exp_id,
        "description": exp.description,
        "checkpoint": str(checkpoint),
        "aggregate_session": int(exp.eval_aggregate_session),
        "output_csv": str(csv_path),
    }
    row.update(metrics)
    return row


def _write_summary(rows: list[dict[str, str | float]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_id",
        "description",
        "checkpoint",
        "aggregate_session",
        "global_rmse_m",
        "inner_rmse_m",
        "outer_rmse_m",
        "output_csv",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written: {SUMMARY_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN improvement experiment matrix")
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

    rows: list[dict[str, str | float]] = []
    for exp in EXPERIMENTS:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== {exp.exp_id}: {exp.description} ===")
        if not args.skip_train:
            _train_experiment(exp)
        rows.append(_eval_experiment(exp))

    _write_summary(rows)


if __name__ == "__main__":
    main()
