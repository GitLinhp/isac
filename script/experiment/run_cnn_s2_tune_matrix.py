#!/usr/bin/env python3
"""S2 结构基线（late + attention）自动调参矩阵。

逐项训练 + Run2 评估，汇总 Global/Inner/Outer RMSE 到 CSV。
已有 checkpoint 则跳过训练；每完成一实验即 flush 汇总表。

示例::

    python script/experiment/run_cnn_s2_tune_matrix.py
    python script/experiment/run_cnn_s2_tune_matrix.py --only late_attn_side3,late_attn_bc48
    python script/experiment/run_cnn_s2_tune_matrix.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import shutil
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
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_s2_tune_summary.csv"
TUNE_ROOT = PROJECT_ROOT / "models/cnn_s2_tune"
# 当前无 aux 的 late+attention 最优权重（勿被矩阵覆盖）
DEPLOY_LATE_ATTN_BASELINE = PROJECT_ROOT / "models/cnn_deploy_strict_roi4/best_model.pth"

COMMON_TRAIN = [
    "--epochs",
    "100",
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
    "--lr-scheduler-patience",
    "5",
    "--feature-mode",
    "real_imag",
    "--fusion-mode",
    "late",
    "--num-layers",
    "3",
    "--dropout",
    "0.3",
    "--lr",
    "0.0005",
    "--no-eval-after-train",
]

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "fusion_mode",
    "pool_mode",
    "aux_range",
    "center_weight",
    "side_weight",
    "corner_weight",
    "aux_range_weight",
    "session_aggregated_loss",
    "base_channels",
    "lr",
    "dropout",
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
    pool_mode: str = "attention"
    aux_range: bool = False
    center_weight: float = 1.0
    side_weight: float = 2.0
    corner_weight: float = 2.0
    aux_range_weight: float = 0.5
    session_aggregated_loss: bool = False
    base_channels: int = 32
    lr: float = 5e-4
    dropout: float = 0.3
    train: bool = True
    # 若给定，则跳过训练并用该权重评测（用于已有基线 ckpt）
    checkpoint: Path | None = None
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_dir(self) -> Path:
        return TUNE_ROOT / self.exp_id

    def resolved_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            return self.checkpoint
        return self.output_dir / "best_model.pth"

    def train_cli_args(self) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--pool-mode",
            self.pool_mode,
            "--center-weight",
            str(self.center_weight),
            "--side-weight",
            str(self.side_weight),
            "--corner-weight",
            str(self.corner_weight),
            "--base-channels",
            str(self.base_channels),
            "--lr",
            str(self.lr),
            "--dropout",
            str(self.dropout),
        ]
        if self.aux_range:
            args.extend(
                [
                    "--aux-range",
                    "--aux-range-weight",
                    str(self.aux_range_weight),
                ]
            )
        else:
            args.append("--no-aux-range")
        if self.session_aggregated_loss:
            args.append("--session-aggregated-loss")
        args.extend(self.extra_train_args)
        return args

    def meta_row(self) -> dict[str, str | float | int]:
        return {
            "exp_id": self.exp_id,
            "description": self.description,
            "fusion_mode": "late",
            "pool_mode": self.pool_mode,
            "aux_range": int(self.aux_range),
            "center_weight": self.center_weight,
            "side_weight": self.side_weight,
            "corner_weight": self.corner_weight,
            "aux_range_weight": self.aux_range_weight if self.aux_range else 0.0,
            "session_aggregated_loss": int(self.session_aggregated_loss),
            "base_channels": self.base_channels,
            "lr": self.lr,
            "dropout": self.dropout,
        }


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        exp_id="late_attn_baseline",
        description="late+attention baseline (no aux, side=corner=2)",
        train=False,
        checkpoint=DEPLOY_LATE_ATTN_BASELINE,
    ),
    Experiment(
        exp_id="late_attn_aux05",
        description="late+attention + aux_range_weight=0.5",
        aux_range=True,
        aux_range_weight=0.5,
    ),
    Experiment(
        exp_id="late_attn_side25",
        description="late+attention side=corner=2.5",
        side_weight=2.5,
        corner_weight=2.5,
    ),
    Experiment(
        exp_id="late_attn_side3",
        description="late+attention side=corner=3.0",
        side_weight=3.0,
        corner_weight=3.0,
    ),
    Experiment(
        exp_id="late_attn_side3_session",
        description="late+attention side=corner=3 + session aggregated loss",
        side_weight=3.0,
        corner_weight=3.0,
        session_aggregated_loss=True,
    ),
    Experiment(
        exp_id="late_softargmax",
        description="late + soft_argmax pooling (no aux)",
        pool_mode="soft_argmax",
    ),
    Experiment(
        exp_id="late_attn_bc48",
        description="late+attention base_channels=48",
        base_channels=48,
    ),
    Experiment(
        exp_id="late_attn_bc64",
        description="late+attention base_channels=64",
        base_channels=64,
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


def _ensure_baseline_link(exp: Experiment) -> None:
    """把 deploy 基线权重同步到矩阵目录，便于统一查找。"""
    if exp.checkpoint is None:
        return
    src = exp.checkpoint
    if not src.is_file():
        raise FileNotFoundError(f"baseline checkpoint missing: {src}")
    exp.output_dir.mkdir(parents=True, exist_ok=True)
    dst = exp.output_dir / "best_model.pth"
    if dst.is_file() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _train_experiment(exp: Experiment) -> None:
    if not exp.train:
        _ensure_baseline_link(exp)
        print(f"[skip train] {exp.exp_id}: using existing checkpoint")
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
        *exp.train_cli_args(),
    ]
    _run(cmd)


def _eval_experiment(exp: Experiment) -> dict[str, str | float | int]:
    checkpoint = exp.resolved_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing for {exp.exp_id}: {checkpoint}")

    out_dir = PROJECT_ROOT / "out/cooperative_monostatic/cnn_s2_tune" / exp.exp_id
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
        "--filter-outliers",
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
    parser = argparse.ArgumentParser(description="S2 late-fusion CNN tune matrix")
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
    if not DEPLOY_LATE_ATTN_BASELINE.is_file():
        raise FileNotFoundError(
            f"late+attn baseline checkpoint missing: {DEPLOY_LATE_ATTN_BASELINE}"
        )

    rows: list[dict[str, str | float | int]] = []
    for exp in EXPERIMENTS:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== {exp.exp_id}: {exp.description} ===", flush=True)
        if not args.skip_train:
            _train_experiment(exp)
        elif exp.checkpoint is not None:
            _ensure_baseline_link(exp)
        rows.append(_eval_experiment(exp))
        _write_summary(rows)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
