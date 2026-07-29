#!/usr/bin/env python3
"""历史最优配方追赶矩阵（统一无过滤全帧协议）。

复现 late_attn_outer3_session（lr=5e-4, jitter=0.05, side=corner=3, no-geom）
并与当前 BEST / 交叉变体对照。验收以 Run2 ``--no-filter-outliers`` Global 为准。

示例::

    python script/experiment/run_cnn_hist_catchup_matrix.py
    python script/experiment/run_cnn_hist_catchup_matrix.py --only hist_outer3_session
    python script/experiment/run_cnn_hist_catchup_matrix.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_hist_catchup_summary.csv"
BEST_TXT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_hist_catchup_best.txt"
MATRIX_ROOT = PROJECT_ROOT / "models/cnn_hist_catchup"
EVAL_ROOT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_hist_catchup"

REF_BEST_AB_CKPT = (
    PROJECT_ROOT / "models/cnn_best_ab/B_geom_w_1_2_2/best_model.pth"
)
HIST_S2_CKPT = (
    PROJECT_ROOT / "models/cnn_s2_tune/late_attn_outer3_session/best_model.pth"
)

GLOBAL_TIE_EPS = 0.005

COMMON_TRAIN = [
    "--epochs",
    "100",
    "--batch-size",
    "128",
    "--weight-decay",
    "1e-4",
    "--feature-noise-std",
    "0.02",
    "--spec-augment-prob",
    "0.3",
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
    "--base-channels",
    "32",
    "--center-weight",
    "1.0",
    "--session-aggregated-loss",
    "--no-filter-outliers",
    "--no-aux-range",
    "--no-cross-attn",
    "--no-stopgrad-geom",
    "--no-eval-after-train",
]

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "center_weight",
    "side_weight",
    "corner_weight",
    "geom_residual",
    "lr",
    "label_jitter_m",
    "early_stop_patience",
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
    side_weight: float = 3.0
    corner_weight: float = 3.0
    geom_residual: bool = False
    lr: float = 5e-4
    label_jitter_m: float = 0.05
    early_stop_patience: int = 15
    train: bool = True
    # If set, skip training and evaluate this checkpoint (copied into output_dir)
    external_checkpoint: Path | None = None

    @property
    def output_dir(self) -> Path:
        return MATRIX_ROOT / self.exp_id

    def resolved_checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_cli_args(self) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--side-weight",
            str(self.side_weight),
            "--corner-weight",
            str(self.corner_weight),
            "--lr",
            str(self.lr),
            "--label-jitter-m",
            str(self.label_jitter_m),
            "--early-stop-patience",
            str(self.early_stop_patience),
        ]
        if self.geom_residual:
            args.append("--geom-residual")
        else:
            args.append("--no-geom-residual")
        return args

    def meta_row(self) -> dict[str, str | float | int]:
        return {
            "exp_id": self.exp_id,
            "description": self.description,
            "center_weight": 1.0,
            "side_weight": self.side_weight,
            "corner_weight": self.corner_weight,
            "geom_residual": int(self.geom_residual),
            "lr": self.lr,
            "label_jitter_m": self.label_jitter_m,
            "early_stop_patience": self.early_stop_patience,
        }


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        exp_id="hist_outer3_session",
        description="reproduce hist: side=corner=3 lr=5e-4 jitter=0.05 no-geom es=10",
        side_weight=3.0,
        corner_weight=3.0,
        geom_residual=False,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=10,
    ),
    Experiment(
        exp_id="hist_es15",
        description="hist recipe with early-stop=15",
        side_weight=3.0,
        corner_weight=3.0,
        geom_residual=False,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="plain_122_lr5e4_j05",
        description="W* weights 1/2/2 + hist lr/jitter, no-geom",
        side_weight=2.0,
        corner_weight=2.0,
        geom_residual=False,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="plain_133_lr5e4_j05",
        description="weights 1/3/3 + hist lr/jitter, es=15",
        side_weight=3.0,
        corner_weight=3.0,
        geom_residual=False,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="geom_133_lr5e4_j05",
        description="hist hyperparams + geom-residual",
        side_weight=3.0,
        corner_weight=3.0,
        geom_residual=True,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="ref_best_ab",
        description="reeval current best_ab geom 1/2/2 (no retrain)",
        side_weight=2.0,
        corner_weight=2.0,
        geom_residual=True,
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        train=False,
        external_checkpoint=REF_BEST_AB_CKPT,
    ),
    Experiment(
        exp_id="ref_hist_s2_ckpt",
        description="reeval original S2 late_attn_outer3_session ckpt (no retrain)",
        side_weight=3.0,
        corner_weight=3.0,
        geom_residual=False,
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=10,
        train=False,
        external_checkpoint=HIST_S2_CKPT,
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


def _better(a: dict[str, str | float | int], b: dict[str, str | float | int]) -> bool:
    ga = float(a["global_rmse_m"])
    gb = float(b["global_rmse_m"])
    if ga < gb - GLOBAL_TIE_EPS:
        return True
    if gb < ga - GLOBAL_TIE_EPS:
        return False
    return float(a["outer_rmse_m"]) < float(b["outer_rmse_m"])


def _pick_best(
    rows: list[dict[str, str | float | int]],
) -> dict[str, str | float | int]:
    best = rows[0]
    for row in rows[1:]:
        if _better(row, best):
            best = row
    return best


def _train_experiment(exp: Experiment) -> None:
    ckpt = exp.resolved_checkpoint()
    if ckpt.is_file():
        print(f"[skip train] checkpoint exists: {ckpt}")
        return
    exp.output_dir.mkdir(parents=True, exist_ok=True)
    if not exp.train:
        if exp.external_checkpoint is None or not exp.external_checkpoint.is_file():
            raise FileNotFoundError(
                f"external checkpoint missing for {exp.exp_id}: "
                f"{exp.external_checkpoint}"
            )
        shutil.copy2(exp.external_checkpoint, ckpt)
        print(f"[reuse ckpt] {exp.external_checkpoint} → {ckpt}")
        return
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


def _print_topk(rows: list[dict[str, str | float | int]], k: int = 7) -> None:
    ranked = sorted(
        rows, key=lambda r: (float(r["global_rmse_m"]), float(r["outer_rmse_m"]))
    )
    print("\n=== Top configs (by Global, then Outer) ===", flush=True)
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"{i}. {row['exp_id']}: G={float(row['global_rmse_m']):.4f} "
            f"I={float(row['inner_rmse_m']):.4f} O={float(row['outer_rmse_m']):.4f} "
            f"| c/s/k=1/{row['side_weight']}/{row['corner_weight']} "
            f"geom={row['geom_residual']} lr={row['lr']} "
            f"jitter={row['label_jitter_m']} es={row['early_stop_patience']}",
            flush=True,
        )


def _write_best(
    best: dict[str, str | float | int],
    *,
    exp: Experiment,
) -> None:
    cli = " ".join(exp.train_cli_args())
    text = (
        f"BEST2 exp_id={best['exp_id']}\n"
        f"description={best['description']}\n"
        f"global_rmse_m={float(best['global_rmse_m']):.6f}\n"
        f"inner_rmse_m={float(best['inner_rmse_m']):.6f}\n"
        f"outer_rmse_m={float(best['outer_rmse_m']):.6f}\n"
        f"center_weight={best['center_weight']}\n"
        f"side_weight={best['side_weight']}\n"
        f"corner_weight={best['corner_weight']}\n"
        f"geom_residual={best['geom_residual']}\n"
        f"lr={best['lr']}\n"
        f"label_jitter_m={best['label_jitter_m']}\n"
        f"early_stop_patience={best['early_stop_patience']}\n"
        f"checkpoint={best['checkpoint']}\n"
        f"train_cli_args={cli}\n"
        f"eval_protocol=Run2_full_frame_no_filter_outliers\n"
    )
    BEST_TXT.parent.mkdir(parents=True, exist_ok=True)
    BEST_TXT.write_text(text, encoding="utf-8")
    print("\n=== BEST2 ===", flush=True)
    print(text, flush=True)
    print(f"Wrote {BEST_TXT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hist catch-up matrix vs 0.5875")
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
    exps_run: list[Experiment] = []
    for exp in EXPERIMENTS:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== {exp.exp_id}: {exp.description} ===", flush=True)
        if not args.skip_train:
            _train_experiment(exp)
        elif not exp.resolved_checkpoint().is_file() and exp.external_checkpoint:
            _train_experiment(exp)  # copies external ckpt
        rows.append(_eval_experiment(exp))
        exps_run.append(exp)
        _write_summary(rows)

    if not rows:
        print("No experiments run.", flush=True)
        return

    best_row = _pick_best(rows)
    best_exp = next(e for e in exps_run if e.exp_id == best_row["exp_id"])
    _print_topk(rows)
    _write_best(best_row, exp=best_exp)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
