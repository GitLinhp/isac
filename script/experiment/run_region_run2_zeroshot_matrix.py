#!/usr/bin/env python3
"""仅 Run1 训练 RegionCNN，完整 Run2 零样本评估对照矩阵。

硬约束：训练不得使用 Run2 任何帧；Run2 仅作最终评估。

示例::

    python script/experiment/run_region_run2_zeroshot_matrix.py
    python script/experiment/run_region_run2_zeroshot_matrix.py --only baseline_drop01,aug_neighbor
    python script/experiment/run_region_run2_zeroshot_matrix.py --skip-train
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
import torch

from isac import PROJECT_ROOT

PYTHON = Path(sys.executable)
RUN1_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5"
)
RUN2_H5 = (
    PROJECT_ROOT
    / "data/experiment/cooperative_monostatic/cooperative_monostatic_dataset.h5"
)
REGION_TRAIN = (
    PROJECT_ROOT
    / "script/model_training/run_train_cooperative_monostatic_region_cnn.py"
)
EVAL_SCRIPT = (
    PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_two_stage_eval.py"
)
BASELINE_CKPT = (
    PROJECT_ROOT / "models/two_stage_tune/region/region_drop01/best_model.pth"
)
FINE_CKPT = (
    PROJECT_ROOT / "models/two_stage_tune/fine/fine_lr1e4/best_model.pth"
)
MATRIX_ROOT = PROJECT_ROOT / "models/region_run2_zeroshot"
SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/region_run2_zeroshot_summary.csv"
)

COMMON_TRAIN = [
    "--h5-path",
    str(RUN1_H5),
    "--epochs",
    "100",
    "--batch-size",
    "128",
    "--val-ratio",
    "0.2",
    "--early-stop-patience",
    "12",
    "--feature-mode",
    "real_imag",
    "--range-roi",
    "0",
    "4",
    "--num-layers",
    "3",
    "--base-channels",
    "64",
    "--lr",
    "5e-5",
    "--dropout",
    "0.1",
    "--pool-mode",
    "attention",
    "--num-workers",
    "4",
    "--device",
    "cuda:0",
]

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "train_h5",
    "checkpoint",
    "run1_val_acc",
    "run1_val_topk_hit",
    "run1_best_epoch",
    "run2_top1_acc",
    "run2_top3_hit",
    "run2_global_rmse_topk_m",
    "run2_global_rmse_top1_m",
    "run2_oracle_region_rmse_m",
    "n",
    "eval_csv",
]


@dataclass(frozen=True)
class ZeroshotExp:
    exp_id: str
    description: str
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    reuse_baseline_ckpt: bool = False

    @property
    def output_dir(self) -> Path:
        return MATRIX_ROOT / "region" / self.exp_id

    @property
    def checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    @property
    def eval_dir(self) -> Path:
        return MATRIX_ROOT / "eval" / self.exp_id

    @property
    def eval_csv(self) -> Path:
        return self.eval_dir / "two_stage_rmse.csv"

    def train_args(self) -> list[str]:
        return [
            *COMMON_TRAIN,
            "--output-dir",
            str(self.output_dir),
            *self.extra_args,
        ]


EXPERIMENTS: tuple[ZeroshotExp, ...] = (
    ZeroshotExp(
        exp_id="baseline_drop01",
        description="reuse region_drop01 (dropout=0.1)",
        reuse_baseline_ckpt=True,
    ),
    ZeroshotExp(
        exp_id="aug_strong",
        description="noise 0.02 + SpecAugment 0.7",
        extra_args=(
            "--feature-noise-std",
            "0.02",
            "--spec-augment-prob",
            "0.7",
        ),
    ),
    ZeroshotExp(
        exp_id="neighbor_smooth",
        description="neighbor-smooth 0.2",
        extra_args=("--neighbor-smooth", "0.2"),
    ),
    ZeroshotExp(
        exp_id="label_smooth",
        description="label-smoothing 0.1",
        extra_args=("--label-smoothing", "0.1"),
    ),
    ZeroshotExp(
        exp_id="mixup",
        description="feature mixup alpha=0.2",
        extra_args=("--mixup-alpha", "0.2"),
    ),
    ZeroshotExp(
        exp_id="aug_neighbor",
        description="strong aug + neighbor-smooth 0.2",
        extra_args=(
            "--feature-noise-std",
            "0.02",
            "--spec-augment-prob",
            "0.7",
            "--neighbor-smooth",
            "0.2",
        ),
    ),
    ZeroshotExp(
        exp_id="topk_ce3",
        description="forced top-k softmax CE k=3",
        extra_args=("--topk-ce", "--topk-ce-k", "3"),
    ),
    ZeroshotExp(
        exp_id="neigh_mild_aug",
        description="neighbor 0.2 + noise 0.01 + SpecAugment 0.55",
        extra_args=(
            "--neighbor-smooth",
            "0.2",
            "--feature-noise-std",
            "0.01",
            "--spec-augment-prob",
            "0.55",
        ),
    ),
    ZeroshotExp(
        exp_id="neigh_mixup",
        description="neighbor 0.2 + mixup 0.1",
        extra_args=(
            "--neighbor-smooth",
            "0.2",
            "--mixup-alpha",
            "0.1",
        ),
    ),
    ZeroshotExp(
        exp_id="combo_best",
        description="neighbor 0.2 + mild aug + mixup 0.1 + wd 0.02",
        extra_args=(
            "--neighbor-smooth",
            "0.2",
            "--feature-noise-std",
            "0.01",
            "--spec-augment-prob",
            "0.55",
            "--mixup-alpha",
            "0.1",
            "--weight-decay",
            "0.02",
        ),
    ),
)


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_ckpt_metrics(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, float | int] = {}
    if "val_acc" in ckpt:
        out["val_acc"] = float(ckpt["val_acc"])
    if "val_topk_hit" in ckpt:
        out["val_topk_hit"] = float(ckpt["val_topk_hit"])
    if "epoch" in ckpt:
        out["epoch"] = int(ckpt["epoch"])
    return out


def _metrics_from_eval_csv(csv_path: Path) -> dict[str, float | int]:
    import sys

    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from cooperative_monostatic_eval_report import load_two_stage_eval_metrics

    m = load_two_stage_eval_metrics(csv_path)
    if not m:
        return {"n": 0}
    out: dict[str, float | int] = {
        "n": m.get("n", 0),
        "run2_global_rmse_topk_m": m.get("global_mean_err_m", float("nan")),
        "run2_global_rmse_top1_m": m.get("global_mean_err_m", float("nan")),
    }
    if "region_top1_acc" in m:
        out["run2_top1_acc"] = m["region_top1_acc"]
    if "region_topk_hit" in m:
        out["run2_top3_hit"] = m["region_topk_hit"]
    if "oracle_region_mean_err_m" in m:
        out["run2_oracle_region_rmse_m"] = m["oracle_region_mean_err_m"]
    return out


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _merge_summary_rows(
    path: Path,
    fields: list[str],
    new_rows: list[dict],
) -> list[dict]:
    """按 exp_id 合并：保留未出现在 new_rows 中的旧行，覆盖同 id。"""
    by_id: dict[str, dict] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eid = str(row.get("exp_id", ""))
                if eid:
                    by_id[eid] = {k: row.get(k, "") for k in fields}
    for row in new_rows:
        eid = str(row.get("exp_id", ""))
        if not eid:
            continue
        by_id[eid] = {k: row.get(k, "") for k in fields}
    # 稳定顺序：已有 EXPERIMENTS 顺序优先，其余按 id
    known = [e.exp_id for e in EXPERIMENTS]
    ordered: list[dict] = []
    seen: set[str] = set()
    for eid in known:
        if eid in by_id:
            ordered.append(by_id[eid])
            seen.add(eid)
    for eid, row in by_id.items():
        if eid not in seen:
            ordered.append(row)
    return ordered


def _ensure_baseline_ckpt(exp: ZeroshotExp, *, force: bool) -> None:
    exp.output_dir.mkdir(parents=True, exist_ok=True)
    if exp.checkpoint.is_file() and not force:
        print(f"[skip-train] {exp.exp_id}: {exp.checkpoint}", flush=True)
        return
    if not BASELINE_CKPT.is_file():
        raise FileNotFoundError(
            f"基线 checkpoint 不存在，无法复用: {BASELINE_CKPT}"
        )
    shutil.copy2(BASELINE_CKPT, exp.checkpoint)
    print(
        f"[reuse-baseline] {exp.exp_id}: copied {BASELINE_CKPT} -> {exp.checkpoint}",
        flush=True,
    )


def _train_exp(exp: ZeroshotExp, *, force: bool, skip_train: bool) -> dict:
    if skip_train and not exp.checkpoint.is_file() and not exp.reuse_baseline_ckpt:
        raise FileNotFoundError(f"--skip-train 但缺少 checkpoint: {exp.checkpoint}")

    if exp.reuse_baseline_ckpt:
        _ensure_baseline_ckpt(exp, force=force and not skip_train)
    elif skip_train:
        print(f"[skip-train] {exp.exp_id}: {exp.checkpoint}", flush=True)
    elif exp.checkpoint.is_file() and not force:
        print(f"[skip-train] {exp.exp_id}: {exp.checkpoint}", flush=True)
    else:
        # 硬约束：始终仅 Run1
        assert str(RUN1_H5) in exp.train_args()
        assert str(RUN2_H5) not in exp.train_args()
        _run([str(PYTHON), str(REGION_TRAIN), *exp.train_args()])

    metrics = _load_ckpt_metrics(exp.checkpoint)
    return {
        "exp_id": exp.exp_id,
        "description": exp.description,
        "train_h5": str(RUN1_H5),
        "checkpoint": str(exp.checkpoint),
        "run1_val_acc": metrics.get("val_acc", ""),
        "run1_val_topk_hit": metrics.get("val_topk_hit", ""),
        "run1_best_epoch": metrics.get("epoch", ""),
    }


def _eval_exp(exp: ZeroshotExp, *, force: bool, skip_eval: bool) -> dict:
    if skip_eval and not exp.eval_csv.is_file():
        raise FileNotFoundError(f"--skip-eval 但缺少评估 CSV: {exp.eval_csv}")
    if exp.eval_csv.is_file() and (skip_eval or not force):
        print(f"[skip-eval] {exp.exp_id}: {exp.eval_csv}", flush=True)
    else:
        if not FINE_CKPT.is_file():
            raise FileNotFoundError(f"Fine checkpoint 不存在: {FINE_CKPT}")
        _run(
            [
                str(PYTHON),
                str(EVAL_SCRIPT),
                "--h5-path",
                str(RUN2_H5),
                "--region-checkpoint",
                str(exp.checkpoint),
                "--fine-checkpoint",
                str(FINE_CKPT),
                "--output-dir",
                str(exp.eval_dir),
                "--region-topk",
                "3",
                "--device",
                "cuda:0",
                "--batch-size",
                "128",
                "--no-plot",
            ]
        )
    metrics = _metrics_from_eval_csv(exp.eval_csv)
    return {
        "eval_csv": str(exp.eval_csv),
        **metrics,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run1-only RegionCNN train + full Run2 zero-shot eval matrix"
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="逗号分隔 exp_id 子集",
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="强制重训 / 重评（覆盖已有 ckpt 与 CSV）",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=SUMMARY_CSV,
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not RUN1_H5.is_file():
        raise FileNotFoundError(f"Run1 H5 不存在: {RUN1_H5}")
    if not RUN2_H5.is_file():
        raise FileNotFoundError(f"Run2 H5 不存在: {RUN2_H5}")

    selected = EXPERIMENTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = tuple(e for e in EXPERIMENTS if e.exp_id in want)
        missing = want - {e.exp_id for e in selected}
        if missing:
            raise ValueError(f"未知 exp_id: {sorted(missing)}")

    print(
        f"Zero-shot matrix | train=Run1 only | eval=full Run2 | "
        f"n_exp={len(selected)} | summary={args.summary_csv}",
        flush=True,
    )

    rows: list[dict] = []
    for exp in selected:
        row = _train_exp(exp, force=args.force, skip_train=args.skip_train)
        row.update(_eval_exp(exp, force=args.force, skip_eval=args.skip_eval))
        rows.append(row)
        merged = _merge_summary_rows(args.summary_csv, SUMMARY_FIELDS, rows)
        _write_rows(args.summary_csv, SUMMARY_FIELDS, merged)
        print(
            f"[summary] {exp.exp_id} | "
            f"run1_val={row.get('run1_val_acc')} | "
            f"run1_topk={row.get('run1_val_topk_hit')} | "
            f"run2_top1={row.get('run2_top1_acc')} | "
            f"run2_top3={row.get('run2_top3_hit')} | "
            f"rmse_topk={row.get('run2_global_rmse_topk_m')}",
            flush=True,
        )

    merged = _merge_summary_rows(args.summary_csv, SUMMARY_FIELDS, rows)
    _write_rows(args.summary_csv, SUMMARY_FIELDS, merged)
    print(f"\nWrote summary: {args.summary_csv}", flush=True)

    baseline = next((r for r in merged if r["exp_id"] == "baseline_drop01"), None)
    if baseline and baseline.get("run2_top1_acc") not in ("", None):
        base_acc = float(baseline["run2_top1_acc"])
        base_top3 = (
            float(baseline["run2_top3_hit"])
            if baseline.get("run2_top3_hit") not in ("", None)
            else float("nan")
        )
        print(
            f"\n相对基线 baseline_drop01 Run2 top-1={base_acc:.4f} "
            f"top-3={base_top3:.4f}:",
            flush=True,
        )
        for r in merged:
            if r["exp_id"] == "baseline_drop01":
                continue
            acc = r.get("run2_top1_acc")
            if acc in ("", None):
                continue
            acc_f = float(acc)
            rel = (acc_f / base_acc - 1.0) * 100.0 if base_acc > 0 else float("nan")
            top3 = r.get("run2_top3_hit")
            top3_s = f"{float(top3):.4f}" if top3 not in ("", None) else "n/a"
            print(
                f"  {r['exp_id']}: top1={acc_f:.4f} ({rel:+.1f}% vs baseline) "
                f"top3={top3_s}",
                flush=True,
            )


if __name__ == "__main__":
    main()
