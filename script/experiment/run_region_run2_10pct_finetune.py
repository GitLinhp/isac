#!/usr/bin/env python3
"""用 10% Run2 session 增强 Region，并在剩余 90% holdout 上对照评估。

流程：
1. seed=42 从 Run2 抽 10% session → aug；其余 → holdout
2. 基线：现优 Run1 ``aug_neighbor`` ckpt，仅在 holdout 上评
3. 微调：resume aug_neighbor + Run1 train ∪ Run2 aug，early-stop 看 Run1 val
4. 汇总 CSV

示例::

    python script/experiment/run_region_run2_10pct_finetune.py
    python script/experiment/run_region_run2_10pct_finetune.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from isac import PROJECT_ROOT
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    DATASET_KEY_SESSION_INDEX,
)

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
REGION_INIT = (
    PROJECT_ROOT
    / "models/region_run2_zeroshot/region/aug_neighbor/best_model.pth"
)
FINE_CKPT = (
    PROJECT_ROOT / "models/two_stage_tune/fine/fine_lr1e4/best_model.pth"
)
OUT_ROOT = PROJECT_ROOT / "models/region_run2_10pct"
SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/region_run2_10pct_summary.csv"
)

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "region_checkpoint",
    "fine_checkpoint",
    "n_aug_sessions",
    "n_holdout_sessions",
    "run1_val_acc",
    "run1_best_epoch",
    "holdout_top1_acc",
    "holdout_top3_hit",
    "holdout_global_rmse_topk_m",
    "holdout_global_rmse_top1_m",
    "n",
    "eval_csv",
]


def _write_session_list(path: Path, sessions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in sessions:
            f.write(f"{int(sid)}\n")


def sample_sessions_by_frac(
    session_indices: np.ndarray,
    *,
    frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac 须在 (0, 1]，收到 {frac}")
    unique = np.unique(np.asarray(session_indices, dtype=np.int64))
    if unique.size == 0:
        raise ValueError("session_indices 为空")
    rng = np.random.default_rng(int(seed))
    n_aug = max(1, int(round(float(frac) * int(unique.size))))
    n_aug = min(n_aug, int(unique.size))
    perm = rng.permutation(unique)
    aug = np.sort(perm[:n_aug])
    holdout = np.sort(perm[n_aug:])
    return aug, holdout


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


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
        "holdout_global_rmse_topk_m": m.get("global_mean_err_m", float("nan")),
        "holdout_global_rmse_top1_m": m.get("global_mean_err_m", float("nan")),
    }
    if "region_top1_acc" in m:
        out["holdout_top1_acc"] = m["region_top1_acc"]
    if "region_topk_hit" in m:
        out["holdout_top3_hit"] = m["region_topk_hit"]
    return out


def _load_ckpt_metrics(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, float | int] = {}
    if "val_acc" in ckpt:
        out["val_acc"] = float(ckpt["val_acc"])
    if "epoch" in ckpt:
        out["epoch"] = int(ckpt["epoch"])
    return out


def _assert_no_session_overlap(aug_path: Path, holdout_path: Path) -> None:
    aug = {
        int(x.strip())
        for x in aug_path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    }
    hold = {
        int(x.strip())
        for x in holdout_path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    }
    inter = aug & hold
    if inter:
        raise RuntimeError(f"aug 与 holdout session 有交集: {sorted(inter)[:10]}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run2 10% session Region finetune compare")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--extra-session-frac", type=float, default=0.1)
    p.add_argument("--extra-session-seed", type=int, default=42)
    p.add_argument("--region-init", type=Path, default=REGION_INIT)
    p.add_argument("--fine-checkpoint", type=Path, default=FINE_CKPT)
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not RUN1_H5.is_file():
        raise FileNotFoundError(RUN1_H5)
    if not RUN2_H5.is_file():
        raise FileNotFoundError(RUN2_H5)
    if not args.region_init.is_file():
        raise FileNotFoundError(args.region_init)
    if not args.fine_checkpoint.is_file():
        raise FileNotFoundError(args.fine_checkpoint)

    out_root = args.output_root.resolve()
    split_dir = out_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    aug_list = split_dir / "run2_aug_sessions.txt"
    holdout_list = split_dir / "run2_holdout_sessions.txt"

    with h5py.File(RUN2_H5, "r") as f:
        run2_sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
    aug_sessions, holdout_sessions = sample_sessions_by_frac(
        run2_sessions,
        frac=float(args.extra_session_frac),
        seed=int(args.extra_session_seed),
    )
    _write_session_list(aug_list, aug_sessions)
    _write_session_list(holdout_list, holdout_sessions)
    _assert_no_session_overlap(aug_list, holdout_list)
    print(
        f"Run2 split seed={args.extra_session_seed} frac={args.extra_session_frac}: "
        f"aug_sessions={len(aug_sessions)} holdout_sessions={len(holdout_sessions)}",
        flush=True,
    )

    ft_dir = out_root / "r1_plus_run2_10"
    ft_ckpt = ft_dir / "best_model.pth"
    if args.skip_train:
        if not ft_ckpt.is_file():
            raise FileNotFoundError(f"--skip-train 但缺少 {ft_ckpt}")
        print(f"[skip-train] {ft_ckpt}", flush=True)
    elif ft_ckpt.is_file() and not args.force:
        print(f"[skip-train] exists {ft_ckpt}", flush=True)
    else:
        _run(
            [
                str(PYTHON),
                str(REGION_TRAIN),
                "--h5-path",
                str(RUN1_H5),
                "--extra-h5",
                str(RUN2_H5),
                "--extra-session-list",
                str(aug_list),
                "--resume-checkpoint",
                str(args.region_init),
                "--output-dir",
                str(ft_dir),
                "--epochs",
                "50",
                "--early-stop-patience",
                "10",
                "--lr",
                "1e-5",
                "--batch-size",
                "128",
                "--dropout",
                "0.1",
                "--base-channels",
                "64",
                "--num-layers",
                "3",
                "--pool-mode",
                "attention",
                "--neighbor-smooth",
                "0.2",
                "--spec-augment-prob",
                "0.5",
                "--feature-noise-std",
                "0.0",
                "--device",
                "cuda:0",
                "--num-workers",
                "4",
            ]
        )

    def _eval(exp_id: str, region_ckpt: Path, description: str) -> dict:
        eval_dir = out_root / "eval" / exp_id
        eval_csv = eval_dir / "two_stage_rmse.csv"
        if args.skip_eval and not eval_csv.is_file():
            raise FileNotFoundError(f"--skip-eval 但缺少 {eval_csv}")
        if eval_csv.is_file() and (args.skip_eval or not args.force):
            print(f"[skip-eval] {exp_id}: {eval_csv}", flush=True)
        else:
            _run(
                [
                    str(PYTHON),
                    str(EVAL_SCRIPT),
                    "--h5-path",
                    str(RUN2_H5),
                    "--region-checkpoint",
                    str(region_ckpt),
                    "--fine-checkpoint",
                    str(args.fine_checkpoint),
                    "--output-dir",
                    str(eval_dir),
                    "--session-list",
                    str(holdout_list),
                    "--region-topk",
                    "3",
                    "--batch-size",
                    "128",
                    "--device",
                    "cuda:0",
                    "--no-plot",
                ]
            )
        metrics = _metrics_from_eval_csv(eval_csv)
        ckpt_m = _load_ckpt_metrics(region_ckpt)
        return {
            "exp_id": exp_id,
            "description": description,
            "region_checkpoint": str(region_ckpt),
            "fine_checkpoint": str(args.fine_checkpoint),
            "n_aug_sessions": len(aug_sessions),
            "n_holdout_sessions": len(holdout_sessions),
            "run1_val_acc": ckpt_m.get("val_acc", ""),
            "run1_best_epoch": ckpt_m.get("epoch", ""),
            "eval_csv": str(eval_csv),
            **metrics,
        }

    rows = [
        _eval(
            "r1_only_aug_neighbor",
            args.region_init,
            "Run1-only aug_neighbor on Run2 90% holdout",
        ),
        _eval(
            "r1_plus_run2_10",
            ft_ckpt,
            "finetune Run1+Run2_10pct_sess on same holdout",
        ),
    ]
    _write_rows(args.summary_csv, SUMMARY_FIELDS, rows)
    print(f"\nWrote summary: {args.summary_csv}", flush=True)

    base = rows[0]
    ft = rows[1]
    b1 = float(base["holdout_top1_acc"])
    f1 = float(ft["holdout_top1_acc"])
    rel = (f1 / b1 - 1.0) * 100.0 if b1 > 0 else float("nan")
    print(
        f"\nHoldout compare: baseline top1={b1:.4f} top3={float(base['holdout_top3_hit']):.4f} "
        f"rmse={float(base['holdout_global_rmse_topk_m']):.4f}",
        flush=True,
    )
    print(
        f"Finetune: top1={f1:.4f} ({rel:+.1f}% vs baseline) "
        f"top3={float(ft['holdout_top3_hit']):.4f} "
        f"rmse={float(ft['holdout_global_rmse_topk_m']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
