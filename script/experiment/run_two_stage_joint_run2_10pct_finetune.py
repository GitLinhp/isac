#!/usr/bin/env python3
"""联合双阶段训后：用 10% Run2 session 联合微调，并在 90% holdout 上对照。

流程：
1. 复用 Region 10% 的 session 划分（seed=42, frac=0.1）；若缺失则生成
2. 基线：``joint_direct`` ckpt 在 holdout 上零样本评估
3. 微调：热启动 Region+Fine，Run1 train ∪ Run2 aug，early-stop 看 Run1 val
4. 微调后同 holdout 评估，写汇总 CSV

示例::

    python script/experiment/run_two_stage_joint_run2_10pct_finetune.py
    python script/experiment/run_two_stage_joint_run2_10pct_finetune.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
import shutil
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
JOINT_TRAIN = (
    PROJECT_ROOT
    / "script/model_training/run_train_cooperative_monostatic_two_stage_joint.py"
)
EVAL_SCRIPT = (
    PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_two_stage_eval.py"
)
REGION_SPLIT_DIR = PROJECT_ROOT / "models/region_run2_10pct/split"
DEFAULT_JOINT_INIT = PROJECT_ROOT / "models/two_stage_joint/joint_direct"
OUT_ROOT = PROJECT_ROOT / "models/two_stage_joint_run2_10pct"
SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/two_stage_joint_run2_10pct_summary.csv"
)

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "region_checkpoint",
    "fine_checkpoint",
    "n_aug_sessions",
    "n_holdout_sessions",
    "run1_val_rmse",
    "run1_best_epoch",
    "holdout_global_rmse",
    "holdout_top1",
    "holdout_top3",
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
    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from cooperative_monostatic_eval_report import load_two_stage_eval_metrics

    m = load_two_stage_eval_metrics(csv_path)
    if not m:
        return {"n": 0}
    out: dict[str, float | int] = {
        "n": m.get("n", 0),
        "holdout_global_rmse": m.get("global_mean_err_m", float("nan")),
    }
    if "region_top1_acc" in m:
        out["holdout_top1"] = m["region_top1_acc"]
    if "region_topk_hit" in m:
        out["holdout_top3"] = m["region_topk_hit"]
    return out


def _load_joint_ckpt_metrics(region_path: Path) -> dict[str, float | int]:
    if not region_path.is_file():
        return {}
    ckpt = torch.load(region_path, map_location="cpu", weights_only=False)
    out: dict[str, float | int] = {}
    if "val_global_rmse" in ckpt:
        out["val_global_rmse"] = float(ckpt["val_global_rmse"])
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


def _read_sessions(path: Path) -> list[int]:
    return [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]


def _ensure_split(
    split_dir: Path,
    *,
    frac: float,
    seed: int,
) -> tuple[Path, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    aug_list = split_dir / "run2_aug_sessions.txt"
    holdout_list = split_dir / "run2_holdout_sessions.txt"
    region_aug = REGION_SPLIT_DIR / "run2_aug_sessions.txt"
    region_hold = REGION_SPLIT_DIR / "run2_holdout_sessions.txt"

    if region_aug.is_file() and region_hold.is_file():
        if not aug_list.is_file() or not holdout_list.is_file():
            shutil.copy2(region_aug, aug_list)
            shutil.copy2(region_hold, holdout_list)
            print(
                f"Reused Region 10% split from {REGION_SPLIT_DIR}",
                flush=True,
            )
        else:
            # 已存在则校验与 Region 一致（可横比）
            if _read_sessions(aug_list) != _read_sessions(region_aug) or _read_sessions(
                holdout_list
            ) != _read_sessions(region_hold):
                raise RuntimeError(
                    f"本地 split 与 Region 10% 划分不一致: {split_dir} vs {REGION_SPLIT_DIR}"
                )
            print(f"Using existing split (matches Region 10%): {split_dir}", flush=True)
    elif not aug_list.is_file() or not holdout_list.is_file():
        with h5py.File(RUN2_H5, "r") as f:
            run2_sessions = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int64)
        aug_sessions, holdout_sessions = sample_sessions_by_frac(
            run2_sessions, frac=frac, seed=seed
        )
        _write_session_list(aug_list, aug_sessions)
        _write_session_list(holdout_list, holdout_sessions)
        print(
            f"Generated Run2 split seed={seed} frac={frac}: "
            f"aug={len(aug_sessions)} holdout={len(holdout_sessions)}",
            flush=True,
        )
    else:
        print(f"Using existing split: {split_dir}", flush=True)

    _assert_no_session_overlap(aug_list, holdout_list)
    return aug_list, holdout_list


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint two-stage Run2 10% session finetune compare"
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--extra-session-frac", type=float, default=0.1)
    p.add_argument("--extra-session-seed", type=int, default=42)
    p.add_argument("--joint-init-dir", type=Path, default=DEFAULT_JOINT_INIT)
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not RUN1_H5.is_file():
        raise FileNotFoundError(RUN1_H5)
    if not RUN2_H5.is_file():
        raise FileNotFoundError(RUN2_H5)

    init_dir = args.joint_init_dir.resolve()
    region_init = init_dir / "best_region.pth"
    fine_init = init_dir / "best_fine.pth"
    if not region_init.is_file():
        raise FileNotFoundError(region_init)
    if not fine_init.is_file():
        raise FileNotFoundError(fine_init)

    out_root = args.output_root.resolve()
    aug_list, holdout_list = _ensure_split(
        out_root / "split",
        frac=float(args.extra_session_frac),
        seed=int(args.extra_session_seed),
    )
    aug_sessions = _read_sessions(aug_list)
    holdout_sessions = _read_sessions(holdout_list)
    print(
        f"Run2 split: aug_sessions={len(aug_sessions)} "
        f"holdout_sessions={len(holdout_sessions)}",
        flush=True,
    )

    ft_dir = out_root / "joint_ft"
    ft_region = ft_dir / "best_region.pth"
    ft_fine = ft_dir / "best_fine.pth"
    if args.skip_train:
        if not ft_region.is_file() or not ft_fine.is_file():
            raise FileNotFoundError(f"--skip-train 但缺少 {ft_region} / {ft_fine}")
        print(f"[skip-train] {ft_dir}", flush=True)
    elif ft_region.is_file() and ft_fine.is_file() and not args.force:
        print(f"[skip-train] exists {ft_dir}", flush=True)
    else:
        _run(
            [
                str(PYTHON),
                str(JOINT_TRAIN),
                "--region-checkpoint",
                str(region_init),
                "--fine-checkpoint",
                str(fine_init),
                "--h5-path",
                str(RUN1_H5),
                "--extra-h5",
                str(RUN2_H5),
                "--extra-session-list",
                str(aug_list),
                "--output-dir",
                str(ft_dir),
                "--epochs",
                "50",
                "--early-stop-patience",
                "10",
                "--lr",
                "1e-5",
                "--region-lr",
                "1e-5",
                "--region-ce-weight",
                "1.0",
                "--neighbor-smooth",
                "0.2",
                "--dropout",
                "0.1",
                "--batch-size",
                "128",
                "--feature-mode",
                "real_imag",
                "--range-roi",
                "0",
                "4",
                "--device",
                str(args.device),
                "--num-workers",
                "4",
            ]
        )

    def _eval(
        exp_id: str,
        region_ckpt: Path,
        fine_ckpt: Path,
        description: str,
    ) -> dict:
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
                    str(fine_ckpt),
                    "--output-dir",
                    str(eval_dir),
                    "--session-list",
                    str(holdout_list),
                    "--region-topk",
                    "3",
                    "--batch-size",
                    "128",
                    "--device",
                    str(args.device),
                    "--no-plot",
                ]
            )
        metrics = _metrics_from_eval_csv(eval_csv)
        ckpt_m = _load_joint_ckpt_metrics(region_ckpt)
        return {
            "exp_id": exp_id,
            "description": description,
            "region_checkpoint": str(region_ckpt),
            "fine_checkpoint": str(fine_ckpt),
            "n_aug_sessions": len(aug_sessions),
            "n_holdout_sessions": len(holdout_sessions),
            "run1_val_rmse": ckpt_m.get("val_global_rmse", ""),
            "run1_best_epoch": ckpt_m.get("epoch", ""),
            "eval_csv": str(eval_csv),
            **metrics,
        }

    rows = [
        _eval(
            "joint_direct_zeroshot",
            region_init,
            fine_init,
            "joint_direct zeroshot on Run2 90% holdout",
        ),
        _eval(
            "joint_ft_run2_10",
            ft_region,
            ft_fine,
            "joint finetune Run1+Run2_10pct_sess on same holdout",
        ),
    ]
    _write_rows(args.summary_csv, SUMMARY_FIELDS, rows)
    print(f"\nWrote summary: {args.summary_csv}", flush=True)

    base = rows[0]
    ft = rows[1]
    b_rmse = float(base["holdout_global_rmse"])
    f_rmse = float(ft["holdout_global_rmse"])
    b1 = float(base.get("holdout_top1", float("nan")))
    f1 = float(ft.get("holdout_top1", float("nan")))
    rel_rmse = (f_rmse / b_rmse - 1.0) * 100.0 if b_rmse > 0 else float("nan")
    print(
        f"\nHoldout compare: baseline rmse={b_rmse:.4f} top1={b1:.4f} "
        f"top3={float(base.get('holdout_top3', float('nan'))):.4f}",
        flush=True,
    )
    print(
        f"Joint FT: rmse={f_rmse:.4f} ({rel_rmse:+.1f}% vs baseline) "
        f"top1={f1:.4f} top3={float(ft.get('holdout_top3', float('nan'))):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
