#!/usr/bin/env python3
"""两阶段联合训练对照：热启动 joint / 直接联合 / 分阶段 Fine。

1. （可选）跑热启动联合 ``joint_rmse`` 与直接联合 ``joint_direct``
2. 在完整 Run2 上评估
3. 写入 ``out/cooperative_monostatic/two_stage_joint_summary.csv``

示例::

    python script/experiment/run_two_stage_joint_compare.py
    python script/experiment/run_two_stage_joint_compare.py --skip-train
    python script/experiment/run_two_stage_joint_compare.py --only-direct
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
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
JOINT_TRAIN = (
    PROJECT_ROOT
    / "script/model_training/run_train_cooperative_monostatic_two_stage_joint.py"
)
EVAL_SCRIPT = (
    PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_two_stage_eval.py"
)
REGION_INIT = (
    PROJECT_ROOT / "models/region_run2_zeroshot/region/aug_neighbor/best_model.pth"
)
FINE_INIT = PROJECT_ROOT / "models/two_stage_tune/fine/fine_lr1e4/best_model.pth"
JOINT_OUT = PROJECT_ROOT / "models/two_stage_joint/joint_rmse"
JOINT_DIRECT_OUT = PROJECT_ROOT / "models/two_stage_joint/joint_direct"
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/two_stage_joint_summary.csv"
EVAL_ROOT = PROJECT_ROOT / "models/two_stage_joint/eval"

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "region_checkpoint",
    "fine_checkpoint",
    "run1_val_global_rmse",
    "run1_best_epoch",
    "run2_top1_acc",
    "run2_top3_hit",
    "run2_global_rmse_topk_m",
    "run2_global_rmse_top1_m",
    "run2_oracle_region_rmse_m",
    "n",
    "eval_csv",
]


def _run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _metrics_from_eval_csv(
    csv_path: Path,
    *,
    diagnostics_path: Path | None = None,
) -> dict[str, float | int]:
    """从主 CSV（7 列）与可选 Region sidecar 汇总指标。"""
    import sys

    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from cooperative_monostatic_eval_report import load_two_stage_eval_metrics

    m = load_two_stage_eval_metrics(csv_path, diagnostics_path=diagnostics_path)
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


def _load_joint_meta(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {}
    meta = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, float | int] = {}
    if "val_global_rmse" in meta:
        out["val_global_rmse"] = float(meta["val_global_rmse"])
    if "epoch" in meta:
        out["epoch"] = int(meta["epoch"])
    return out


def _eval_pair(
    *,
    exp_id: str,
    description: str,
    region_ckpt: Path,
    fine_ckpt: Path,
    force: bool,
    skip_eval: bool,
    run1_val_rmse: float | str = "",
    run1_epoch: int | str = "",
) -> dict:
    out_dir = EVAL_ROOT / exp_id
    out_csv = out_dir / "two_stage_rmse.csv"
    if skip_eval and not out_csv.is_file():
        raise FileNotFoundError(f"--skip-eval 但缺少 {out_csv}")
    if out_csv.is_file() and (skip_eval or not force):
        print(f"[skip-eval] {exp_id}: {out_csv}", flush=True)
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
                str(out_dir),
                "--region-topk",
                "3",
                "--device",
                "cuda:0",
                "--batch-size",
                "128",
                "--no-plot",
            ]
        )
    metrics = _metrics_from_eval_csv(out_csv)
    return {
        "exp_id": exp_id,
        "description": description,
        "region_checkpoint": str(region_ckpt),
        "fine_checkpoint": str(fine_ckpt),
        "run1_val_global_rmse": run1_val_rmse,
        "run1_best_epoch": run1_epoch,
        "eval_csv": str(out_csv),
        **metrics,
    }


def _train_warm_start(args: argparse.Namespace) -> None:
    joint_region = args.joint_output_dir / "best_region.pth"
    joint_fine = args.joint_output_dir / "best_fine.pth"
    if args.skip_train:
        if not joint_region.is_file() or not joint_fine.is_file():
            raise FileNotFoundError(
                f"--skip-train 但缺少 joint ckpt: {joint_region} / {joint_fine}"
            )
        print(f"[skip-train] joint: {args.joint_output_dir}", flush=True)
        return
    if joint_region.is_file() and joint_fine.is_file() and not args.force:
        print(f"[skip-train] joint exists: {args.joint_output_dir}", flush=True)
        return
    if not args.region_init.is_file():
        raise FileNotFoundError(f"Region init 不存在: {args.region_init}")
    cmd = [
        str(PYTHON),
        str(JOINT_TRAIN),
        "--h5-path",
        str(RUN1_H5),
        "--region-checkpoint",
        str(args.region_init),
        "--output-dir",
        str(args.joint_output_dir),
        "--epochs",
        str(args.epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--batch-size",
        "128",
        "--lr",
        "1e-4",
        "--region-lr",
        "1e-5",
        "--dropout",
        "0.1",
        "--region-topk",
        "3",
        "--device",
        "cuda:0",
        "--num-workers",
        "4",
    ]
    if args.fine_init.is_file():
        cmd.extend(["--fine-checkpoint", str(args.fine_init)])
    _run(cmd)


def _train_direct(args: argparse.Namespace) -> None:
    joint_region = args.direct_output_dir / "best_region.pth"
    joint_fine = args.direct_output_dir / "best_fine.pth"
    if args.skip_train:
        if not joint_region.is_file() or not joint_fine.is_file():
            raise FileNotFoundError(
                f"--skip-train 但缺少 direct ckpt: {joint_region} / {joint_fine}"
            )
        print(f"[skip-train] direct: {args.direct_output_dir}", flush=True)
        return
    if joint_region.is_file() and joint_fine.is_file() and not args.force:
        print(f"[skip-train] direct exists: {args.direct_output_dir}", flush=True)
        return
    _run(
        [
            str(PYTHON),
            str(JOINT_TRAIN),
            "--h5-path",
            str(RUN1_H5),
            "--output-dir",
            str(args.direct_output_dir),
            "--epochs",
            str(args.epochs),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--batch-size",
            "128",
            "--lr",
            "1e-4",
            "--region-lr",
            "1e-4",
            "--region-ce-weight",
            "1.0",
            "--neighbor-smooth",
            "0.2",
            "--dropout",
            "0.1",
            "--region-topk",
            "3",
            "--device",
            "cuda:0",
            "--num-workers",
            "4",
        ]
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint two-stage RMSE compare matrix")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--only-direct",
        action="store_true",
        help="只跑直接联合（跳过热启动 joint 与 tf_fine）",
    )
    p.add_argument(
        "--skip-direct",
        action="store_true",
        help="跳过直接联合实验",
    )
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--region-init", type=Path, default=REGION_INIT)
    p.add_argument("--fine-init", type=Path, default=FINE_INIT)
    p.add_argument("--joint-output-dir", type=Path, default=JOINT_OUT)
    p.add_argument("--direct-output-dir", type=Path, default=JOINT_DIRECT_OUT)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not RUN1_H5.is_file():
        raise FileNotFoundError(f"Run1 H5 不存在: {RUN1_H5}")
    if not RUN2_H5.is_file():
        raise FileNotFoundError(f"Run2 H5 不存在: {RUN2_H5}")

    rows: list[dict] = []

    if not args.only_direct:
        _train_warm_start(args)
        meta = _load_joint_meta(args.joint_output_dir / "joint_meta.pth")
        if args.fine_init.is_file() and args.region_init.is_file():
            rows.append(
                _eval_pair(
                    exp_id="tf_fine",
                    description="frozen region + serial fine (region softmax)",
                    region_ckpt=args.region_init,
                    fine_ckpt=args.fine_init,
                    force=args.force,
                    skip_eval=args.skip_eval,
                )
            )
        rows.append(
            _eval_pair(
                exp_id="joint_rmse",
                description="warm-start e2e joint global RMSE",
                region_ckpt=args.joint_output_dir / "best_region.pth",
                fine_ckpt=args.joint_output_dir / "best_fine.pth",
                force=args.force,
                skip_eval=args.skip_eval,
                run1_val_rmse=meta.get("val_global_rmse", ""),
                run1_epoch=meta.get("epoch", ""),
            )
        )

    if not args.skip_direct:
        _train_direct(args)
        dmeta = _load_joint_meta(args.direct_output_dir / "joint_meta.pth")
        rows.append(
            _eval_pair(
                exp_id="joint_direct",
                description="direct from-scratch joint RMSE+RegionCE",
                region_ckpt=args.direct_output_dir / "best_region.pth",
                fine_ckpt=args.direct_output_dir / "best_fine.pth",
                force=args.force,
                skip_eval=args.skip_eval,
                run1_val_rmse=dmeta.get("val_global_rmse", ""),
                run1_epoch=dmeta.get("epoch", ""),
            )
        )

    _write_rows(args.summary_csv, SUMMARY_FIELDS, rows)
    print(f"\nWrote summary: {args.summary_csv}", flush=True)
    for r in rows:
        print(
            f"[summary] {r['exp_id']} | "
            f"run2_top1={r.get('run2_top1_acc')} | "
            f"run2_top3={r.get('run2_top3_hit')} | "
            f"rmse={r.get('run2_global_rmse_topk_m')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
