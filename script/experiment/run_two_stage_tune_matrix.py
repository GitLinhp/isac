#!/usr/bin/env python3
"""两阶段（Region 4×4 + Fine）初步训练与调优矩阵。

1. 训练若干 Region / Fine 超参配置
2. 用 Run2 H5 做两阶段联合评估
3. 汇总 accuracy / RMSE 到 CSV（每完成一实验即 flush）

示例::

    python script/experiment/run_two_stage_tune_matrix.py
    python script/experiment/run_two_stage_tune_matrix.py --only region_baseline,fine_baseline
    python script/experiment/run_two_stage_tune_matrix.py --skip-train
"""

from __future__ import annotations

import argparse
import csv
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
FINE_TRAIN = (
    PROJECT_ROOT
    / "script/model_training/run_train_cooperative_monostatic_fine_cnn.py"
)
EVAL_SCRIPT = (
    PROJECT_ROOT / "script/experiment/run_cooperative_monostatic_two_stage_eval.py"
)
TUNE_ROOT = PROJECT_ROOT / "models/two_stage_tune"
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/two_stage_tune_summary.csv"
REGION_SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/two_stage_region_tune_summary.csv"
)
FINE_SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/two_stage_fine_tune_summary.csv"
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
    "--spec-augment-prob",
    "0.5",
    "--num-workers",
    "4",
    "--device",
    "cuda:0",
]

REGION_SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "base_channels",
    "lr",
    "dropout",
    "pool_mode",
    "class_weight",
    "checkpoint",
    "val_acc",
    "epoch",
]

FINE_SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "base_channels",
    "lr",
    "dropout",
    "pool_mode",
    "region_checkpoint",
    "checkpoint",
    "val_global_rmse",
    "epoch",
]

COMBO_SUMMARY_FIELDS = [
    "combo_id",
    "region_exp_id",
    "fine_exp_id",
    "description",
    "region_checkpoint",
    "fine_checkpoint",
    "region_top1_acc",
    "global_mean_err_m",
    "global_mean_err_median_m",
    "global_mean_err_p90_m",
    "oracle_region_mean_err_m",
    "mean_err_when_region_correct_m",
    "mean_err_when_region_wrong_m",
    "inner_mean_err_m",
    "outer_mean_err_m",
    "n",
    "output_csv",
]


@dataclass(frozen=True)
class RegionExp:
    exp_id: str
    description: str
    base_channels: int = 64
    lr: float = 5e-5
    dropout: float = 0.3
    pool_mode: str = "attention"
    class_weight: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_dir(self) -> Path:
        return TUNE_ROOT / "region" / self.exp_id

    @property
    def checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_args(self) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--output-dir",
            str(self.output_dir),
            "--base-channels",
            str(self.base_channels),
            "--lr",
            str(self.lr),
            "--dropout",
            str(self.dropout),
            "--pool-mode",
            self.pool_mode,
        ]
        if self.class_weight:
            args.append("--class-weight")
        args.extend(self.extra_args)
        return args


@dataclass(frozen=True)
class FineExp:
    exp_id: str
    description: str
    base_channels: int = 64
    lr: float = 5e-5
    dropout: float = 0.3
    pool_mode: str = "attention"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_dir(self) -> Path:
        return TUNE_ROOT / "fine" / self.exp_id

    @property
    def checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_args(self, *, region_checkpoint: Path) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--output-dir",
            str(self.output_dir),
            "--base-channels",
            str(self.base_channels),
            "--lr",
            str(self.lr),
            "--dropout",
            str(self.dropout),
            "--pool-mode",
            self.pool_mode,
            "--region-checkpoint",
            str(region_checkpoint),
        ]
        args.extend(self.extra_args)
        return args


REGION_EXPERIMENTS: tuple[RegionExp, ...] = (
    RegionExp(
        exp_id="region_baseline",
        description="bc64 lr5e-5 attn dropout0.3",
    ),
    RegionExp(
        exp_id="region_lr1e4",
        description="bc64 lr1e-4 attn",
        lr=1e-4,
    ),
    RegionExp(
        exp_id="region_lr2e5",
        description="bc64 lr2e-5 attn",
        lr=2e-5,
    ),
    RegionExp(
        exp_id="region_bc32",
        description="bc32 lr5e-5 attn",
        base_channels=32,
    ),
    RegionExp(
        exp_id="region_class_weight",
        description="bc64 + inverse-freq class weight",
        class_weight=True,
    ),
    RegionExp(
        exp_id="region_gap",
        description="bc64 pool=gap",
        pool_mode="gap",
    ),
    RegionExp(
        exp_id="region_drop01",
        description="bc64 dropout=0.1",
        dropout=0.1,
    ),
)

FINE_EXPERIMENTS: tuple[FineExp, ...] = (
    FineExp(
        exp_id="fine_baseline",
        description="bc64 lr5e-5 attn",
    ),
    FineExp(
        exp_id="fine_lr1e4",
        description="bc64 lr1e-4",
        lr=1e-4,
    ),
    FineExp(
        exp_id="fine_lr2e5",
        description="bc64 lr2e-5",
        lr=2e-5,
    ),
    FineExp(
        exp_id="fine_bc32",
        description="bc32 lr5e-5",
        base_channels=32,
    ),
    FineExp(
        exp_id="fine_gap",
        description="bc64 pool=gap",
        pool_mode="gap",
    ),
    FineExp(
        exp_id="fine_drop01",
        description="bc64 dropout=0.1",
        dropout=0.1,
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
    if "val_local_rmse" in ckpt:
        out["val_local_rmse"] = float(ckpt["val_local_rmse"])
    if "val_global_rmse" in ckpt:
        out["val_global_rmse"] = float(ckpt["val_global_rmse"])
    if "epoch" in ckpt:
        out["epoch"] = int(ckpt["epoch"])
    return out


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

    return load_two_stage_eval_metrics(csv_path)


def _train_region(exp: RegionExp, *, force: bool) -> dict:
    if exp.checkpoint.is_file() and not force:
        print(f"[skip-train] {exp.exp_id}: {exp.checkpoint}", flush=True)
    else:
        _run([str(PYTHON), str(REGION_TRAIN), *exp.train_args()])
    metrics = _load_ckpt_metrics(exp.checkpoint)
    return {
        "exp_id": exp.exp_id,
        "description": exp.description,
        "base_channels": exp.base_channels,
        "lr": exp.lr,
        "dropout": exp.dropout,
        "pool_mode": exp.pool_mode,
        "class_weight": int(exp.class_weight),
        "checkpoint": str(exp.checkpoint),
        "val_acc": metrics.get("val_acc", ""),
        "epoch": metrics.get("epoch", ""),
    }


def _train_fine(
    exp: FineExp,
    *,
    region_checkpoint: Path,
    force: bool,
) -> dict:
    if exp.checkpoint.is_file() and not force:
        print(f"[skip-train] {exp.exp_id}: {exp.checkpoint}", flush=True)
    else:
        if not region_checkpoint.is_file():
            raise FileNotFoundError(
                f"Fine 训练需要 Region checkpoint: {region_checkpoint}"
            )
        _run(
            [
                str(PYTHON),
                str(FINE_TRAIN),
                *exp.train_args(region_checkpoint=region_checkpoint),
            ]
        )
    metrics = _load_ckpt_metrics(exp.checkpoint)
    return {
        "exp_id": exp.exp_id,
        "description": exp.description,
        "base_channels": exp.base_channels,
        "lr": exp.lr,
        "dropout": exp.dropout,
        "pool_mode": exp.pool_mode,
        "region_checkpoint": str(region_checkpoint),
        "checkpoint": str(exp.checkpoint),
        "val_global_rmse": metrics.get("val_global_rmse", ""),
        "epoch": metrics.get("epoch", ""),
    }


def _eval_combo(
    region_exp: RegionExp,
    fine_exp: FineExp,
    *,
    force: bool,
) -> dict:
    combo_id = f"{region_exp.exp_id}__{fine_exp.exp_id}"
    out_dir = TUNE_ROOT / "eval" / combo_id
    out_csv = out_dir / "two_stage_rmse.csv"
    if out_csv.is_file() and not force:
        print(f"[skip-eval] {combo_id}", flush=True)
    else:
        _run(
            [
                str(PYTHON),
                str(EVAL_SCRIPT),
                "--h5-path",
                str(RUN2_H5),
                "--region-checkpoint",
                str(region_exp.checkpoint),
                "--fine-checkpoint",
                str(fine_exp.checkpoint),
                "--output-dir",
                str(out_dir),
                "--device",
                "cuda:0",
                "--batch-size",
                "128",
                "--no-plot",
            ]
        )
    metrics = _metrics_from_eval_csv(out_csv)
    return {
        "combo_id": combo_id,
        "region_exp_id": region_exp.exp_id,
        "fine_exp_id": fine_exp.exp_id,
        "description": f"{region_exp.description} + {fine_exp.description}",
        "region_checkpoint": str(region_exp.checkpoint),
        "fine_checkpoint": str(fine_exp.checkpoint),
        "output_csv": str(out_csv),
        **metrics,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage region+fine tune matrix")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="逗号分隔 exp_id 子集（region_* / fine_*）",
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="即使已有 checkpoint / eval CSV 也重跑",
    )
    p.add_argument(
        "--eval-mode",
        choices=["matched", "best_cross", "full"],
        default="best_cross",
        help=(
            "matched: 同名超参对；best_cross: 最优 region×各 fine + 各 region×最优 fine；"
            "full: 全笛卡尔积"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    only: set[str] | None = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    region_exps = [
        e for e in REGION_EXPERIMENTS if only is None or e.exp_id in only
    ]
    fine_exps = [
        e for e in FINE_EXPERIMENTS if only is None or e.exp_id in only
    ]
    # 若 only 只点了 region 或 fine，另一侧仍用全集做交叉评估
    if only is not None:
        if not region_exps:
            region_exps = list(REGION_EXPERIMENTS)
        if not fine_exps:
            fine_exps = list(FINE_EXPERIMENTS)

    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    region_rows: list[dict] = []
    fine_rows: list[dict] = []

    if not args.skip_train:
        print("=== Train Region CNNs ===", flush=True)
        for exp in region_exps:
            row = _train_region(exp, force=args.force)
            region_rows.append(row)
            _write_rows(REGION_SUMMARY_CSV, REGION_SUMMARY_FIELDS, region_rows)
            print(
                f"[region] {exp.exp_id} val_acc={row.get('val_acc')} "
                f"epoch={row.get('epoch')}",
                flush=True,
            )

        # Fine 串联训练：绑定当前最优 Region（按 val_acc）
        def _best_region_for_fine() -> RegionExp:
            scored: list[tuple[float, RegionExp]] = []
            for exp in region_exps:
                m = _load_ckpt_metrics(exp.checkpoint)
                scored.append((float(m.get("val_acc", -1.0)), exp))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        region_for_fine = _best_region_for_fine()
        print(
            f"=== Train Fine CNNs (serial region={region_for_fine.exp_id}) ===",
            flush=True,
        )
        for exp in fine_exps:
            row = _train_fine(
                exp,
                region_checkpoint=region_for_fine.checkpoint,
                force=args.force,
            )
            fine_rows.append(row)
            _write_rows(FINE_SUMMARY_CSV, FINE_SUMMARY_FIELDS, fine_rows)
            print(
                f"[fine] {exp.exp_id} val_global={row.get('val_global_rmse')} "
                f"epoch={row.get('epoch')}",
                flush=True,
            )
    else:
        for exp in region_exps:
            region_rows.append(_train_region(exp, force=False))
        # skip-train：尽量从已有 fine ckpt meta 读 region_checkpoint
        for exp in fine_exps:
            region_ckpt = region_exps[0].checkpoint
            if exp.checkpoint.is_file():
                ckpt = torch.load(
                    exp.checkpoint, map_location="cpu", weights_only=False
                )
                meta_rc = ckpt.get("region_checkpoint")
                if meta_rc:
                    region_ckpt = Path(str(meta_rc))
            fine_rows.append(
                _train_fine(exp, region_checkpoint=region_ckpt, force=False)
            )
        _write_rows(REGION_SUMMARY_CSV, REGION_SUMMARY_FIELDS, region_rows)
        _write_rows(FINE_SUMMARY_CSV, FINE_SUMMARY_FIELDS, fine_rows)

    # 选最优（按 val 指标）
    def _best_region() -> RegionExp:
        scored: list[tuple[float, RegionExp]] = []
        for exp in region_exps:
            m = _load_ckpt_metrics(exp.checkpoint)
            scored.append((float(m.get("val_acc", -1.0)), exp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _best_fine() -> FineExp:
        scored: list[tuple[float, FineExp]] = []
        for exp in fine_exps:
            m = _load_ckpt_metrics(exp.checkpoint)
            # 越小越好（全局 RMSE）
            scored.append((float(m.get("val_global_rmse", 1e9)), exp))
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    best_region = _best_region()
    best_fine = _best_fine()
    print(
        f"Best region: {best_region.exp_id} | Best fine: {best_fine.exp_id}",
        flush=True,
    )

    combos: list[tuple[RegionExp, FineExp]] = []
    if args.eval_mode == "matched":
        # 仅同后缀配对（baseline↔baseline 等）
        fine_by_suffix = {
            e.exp_id.replace("fine_", ""): e for e in fine_exps
        }
        for rexp in region_exps:
            suffix = rexp.exp_id.replace("region_", "")
            fexp = fine_by_suffix.get(suffix)
            if fexp is not None:
                combos.append((rexp, fexp))
        if not combos:
            combos.append((best_region, best_fine))
    elif args.eval_mode == "full":
        for rexp in region_exps:
            for fexp in fine_exps:
                combos.append((rexp, fexp))
    else:  # best_cross
        seen: set[str] = set()
        for fexp in fine_exps:
            key = f"{best_region.exp_id}__{fexp.exp_id}"
            if key not in seen:
                combos.append((best_region, fexp))
                seen.add(key)
        for rexp in region_exps:
            key = f"{rexp.exp_id}__{best_fine.exp_id}"
            if key not in seen:
                combos.append((rexp, best_fine))
                seen.add(key)

    combo_rows: list[dict] = []
    if not args.skip_eval:
        print("=== Two-stage eval ===", flush=True)
        for rexp, fexp in combos:
            if not rexp.checkpoint.is_file() or not fexp.checkpoint.is_file():
                print(
                    f"[skip-eval missing ckpt] {rexp.exp_id} + {fexp.exp_id}",
                    flush=True,
                )
                continue
            row = _eval_combo(rexp, fexp, force=args.force)
            combo_rows.append(row)
            _write_rows(SUMMARY_CSV, COMBO_SUMMARY_FIELDS, combo_rows)
            print(
                f"[eval] {row['combo_id']} acc={row['region_top1_acc']:.4f} "
                f"mean_err={row['global_mean_err_m']:.4f} "
                f"oracle={row['oracle_region_mean_err_m']:.4f}",
                flush=True,
            )
    else:
        for rexp, fexp in combos:
            out_csv = TUNE_ROOT / "eval" / f"{rexp.exp_id}__{fexp.exp_id}" / "two_stage_rmse.csv"
            if out_csv.is_file():
                metrics = _metrics_from_eval_csv(out_csv)
                combo_rows.append(
                    {
                        "combo_id": f"{rexp.exp_id}__{fexp.exp_id}",
                        "region_exp_id": rexp.exp_id,
                        "fine_exp_id": fexp.exp_id,
                        "description": f"{rexp.description} + {fexp.description}",
                        "region_checkpoint": str(rexp.checkpoint),
                        "fine_checkpoint": str(fexp.checkpoint),
                        "output_csv": str(out_csv),
                        **metrics,
                    }
                )
        _write_rows(SUMMARY_CSV, COMBO_SUMMARY_FIELDS, combo_rows)

    print("\n=== Summary paths ===", flush=True)
    print(f"Region: {REGION_SUMMARY_CSV}", flush=True)
    print(f"Fine:   {FINE_SUMMARY_CSV}", flush=True)
    print(f"Combo:  {SUMMARY_CSV}", flush=True)

    if combo_rows:
        best = min(combo_rows, key=lambda r: float(r["global_mean_err_m"]))
        print(
            f"Best combo by global mean error: {best['combo_id']} "
            f"mean_err={best['global_mean_err_m']:.4f} "
            f"acc={best['region_top1_acc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
