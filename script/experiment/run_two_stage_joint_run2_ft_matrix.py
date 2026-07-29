#!/usr/bin/env python3
"""联合微调小矩阵：val_aug 选模 + Region-only / oversample 对照。

协议：
- holdout 195 复用 Region 10% 划分（禁止进训/选模）
- 原 22 aug → seed=43 拆成 18 train_aug / 4 val_aug
- 热启动 ``joint_direct``

实验行：
- zeroshot：不训
- ft_run1stop：early-stop Run1（对照已有 ~0.559）
- ft_valg：early-stop val_aug
- ft_valg_reg：freeze Fine + region_lr=3e-5
- ft_valg_os3：extra-oversample 3

示例::

    python script/experiment/run_two_stage_joint_run2_ft_matrix.py --device cuda:0
    python script/experiment/run_two_stage_joint_run2_ft_matrix.py --only ft_valg,ft_valg_reg
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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
REGION_SPLIT_DIR = PROJECT_ROOT / "models/region_run2_10pct/split"
DEFAULT_JOINT_INIT = PROJECT_ROOT / "models/two_stage_joint/joint_direct"
OUT_ROOT = PROJECT_ROOT / "models/two_stage_joint_run2_10pct/matrix"
SUMMARY_CSV = (
    PROJECT_ROOT / "out/cooperative_monostatic/two_stage_joint_run2_ft_matrix_summary.csv"
)
AUG_VAL_SEED = 43
AUG_N_VAL = 4

SUMMARY_FIELDS = [
    "exp_id",
    "description",
    "select_metric",
    "region_checkpoint",
    "fine_checkpoint",
    "n_aug_train",
    "n_aug_val",
    "n_holdout",
    "best_epoch",
    "run1_val_rmse",
    "val_aug_rmse",
    "holdout_global_rmse",
    "holdout_top1",
    "holdout_top3",
    "n",
    "eval_csv",
]


@dataclass(frozen=True)
class ExpSpec:
    exp_id: str
    description: str
    train: bool
    early_stop_on: str
    freeze_fine: bool
    region_lr: float
    fine_lr: float
    oversample: int
    epochs: int
    patience: int
    use_aug_split: bool  # False → 全部 22 aug 进 train（ft_run1stop）


EXPERIMENTS: list[ExpSpec] = [
    ExpSpec(
        exp_id="zeroshot",
        description="joint_direct zeroshot on holdout",
        train=False,
        early_stop_on="run1_val",
        freeze_fine=False,
        region_lr=1e-5,
        fine_lr=1e-5,
        oversample=1,
        epochs=0,
        patience=0,
        use_aug_split=False,
    ),
    ExpSpec(
        exp_id="ft_run1stop",
        description="FT early-stop Run1 val, both towers 1e-5, all 22 aug train",
        train=True,
        early_stop_on="run1_val",
        freeze_fine=False,
        region_lr=1e-5,
        fine_lr=1e-5,
        oversample=1,
        epochs=50,
        patience=10,
        use_aug_split=False,
    ),
    ExpSpec(
        exp_id="ft_valg",
        description="FT early-stop val_aug, both towers 1e-5",
        train=True,
        early_stop_on="extra_val",
        freeze_fine=False,
        region_lr=1e-5,
        fine_lr=1e-5,
        oversample=1,
        epochs=30,
        patience=8,
        use_aug_split=True,
    ),
    ExpSpec(
        exp_id="ft_valg_reg",
        description="FT early-stop val_aug, freeze Fine, region_lr=3e-5",
        train=True,
        early_stop_on="extra_val",
        freeze_fine=True,
        region_lr=3e-5,
        fine_lr=1e-5,
        oversample=1,
        epochs=30,
        patience=8,
        use_aug_split=True,
    ),
    ExpSpec(
        exp_id="ft_valg_os3",
        description="FT early-stop val_aug, oversample=3",
        train=True,
        early_stop_on="extra_val",
        freeze_fine=False,
        region_lr=1e-5,
        fine_lr=1e-5,
        oversample=3,
        epochs=30,
        patience=8,
        use_aug_split=True,
    ),
]


def _load_joint_helpers():
    path = JOINT_TRAIN
    spec = importlib.util.spec_from_file_location("joint_train_ft_matrix_helpers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_session_list(path: Path, sessions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in sessions:
            f.write(f"{int(sid)}\n")


def _read_sessions(path: Path) -> list[int]:
    return [
        int(x.strip())
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]


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


def _load_ckpt_metrics(region_path: Path) -> dict[str, float | int]:
    if not region_path.is_file():
        return {}
    ckpt = torch.load(region_path, map_location="cpu", weights_only=False)
    out: dict[str, float | int] = {}
    for key in ("epoch", "val_global_rmse", "val_extra_rmse", "select_rmse"):
        if key in ckpt:
            out[key] = ckpt[key]
    return out


def _ensure_splits(split_dir: Path) -> tuple[Path, Path, Path, Path]:
    """返回 (aug_all, holdout, aug_train, aug_val)。"""
    split_dir.mkdir(parents=True, exist_ok=True)
    aug_all = split_dir / "run2_aug_sessions.txt"
    holdout = split_dir / "run2_holdout_sessions.txt"
    aug_train = split_dir / "run2_aug_train.txt"
    aug_val = split_dir / "run2_aug_val.txt"

    region_aug = REGION_SPLIT_DIR / "run2_aug_sessions.txt"
    region_hold = REGION_SPLIT_DIR / "run2_holdout_sessions.txt"
    if not region_aug.is_file() or not region_hold.is_file():
        raise FileNotFoundError(
            f"缺少 Region 10% split: {REGION_SPLIT_DIR} "
            "(先跑 run_region_run2_10pct_finetune.py 或手动生成)"
        )
    if not aug_all.is_file() or not holdout.is_file():
        shutil.copy2(region_aug, aug_all)
        shutil.copy2(region_hold, holdout)
    elif _read_sessions(aug_all) != _read_sessions(region_aug) or _read_sessions(
        holdout
    ) != _read_sessions(region_hold):
        raise RuntimeError(
            f"matrix split 与 Region 10% 不一致: {split_dir} vs {REGION_SPLIT_DIR}"
        )

    helpers = _load_joint_helpers()
    all_aug = _read_sessions(aug_all)
    if not aug_train.is_file() or not aug_val.is_file():
        train_s, val_s = helpers.split_session_list_train_val(
            all_aug, n_val=AUG_N_VAL, seed=AUG_VAL_SEED
        )
        _write_session_list(aug_train, train_s)
        _write_session_list(aug_val, val_s)
        print(
            f"Split aug seed={AUG_VAL_SEED}: train={len(train_s)} val={len(val_s)}",
            flush=True,
        )
    else:
        train_s = _read_sessions(aug_train)
        val_s = _read_sessions(aug_val)
        if set(train_s) & set(val_s):
            raise RuntimeError("aug train/val 有交集")
        if set(train_s) | set(val_s) != set(all_aug):
            raise RuntimeError("aug train∪val 未覆盖全部 aug sessions")
        print(
            f"Using existing aug split: train={len(train_s)} val={len(val_s)}",
            flush=True,
        )

    hold_s = set(_read_sessions(holdout))
    if set(all_aug) & hold_s:
        raise RuntimeError("aug 与 holdout 有交集")
    return aug_all, holdout, aug_train, aug_val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint Run2 FT matrix (val_aug select + recipes)"
    )
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="逗号分隔 exp_id，空=全部",
    )
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

    only = {
        x.strip()
        for x in str(args.only).split(",")
        if x.strip()
    }
    specs = [e for e in EXPERIMENTS if not only or e.exp_id in only]
    if not specs:
        raise ValueError(f"--only 无匹配实验: {args.only}")

    out_root = args.output_root.resolve()
    aug_all, holdout, aug_train, aug_val = _ensure_splits(out_root / "split")
    n_aug_all = len(_read_sessions(aug_all))
    n_aug_train = len(_read_sessions(aug_train))
    n_aug_val = len(_read_sessions(aug_val))
    n_holdout = len(_read_sessions(holdout))
    print(
        f"Split: aug_all={n_aug_all} train={n_aug_train} val={n_aug_val} "
        f"holdout={n_holdout}",
        flush=True,
    )

    rows: list[dict] = []
    for exp in specs:
        print(f"\n===== {exp.exp_id}: {exp.description} =====", flush=True)
        if exp.train:
            ft_dir = out_root / exp.exp_id
            region_ckpt = ft_dir / "best_region.pth"
            fine_ckpt = ft_dir / "best_fine.pth"
            if args.skip_train:
                if not region_ckpt.is_file() or not fine_ckpt.is_file():
                    raise FileNotFoundError(
                        f"--skip-train 但缺少 {region_ckpt} / {fine_ckpt}"
                    )
                print(f"[skip-train] {ft_dir}", flush=True)
            elif region_ckpt.is_file() and fine_ckpt.is_file() and not args.force:
                print(f"[skip-train] exists {ft_dir}", flush=True)
            else:
                train_list = aug_train if exp.use_aug_split else aug_all
                cmd = [
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
                    str(train_list),
                    "--output-dir",
                    str(ft_dir),
                    "--epochs",
                    str(exp.epochs),
                    "--early-stop-patience",
                    str(exp.patience),
                    "--early-stop-on",
                    exp.early_stop_on,
                    "--lr",
                    str(exp.fine_lr),
                    "--region-lr",
                    str(exp.region_lr),
                    "--extra-oversample",
                    str(exp.oversample),
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
                if exp.use_aug_split:
                    cmd.extend(["--extra-val-session-list", str(aug_val)])
                if exp.freeze_fine:
                    cmd.append("--freeze-fine")
                _run(cmd)
        else:
            region_ckpt = region_init
            fine_ckpt = fine_init

        eval_dir = out_root / "eval" / exp.exp_id
        eval_csv = eval_dir / "two_stage_rmse.csv"
        if args.skip_eval and not eval_csv.is_file():
            raise FileNotFoundError(f"--skip-eval 但缺少 {eval_csv}")
        if eval_csv.is_file() and (args.skip_eval or not args.force):
            print(f"[skip-eval] {exp.exp_id}: {eval_csv}", flush=True)
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
                    str(holdout),
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
        ckpt_m = _load_ckpt_metrics(region_ckpt)
        rows.append(
            {
                "exp_id": exp.exp_id,
                "description": exp.description,
                "select_metric": exp.early_stop_on if exp.train else "n/a",
                "region_checkpoint": str(region_ckpt),
                "fine_checkpoint": str(fine_ckpt),
                "n_aug_train": n_aug_train if exp.use_aug_split else n_aug_all,
                "n_aug_val": n_aug_val if exp.use_aug_split else 0,
                "n_holdout": n_holdout,
                "best_epoch": ckpt_m.get("epoch", ""),
                "run1_val_rmse": ckpt_m.get("val_global_rmse", ""),
                "val_aug_rmse": ckpt_m.get("val_extra_rmse", ""),
                "eval_csv": str(eval_csv),
                **metrics,
            }
        )

    _write_rows(args.summary_csv, SUMMARY_FIELDS, rows)
    print(f"\nWrote summary: {args.summary_csv}", flush=True)
    for row in rows:
        rmse = row.get("holdout_global_rmse", float("nan"))
        top1 = row.get("holdout_top1", float("nan"))
        print(
            f"  {row['exp_id']}: holdout_rmse={rmse} top1={top1} "
            f"epoch={row.get('best_epoch', '')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
