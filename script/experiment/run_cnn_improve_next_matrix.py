#!/usr/bin/env python3
"""下一轮性能改善对照矩阵（trim / Outer / hparam / aux / xattn / aug）。

统一协议：Run1 训练、Run2 全帧、--no-filter-outliers。
主指标 Global RMSE；若 Global 差 ≤0.005 则 Outer 更低者胜。

实验覆盖：
  - hist_recipe / this_run + lr×jitter 2×2
  - trim_best_rmse frac 0.8 / 0.9（本次配方）
  - corner-weight=4、geom-only、aux-range=0.3
  - plain_xattn（this_run + cross-attn，无 geom）
  - aug_off / aug_strong / aug_spec_only / aug_noise_only（特征增强消融）
  - cpi_aug_mild / cpi_aug_strong（raw CPI 幅度缩放+复噪声）
  - feat_complex_roi / feat_legacy_4ch / feat_real_imag_cnorm（特征预处理）
  - arch_range_geom（geom-only 交会）/ arch_session_median|ema（评测时序平滑）

示例::

    python script/experiment/run_cnn_improve_next_matrix.py
    python script/experiment/run_cnn_improve_next_matrix.py --only trim_0.8,trim_0.9
    python script/experiment/run_cnn_improve_next_matrix.py --only plain_xattn
    python script/experiment/run_cnn_improve_next_matrix.py --only aug_off,aug_strong,aug_spec_only,aug_noise_only
    python script/experiment/run_cnn_improve_next_matrix.py --only cpi_aug_mild,cpi_aug_strong
    python script/experiment/run_cnn_improve_next_matrix.py --only feat_complex_roi,feat_legacy_4ch,feat_real_imag_cnorm
    python script/experiment/run_cnn_improve_next_matrix.py --only arch_range_geom,arch_session_median,arch_session_ema
    python script/experiment/run_cnn_improve_next_matrix.py --only arch_light_tf
    python script/experiment/run_cnn_improve_next_matrix.py --skip-train
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
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_improve_next_summary.csv"
BEST_TXT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_improve_next_best.txt"
MATRIX_ROOT = PROJECT_ROOT / "models/cnn_improve_next"
EVAL_ROOT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_improve_next"

THIS_RUN_CKPT = (
    PROJECT_ROOT / "models/cnn_deploy_strict_roi4/best_model.pth"
)
HIST_CATCHUP_CKPT = (
    PROJECT_ROOT / "models/cnn_hist_catchup/hist_outer3_session/best_model.pth"
)
FEAT_COMPLEX_ROI_CKPT = (
    PROJECT_ROOT / "models/cnn_improve_next/feat_complex_roi/best_model.pth"
)
AUG_SPEC_ONLY_CKPT = (
    PROJECT_ROOT / "models/cnn_improve_next/aug_spec_only/best_model.pth"
)

GLOBAL_TIE_EPS = 0.005

COMMON_TRAIN = [
    "--epochs",
    "100",
    "--batch-size",
    "128",
    "--weight-decay",
    "1e-4",
    "--lr-scheduler-patience",
    "5",
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
    "geom_only",
    "cross_attn",
    "aux_range",
    "aux_range_weight",
    "trim_best_rmse",
    "trim_best_frac",
    "lr",
    "label_jitter_m",
    "feature_noise_std",
    "spec_augment_prob",
    "cpi_aug",
    "cpi_amp_scale",
    "cpi_complex_noise_std",
    "feature_mode",
    "feature_norm",
    "backbone",
    "session_smooth",
    "session_smooth_alpha",
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
    geom_only: bool = False
    cross_attn: bool = False
    aux_range: bool = False
    aux_range_weight: float = 0.3
    trim_best_rmse: bool = False
    trim_best_frac: float = 0.8
    lr: float = 3e-4
    label_jitter_m: float = 0.02
    feature_noise_std: float = 0.02
    spec_augment_prob: float = 0.3
    cpi_aug: bool = False
    cpi_amp_scale: float = 0.2
    cpi_complex_noise_std: float = 0.02
    feature_mode: str = "real_imag"
    feature_norm: str = "none"
    backbone: str = "cnn"
    session_smooth: str = "none"
    session_smooth_alpha: float = 0.3
    early_stop_patience: int = 15
    train: bool = True
    external_checkpoint: Path | None = None

    @property
    def output_dir(self) -> Path:
        return MATRIX_ROOT / self.exp_id

    def resolved_checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_cli_args(self) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--feature-mode",
            self.feature_mode,
            "--feature-norm",
            self.feature_norm,
            "--model-type",
            self.backbone,
            "--side-weight",
            str(self.side_weight),
            "--corner-weight",
            str(self.corner_weight),
            "--lr",
            str(self.lr),
            "--feature-noise-std",
            str(self.feature_noise_std),
            "--spec-augment-prob",
            str(self.spec_augment_prob),
            "--early-stop-patience",
            str(self.early_stop_patience),
            "--trim-best-frac",
            str(self.trim_best_frac),
        ]
        if self.label_jitter_m <= 0.0:
            args.append("--no-label-jitter")
        else:
            args.extend(["--label-jitter-m", str(self.label_jitter_m)])
        if self.geom_residual:
            args.append("--geom-residual")
        else:
            args.append("--no-geom-residual")
        if self.geom_only:
            args.append("--geom-only")
        else:
            args.append("--no-geom-only")
        if self.cross_attn:
            args.append("--cross-attn")
        else:
            args.append("--no-cross-attn")
        if self.aux_range:
            args.extend(["--aux-range", "--aux-range-weight", str(self.aux_range_weight)])
        else:
            args.append("--no-aux-range")
        if self.trim_best_rmse:
            args.append("--trim-best-rmse")
        else:
            args.append("--no-trim-best-rmse")
        if self.cpi_aug:
            args.extend(
                [
                    "--cpi-aug",
                    "--cpi-amp-scale",
                    str(self.cpi_amp_scale),
                    "--cpi-complex-noise-std",
                    str(self.cpi_complex_noise_std),
                ]
            )
        else:
            args.append("--no-cpi-aug")
        return args

    def meta_row(self) -> dict[str, str | float | int]:
        return {
            "exp_id": self.exp_id,
            "description": self.description,
            "center_weight": 1.0,
            "side_weight": self.side_weight,
            "corner_weight": self.corner_weight,
            "geom_residual": int(self.geom_residual),
            "geom_only": int(self.geom_only),
            "cross_attn": int(self.cross_attn),
            "aux_range": int(self.aux_range),
            "aux_range_weight": self.aux_range_weight if self.aux_range else 0.0,
            "trim_best_rmse": int(self.trim_best_rmse),
            "trim_best_frac": self.trim_best_frac if self.trim_best_rmse else 1.0,
            "lr": self.lr,
            "label_jitter_m": self.label_jitter_m,
            "feature_noise_std": self.feature_noise_std,
            "spec_augment_prob": self.spec_augment_prob,
            "cpi_aug": int(self.cpi_aug),
            "cpi_amp_scale": self.cpi_amp_scale if self.cpi_aug else 0.0,
            "cpi_complex_noise_std": (
                self.cpi_complex_noise_std if self.cpi_aug else 0.0
            ),
            "feature_mode": self.feature_mode,
            "feature_norm": self.feature_norm,
            "backbone": self.backbone,
            "session_smooth": self.session_smooth,
            "session_smooth_alpha": (
                self.session_smooth_alpha if self.session_smooth == "ema" else 0.0
            ),
            "early_stop_patience": self.early_stop_patience,
        }


EXPERIMENTS: tuple[Experiment, ...] = (
    # --- hparam 2x2 + anchors ---
    Experiment(
        exp_id="hist_recipe",
        description="hist: lr=5e-4 jitter=0.05 es=10 w=1/3/3 no-geom",
        lr=5e-4,
        label_jitter_m=0.05,
        early_stop_patience=10,
        train=False,
        external_checkpoint=HIST_CATCHUP_CKPT,
    ),
    Experiment(
        exp_id="this_run",
        description="user run: lr=3e-4 jitter=0.02 es=15 w=1/3/3 (reuse ckpt)",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        train=False,
        external_checkpoint=THIS_RUN_CKPT,
    ),
    Experiment(
        exp_id="lr5e4_j02",
        description="hparam: lr=5e-4 jitter=0.02 es=15",
        lr=5e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="lr3e4_j05",
        description="hparam: lr=3e-4 jitter=0.05 es=15",
        lr=3e-4,
        label_jitter_m=0.05,
        early_stop_patience=15,
    ),
    # --- trim (this_run recipe) ---
    Experiment(
        exp_id="trim_0.8",
        description="this_run + trim-best-rmse frac=0.8",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        trim_best_rmse=True,
        trim_best_frac=0.8,
    ),
    Experiment(
        exp_id="trim_0.9",
        description="this_run + trim-best-rmse frac=0.9",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        trim_best_rmse=True,
        trim_best_frac=0.9,
    ),
    # --- Outer levers (this_run base) ---
    Experiment(
        exp_id="corner4",
        description="this_run + corner-weight=4",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        corner_weight=4.0,
    ),
    Experiment(
        exp_id="geom_plain",
        description="this_run + geom-residual (no xattn)",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        geom_residual=True,
    ),
    Experiment(
        exp_id="plain_xattn",
        description="this_run + cross-attn (no geom)",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        cross_attn=True,
    ),
    # --- aux ---
    Experiment(
        exp_id="aux_0.3",
        description="this_run + aux-range weight=0.3",
        lr=3e-4,
        label_jitter_m=0.02,
        early_stop_patience=15,
        aux_range=True,
        aux_range_weight=0.3,
    ),
    # --- feature augmentation ablation (this_run base) ---
    Experiment(
        exp_id="aug_off",
        description="this_run + no aug (jitter=0 noise=0 spec=0)",
        lr=3e-4,
        label_jitter_m=0.0,
        feature_noise_std=0.0,
        spec_augment_prob=0.0,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="aug_strong",
        description="this_run + strong aug (noise=0.05 spec=0.5 jitter=0.03)",
        lr=3e-4,
        label_jitter_m=0.03,
        feature_noise_std=0.05,
        spec_augment_prob=0.5,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="aug_spec_only",
        description="this_run + SpecAug only (noise=0 spec=0.5)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="aug_noise_only",
        description="this_run + feature noise only (noise=0.05 spec=0)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.05,
        spec_augment_prob=0.0,
        early_stop_patience=15,
    ),
    # --- CPI-level aug on raw path (this_run base) ---
    Experiment(
        exp_id="cpi_aug_mild",
        description="this_run + raw CPI aug (amp=0.2 noise_rel=0.02)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.02,
        spec_augment_prob=0.3,
        cpi_aug=True,
        cpi_amp_scale=0.2,
        cpi_complex_noise_std=0.02,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="cpi_aug_strong",
        description="this_run + raw CPI aug strong (amp=0.3 noise_rel=0.05)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.02,
        spec_augment_prob=0.3,
        cpi_aug=True,
        cpi_amp_scale=0.3,
        cpi_complex_noise_std=0.05,
        early_stop_patience=15,
    ),
    # --- feature preprocess (aug_spec_only recipe) ---
    Experiment(
        exp_id="feat_complex_roi",
        description="aug_spec_only + feature_mode=complex_roi (reuse ckpt)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        feature_mode="complex_roi",
        early_stop_patience=15,
        train=False,
        external_checkpoint=FEAT_COMPLEX_ROI_CKPT,
    ),
    Experiment(
        exp_id="feat_legacy_4ch",
        description="aug_spec_only + feature_mode=legacy_4ch",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        feature_mode="legacy_4ch",
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="feat_real_imag_cnorm",
        description="aug_spec_only + real_imag RMS per-station norm",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        feature_mode="real_imag",
        feature_norm="rms",
        early_stop_patience=15,
    ),
    # --- non-CNN structure levers (aug_spec_only recipe) ---
    Experiment(
        exp_id="arch_range_geom",
        description="aug_spec_only + geom-only (range→交会, no Δxy)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        geom_only=True,
        early_stop_patience=15,
    ),
    Experiment(
        exp_id="arch_session_median",
        description="aug_spec_only ckpt + session median smooth (eval only)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        session_smooth="median",
        early_stop_patience=15,
        train=False,
        external_checkpoint=AUG_SPEC_ONLY_CKPT,
    ),
    Experiment(
        exp_id="arch_session_ema",
        description="aug_spec_only ckpt + session EMA smooth α=0.3 (eval only)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        session_smooth="ema",
        session_smooth_alpha=0.3,
        early_stop_patience=15,
        train=False,
        external_checkpoint=AUG_SPEC_ONLY_CKPT,
    ),
    Experiment(
        exp_id="arch_light_tf",
        description="aug_spec_only + light late-fusion Transformer (d=64, 2L, 4H)",
        lr=3e-4,
        label_jitter_m=0.02,
        feature_noise_std=0.0,
        spec_augment_prob=0.5,
        backbone="transformer",
        early_stop_patience=15,
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
    if exp.session_smooth != "none":
        cmd.extend(["--session-smooth", exp.session_smooth])
        if exp.session_smooth == "ema":
            cmd.extend(["--session-smooth-alpha", str(exp.session_smooth_alpha)])
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


def _normalize_summary_row(row: dict[str, str]) -> dict[str, str | float | int]:
    """Coerce CSV strings; fill ``cross_attn=0`` for pre-xattn summary rows."""
    out: dict[str, str | float | int] = dict(row)
    if "cross_attn" not in out or out["cross_attn"] in ("", None):
        out["cross_attn"] = 0
    if "geom_only" not in out or out["geom_only"] in ("", None):
        out["geom_only"] = 0
    if "session_smooth" not in out or out["session_smooth"] in ("", None):
        out["session_smooth"] = "none"
    if "session_smooth_alpha" not in out or out["session_smooth_alpha"] in ("", None):
        out["session_smooth_alpha"] = 0.0
    if "feature_noise_std" not in out or out["feature_noise_std"] in ("", None):
        out["feature_noise_std"] = 0.02
    if "spec_augment_prob" not in out or out["spec_augment_prob"] in ("", None):
        out["spec_augment_prob"] = 0.3
    if "cpi_aug" not in out or out["cpi_aug"] in ("", None):
        out["cpi_aug"] = 0
    if "cpi_amp_scale" not in out or out["cpi_amp_scale"] in ("", None):
        out["cpi_amp_scale"] = 0.0
    if "cpi_complex_noise_std" not in out or out["cpi_complex_noise_std"] in ("", None):
        out["cpi_complex_noise_std"] = 0.0
    if "feature_mode" not in out or out["feature_mode"] in ("", None):
        out["feature_mode"] = "real_imag"
    if "feature_norm" not in out or out["feature_norm"] in ("", None):
        out["feature_norm"] = "none"
    if "backbone" not in out or out["backbone"] in ("", None):
        out["backbone"] = "cnn"
    for key in (
        "center_weight",
        "side_weight",
        "corner_weight",
        "aux_range_weight",
        "trim_best_frac",
        "lr",
        "label_jitter_m",
        "feature_noise_std",
        "spec_augment_prob",
        "cpi_amp_scale",
        "cpi_complex_noise_std",
        "session_smooth_alpha",
        "global_rmse_m",
        "inner_rmse_m",
        "outer_rmse_m",
    ):
        if key in out and out[key] not in ("", None):
            out[key] = float(out[key])
    for key in (
        "geom_residual",
        "geom_only",
        "cross_attn",
        "aux_range",
        "trim_best_rmse",
        "cpi_aug",
        "early_stop_patience",
    ):
        if key in out and out[key] not in ("", None):
            out[key] = int(float(out[key]))
    return out


def _load_summary() -> list[dict[str, str | float | int]]:
    if not SUMMARY_CSV.is_file():
        return []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return [_normalize_summary_row(row) for row in csv.DictReader(f)]


def _merge_summary_rows(
    existing: list[dict[str, str | float | int]],
    updates: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    """Upsert by exp_id; keep EXPERIMENTS order, then any unknown legacy ids."""
    by_id = {str(r["exp_id"]): r for r in existing}
    for row in updates:
        by_id[str(row["exp_id"])] = row
    ordered: list[dict[str, str | float | int]] = []
    seen: set[str] = set()
    for exp in EXPERIMENTS:
        if exp.exp_id in by_id:
            ordered.append(by_id[exp.exp_id])
            seen.add(exp.exp_id)
    for exp_id, row in by_id.items():
        if exp_id not in seen:
            ordered.append(row)
    return ordered


def _write_summary(rows: list[dict[str, str | float | int]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    print(f"\nSummary written ({len(rows)} rows): {SUMMARY_CSV}", flush=True)


def _print_topk(rows: list[dict[str, str | float | int]], k: int = 10) -> None:
    ranked = sorted(
        rows, key=lambda r: (float(r["global_rmse_m"]), float(r["outer_rmse_m"]))
    )
    print("\n=== Top configs (by Global, then Outer) ===", flush=True)
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"{i}. {row['exp_id']}: G={float(row['global_rmse_m']):.4f} "
            f"I={float(row['inner_rmse_m']):.4f} O={float(row['outer_rmse_m']):.4f} "
            f"| c/s/k=1/{row['side_weight']}/{row['corner_weight']} "
            f"geom={row['geom_residual']} geom_only={row.get('geom_only', 0)} "
            f"xattn={row.get('cross_attn', 0)} "
            f"aux={row['aux_range']} "
            f"trim={row['trim_best_rmse']}/{row['trim_best_frac']} "
            f"lr={row['lr']} jitter={row['label_jitter_m']} "
            f"noise={row.get('feature_noise_std', 0.02)} "
            f"spec={row.get('spec_augment_prob', 0.3)} "
            f"cpi={row.get('cpi_aug', 0)}/"
            f"{row.get('cpi_amp_scale', 0)}/"
            f"{row.get('cpi_complex_noise_std', 0)} "
            f"feat={row.get('feature_mode', 'real_imag')}/"
            f"{row.get('feature_norm', 'none')} "
            f"backbone={row.get('backbone', 'cnn')} "
            f"smooth={row.get('session_smooth', 'none')}",
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
        f"geom_only={best.get('geom_only', 0)}\n"
        f"cross_attn={best.get('cross_attn', 0)}\n"
        f"aux_range={best['aux_range']}\n"
        f"aux_range_weight={best['aux_range_weight']}\n"
        f"trim_best_rmse={best['trim_best_rmse']}\n"
        f"trim_best_frac={best['trim_best_frac']}\n"
        f"lr={best['lr']}\n"
        f"label_jitter_m={best['label_jitter_m']}\n"
        f"feature_noise_std={best.get('feature_noise_std', 0.02)}\n"
        f"spec_augment_prob={best.get('spec_augment_prob', 0.3)}\n"
        f"cpi_aug={best.get('cpi_aug', 0)}\n"
        f"cpi_amp_scale={best.get('cpi_amp_scale', 0.0)}\n"
        f"cpi_complex_noise_std={best.get('cpi_complex_noise_std', 0.0)}\n"
        f"feature_mode={best.get('feature_mode', 'real_imag')}\n"
        f"feature_norm={best.get('feature_norm', 'none')}\n"
        f"backbone={best.get('backbone', 'cnn')}\n"
        f"session_smooth={best.get('session_smooth', 'none')}\n"
        f"session_smooth_alpha={best.get('session_smooth_alpha', 0.0)}\n"
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
    parser = argparse.ArgumentParser(
        description="Improve-next matrix: trim/outer/hparam/aux/xattn/aug/cpi/feat-preproc"
    )
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

    updates: list[dict[str, str | float | int]] = []
    exps_by_id = {e.exp_id: e for e in EXPERIMENTS}
    for exp in EXPERIMENTS:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== {exp.exp_id}: {exp.description} ===", flush=True)
        if not args.skip_train:
            _train_experiment(exp)
        elif not exp.resolved_checkpoint().is_file() and exp.external_checkpoint:
            _train_experiment(exp)
        updates.append(_eval_experiment(exp))
        merged = _merge_summary_rows(_load_summary(), updates)
        _write_summary(merged)

    if not updates:
        print("No experiments run.", flush=True)
        return

    rows = _merge_summary_rows(_load_summary(), updates)
    best_row = _pick_best(rows)
    best_exp = exps_by_id.get(str(best_row["exp_id"]))
    if best_exp is None:
        best_exp = next(e for e in EXPERIMENTS if e.exp_id == updates[0]["exp_id"])
    _print_topk(rows)
    _write_best(best_row, exp=best_exp)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
