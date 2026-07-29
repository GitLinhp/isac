#!/usr/bin/env python3
"""最优配置分阶段对照扫描（区权重 → geom → lr/jitter）。

统一协议：Run1 训练、Run2 全帧、--no-filter-outliers、新默认底座。
主指标 Global RMSE；若 Global 差 ≤0.005 则 Outer 更低者胜。

示例::

    python script/experiment/run_cnn_best_config_matrix.py
    python script/experiment/run_cnn_best_config_matrix.py --phase A
    python script/experiment/run_cnn_best_config_matrix.py --skip-train
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
SUMMARY_CSV = PROJECT_ROOT / "out/cooperative_monostatic/cnn_best_config_summary.csv"
BEST_TXT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_best_config_best.txt"
MATRIX_ROOT = PROJECT_ROOT / "models/cnn_best_ab"
EVAL_ROOT = PROJECT_ROOT / "out/cooperative_monostatic/cnn_best_ab"

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
    "--early-stop-patience",
    "10",
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
    "--session-aggregated-loss",
    "--no-filter-outliers",
    "--no-aux-range",
    "--no-cross-attn",
    "--no-stopgrad-geom",
    "--no-eval-after-train",
]

SUMMARY_FIELDS = [
    "phase",
    "exp_id",
    "description",
    "center_weight",
    "side_weight",
    "corner_weight",
    "geom_residual",
    "lr",
    "label_jitter_m",
    "checkpoint",
    "global_rmse_m",
    "inner_rmse_m",
    "outer_rmse_m",
    "output_csv",
]


@dataclass(frozen=True)
class Experiment:
    phase: str
    exp_id: str
    description: str
    center_weight: float = 1.0
    side_weight: float = 3.0
    corner_weight: float = 3.0
    geom_residual: bool = False
    lr: float = 5e-4
    label_jitter_m: float = 0.05

    @property
    def output_dir(self) -> Path:
        return MATRIX_ROOT / self.exp_id

    def resolved_checkpoint(self) -> Path:
        return self.output_dir / "best_model.pth"

    def train_cli_args(self) -> list[str]:
        args = [
            *COMMON_TRAIN,
            "--center-weight",
            str(self.center_weight),
            "--side-weight",
            str(self.side_weight),
            "--corner-weight",
            str(self.corner_weight),
            "--lr",
            str(self.lr),
            "--label-jitter-m",
            str(self.label_jitter_m),
        ]
        if self.geom_residual:
            args.append("--geom-residual")
        else:
            args.append("--no-geom-residual")
        return args

    def meta_row(self) -> dict[str, str | float | int]:
        return {
            "phase": self.phase,
            "exp_id": self.exp_id,
            "description": self.description,
            "center_weight": self.center_weight,
            "side_weight": self.side_weight,
            "corner_weight": self.corner_weight,
            "geom_residual": int(self.geom_residual),
            "lr": self.lr,
            "label_jitter_m": self.label_jitter_m,
        }


PHASE_A: tuple[Experiment, ...] = (
    Experiment(
        phase="A",
        exp_id="w_1_3_2",
        description="zone weights 1/3/2 (previous default)",
        side_weight=3.0,
        corner_weight=2.0,
    ),
    Experiment(
        phase="A",
        exp_id="w_1_3_1",
        description="zone weights 1/3/1",
        side_weight=3.0,
        corner_weight=1.0,
    ),
    Experiment(
        phase="A",
        exp_id="w_1_3_5",
        description="zone weights 1/3/5",
        side_weight=3.0,
        corner_weight=5.0,
    ),
    Experiment(
        phase="A",
        exp_id="w_1_2_2",
        description="zone weights 1/2/2",
        side_weight=2.0,
        corner_weight=2.0,
    ),
    Experiment(
        phase="A",
        exp_id="w_1_3_3",
        description="zone weights 1/3/3",
        side_weight=3.0,
        corner_weight=3.0,
    ),
    Experiment(
        phase="A",
        exp_id="w_1_4_2",
        description="zone weights 1/4/2",
        side_weight=4.0,
        corner_weight=2.0,
    ),
)


def _phase_b_exps(w_star: Experiment) -> tuple[Experiment, Experiment]:
    plain = Experiment(
        phase="B",
        exp_id=f"B_plain_{w_star.exp_id}",
        description=f"plain (reuse {w_star.exp_id})",
        center_weight=w_star.center_weight,
        side_weight=w_star.side_weight,
        corner_weight=w_star.corner_weight,
        geom_residual=False,
        lr=w_star.lr,
        label_jitter_m=w_star.label_jitter_m,
    )
    geom = Experiment(
        phase="B",
        exp_id=f"B_geom_{w_star.exp_id}",
        description=f"geom residual on {w_star.exp_id} weights",
        center_weight=w_star.center_weight,
        side_weight=w_star.side_weight,
        corner_weight=w_star.corner_weight,
        geom_residual=True,
        lr=w_star.lr,
        label_jitter_m=w_star.label_jitter_m,
    )
    return plain, geom


def _phase_c_exps(s_star: Experiment) -> list[Experiment]:
    combos = [
        (3e-4, 0.02),
        (3e-4, 0.05),
        (5e-4, 0.02),
        (5e-4, 0.05),
    ]
    exps: list[Experiment] = []
    for lr, jitter in combos:
        same_as_s = (
            abs(lr - s_star.lr) < 1e-12
            and abs(jitter - s_star.label_jitter_m) < 1e-12
        )
        tag = f"lr{lr:g}_j{jitter:g}".replace(".", "p")
        geom_tag = "geom" if s_star.geom_residual else "plain"
        exp_id = f"C_{geom_tag}_{s_star.exp_id}_{tag}"
        if same_as_s:
            # Reuse S* checkpoint path via alias exp that points to same dir content
            exp_id = f"C_reuse_{s_star.exp_id}"
        exps.append(
            Experiment(
                phase="C",
                exp_id=exp_id,
                description=(
                    f"reuse {s_star.exp_id}"
                    if same_as_s
                    else f"{geom_tag} lr={lr:g} jitter={jitter:g}"
                ),
                center_weight=s_star.center_weight,
                side_weight=s_star.side_weight,
                corner_weight=s_star.corner_weight,
                geom_residual=s_star.geom_residual,
                lr=lr,
                label_jitter_m=jitter,
            )
        )
    return exps


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
    """Return True if a is better than b (Global primary, Outer tie-break)."""
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


def _train_experiment(exp: Experiment, *, reuse_ckpt_from: Path | None = None) -> None:
    ckpt = exp.resolved_checkpoint()
    if ckpt.is_file():
        print(f"[skip train] checkpoint exists: {ckpt}")
        return
    if reuse_ckpt_from is not None and reuse_ckpt_from.is_file():
        exp.output_dir.mkdir(parents=True, exist_ok=True)
        # Hardlink/copy best_model for independent path bookkeeping
        import shutil

        shutil.copy2(reuse_ckpt_from, ckpt)
        print(f"[reuse ckpt] {reuse_ckpt_from} → {ckpt}")
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


def _print_topk(rows: list[dict[str, str | float | int]], k: int = 5) -> None:
    ranked = sorted(rows, key=lambda r: (float(r["global_rmse_m"]), float(r["outer_rmse_m"])))
    print("\n=== Top configs (by Global, then Outer) ===", flush=True)
    for i, row in enumerate(ranked[:k], start=1):
        print(
            f"{i}. {row['exp_id']}: G={float(row['global_rmse_m']):.4f} "
            f"I={float(row['inner_rmse_m']):.4f} O={float(row['outer_rmse_m']):.4f} "
            f"| c/s/k={row['center_weight']}/{row['side_weight']}/{row['corner_weight']} "
            f"geom={row['geom_residual']} lr={row['lr']} jitter={row['label_jitter_m']}",
            flush=True,
        )


def _write_best(
    best: dict[str, str | float | int],
    *,
    exp: Experiment,
) -> None:
    cli = " ".join(exp.train_cli_args())
    text = (
        f"BEST exp_id={best['exp_id']}\n"
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
        f"checkpoint={best['checkpoint']}\n"
        f"train_cli_args={cli}\n"
    )
    BEST_TXT.parent.mkdir(parents=True, exist_ok=True)
    BEST_TXT.write_text(text, encoding="utf-8")
    print("\n=== BEST ===", flush=True)
    print(text, flush=True)
    print(f"Wrote {BEST_TXT}", flush=True)


def _run_exps(
    exps: list[Experiment] | tuple[Experiment, ...],
    *,
    skip_train: bool,
    only_ids: set[str] | None,
    rows: list[dict[str, str | float | int]],
    reuse_map: dict[str, Path] | None = None,
) -> list[dict[str, str | float | int]]:
    phase_rows: list[dict[str, str | float | int]] = []
    reuse_map = reuse_map or {}
    for exp in exps:
        if only_ids is not None and exp.exp_id not in only_ids:
            continue
        print(f"\n=== [{exp.phase}] {exp.exp_id}: {exp.description} ===", flush=True)
        if not skip_train:
            _train_experiment(exp, reuse_ckpt_from=reuse_map.get(exp.exp_id))
        row = _eval_experiment(exp)
        rows.append(row)
        phase_rows.append(row)
        _write_summary(rows)
    return phase_rows


def _exp_by_id(
    exps: list[Experiment] | tuple[Experiment, ...], exp_id: str
) -> Experiment:
    for exp in exps:
        if exp.exp_id == exp_id:
            return exp
    raise KeyError(exp_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-config phased A/B/C matrix")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="skip training; evaluate existing checkpoints only",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated exp_id filter (disables auto phase chaining)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=("all", "A", "B", "C"),
        help="run only this phase (default: all chained)",
    )
    args = parser.parse_args()

    only_ids = None if args.only is None else {s.strip() for s in args.only.split(",")}

    if not RUN1_H5.is_file():
        raise FileNotFoundError(f"train HDF5 missing: {RUN1_H5}")
    if not RUN2_H5.is_file():
        raise FileNotFoundError(f"eval HDF5 missing: {RUN2_H5}")

    rows: list[dict[str, str | float | int]] = []
    run_all = args.phase == "all" and only_ids is None

    # Phase A
    if args.phase in ("all", "A") or only_ids is not None:
        a_rows = _run_exps(
            PHASE_A, skip_train=args.skip_train, only_ids=only_ids, rows=rows
        )
    else:
        a_rows = []

    if only_ids is not None:
        _print_topk(rows)
        print("\nDone (--only mode).", flush=True)
        return

    if not a_rows and args.phase in ("all", "A"):
        raise RuntimeError("Phase A produced no rows")

    if args.phase == "A":
        w_star_row = _pick_best(a_rows)
        print(
            f"\nW* = {w_star_row['exp_id']} "
            f"G={float(w_star_row['global_rmse_m']):.4f} "
            f"O={float(w_star_row['outer_rmse_m']):.4f}",
            flush=True,
        )
        _print_topk(rows)
        print("\nDone (phase A only).", flush=True)
        return

    # Resolve W* from Phase A rows (or from summary if only B/C)
    if not a_rows:
        # Load summary for prior A results
        if not SUMMARY_CSV.is_file():
            raise FileNotFoundError(
                f"need Phase A results in {SUMMARY_CSV} before phase {args.phase}"
            )
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
            prior = list(csv.DictReader(f))
        a_rows = [r for r in prior if r.get("phase") == "A"]
        rows = [
            {
                **r,
                "center_weight": float(r["center_weight"]),
                "side_weight": float(r["side_weight"]),
                "corner_weight": float(r["corner_weight"]),
                "geom_residual": int(float(r["geom_residual"])),
                "lr": float(r["lr"]),
                "label_jitter_m": float(r["label_jitter_m"]),
                "global_rmse_m": float(r["global_rmse_m"]),
                "inner_rmse_m": float(r["inner_rmse_m"]),
                "outer_rmse_m": float(r["outer_rmse_m"]),
            }
            for r in prior
        ]
        if not a_rows:
            raise RuntimeError("No Phase A rows in summary")

    w_star_row = _pick_best(a_rows)
    w_star = _exp_by_id(PHASE_A, str(w_star_row["exp_id"]))
    print(
        f"\n>>> W* = {w_star.exp_id} "
        f"(G={float(w_star_row['global_rmse_m']):.4f}, "
        f"O={float(w_star_row['outer_rmse_m']):.4f})",
        flush=True,
    )

    # Phase B
    plain_b, geom_b = _phase_b_exps(w_star)
    # Plain reuses W* checkpoint
    reuse_b = {plain_b.exp_id: w_star.resolved_checkpoint()}
    if args.phase in ("all", "B"):
        b_rows = _run_exps(
            [plain_b, geom_b],
            skip_train=args.skip_train,
            only_ids=None,
            rows=rows,
            reuse_map=reuse_b,
        )
    else:
        b_rows = [r for r in rows if r.get("phase") == "B"]

    if args.phase == "B":
        s_star_row = _pick_best(b_rows)
        print(f"\nS* = {s_star_row['exp_id']}", flush=True)
        _print_topk(rows)
        print("\nDone (phase B only).", flush=True)
        return

    s_star_row = _pick_best(b_rows)
    # Reconstruct S* experiment from winner
    if str(s_star_row["exp_id"]) == plain_b.exp_id:
        s_star = plain_b
    else:
        s_star = geom_b
    print(
        f"\n>>> S* = {s_star.exp_id} "
        f"(G={float(s_star_row['global_rmse_m']):.4f}, "
        f"O={float(s_star_row['outer_rmse_m']):.4f})",
        flush=True,
    )

    # Phase C
    c_exps = _phase_c_exps(s_star)
    reuse_c: dict[str, Path] = {}
    for exp in c_exps:
        if exp.exp_id.startswith("C_reuse_"):
            reuse_c[exp.exp_id] = s_star.resolved_checkpoint()
    c_rows = _run_exps(
        c_exps,
        skip_train=args.skip_train,
        only_ids=None,
        rows=rows,
        reuse_map=reuse_c,
    )

    best_row = _pick_best(c_rows if c_rows else rows)
    # Find matching Experiment for CLI dump
    all_exps = list(PHASE_A) + [plain_b, geom_b] + c_exps
    try:
        best_exp = _exp_by_id(all_exps, str(best_row["exp_id"]))
    except KeyError:
        best_exp = s_star
    _print_topk(rows)
    _write_best(best_row, exp=best_exp)
    if run_all:
        print("\nDone (all phases).", flush=True)
    else:
        print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
