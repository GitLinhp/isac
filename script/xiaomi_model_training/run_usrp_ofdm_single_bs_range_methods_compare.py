#!/usr/bin/env python3
"""单站测距 CNN / MUSIC / ESPRIT 距离估计性能对比。

默认在 val H5 上评测；可通过 ``--h5-path`` 切换训练集或其他 HDF5。

示例::

    python script/xiaomi_model_training/run_usrp_ofdm_single_bs_range_methods_compare.py
    python script/xiaomi_model_training/run_usrp_ofdm_single_bs_range_methods_compare.py \\
        --methods music,cnn --max-samples 256 --skip-plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from isac import (
    DEFAULT_XIAOMI_SINGLE_BS_RANGE_CNN_MODEL,
    DEFAULT_XIAOMI_SINGLE_BS_RANGE_VAL_H5,
    PROJECT_ROOT,
)
from isac.xiaomi_models import (
    DEFAULT_RANGE_ROI,
    default_range_bin_step,
    load_single_bs_range_cnn_checkpoint,
    profile_to_features,
    profile_to_roi,
)
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_ESPRIT_NUM_SOURCES,
    DEFAULT_ESPRIT_SUBARRAY_SIZE,
    DEFAULT_ESPRIT_WINDOW_SIZE,
    DEFAULT_MUSIC_NUM_SOURCES,
    DEFAULT_MUSIC_SUBARRAY_SIZE,
    DEFAULT_MUSIC_THRESHOLD,
    estimate_monostatic_range_esprit_m,
    estimate_monostatic_range_m,
)
from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    DATASET_KEY_PROFILES,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_RANGE,
    META_KEY_FFT_LEN,
    META_KEY_ZEROPADDING_FAC,
)

VALID_METHODS = ("music", "esprit", "cnn")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "out" / "xiaomi" / "single_bs_range_methods_compare"
SUMMARY_FIELDS = (
    "method",
    "n_valid",
    "fail_rate",
    "rmse_m",
    "mae_m",
    "min_abs_err_m",
    "max_abs_err_m",
    "var_abs_err_m",
    "n_total",
)


def _parse_methods(raw: str) -> list[str]:
    parts = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--methods 不能为空")
    unknown = [m for m in parts if m not in VALID_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"--methods 仅支持 {VALID_METHODS}，收到未知项 {unknown}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for m in parts:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"range-roi 须满足 min < max，收到 {lo} {hi}")
    return lo, hi


def rmse_mae_from_preds(
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    """由预测与真值计算 n_valid / fail_rate / rmse / mae / abs 误差统计。

    ``pred`` 中 ``nan`` / ``inf`` 视为失败，不计入误差统计。
    """
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_arr.shape} vs {target_arr.shape}"
        )
    n_total = int(pred_arr.size)
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr)
    n_valid = int(valid.sum())
    fail_rate = 1.0 - (n_valid / n_total) if n_total else 0.0
    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "fail_rate": float(fail_rate),
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "min_abs_err_m": float("nan"),
            "max_abs_err_m": float("nan"),
            "var_abs_err_m": float("nan"),
        }
    err = pred_arr[valid] - target_arr[valid]
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(abs_err))
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "fail_rate": float(fail_rate),
        "rmse_m": rmse,
        "mae_m": mae,
        "min_abs_err_m": float(np.min(abs_err)),
        "max_abs_err_m": float(np.max(abs_err)),
        "var_abs_err_m": float(np.var(abs_err)),
    }


def _resolve_range_bin_step(
    *,
    h5_attrs: dict[str, Any],
    ckpt: dict[str, Any] | None,
    override: float | None,
) -> float:
    if override is not None:
        return float(override)
    if ckpt is not None and "range_bin_step" in ckpt:
        return float(ckpt["range_bin_step"])
    fft_len = int(h5_attrs.get(META_KEY_FFT_LEN, 4096))
    zp = int(h5_attrs.get(META_KEY_ZEROPADDING_FAC, 4))
    return default_range_bin_step(fft_len=fft_len, zeropadding_fac=zp)


def _load_h5(h5_path: Path, *, max_samples: int | None) -> dict[str, np.ndarray | dict]:
    with h5py.File(h5_path, "r") as f:
        profiles = np.asarray(f[DATASET_KEY_PROFILES][:], dtype=np.complex64)
        target_range = np.asarray(f[DATASET_KEY_TARGET_RANGE][:], dtype=np.float64)
        session_index = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int32)
        attrs = dict(f.attrs)
    n = profiles.shape[0]
    if max_samples is not None:
        n = min(n, max(1, int(max_samples)))
        profiles = profiles[:n]
        target_range = target_range[:n]
        session_index = session_index[:n]
    return {
        "profiles": profiles,
        "target_range": target_range,
        "session_index": session_index,
        "attrs": attrs,
    }


@torch.no_grad()
def _run_cnn(
    profiles: np.ndarray,
    *,
    model: torch.nn.Module,
    feature_mode: str,
    range_roi: tuple[float, float],
    range_bin_step: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    n = profiles.shape[0]
    for start in tqdm(range(0, n, batch_size), desc="CNN", leave=False):
        end = min(start + batch_size, n)
        feats = []
        for i in range(start, end):
            roi = profile_to_roi(
                profiles[i],
                range_roi=range_roi,
                range_bin_step=range_bin_step,
            )
            feats.append(profile_to_features(roi, mode=feature_mode))  # type: ignore[arg-type]
        batch = torch.stack(feats, dim=0).to(device)
        out = model(batch).detach().cpu().numpy().reshape(-1)
        preds.append(out.astype(np.float64))
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float64)


def _run_subspace(
    profiles: np.ndarray,
    *,
    method: str,
    range_roi: tuple[float, float],
    range_bin_step: float,
    device: torch.device | str | None,
) -> np.ndarray:
    preds = np.full(profiles.shape[0], np.nan, dtype=np.float64)
    desc = method.upper()
    for i in tqdm(range(profiles.shape[0]), desc=desc, leave=False):
        if method == "music":
            preds[i] = estimate_monostatic_range_m(
                profiles[i],
                range_bin_step=range_bin_step,
                range_roi=range_roi,
                num_sources=DEFAULT_MUSIC_NUM_SOURCES,
                subarray_size=DEFAULT_MUSIC_SUBARRAY_SIZE,
                threshold=DEFAULT_MUSIC_THRESHOLD,
                device=device,
            )
        else:
            preds[i] = estimate_monostatic_range_esprit_m(
                profiles[i],
                range_bin_step=range_bin_step,
                range_roi=range_roi,
                num_sources=DEFAULT_ESPRIT_NUM_SOURCES,
                subarray_size=DEFAULT_ESPRIT_SUBARRAY_SIZE,
                window_size=DEFAULT_ESPRIT_WINDOW_SIZE,
                device=device,
            )
    return preds


def _fmt_metric(value: float, *, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = (
        "rmse_m",
        "mae_m",
        "min_abs_err_m",
        "max_abs_err_m",
        "var_abs_err_m",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in SUMMARY_FIELDS:
                value = row.get(key, "")
                if key in metric_fields and isinstance(value, (int, float)):
                    out[key] = _fmt_metric(float(value), digits=3)
                elif key == "fail_rate" and isinstance(value, (int, float)):
                    out[key] = _fmt_metric(float(value), digits=3)
                else:
                    out[key] = value
            writer.writerow(out)


def _write_per_sample_csv(
    path: Path,
    *,
    session_index: np.ndarray,
    target_range: np.ndarray,
    preds: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(preds.keys())
    fieldnames = ["frame_idx", "session_index", "target_range_m"]
    for m in methods:
        fieldnames.append(f"{m}_m")
        fieldnames.append(f"{m}_abs_err_m")
    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(target_range.shape[0]):
            row: dict[str, Any] = {
                "frame_idx": i,
                "session_index": int(session_index[i]),
                "target_range_m": float(target_range[i]),
            }
            for m in methods:
                p = float(preds[m][i])
                row[f"{m}_m"] = p
                row[f"{m}_abs_err_m"] = (
                    abs(p - float(target_range[i])) if np.isfinite(p) else float("nan")
                )
            writer.writerow(row)


def _plot_error_cdf(
    preds: dict[str, np.ndarray],
    target_range: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, pred in preds.items():
        valid = np.isfinite(pred) & np.isfinite(target_range)
        if not np.any(valid):
            continue
        abs_err = np.sort(np.abs(pred[valid] - target_range[valid]))
        cdf = np.arange(1, abs_err.size + 1, dtype=np.float64) / abs_err.size
        ax.plot(abs_err, cdf, label=method.upper())
    ax.set_xlabel("Absolute range error (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Single-BS range estimation error CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CNN / MUSIC / ESPRIT single-BS range estimation"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=DEFAULT_XIAOMI_SINGLE_BS_RANGE_VAL_H5,
        help="input single-BS range HDF5 (default: val set)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_XIAOMI_SINGLE_BS_RANGE_CNN_MODEL,
        help="CNN checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory for CSV / plots",
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=_parse_methods("music,esprit,cnn"),
        help="comma-separated subset of music,esprit,cnn",
    )
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_M", "MAX_M"),
        help="range ROI (default: checkpoint ROI or 0 8)",
    )
    parser.add_argument(
        "--range-bin-step",
        type=float,
        default=None,
        help="override range bin step (m)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    h5_path = args.h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    methods: list[str] = list(args.methods)
    need_cnn = "cnn" in methods
    ckpt: dict[str, Any] | None = None
    model = None
    device = torch.device(args.device)

    if need_cnn:
        ckpt_path = args.checkpoint.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        model, ckpt = load_single_bs_range_cnn_checkpoint(
            ckpt_path, map_location=device
        )
        model = model.to(device)

    data = _load_h5(h5_path, max_samples=args.max_samples)
    profiles = data["profiles"]  # type: ignore[assignment]
    target_range = data["target_range"]  # type: ignore[assignment]
    session_index = data["session_index"]  # type: ignore[assignment]
    h5_attrs = data["attrs"]  # type: ignore[assignment]
    assert isinstance(profiles, np.ndarray)
    assert isinstance(target_range, np.ndarray)
    assert isinstance(session_index, np.ndarray)
    assert isinstance(h5_attrs, dict)

    if args.range_roi is not None:
        range_roi = _parse_range_roi(list(args.range_roi))
    elif ckpt is not None and "range_roi" in ckpt:
        roi = ckpt["range_roi"]
        range_roi = (float(roi[0]), float(roi[1]))
    else:
        range_roi = DEFAULT_RANGE_ROI

    range_bin_step = _resolve_range_bin_step(
        h5_attrs=h5_attrs,
        ckpt=ckpt,
        override=args.range_bin_step,
    )
    feature_mode = str(ckpt.get("feature_mode", "real_imag")) if ckpt else "real_imag"

    print(
        f"h5={h5_path} N={profiles.shape[0]} methods={methods} "
        f"roi={range_roi} step={range_bin_step:.6f} device={device}"
    )

    preds: dict[str, np.ndarray] = {}
    subspace_device: torch.device | str | None = (
        device if device.type == "cuda" else None
    )

    if "music" in methods:
        preds["music"] = _run_subspace(
            profiles,
            method="music",
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            device=subspace_device,
        )
    if "esprit" in methods:
        preds["esprit"] = _run_subspace(
            profiles,
            method="esprit",
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            device=subspace_device,
        )
    if "cnn" in methods:
        assert model is not None
        preds["cnn"] = _run_cnn(
            profiles,
            model=model,
            feature_mode=feature_mode,
            range_roi=range_roi,
            range_bin_step=range_bin_step,
            device=device,
            batch_size=args.batch_size,
        )

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        metrics = rmse_mae_from_preds(preds[method], target_range)
        row = {"method": method, **metrics}
        summary_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    per_sample_path = output_dir / "per_sample.csv"
    _write_summary_csv(summary_path, summary_rows)
    _write_per_sample_csv(
        per_sample_path,
        session_index=session_index,
        target_range=target_range,
        preds=preds,
    )

    display_rows = [
        {
            "method": r["method"],
            "n_valid": r["n_valid"],
            "fail_rate": f"{100.0 * float(r['fail_rate']):.2f}%",
            "rmse_m": _fmt_metric(float(r["rmse_m"])),
            "mae_m": _fmt_metric(float(r["mae_m"])),
            "min_abs": _fmt_metric(float(r["min_abs_err_m"])),
            "max_abs": _fmt_metric(float(r["max_abs_err_m"])),
            "var_abs": _fmt_metric(float(r["var_abs_err_m"])),
        }
        for r in summary_rows
    ]
    print(tabulate(display_rows, headers="keys", tablefmt="simple_grid"))
    print(f"summary → {summary_path}")
    print(f"per_sample → {per_sample_path}")

    if not args.skip_plots:
        cdf_path = output_dir / "error_cdf.png"
        _plot_error_cdf(preds, target_range, cdf_path)
        print(f"cdf → {cdf_path}")


if __name__ == "__main__":
    main()
