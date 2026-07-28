"""Cooperative monostatic MUSIC/ESPRIT 单站距离偏置校准。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

DEFAULT_CALIB_SEARCH_MIN_M = -2.0
DEFAULT_CALIB_SEARCH_MAX_M = 2.0
DEFAULT_CALIB_STEP_M = 0.01


@dataclass(frozen=True)
class RangeBiasCalibRoi:
    """校准目标 ROI（目标 true_xy 矩形区域，m）。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x_m: float, y_m: float) -> bool:
        return (
            self.x_min <= float(x_m) <= self.x_max
            and self.y_min <= float(y_m) <= self.y_max
        )


DEFAULT_CALIB_ROI_DEV0 = RangeBiasCalibRoi(-0.5, 0.5, -1.0, 1.0)
DEFAULT_CALIB_ROI_DEV1 = RangeBiasCalibRoi(-1.0, 1.0, -0.5, 0.5)


@dataclass(frozen=True)
class DevRangeBiasFit:
    """单 dev 偏置拟合结果。"""

    bias_m: float
    mae_m: float
    n_samples: int
    roi: RangeBiasCalibRoi


@dataclass(frozen=True)
class RangeBiasCalibResult:
    """双 dev 距离偏置校准结果。"""

    bias_dev0_m: float
    bias_dev1_m: float
    mae_dev0_m: float
    mae_dev1_m: float
    n_dev0: int
    n_dev1: int
    roi_dev0: RangeBiasCalibRoi
    roi_dev1: RangeBiasCalibRoi
    search_min_m: float
    search_max_m: float
    step_m: float
    dev0_xy: tuple[float, float]
    dev1_xy: tuple[float, float]

    @property
    def dev0(self) -> DevRangeBiasFit:
        return DevRangeBiasFit(
            self.bias_dev0_m,
            self.mae_dev0_m,
            self.n_dev0,
            self.roi_dev0,
        )

    @property
    def dev1(self) -> DevRangeBiasFit:
        return DevRangeBiasFit(
            self.bias_dev1_m,
            self.mae_dev1_m,
            self.n_dev1,
            self.roi_dev1,
        )


def true_monostatic_range_m(
    target_xy: tuple[float, float],
    dev_xy: tuple[float, float],
) -> float:
    """目标到单站传感器的几何距离 (m)。"""
    dx = float(target_xy[0]) - float(dev_xy[0])
    dy = float(target_xy[1]) - float(dev_xy[1])
    return math.hypot(dx, dy)


def target_in_calib_roi(
    x_m: float,
    y_m: float,
    roi: RangeBiasCalibRoi,
) -> bool:
    return roi.contains(x_m, y_m)


def calib_roi_mask(
    true_x_m: np.ndarray,
    true_y_m: np.ndarray,
    roi: RangeBiasCalibRoi,
) -> np.ndarray:
    x = np.asarray(true_x_m, dtype=np.float64)
    y = np.asarray(true_y_m, dtype=np.float64)
    return (
        (x >= roi.x_min)
        & (x <= roi.x_max)
        & (y >= roi.y_min)
        & (y <= roi.y_max)
    )


def parse_calib_roi(values: list[float]) -> RangeBiasCalibRoi:
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "calib ROI 须为四个浮点数：x_min x_max y_min y_max"
        )
    x_min, x_max, y_min, y_max = (float(v) for v in values)
    if x_min >= x_max or y_min >= y_max:
        raise argparse.ArgumentTypeError(
            f"calib ROI 须满足 x_min < x_max 且 y_min < y_max，收到 {values}"
        )
    return RangeBiasCalibRoi(x_min, x_max, y_min, y_max)


def fit_range_bias_1d(
    r_est: np.ndarray,
    r_true: np.ndarray,
    mask: np.ndarray,
    *,
    search_min: float = DEFAULT_CALIB_SEARCH_MIN_M,
    search_max: float = DEFAULT_CALIB_SEARCH_MAX_M,
    step: float = DEFAULT_CALIB_STEP_M,
) -> tuple[float, float]:
    """网格搜索 bias，最小化 ROI 内 ``mean(|r_est + bias - r_true|)``。"""
    r_est = np.asarray(r_est, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if r_est.shape != r_true.shape or r_est.shape != mask.shape:
        raise ValueError("r_est, r_true, mask 须同形状")

    valid = mask & np.isfinite(r_est) & np.isfinite(r_true)
    if not valid.any():
        raise ValueError("calibration ROI 内无有效距离样本")

    delta = r_est[valid] - r_true[valid]
    biases = np.arange(float(search_min), float(search_max) + step * 0.5, float(step))
    if biases.size == 0:
        raise ValueError("bias 搜索网格为空，请检查 search_min/max/step")

    mae_vals = np.array([np.mean(np.abs(delta + b)) for b in biases], dtype=np.float64)
    best_idx = int(np.argmin(mae_vals))
    return float(biases[best_idx]), float(mae_vals[best_idx])


def fit_dual_dev_range_biases(
    df: pd.DataFrame,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    roi_dev0: RangeBiasCalibRoi = DEFAULT_CALIB_ROI_DEV0,
    roi_dev1: RangeBiasCalibRoi = DEFAULT_CALIB_ROI_DEV1,
    search_min: float = DEFAULT_CALIB_SEARCH_MIN_M,
    search_max: float = DEFAULT_CALIB_SEARCH_MAX_M,
    step: float = DEFAULT_CALIB_STEP_M,
) -> RangeBiasCalibResult:
    """在各自 ROI 内独立拟合 dev0/dev1 距离偏置。"""
    required = ("true_x_m", "true_y_m", "r_dev0_m", "r_dev1_m")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少列: {', '.join(missing)}")

    true_x = df["true_x_m"].to_numpy(dtype=np.float64)
    true_y = df["true_y_m"].to_numpy(dtype=np.float64)
    r0 = df["r_dev0_m"].to_numpy(dtype=np.float64)
    r1 = df["r_dev1_m"].to_numpy(dtype=np.float64)

    true_r0 = np.array(
        [
            true_monostatic_range_m((float(x), float(y)), dev0_xy)
            for x, y in zip(true_x, true_y, strict=True)
        ],
        dtype=np.float64,
    )
    true_r1 = np.array(
        [
            true_monostatic_range_m((float(x), float(y)), dev1_xy)
            for x, y in zip(true_x, true_y, strict=True)
        ],
        dtype=np.float64,
    )

    mask0 = calib_roi_mask(true_x, true_y, roi_dev0)
    mask1 = calib_roi_mask(true_x, true_y, roi_dev1)
    bias0, mae0 = fit_range_bias_1d(
        r0, true_r0, mask0, search_min=search_min, search_max=search_max, step=step
    )
    bias1, mae1 = fit_range_bias_1d(
        r1, true_r1, mask1, search_min=search_min, search_max=search_max, step=step
    )

    return RangeBiasCalibResult(
        bias_dev0_m=bias0,
        bias_dev1_m=bias1,
        mae_dev0_m=mae0,
        mae_dev1_m=mae1,
        n_dev0=int(mask0.sum()),
        n_dev1=int(mask1.sum()),
        roi_dev0=roi_dev0,
        roi_dev1=roi_dev1,
        search_min_m=float(search_min),
        search_max_m=float(search_max),
        step_m=float(step),
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
    )


def apply_range_bias(
    df: pd.DataFrame,
    *,
    bias_dev0_m: float,
    bias_dev1_m: float,
    overwrite: bool = False,
) -> pd.DataFrame:
    """应用偏置；默认保留 raw 列并新增 ``r_dev0_cal_m`` / ``r_dev1_cal_m``。"""
    if "r_dev0_m" not in df.columns or "r_dev1_m" not in df.columns:
        raise ValueError("DataFrame 须含 r_dev0_m / r_dev1_m")

    out = df.copy()
    r0_cal, r1_cal = correct_monostatic_range_pair(
        out["r_dev0_m"].to_numpy(dtype=np.float64),
        out["r_dev1_m"].to_numpy(dtype=np.float64),
        bias_dev0_m=bias_dev0_m,
        bias_dev1_m=bias_dev1_m,
    )
    out["r_dev0_cal_m"] = r0_cal
    out["r_dev1_cal_m"] = r1_cal
    if overwrite:
        out["r_dev0_m"] = r0_cal
        out["r_dev1_m"] = r1_cal
    return out


def correct_monostatic_range_pair(
    r0_m: float | np.ndarray,
    r1_m: float | np.ndarray,
    *,
    bias_dev0_m: float,
    bias_dev1_m: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """对 dev0/dev1 单站距离应用常数偏置，返回校准距离。"""
    scalar_in = np.isscalar(r0_m) and np.isscalar(r1_m)
    r0 = np.asarray(r0_m, dtype=np.float64)
    r1 = np.asarray(r1_m, dtype=np.float64)
    r0_cal = r0 + float(bias_dev0_m)
    r1_cal = r1 + float(bias_dev1_m)
    if r0_cal.ndim == 0:
        if not np.isfinite(r0):
            r0_cal = np.float64(np.nan)
    else:
        r0_cal = r0_cal.copy()
        r0_cal[~np.isfinite(r0)] = np.nan
    if r1_cal.ndim == 0:
        if not np.isfinite(r1):
            r1_cal = np.float64(np.nan)
    else:
        r1_cal = r1_cal.copy()
        r1_cal[~np.isfinite(r1)] = np.nan
    if scalar_in:
        return float(r0_cal), float(r1_cal)
    return r0_cal, r1_cal


def biases_from_calib_result(result: RangeBiasCalibResult) -> tuple[float, float]:
    return float(result.bias_dev0_m), float(result.bias_dev1_m)


def resolve_loaded_range_biases(
    args: argparse.Namespace,
) -> tuple[float, float] | None:
    """若 ``args.calib_json`` 已设则加载并返回 ``(bias_dev0, bias_dev1)``。"""
    if args.calib_json is None:
        return None
    result = load_range_bias_calib(args.calib_json.resolve())
    return biases_from_calib_result(result)


def effective_range_columns(df: pd.DataFrame) -> tuple[str, str]:
    """返回用于 MAE/定位的有效距离列名（优先校准列）。"""
    if "r_dev0_cal_m" in df.columns and "r_dev1_cal_m" in df.columns:
        return "r_dev0_cal_m", "r_dev1_cal_m"
    return "r_dev0_m", "r_dev1_m"


def dataframe_for_range_mae(df: pd.DataFrame) -> pd.DataFrame:
    """将有效距离列映射为 ``r_dev0_m`` / ``r_dev1_m`` 供 MAE 绘图。"""
    out = df.copy()
    col0, col1 = effective_range_columns(out)
    out["r_dev0_m"] = out[col0].to_numpy(dtype=np.float64)
    out["r_dev1_m"] = out[col1].to_numpy(dtype=np.float64)
    return out


def _roi_to_dict(roi: RangeBiasCalibRoi) -> dict[str, float]:
    return asdict(roi)


def _roi_from_dict(data: dict[str, Any]) -> RangeBiasCalibRoi:
    return RangeBiasCalibRoi(
        float(data["x_min"]),
        float(data["x_max"]),
        float(data["y_min"]),
        float(data["y_max"]),
    )


def save_range_bias_calib(path: str | Path, result: RangeBiasCalibResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bias_dev0_m": result.bias_dev0_m,
        "bias_dev1_m": result.bias_dev1_m,
        "mae_dev0_m": result.mae_dev0_m,
        "mae_dev1_m": result.mae_dev1_m,
        "n_dev0": result.n_dev0,
        "n_dev1": result.n_dev1,
        "roi_dev0": _roi_to_dict(result.roi_dev0),
        "roi_dev1": _roi_to_dict(result.roi_dev1),
        "search_min_m": result.search_min_m,
        "search_max_m": result.search_max_m,
        "step_m": result.step_m,
        "dev0_xy": list(result.dev0_xy),
        "dev1_xy": list(result.dev1_xy),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_range_bias_calib(path: str | Path) -> RangeBiasCalibResult:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    return RangeBiasCalibResult(
        bias_dev0_m=float(data["bias_dev0_m"]),
        bias_dev1_m=float(data["bias_dev1_m"]),
        mae_dev0_m=float(data["mae_dev0_m"]),
        mae_dev1_m=float(data["mae_dev1_m"]),
        n_dev0=int(data["n_dev0"]),
        n_dev1=int(data["n_dev1"]),
        roi_dev0=_roi_from_dict(data["roi_dev0"]),
        roi_dev1=_roi_from_dict(data["roi_dev1"]),
        search_min_m=float(data.get("search_min_m", DEFAULT_CALIB_SEARCH_MIN_M)),
        search_max_m=float(data.get("search_max_m", DEFAULT_CALIB_SEARCH_MAX_M)),
        step_m=float(data.get("step_m", DEFAULT_CALIB_STEP_M)),
        dev0_xy=(float(data["dev0_xy"][0]), float(data["dev0_xy"][1])),
        dev1_xy=(float(data["dev1_xy"][0]), float(data["dev1_xy"][1])),
    )


def format_calib_summary(result: RangeBiasCalibResult) -> str:
    return (
        f"bias_dev0={result.bias_dev0_m:.3f} m (ROI MAE={result.mae_dev0_m:.3f}, n={result.n_dev0}), "
        f"bias_dev1={result.bias_dev1_m:.3f} m (ROI MAE={result.mae_dev1_m:.3f}, n={result.n_dev1})"
    )


def add_range_bias_calib_arguments(parser: argparse.ArgumentParser) -> None:
    """向 argparse 注册距离偏置校准相关参数。"""
    parser.add_argument(
        "--calibrate-range",
        action="store_true",
        help="fit per-device range bias on facing ROI before plotting/eval output",
    )
    parser.add_argument(
        "--calib-step",
        type=float,
        default=DEFAULT_CALIB_STEP_M,
        help=f"bias grid step in meters (default: {DEFAULT_CALIB_STEP_M})",
    )
    parser.add_argument(
        "--calib-search-min",
        type=float,
        default=DEFAULT_CALIB_SEARCH_MIN_M,
        help=f"bias search min in meters (default: {DEFAULT_CALIB_SEARCH_MIN_M})",
    )
    parser.add_argument(
        "--calib-search-max",
        type=float,
        default=DEFAULT_CALIB_SEARCH_MAX_M,
        help=f"bias search max in meters (default: {DEFAULT_CALIB_SEARCH_MAX_M})",
    )
    parser.add_argument(
        "--calib-roi-dev0",
        type=float,
        nargs=4,
        default=[-0.5, 0.5, -1.0, 1.0],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="dev0 calibration target ROI (default: -0.5 0.5 -1.0 1.0)",
    )
    parser.add_argument(
        "--calib-roi-dev1",
        type=float,
        nargs=4,
        default=[-1.0, 1.0, -0.5, 0.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="dev1 calibration target ROI (default: -1.0 1.0 -0.5 0.5)",
    )
    parser.add_argument(
        "--calib-json",
        type=Path,
        default=None,
        help="load saved range bias calibration JSON (overrides --calibrate-range)",
    )
    parser.add_argument(
        "--write-calib-json",
        type=Path,
        default=None,
        help="write fitted range bias calibration to JSON",
    )


def calib_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "roi_dev0": parse_calib_roi(list(args.calib_roi_dev0)),
        "roi_dev1": parse_calib_roi(list(args.calib_roi_dev1)),
        "search_min": float(args.calib_search_min),
        "search_max": float(args.calib_search_max),
        "step": float(args.calib_step),
    }


def apply_calibration_to_eval_rows(
    rows: list[dict[str, float | int]],
    result: RangeBiasCalibResult,
    *,
    localize_fn,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
) -> None:
    """为评估行写入 cal 距离列并用校准距离重算 xy 定位。"""
    bias_dev0_m, bias_dev1_m = biases_from_calib_result(result)
    for row in rows:
        r0_raw = float(row["r_dev0_m"])
        r1_raw = float(row["r_dev1_m"])
        r0_cal, r1_cal = correct_monostatic_range_pair(
            r0_raw,
            r1_raw,
            bias_dev0_m=bias_dev0_m,
            bias_dev1_m=bias_dev1_m,
        )
        row["r_dev0_cal_m"] = r0_cal
        row["r_dev1_cal_m"] = r1_cal
        true_xy = (float(row["true_x_m"]), float(row["true_y_m"]))
        est_x, est_y, rmse = localize_fn(
            r0_raw,
            r1_raw,
            true_xy,
            dev0_xy=dev0_xy,
            dev1_xy=dev1_xy,
            bias_dev0_m=bias_dev0_m,
            bias_dev1_m=bias_dev1_m,
        )
        row["est_x_m"] = est_x
        row["est_y_m"] = est_y
        row["rmse_xy_m"] = rmse


def ensure_eval_row_cal_columns(rows: list[dict[str, float | int]]) -> None:
    """无校准时 cal 列等于 raw。"""
    for row in rows:
        row["r_dev0_cal_m"] = float(row["r_dev0_m"])
        row["r_dev1_cal_m"] = float(row["r_dev1_m"])


def resolve_and_apply_eval_row_calibration(
    rows: list[dict[str, float | int]],
    args: argparse.Namespace,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
    localize_fn,
    calibration_preapplied: bool = False,
) -> RangeBiasCalibResult | None:
    """评估行列表：拟合/加载偏置，写入 cal 列并重算 xy 定位。"""
    df = pd.DataFrame(rows)
    if args.calib_json is not None:
        result = load_range_bias_calib(args.calib_json.resolve())
        if not calibration_preapplied:
            apply_calibration_to_eval_rows(
                rows,
                result,
                localize_fn=localize_fn,
                dev0_xy=dev0_xy,
                dev1_xy=dev1_xy,
            )
        print(f"loaded calibration: {args.calib_json.resolve()}")
        print(format_calib_summary(result))
        return result

    if not args.calibrate_range:
        ensure_eval_row_cal_columns(rows)
        return None

    opts = calib_options_from_args(args)
    result = fit_dual_dev_range_biases(
        df,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        **opts,
    )
    print(f"fitted calibration: {format_calib_summary(result)}")
    if args.write_calib_json is not None:
        path = save_range_bias_calib(args.write_calib_json.resolve(), result)
        print(f"output calib json: {path}")
    apply_calibration_to_eval_rows(
        rows,
        result,
        localize_fn=localize_fn,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
    )
    return result


def resolve_range_bias_calibration(
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    dev0_xy: tuple[float, float],
    dev1_xy: tuple[float, float],
) -> tuple[pd.DataFrame, RangeBiasCalibResult | None]:
    """按 CLI 拟合或加载偏置，返回带 cal 列的 DataFrame 与校准结果。"""
    if args.calib_json is not None:
        result = load_range_bias_calib(args.calib_json.resolve())
        out = apply_range_bias(
            df,
            bias_dev0_m=result.bias_dev0_m,
            bias_dev1_m=result.bias_dev1_m,
        )
        print(f"loaded calibration: {args.calib_json.resolve()}")
        print(format_calib_summary(result))
        return out, result

    if not args.calibrate_range:
        out = apply_range_bias(df, bias_dev0_m=0.0, bias_dev1_m=0.0)
        return out, None

    opts = calib_options_from_args(args)
    result = fit_dual_dev_range_biases(
        df,
        dev0_xy=dev0_xy,
        dev1_xy=dev1_xy,
        **opts,
    )
    print(f"fitted calibration: {format_calib_summary(result)}")
    if args.write_calib_json is not None:
        path = save_range_bias_calib(args.write_calib_json.resolve(), result)
        print(f"output calib json: {path}")
    out = apply_range_bias(
        df,
        bias_dev0_m=result.bias_dev0_m,
        bias_dev1_m=result.bias_dev1_m,
    )
    return out, result
