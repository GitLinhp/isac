#!/usr/bin/env python3
"""从 cooperative monostatic raw CPI HDF5 构建 CNN 训练用特征 sidecar。

示例::

    python script/experiment/build_cooperative_monostatic_features_h5.py \\
        --input-h5 data/experiment/cooperative_monostatic_measurement0/cooperative_monostatic_dataset.h5 \\
        --range-roi 0 4.5 --feature-mode real_imag
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isac_imp.cooperative_monostatic_pipeline import DEFAULT_RANGE_ROI
from isac_imp.data_collection.cooperative_monostatic_dataset import (
    SIDECAR_FEATURE_MODES,
    build_cooperative_monostatic_features_h5,
    default_features_h5_path,
    summarize_cooperative_monostatic_h5,
)


def _parse_range_roi(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range-roi 须为两个浮点数：min_m max_m")
    lo, hi = float(values[0]), float(values[1])
    if lo >= hi:
        raise argparse.ArgumentTypeError(f"range-roi 须满足 min < max，收到 {lo} {hi}")
    return lo, hi


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cooperative monostatic CNN features sidecar HDF5 from raw CPI dataset"
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="source raw cooperative_monostatic_dataset.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output features HDF5 (default: derived from input stem / ROI / feature-mode)",
    )
    parser.add_argument(
        "--range-roi",
        type=float,
        nargs=2,
        default=list(DEFAULT_RANGE_ROI),
        metavar=("MIN_M", "MAX_M"),
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=list(SIDECAR_FEATURE_MODES),
        default="legacy_4ch",
        help="offline feature mode (default: legacy_4ch)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bar",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    input_h5 = args.input_h5.resolve()
    range_roi = _parse_range_roi(list(args.range_roi))
    feature_mode = str(args.feature_mode)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_features_h5_path(
            input_h5, range_roi=range_roi, feature_mode=feature_mode
        )
    )

    build_cooperative_monostatic_features_h5(
        input_h5,
        output_path,
        range_roi=range_roi,
        feature_mode=feature_mode,
        show_progress=not args.no_progress,
    )

    summary = summarize_cooperative_monostatic_h5(output_path)
    file_size_mb = summary["file_size_bytes"] / (1024 * 1024)
    print(
        f"output features h5: {summary['path']} "
        f"({file_size_mb:.2f} MiB, frames={summary['total_frames']}, "
        f"features_shape={summary['features_shape']}, "
        f"feature_mode={feature_mode}, roi={range_roi})"
    )


if __name__ == "__main__":
    main()
