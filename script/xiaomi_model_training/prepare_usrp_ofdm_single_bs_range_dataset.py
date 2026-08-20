#!/usr/bin/env python3
"""整理单站 OFDM 测距实验目录并构建 HDF5 数据集。

步骤：整理 target_positions.csv → 标签四舍五入两位 → 删除未引用二进制
→ 对齐帧数 → 构建 HDF5。

示例::

    python script/experiment/prepare_usrp_ofdm_single_bs_range_dataset.py --dry-run
    python script/experiment/prepare_usrp_ofdm_single_bs_range_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    DEFAULT_SINGLE_BS_RANGE_VLEN,
    build_usrp_ofdm_single_bs_range_h5,
    infer_single_bs_range_vlen,
    summarize_usrp_ofdm_single_bs_range_h5,
)
from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    MONO_RANGE_TARGET_CSV_COLUMNS,
    _count_divide_cpi_frames,
    _read_mono_range_target_rows,
    _write_mono_range_target_rows,
    prune_unreferenced_mono_range_data,
    round_mono_range_labels,
    sort_mono_range_target_csv,
)


def _default_input_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "xiaomi"
        / "usrp_ofdm_single_bs_range_train"
    )


def _count_csv_rows(csv_path: Path, *, include_empty: bool = False) -> int:
    with csv_path.open(newline="", encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    if include_empty:
        return len(rows)
    return sum(
        1
        for row in rows
        if any((row.get(col) or "").strip() for col in MONO_RANGE_TARGET_CSV_COLUMNS)
    )


def _resolve_vlen_for_dir(input_dir: Path, rows: list[dict[str, str]], vlen: int | None) -> int:
    if vlen is not None:
        return int(vlen)
    if not rows:
        return DEFAULT_SINGLE_BS_RANGE_VLEN
    first_path = input_dir / rows[0]["data_file"]
    if first_path.is_file():
        return infer_single_bs_range_vlen(first_path)
    return DEFAULT_SINGLE_BS_RANGE_VLEN


def _harmonize_mono_range_frame_counts(
    input_dir: Path,
    *,
    vlen: int,
    max_frames: int = 50,
    dry_run: bool = False,
) -> tuple[int, int]:
    """将会话二进制截断至 min(文件帧数, record_max_frames, max_frames)。"""
    max_frames = int(max_frames)
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")

    input_dir = input_dir.resolve()
    csv_path = input_dir / COOPERATIVE_TARGET_CSV
    rows = _read_mono_range_target_rows(csv_path)
    item_bytes = vlen * 8
    files_harmonized = 0

    for session_index, row in enumerate(rows):
        data_path = input_dir / row["data_file"]
        n_frames = _count_divide_cpi_frames(data_path, vlen=vlen)
        declared = int(row["record_max_frames"])
        target_frames = min(n_frames, declared, max_frames)
        if target_frames < 1:
            raise ValueError(
                f"session {session_index} has no frames: "
                f"{row['data_file']} ({n_frames})"
            )
        row["record_max_frames"] = str(target_frames)
        new_size = target_frames * item_bytes
        if data_path.stat().st_size != new_size:
            files_harmonized += 1
            if not dry_run:
                with data_path.open("r+b") as file_obj:
                    file_obj.truncate(new_size)

    if not dry_run and rows:
        _write_mono_range_target_rows(csv_path, rows)

    return len(rows), files_harmonized


def prepare_usrp_ofdm_single_bs_range_dataset(
    input_dir: Path,
    *,
    output_path: Path,
    dry_run: bool = False,
    skip_h5: bool = False,
    compression: str | None = None,
    show_progress: bool = True,
    vlen: int | None = None,
    max_frames: int = 50,
    label_ndigits: int = 2,
) -> dict[str, int | str | Path]:
    """整理 CSV、删除孤儿文件，并可选构建 HDF5。"""
    input_dir = input_dir.resolve()
    csv_path = input_dir / COOPERATIVE_TARGET_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    rows_before = _count_csv_rows(csv_path, include_empty=True)
    rows_preview = _read_mono_range_target_rows(csv_path)
    resolved_vlen = _resolve_vlen_for_dir(input_dir, rows_preview, vlen)

    if dry_run:
        deleted, missing = prune_unreferenced_mono_range_data(input_dir, dry_run=True)
        _, files_harmonized = _harmonize_mono_range_frame_counts(
            input_dir,
            vlen=resolved_vlen,
            max_frames=max_frames,
            dry_run=True,
        )
        result: dict[str, int | str | Path] = {
            "input_dir": input_dir,
            "csv_rows_before": rows_before,
            "orphans_would_delete": len(deleted),
            "files_would_harmonize": files_harmonized,
            "missing_referenced": len(missing),
            "vlen": resolved_vlen,
            "max_frames": int(max_frames),
        }
        if missing:
            missing_paths = ", ".join(
                str(path.relative_to(input_dir)) for path in missing[:5]
            )
            raise FileNotFoundError(
                f"{len(missing)} referenced files missing under {input_dir}; "
                f"examples: {missing_paths}"
            )
        return result

    sort_mono_range_target_csv(csv_path)
    round_mono_range_labels(csv_path, ndigits=label_ndigits, backup_suffix="")
    rows_after = _count_csv_rows(csv_path)

    deleted, missing = prune_unreferenced_mono_range_data(input_dir, dry_run=False)
    if missing:
        missing_paths = ", ".join(
            str(path.relative_to(input_dir)) for path in missing[:5]
        )
        raise FileNotFoundError(
            f"{len(missing)} referenced files missing under {input_dir}; "
            f"examples: {missing_paths}"
        )

    rows_after_prune = _read_mono_range_target_rows(csv_path)
    resolved_vlen = _resolve_vlen_for_dir(input_dir, rows_after_prune, vlen)
    _, files_harmonized = _harmonize_mono_range_frame_counts(
        input_dir,
        vlen=resolved_vlen,
        max_frames=max_frames,
        dry_run=False,
    )

    result = {
        "input_dir": input_dir,
        "csv_rows_before": rows_before,
        "csv_rows_after": rows_after,
        "orphans_deleted": len(deleted),
        "files_harmonized": files_harmonized,
        "missing_referenced": len(missing),
        "vlen": resolved_vlen,
        "max_frames": int(max_frames),
    }

    if skip_h5:
        return result

    h5_path = build_usrp_ofdm_single_bs_range_h5(
        input_dir,
        output_path.resolve(),
        vlen=resolved_vlen,
        compression=compression,
        label_ndigits=label_ndigits,
        show_progress=show_progress,
    )
    summary = summarize_usrp_ofdm_single_bs_range_h5(h5_path)
    result["h5_path"] = h5_path
    result["num_sessions"] = summary["num_sessions"]
    result["frames_per_session"] = summary["frames_per_session"]
    result["total_frames"] = summary["total_frames"]
    return result


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare USRP OFDM single-BS range experiment dir and build HDF5 dataset"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help="directory with target_positions.csv and divide_profiles_*",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output HDF5 path "
            "(default: {input-dir}/usrp_ofdm_single_bs_range_dataset.h5)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report orphan files without modifying CSV, deleting files, or building H5",
    )
    parser.add_argument(
        "--skip-h5",
        action="store_true",
        help="only sort CSV, round labels, and prune orphan files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable HDF5 build progress bar",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=("lzf", "gzip", "none"),
        help="HDF5 compression (default: none)",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=None,
        help=(
            "complex profile length per CPI frame "
            f"(default: auto-infer, fallback {DEFAULT_SINGLE_BS_RANGE_VLEN})"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="CPI frames to keep per sampling point (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    input_dir = args.input_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "usrp_ofdm_single_bs_range_dataset.h5"
    )
    compression = None if args.compression == "none" else args.compression

    result = prepare_usrp_ofdm_single_bs_range_dataset(
        input_dir,
        output_path=output_path,
        dry_run=args.dry_run,
        skip_h5=args.skip_h5,
        compression=compression,
        show_progress=not args.no_progress,
        vlen=args.vlen,
        max_frames=args.max_frames,
    )

    if args.dry_run:
        print(
            f"dry-run: csv_rows={result['csv_rows_before']} "
            f"orphans_would_delete={result['orphans_would_delete']} "
            f"files_would_harmonize={result['files_would_harmonize']} "
            f"vlen={result['vlen']} max_frames={result['max_frames']}"
        )
        return

    print(
        f"sorted+rounded csv: {result['csv_rows_before']} -> {result['csv_rows_after']} "
        f"rows ({input_dir / COOPERATIVE_TARGET_CSV})"
    )
    print(f"pruned orphans: {result['orphans_deleted']} files deleted")
    print(f"harmonized frames: {result['files_harmonized']} files truncated")
    print(f"vlen: {result['vlen']}")
    print(f"max_frames: {result['max_frames']}")

    if args.skip_h5:
        return

    file_size_mb = Path(result["h5_path"]).stat().st_size / (1024 * 1024)
    print(
        f"output h5: {result['h5_path']} "
        f"({file_size_mb:.1f} MiB, sessions={result['num_sessions']}, "
        f"frames_per_session={result['frames_per_session']}, "
        f"total_frames={result['total_frames']})"
    )


if __name__ == "__main__":
    main()
