#!/usr/bin/env python3
"""整理 cooperative monostatic 实验目录并构建 HDF5 数据集。

步骤：整理 target_positions.csv → 删除未引用二进制 → 对齐 dev0/dev1 帧数 → 构建 HDF5。

示例::

    python script/experiment/prepare_cooperative_monostatic_dataset.py --dry-run
    python script/experiment/prepare_cooperative_monostatic_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    build_cooperative_monostatic_h5,
    DEFAULT_COOPERATIVE_VLEN,
    summarize_cooperative_monostatic_h5,
)
from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    CSV_COLUMNS,
    _count_divide_cpi_frames,
    _read_cooperative_target_rows,
    _write_cooperative_target_rows,
    prune_unreferenced_cooperative_data,
    sort_cooperative_target_csv,
)


def _default_input_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "experiment" / "cooperative_monostatic"


def _count_csv_rows(csv_path: Path, *, include_empty: bool = False) -> int:
    with csv_path.open(newline="", encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    if include_empty:
        return len(rows)
    return sum(1 for row in rows if any(row.get(col) for col in CSV_COLUMNS))


def _harmonize_cooperative_frame_counts(
    input_dir: Path,
    *,
    vlen: int = DEFAULT_COOPERATIVE_VLEN,
    dry_run: bool = False,
) -> tuple[int, int]:
    """将会话 dev0/dev1 截断至相同帧数：min(n_dev0, n_dev1, record_max_frames)。"""
    input_dir = input_dir.resolve()
    csv_path = input_dir / COOPERATIVE_TARGET_CSV
    rows = _read_cooperative_target_rows(csv_path)
    item_bytes = vlen * 8
    files_harmonized = 0

    for session_index, row in enumerate(rows):
        dev0_path = input_dir / row["dev0_file"]
        dev1_path = input_dir / row["dev1_file"]
        n_dev0 = _count_divide_cpi_frames(dev0_path, vlen=vlen)
        n_dev1 = _count_divide_cpi_frames(dev1_path, vlen=vlen)
        declared = int(row["record_max_frames"])
        target_frames = min(n_dev0, n_dev1, declared)
        if target_frames < 1:
            raise ValueError(
                f"session {session_index} has no frames: "
                f"{row['dev0_file']} ({n_dev0}), {row['dev1_file']} ({n_dev1})"
            )
        row["record_max_frames"] = str(target_frames)
        new_size = target_frames * item_bytes
        for path in (dev0_path, dev1_path):
            if path.stat().st_size != new_size:
                files_harmonized += 1
                if not dry_run:
                    with path.open("r+b") as file_obj:
                        file_obj.truncate(new_size)

    if not dry_run and rows:
        _write_cooperative_target_rows(csv_path, rows)

    return len(rows), files_harmonized


def prepare_cooperative_monostatic_dataset(
    input_dir: Path,
    *,
    output_path: Path,
    dry_run: bool = False,
    skip_h5: bool = False,
    compression: str | None = None,
    show_progress: bool = True,
    vlen: int = DEFAULT_COOPERATIVE_VLEN,
) -> dict[str, int | str | Path]:
    """整理 CSV、删除孤儿文件，并可选构建 HDF5。"""
    input_dir = input_dir.resolve()
    csv_path = input_dir / COOPERATIVE_TARGET_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    rows_before = _count_csv_rows(csv_path, include_empty=True)
    if dry_run:
        deleted, missing = prune_unreferenced_cooperative_data(input_dir, dry_run=True)
        _, files_harmonized = _harmonize_cooperative_frame_counts(
            input_dir,
            vlen=vlen,
            dry_run=True,
        )
        result: dict[str, int | str | Path] = {
            "input_dir": input_dir,
            "csv_rows_before": rows_before,
            "orphans_would_delete": len(deleted),
            "files_would_harmonize": files_harmonized,
            "missing_referenced": len(missing),
        }
        if missing:
            missing_paths = ", ".join(str(path.relative_to(input_dir)) for path in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} referenced files missing under {input_dir}; "
                f"examples: {missing_paths}"
            )
        return result

    sort_cooperative_target_csv(csv_path)
    rows_after = _count_csv_rows(csv_path)

    deleted, missing = prune_unreferenced_cooperative_data(input_dir, dry_run=False)
    if missing:
        missing_paths = ", ".join(str(path.relative_to(input_dir)) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} referenced files missing under {input_dir}; "
            f"examples: {missing_paths}"
        )

    _, files_harmonized = _harmonize_cooperative_frame_counts(
        input_dir,
        vlen=vlen,
        dry_run=False,
    )

    result = {
        "input_dir": input_dir,
        "csv_rows_before": rows_before,
        "csv_rows_after": rows_after,
        "orphans_deleted": len(deleted),
        "files_harmonized": files_harmonized,
        "missing_referenced": len(missing),
    }

    if skip_h5:
        return result

    h5_path = build_cooperative_monostatic_h5(
        input_dir,
        output_path.resolve(),
        vlen=vlen,
        compression=compression,
        show_progress=show_progress,
    )
    summary = summarize_cooperative_monostatic_h5(h5_path)
    result["h5_path"] = h5_path
    result["num_sessions"] = summary["num_sessions"]
    result["frames_per_session"] = summary["frames_per_session"]
    result["total_frames"] = summary["total_frames"]
    return result


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cooperative monostatic experiment dir and build HDF5 dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help="directory with target_positions.csv and dev0/dev1/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output HDF5 path (default: {input-dir}/cooperative_monostatic_dataset.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report orphan files without modifying CSV, deleting files, or building H5",
    )
    parser.add_argument(
        "--skip-h5",
        action="store_true",
        help="only sort CSV and prune orphan files",
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
    return parser.parse_args()


def main() -> None:
    args = argument_parser()
    input_dir = args.input_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "cooperative_monostatic_dataset.h5"
    )
    compression = None if args.compression == "none" else args.compression

    result = prepare_cooperative_monostatic_dataset(
        input_dir,
        output_path=output_path,
        dry_run=args.dry_run,
        skip_h5=args.skip_h5,
        compression=compression,
        show_progress=not args.no_progress,
    )

    if args.dry_run:
        print(
            f"dry-run: csv_rows={result['csv_rows_before']} "
            f"orphans_would_delete={result['orphans_would_delete']} "
            f"files_would_harmonize={result['files_would_harmonize']}"
        )
        return

    print(
        f"sorted csv: {result['csv_rows_before']} -> {result['csv_rows_after']} rows "
        f"({input_dir / COOPERATIVE_TARGET_CSV})"
    )
    print(f"pruned orphans: {result['orphans_deleted']} files deleted")
    print(f"harmonized frames: {result['files_harmonized']} files truncated")

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
