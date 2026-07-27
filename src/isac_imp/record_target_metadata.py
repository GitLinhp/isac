"""Cooperative monostatic 录制元数据：目标位置 CSV。"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

COOPERATIVE_TARGET_CSV = "target_positions.csv"
INNER_RADIUS_CM = 50.0
WITHIN_50CM_DIR = "within_50cm"
DEFAULT_COOPERATIVE_VLEN = 32768
_COMPLEX64_ITEMSIZE = 8

CSV_COLUMNS = (
    "recorded_at_utc",
    "target_x_cm",
    "target_y_cm",
    "dev0_file",
    "dev1_file",
    "record_max_frames",
)

_LOG_PREFIX = "[RecordTargetMetadata]"


def _relative_to_parent(parent_dir: Path, data_path: str) -> str:
    return str(Path(data_path).resolve().relative_to(parent_dir.resolve()))


def append_cooperative_target_row(
    parent_dir: str | Path,
    *,
    target_x_cm: float,
    target_y_cm: float,
    dev0_file: str,
    dev1_file: str,
    record_max_frames: int,
) -> Path:
    """向 ``parent_dir/target_positions.csv`` 追加一行会话元数据。"""
    parent = Path(parent_dir)
    parent.mkdir(parents=True, exist_ok=True)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_x_cm": float(target_x_cm),
        "target_y_cm": float(target_y_cm),
        "dev0_file": _relative_to_parent(parent, dev0_file),
        "dev1_file": _relative_to_parent(parent, dev1_file),
        "record_max_frames": int(record_max_frames),
    }

    with csv_path.open("a", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"{_LOG_PREFIX} appended → {csv_path}", file=sys.stderr)
    return csv_path


def _is_empty_csv_row(row: dict[str, str]) -> bool:
    return not any((row.get(col) or "").strip() for col in CSV_COLUMNS)


def _sort_key(row: dict[str, str]) -> tuple[float, float, str]:
    return (
        float(row["target_y_cm"]),
        float(row["target_x_cm"]),
        row["recorded_at_utc"],
    )


def sort_cooperative_target_csv(
    csv_path: str | Path,
    *,
    backup_suffix: str = ".bak",
) -> Path:
    """按 target_y_cm、target_x_cm、recorded_at_utc 升序整理 CSV，删除空行。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as csv_f:
        rows = [
            row
            for row in csv.DictReader(csv_f)
            if not _is_empty_csv_row(row)
        ]

    if backup_suffix:
        backup_path = path.with_name(path.name + backup_suffix)
        shutil.copy2(path, backup_path)

    rows.sort(key=_sort_key)

    with path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{_LOG_PREFIX} sorted {len(rows)} rows → {path}", file=sys.stderr)
    return path


def is_inner_target_xy_m(
    x_m: float,
    y_m: float,
    *,
    radius_m: float = INNER_RADIUS_CM / 100.0,
) -> bool:
    """目标位置是否落在内侧区域（默认 |x|,|y| <= 0.5 m）。"""
    return abs(x_m) <= radius_m and abs(y_m) <= radius_m


def _is_inner_coord(
    x_cm: float,
    y_cm: float,
    *,
    radius_cm: float = INNER_RADIUS_CM,
) -> bool:
    return is_inner_target_xy_m(
        x_cm / 100.0,
        y_cm / 100.0,
        radius_m=radius_cm / 100.0,
    )


def _is_inner_boundary_coord(
    x_cm: float,
    y_cm: float,
    *,
    radius_cm: float = INNER_RADIUS_CM,
) -> bool:
    return _is_inner_coord(x_cm, y_cm, radius_cm=radius_cm) and (
        abs(x_cm) == radius_cm or abs(y_cm) == radius_cm
    )


def _read_cooperative_target_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_f:
        return [
            row
            for row in csv.DictReader(csv_f)
            if not _is_empty_csv_row(row)
        ]


def _write_cooperative_target_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def split_cooperative_target_csv(
    parent_dir: str | Path,
    *,
    backup_suffix: str = ".bak",
) -> tuple[Path, Path]:
    """按 ±50cm 规则拆分 target_positions.csv。"""
    parent = Path(parent_dir)
    root_csv = parent / COOPERATIVE_TARGET_CSV
    inner_csv = parent / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV

    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)

    rows = _read_cooperative_target_rows(root_csv)
    rows.sort(key=_sort_key)

    inner_groups: dict[tuple[float, float], list[dict[str, str]]] = {}
    root_rows: list[dict[str, str]] = []

    for row in rows:
        x_cm = float(row["target_x_cm"])
        y_cm = float(row["target_y_cm"])
        if _is_inner_coord(x_cm, y_cm):
            inner_groups.setdefault((x_cm, y_cm), []).append(row)
        else:
            root_rows.append(row)

    inner_second_rows: list[dict[str, str]] = []
    for coord_rows in inner_groups.values():
        coord_rows.sort(key=lambda row: row["recorded_at_utc"])
        for index, row in enumerate(coord_rows):
            x_cm = float(row["target_x_cm"])
            y_cm = float(row["target_y_cm"])
            if index == 1:
                inner_second_rows.append(row)
            elif index == 0 and _is_inner_boundary_coord(x_cm, y_cm):
                inner_second_rows.append(row)
            else:
                root_rows.append(row)

    root_rows.sort(key=_sort_key)
    inner_second_rows.sort(key=_sort_key)

    if backup_suffix:
        backup_path = root_csv.with_name(root_csv.name + backup_suffix)
        shutil.copy2(root_csv, backup_path)

    _write_cooperative_target_rows(root_csv, root_rows)
    _write_cooperative_target_rows(inner_csv, inner_second_rows)

    print(
        f"{_LOG_PREFIX} split root={len(root_rows)} rows → {root_csv}",
        file=sys.stderr,
    )
    print(
        f"{_LOG_PREFIX} split within_50cm={len(inner_second_rows)} rows → {inner_csv}",
        file=sys.stderr,
    )
    return root_csv, inner_csv


def dedupe_cooperative_target_csv_latest(
    csv_path: str | Path,
    *,
    backup_suffix: str = ".bak",
) -> Path:
    """同坐标多条记录时仅保留 recorded_at_utc 最新的一条。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = _read_cooperative_target_rows(path)
    original_count = len(rows)

    deduped_rows = _keep_latest_row_per_coord(rows)

    if backup_suffix:
        backup_path = path.with_name(path.name + backup_suffix)
        shutil.copy2(path, backup_path)

    _write_cooperative_target_rows(path, deduped_rows)

    print(
        f"{_LOG_PREFIX} deduped {original_count} -> {len(deduped_rows)} rows → {path}",
        file=sys.stderr,
    )
    return path


def _keep_latest_row_per_coord(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    latest_by_coord: dict[tuple[float, float], dict[str, str]] = {}
    for row in rows:
        coord = (float(row["target_x_cm"]), float(row["target_y_cm"]))
        current = latest_by_coord.get(coord)
        if current is None or row["recorded_at_utc"] > current["recorded_at_utc"]:
            latest_by_coord[coord] = row
    return sorted(latest_by_coord.values(), key=_sort_key)


def merge_cooperative_target_csv_plan_b(
    parent_dir: str | Path,
    *,
    backup_suffix: str = ".bak",
    clear_within: bool = True,
) -> Path:
    """合并根目录与 within_50cm CSV，每坐标保留最新一条写入根目录。"""
    parent = Path(parent_dir)
    root_csv = parent / COOPERATIVE_TARGET_CSV
    inner_csv = parent / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV

    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)

    root_rows = _read_cooperative_target_rows(root_csv)
    inner_rows = _read_cooperative_target_rows(inner_csv) if inner_csv.is_file() else []
    merged_count = len(root_rows) + len(inner_rows)

    merged_rows = _keep_latest_row_per_coord(root_rows + inner_rows)

    if backup_suffix:
        backup_path = root_csv.with_name(root_csv.name + backup_suffix)
        shutil.copy2(root_csv, backup_path)

    _write_cooperative_target_rows(root_csv, merged_rows)
    if clear_within:
        _write_cooperative_target_rows(inner_csv, [])

    print(
        f"{_LOG_PREFIX} merged {merged_count} -> {len(merged_rows)} rows → {root_csv}",
        file=sys.stderr,
    )
    if clear_within:
        print(
            f"{_LOG_PREFIX} cleared within_50cm → {inner_csv}",
            file=sys.stderr,
        )
    return root_csv


def migrate_exact_50cm_boundary_to_within_50cm(
    parent_dir: str | Path,
    *,
    backup_suffix: str = ".bak",
) -> tuple[Path, Path]:
    """将根目录 CSV 中 |X|或|Y|恰为 50cm 的内侧点迁入 within_50cm/target_positions.csv。"""
    parent = Path(parent_dir)
    root_csv = parent / COOPERATIVE_TARGET_CSV
    inner_csv = parent / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV

    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)

    root_rows = _read_cooperative_target_rows(root_csv)
    inner_rows = _read_cooperative_target_rows(inner_csv) if inner_csv.is_file() else []

    to_move: list[dict[str, str]] = []
    kept_root: list[dict[str, str]] = []
    for row in root_rows:
        x_cm = float(row["target_x_cm"])
        y_cm = float(row["target_y_cm"])
        if _is_inner_boundary_coord(x_cm, y_cm):
            to_move.append(row)
        else:
            kept_root.append(row)

    merged_inner = inner_rows + to_move
    kept_root.sort(key=_sort_key)
    merged_inner.sort(key=_sort_key)

    if backup_suffix:
        backup_path = root_csv.with_name(root_csv.name + backup_suffix)
        shutil.copy2(root_csv, backup_path)

    _write_cooperative_target_rows(root_csv, kept_root)
    _write_cooperative_target_rows(inner_csv, merged_inner)

    print(
        f"{_LOG_PREFIX} migrated {len(to_move)} boundary rows root -> within_50cm",
        file=sys.stderr,
    )
    print(
        f"{_LOG_PREFIX} root={len(kept_root)} rows → {root_csv}",
        file=sys.stderr,
    )
    print(
        f"{_LOG_PREFIX} within_50cm={len(merged_inner)} rows → {inner_csv}",
        file=sys.stderr,
    )
    return root_csv, inner_csv


def prune_unreferenced_cooperative_data(
    parent_dir: str | Path,
    *,
    dry_run: bool = False,
) -> tuple[list[Path], list[Path]]:
    """删除 ``dev0/``、``dev1/`` 中未被 ``target_positions.csv`` 引用的文件。"""
    parent = Path(parent_dir)
    root_csv = parent / COOPERATIVE_TARGET_CSV
    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)

    rows = _read_cooperative_target_rows(root_csv)
    referenced: set[str] = set()
    for row in rows:
        referenced.add(row["dev0_file"])
        referenced.add(row["dev1_file"])

    dev_dirs = ("dev0", "dev1")
    on_disk: list[Path] = []
    for dev_name in dev_dirs:
        dev_dir = parent / dev_name
        if not dev_dir.is_dir():
            continue
        for path in dev_dir.iterdir():
            if path.is_file():
                on_disk.append(path)

    deleted: list[Path] = []
    for path in on_disk:
        rel = str(path.relative_to(parent))
        if rel not in referenced:
            deleted.append(path)
            if not dry_run:
                path.unlink()

    missing_referenced: list[Path] = []
    for rel in sorted(referenced):
        path = parent / rel
        if not path.is_file():
            missing_referenced.append(path)

    action = "would delete" if dry_run else "deleted"
    print(
        f"{_LOG_PREFIX} referenced={len(referenced)} on_disk={len(on_disk)} "
        f"{action}={len(deleted)} missing={len(missing_referenced)}",
        file=sys.stderr,
    )
    if missing_referenced:
        print(
            f"{_LOG_PREFIX} warning: {len(missing_referenced)} referenced files missing",
            file=sys.stderr,
        )
    return deleted, missing_referenced


def _count_divide_cpi_frames(path: Path, *, vlen: int) -> int:
    item_bytes = vlen * _COMPLEX64_ITEMSIZE
    size = os.path.getsize(path)
    if size % item_bytes != 0:
        raise ValueError(
            f"file size {size} is not a multiple of frame size {item_bytes}; "
            f"path={path!r}"
        )
    return size // item_bytes


def truncate_cooperative_recordings(
    parent_dir: str | Path,
    *,
    max_frames: int = 50,
    vlen: int = DEFAULT_COOPERATIVE_VLEN,
    dry_run: bool = False,
    show_progress: bool = True,
    backup_suffix: str = ".bak",
) -> tuple[int, int]:
    """将每个采样会话的 dev0/dev1 二进制文件截断为前 ``max_frames`` 帧 CPI。

    同步更新 ``target_positions.csv`` 中 ``record_max_frames`` 列。
    """
    parent = Path(parent_dir)
    root_csv = parent / COOPERATIVE_TARGET_CSV
    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)

    max_frames = max(1, int(max_frames))
    rows = _read_cooperative_target_rows(root_csv)
    if not rows:
        raise ValueError(f"no data rows in {root_csv}")

    files_truncated = 0
    iterator = tqdm(
        rows,
        desc="截断 CPI 帧",
        unit="session",
        disable=not show_progress,
    )
    for session_index, row in enumerate(iterator):
        dev0_path = parent / row["dev0_file"]
        dev1_path = parent / row["dev1_file"]
        if not dev0_path.is_file():
            raise FileNotFoundError(dev0_path)
        if not dev1_path.is_file():
            raise FileNotFoundError(dev1_path)

        n_dev0 = _count_divide_cpi_frames(dev0_path, vlen=vlen)
        n_dev1 = _count_divide_cpi_frames(dev1_path, vlen=vlen)
        if n_dev0 < max_frames or n_dev1 < max_frames:
            raise ValueError(
                f"session {session_index} has insufficient frames for truncate to "
                f"{max_frames}: {row['dev0_file']} ({n_dev0}), "
                f"{row['dev1_file']} ({n_dev1})"
            )
        if n_dev0 != n_dev1:
            print(
                f"{_LOG_PREFIX} warning: session {session_index} frame mismatch "
                f"{n_dev0} vs {n_dev1}, truncating both to {max_frames}",
                file=sys.stderr,
            )

        iterator.set_postfix(
            session=f"{session_index + 1}/{len(rows)}",
            target=f"({row['target_x_cm']},{row['target_y_cm']})cm",
            refresh=False,
        )

        new_size = max_frames * vlen * _COMPLEX64_ITEMSIZE
        for path, n_frames in ((dev0_path, n_dev0), (dev1_path, n_dev1)):
            if n_frames < max_frames:
                raise ValueError(
                    f"{path} has {n_frames} frames, need at least {max_frames} to truncate"
                )
            if path.stat().st_size != new_size:
                files_truncated += 1
                if not dry_run:
                    with path.open("r+b") as f:
                        f.truncate(new_size)

    for row in rows:
        row["record_max_frames"] = str(max_frames)

    if not dry_run:
        if backup_suffix:
            backup_path = root_csv.with_name(root_csv.name + backup_suffix)
            shutil.copy2(root_csv, backup_path)
        _write_cooperative_target_rows(root_csv, rows)

    action = "would truncate" if dry_run else "truncated"
    print(
        f"{_LOG_PREFIX} {action} sessions={len(rows)} files={files_truncated} "
        f"max_frames={max_frames} → {root_csv}",
        file=sys.stderr,
    )
    return len(rows), files_truncated
