"""record_target_metadata CSV 单元测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    CSV_COLUMNS,
    MONO_RANGE_TARGET_CSV_COLUMNS,
    WITHIN_50CM_DIR,
    append_cooperative_target_row,
    append_mono_range_target_row,
    dedupe_cooperative_target_csv_latest,
    merge_cooperative_target_csv_plan_b,
    migrate_exact_50cm_boundary_to_within_50cm,
    prune_unreferenced_cooperative_data,
    sort_cooperative_target_csv,
    split_cooperative_target_csv,
)


def test_append_creates_header_and_first_row(tmp_path: Path) -> None:
    dev0 = tmp_path / "dev0" / "divide_profiles_001"
    dev1 = tmp_path / "dev1" / "divide_profiles_001"
    dev0.parent.mkdir(parents=True)
    dev1.parent.mkdir(parents=True)
    dev0.touch()
    dev1.touch()

    csv_path = append_cooperative_target_row(
        tmp_path,
        target_x_cm=12.0,
        target_y_cm=-5.0,
        dev0_file=str(dev0),
        dev1_file=str(dev1),
        record_max_frames=100,
    )

    assert csv_path == tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert len(rows) == 1
    assert rows[0]["target_x_cm"] == "12.0"
    assert rows[0]["target_y_cm"] == "-5.0"
    assert rows[0]["dev0_file"] == "dev0/divide_profiles_001"
    assert rows[0]["dev1_file"] == "dev1/divide_profiles_001"
    assert rows[0]["record_max_frames"] == "100"
    assert rows[0]["recorded_at_utc"]


def test_append_second_row_without_duplicate_header(tmp_path: Path) -> None:
    dev0 = tmp_path / "dev0" / "divide_profiles_001"
    dev1 = tmp_path / "dev1" / "divide_profiles_001"
    dev0.parent.mkdir(parents=True)
    dev1.parent.mkdir(parents=True)
    dev0.touch()
    dev1.touch()

    append_cooperative_target_row(
        tmp_path,
        target_x_cm=0.0,
        target_y_cm=0.0,
        dev0_file=str(dev0),
        dev1_file=str(dev1),
        record_max_frames=50,
    )
    dev0_2 = tmp_path / "dev0" / "divide_profiles_002"
    dev1_2 = tmp_path / "dev1" / "divide_profiles_002"
    dev0_2.touch()
    dev1_2.touch()
    append_cooperative_target_row(
        tmp_path,
        target_x_cm=3.0,
        target_y_cm=4.0,
        dev0_file=str(dev0_2),
        dev1_file=str(dev1_2),
        record_max_frames=80,
    )

    with (tmp_path / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert len(rows) == 2
    assert rows[1]["dev0_file"] == "dev0/divide_profiles_002"
    assert rows[1]["target_x_cm"] == "3.0"

    with (tmp_path / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        header = csv_f.readline().strip().split(",")
    assert header == list(CSV_COLUMNS)


def test_append_mono_range_target_row(tmp_path: Path) -> None:
    data = tmp_path / "divide_profiles_001"
    data.touch()

    csv_path = append_mono_range_target_row(
        tmp_path,
        target_range_m=1.25,
        data_file=str(data),
        record_max_frames=55,
    )

    assert csv_path == tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open(encoding="utf-8") as csv_f:
        reader = csv.DictReader(csv_f)
        assert list(reader.fieldnames or ()) == list(MONO_RANGE_TARGET_CSV_COLUMNS)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["target_range_m"] == "1.25"
    assert rows[0]["data_file"] == "divide_profiles_001"
    assert rows[0]["record_max_frames"] == "55"
    assert rows[0]["recorded_at_utc"]
    assert "target_x_cm" not in rows[0]


def test_sort_cooperative_target_csv_removes_empty_and_sorts(tmp_path: Path) -> None:
    csv_path = tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "recorded_at_utc": "2026-07-26T10:00:00+00:00",
                "target_x_cm": "20.0",
                "target_y_cm": "10.0",
                "dev0_file": "dev0/divide_profiles_002",
                "dev1_file": "dev1/divide_profiles_002",
                "record_max_frames": "100",
            }
        )
        writer.writerow(
            {
                "recorded_at_utc": "",
                "target_x_cm": "",
                "target_y_cm": "",
                "dev0_file": "",
                "dev1_file": "",
                "record_max_frames": "",
            }
        )
        writer.writerow(
            {
                "recorded_at_utc": "2026-07-26T09:00:00+00:00",
                "target_x_cm": "0.0",
                "target_y_cm": "0.0",
                "dev0_file": "dev0/divide_profiles_001",
                "dev1_file": "dev1/divide_profiles_001",
                "record_max_frames": "100",
            }
        )
        writer.writerow(
            {
                "recorded_at_utc": "2026-07-26T11:00:00+00:00",
                "target_x_cm": "0.0",
                "target_y_cm": "0.0",
                "dev0_file": "dev0/divide_profiles_003",
                "dev1_file": "dev1/divide_profiles_003",
                "record_max_frames": "100",
            }
        )
        writer.writerow(
            {
                "recorded_at_utc": "2026-07-26T08:00:00+00:00",
                "target_x_cm": "-10.0",
                "target_y_cm": "-20.0",
                "dev0_file": "dev0/divide_profiles_004",
                "dev1_file": "dev1/divide_profiles_004",
                "record_max_frames": "100",
            }
        )

    sort_cooperative_target_csv(csv_path)

    backup_path = csv_path.with_name(csv_path.name + ".bak")
    assert backup_path.is_file()

    with csv_path.open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))

    assert len(rows) == 4
    assert [(float(r["target_y_cm"]), float(r["target_x_cm"])) for r in rows] == [
        (-20.0, -10.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (10.0, 20.0),
    ]
    assert rows[1]["recorded_at_utc"] == "2026-07-26T09:00:00+00:00"
    assert rows[2]["recorded_at_utc"] == "2026-07-26T11:00:00+00:00"


def _row(
    *,
    recorded_at_utc: str,
    target_x_cm: str,
    target_y_cm: str,
    dev_suffix: str,
) -> dict[str, str]:
    return {
        "recorded_at_utc": recorded_at_utc,
        "target_x_cm": target_x_cm,
        "target_y_cm": target_y_cm,
        "dev0_file": f"dev0/divide_profiles_{dev_suffix}",
        "dev1_file": f"dev1/divide_profiles_{dev_suffix}",
        "record_max_frames": "100",
    }


def test_split_cooperative_target_csv_routes_inner_second_samples(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="60.0",
                target_y_cm="0.0",
                dev_suffix="001",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T09:00:00+00:00",
                target_x_cm="10.0",
                target_y_cm="10.0",
                dev_suffix="002",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="10.0",
                target_y_cm="10.0",
                dev_suffix="003",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T08:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="004",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T09:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="005",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="006",
            )
        )

    root_csv, inner_csv = split_cooperative_target_csv(tmp_path)

    assert root_csv == csv_path
    assert inner_csv == tmp_path / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV
    assert (csv_path.with_name(csv_path.name + ".bak")).is_file()

    with root_csv.open(encoding="utf-8") as csv_f:
        root_rows = list(csv.DictReader(csv_f))
    with inner_csv.open(encoding="utf-8") as csv_f:
        inner_rows = list(csv.DictReader(csv_f))

    assert len(root_rows) == 4
    assert len(inner_rows) == 2
    assert {r["dev0_file"] for r in root_rows} == {
        "dev0/divide_profiles_001",
        "dev0/divide_profiles_002",
        "dev0/divide_profiles_004",
        "dev0/divide_profiles_006",
    }
    assert {r["dev0_file"] for r in inner_rows} == {
        "dev0/divide_profiles_003",
        "dev0/divide_profiles_005",
    }

    root_keys = [
        (float(r["target_y_cm"]), float(r["target_x_cm"]), r["recorded_at_utc"])
        for r in root_rows
    ]
    inner_keys = [
        (float(r["target_y_cm"]), float(r["target_x_cm"]), r["recorded_at_utc"])
        for r in inner_rows
    ]
    assert root_keys == sorted(root_keys)
    assert inner_keys == sorted(inner_keys)


def test_dedupe_cooperative_target_csv_latest_keeps_newest_per_coord(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T08:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="001",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T09:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="002",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="20.0",
                target_y_cm="20.0",
                dev_suffix="003",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T11:00:00+00:00",
                target_x_cm="0.0",
                target_y_cm="10.0",
                dev_suffix="004",
            )
        )

    dedupe_cooperative_target_csv_latest(csv_path)

    with csv_path.open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))

    assert len(rows) == 2
    assert rows[0]["dev0_file"] == "dev0/divide_profiles_004"
    assert rows[1]["dev0_file"] == "dev0/divide_profiles_003"
    coords = [(float(r["target_x_cm"]), float(r["target_y_cm"])) for r in rows]
    assert len(coords) == len(set(coords))


def test_migrate_exact_50cm_boundary_to_within_50cm(tmp_path: Path) -> None:
    root_csv = tmp_path / COOPERATIVE_TARGET_CSV
    inner_csv = tmp_path / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV
    with root_csv.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T09:00:00+00:00",
                target_x_cm="50.0",
                target_y_cm="0.0",
                dev_suffix="001",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="10.0",
                target_y_cm="10.0",
                dev_suffix="002",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T11:00:00+00:00",
                target_x_cm="60.0",
                target_y_cm="0.0",
                dev_suffix="003",
            )
        )
    inner_csv.parent.mkdir(parents=True, exist_ok=True)
    with inner_csv.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:30:00+00:00",
                target_x_cm="-50.0",
                target_y_cm="-40.0",
                dev_suffix="004",
            )
        )

    migrate_exact_50cm_boundary_to_within_50cm(tmp_path)

    with root_csv.open(encoding="utf-8") as csv_f:
        root_rows = list(csv.DictReader(csv_f))
    with inner_csv.open(encoding="utf-8") as csv_f:
        inner_rows = list(csv.DictReader(csv_f))

    assert len(root_rows) == 2
    assert {r["dev0_file"] for r in root_rows} == {
        "dev0/divide_profiles_002",
        "dev0/divide_profiles_003",
    }
    assert len(inner_rows) == 2
    assert {r["dev0_file"] for r in inner_rows} == {
        "dev0/divide_profiles_001",
        "dev0/divide_profiles_004",
    }


def test_merge_cooperative_target_csv_plan_b_keeps_latest_and_clears_within(
    tmp_path: Path,
) -> None:
    root_csv = tmp_path / COOPERATIVE_TARGET_CSV
    inner_csv = tmp_path / WITHIN_50CM_DIR / COOPERATIVE_TARGET_CSV
    inner_csv.parent.mkdir(parents=True, exist_ok=True)

    with root_csv.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T09:00:00+00:00",
                target_x_cm="10.0",
                target_y_cm="10.0",
                dev_suffix="001",
            )
        )
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="60.0",
                target_y_cm="0.0",
                dev_suffix="002",
            )
        )
    with inner_csv.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T11:00:00+00:00",
                target_x_cm="10.0",
                target_y_cm="10.0",
                dev_suffix="003",
            )
        )

    merge_cooperative_target_csv_plan_b(tmp_path)

    with root_csv.open(encoding="utf-8") as csv_f:
        root_rows = list(csv.DictReader(csv_f))
    with inner_csv.open(encoding="utf-8") as csv_f:
        inner_rows = list(csv.DictReader(csv_f))

    assert len(root_rows) == 2
    assert root_rows[0]["dev0_file"] == "dev0/divide_profiles_002"
    assert root_rows[1]["dev0_file"] == "dev0/divide_profiles_003"
    assert len(inner_rows) == 0


def test_prune_unreferenced_cooperative_data(tmp_path: Path) -> None:
    csv_path = tmp_path / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _row(
                recorded_at_utc="2026-07-26T10:00:00+00:00",
                target_x_cm="0.0",
                target_y_cm="0.0",
                dev_suffix="001",
            )
        )

    for dev_name in ("dev0", "dev1"):
        dev_dir = tmp_path / dev_name
        dev_dir.mkdir(parents=True)
        for suffix in ("001", "002", "003"):
            (dev_dir / f"divide_profiles_{suffix}").touch()

    deleted, missing = prune_unreferenced_cooperative_data(tmp_path)

    assert len(deleted) == 4
    assert missing == []
    for dev_name in ("dev0", "dev1"):
        dev_dir = tmp_path / dev_name
        assert (dev_dir / "divide_profiles_001").is_file()
        assert not (dev_dir / "divide_profiles_002").exists()
        assert not (dev_dir / "divide_profiles_003").exists()
