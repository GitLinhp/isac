"""方案 C Run2 采集路书 CSV 生成测试。"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from isac import PROJECT_ROOT

_MANIFEST_SCRIPT = (
    PROJECT_ROOT / "script" / "experiment" / "generate_plan_c_collection_manifest.py"
)
_EXPECTED_CSV = (
    PROJECT_ROOT
    / "data"
    / "experiment"
    / "cooperative_monostatic_plan_c"
    / "collection_run2_targets.csv"
)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "generate_plan_c_collection_manifest",
        _MANIFEST_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest_mod():
    return _load_manifest_module()


def test_master_grid_has_217_points(manifest_mod) -> None:
    master = manifest_mod.build_master_grid_xy_m()
    assert len(master) == 217


def test_run2_manifest_row_counts(manifest_mod) -> None:
    rows = manifest_mod.build_run2_manifest_rows()
    assert len(rows) == 237
    resample = [row for row in rows if row.role == manifest_mod.ROLE_RESAMPLE_RUN1]
    extra = [row for row in rows if row.role == manifest_mod.ROLE_EXTRA_30PCT]
    assert len(resample) == 217
    assert len(extra) == 20


def test_extra_points_disjoint_from_master_grid(manifest_mod) -> None:
    master = manifest_mod.build_master_grid_xy_m()
    rows = manifest_mod.build_run2_manifest_rows()
    extra_rows = [row for row in rows if row.role == manifest_mod.ROLE_EXTRA_30PCT]

    for row in extra_rows:
        x_m = row.target_x_cm / 100.0
        y_m = row.target_y_cm / 100.0
        assert (x_m, y_m) not in master


def test_unique_coordinates_in_run2_manifest(manifest_mod) -> None:
    rows = manifest_mod.build_run2_manifest_rows()
    coords = {(row.target_x_cm, row.target_y_cm) for row in rows}
    assert len(coords) == 237


def test_session_ids_are_sequential(manifest_mod) -> None:
    rows = manifest_mod.build_run2_manifest_rows()
    assert [row.session_id for row in rows] == list(range(1, len(rows) + 1))


def test_committed_csv_matches_generator(manifest_mod) -> None:
    assert _EXPECTED_CSV.is_file(), f"missing committed CSV: {_EXPECTED_CSV}"

    rows = manifest_mod.build_run2_manifest_rows()
    with _EXPECTED_CSV.open(encoding="utf-8") as csv_f:
        on_disk = list(csv.DictReader(csv_f))

    assert len(on_disk) == len(rows)
    for disk_row, gen_row in zip(on_disk, rows, strict=True):
        assert int(disk_row["session_id"]) == gen_row.session_id
        assert int(disk_row["target_x_cm"]) == gen_row.target_x_cm
        assert int(disk_row["target_y_cm"]) == gen_row.target_y_cm
        assert disk_row["region"] == gen_row.region
        assert disk_row["role"] == gen_row.role


def test_write_manifest_csv_roundtrip(tmp_path: Path, manifest_mod) -> None:
    rows = manifest_mod.build_run2_manifest_rows()
    output_path = tmp_path / "collection_run2_targets.csv"
    manifest_mod.write_run2_manifest_csv(output_path, rows)

    with output_path.open(encoding="utf-8") as csv_f:
        written = list(csv.DictReader(csv_f))

    assert list(written[0].keys()) == list(manifest_mod.MANIFEST_COLUMNS)
    assert len(written) == 237
