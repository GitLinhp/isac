"""prepare_cooperative_monostatic_dataset 整理脚本测试。"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

from isac_imp.data_collection.cooperative_monostatic_dataset import summarize_cooperative_monostatic_h5
from isac_imp.record_target_metadata import COOPERATIVE_TARGET_CSV, CSV_COLUMNS

_VLEN = 8
_N_FRAMES = 2


def _load_prepare_module():
    prepare_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "prepare_cooperative_monostatic_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_cooperative_dataset", prepare_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {prepare_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_binary_frames(path: Path, n_frames: int, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with path.open("wb") as f:
        for _ in range(n_frames):
            frame = rng.standard_normal(_VLEN).astype(np.complex64)
            f.write(frame.tobytes())


def _write_csv(parent: Path, rows: list[dict[str, str]]) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_messy_experiment_dir(tmp_path: Path) -> Path:
    rows = [
        {
            "recorded_at_utc": "2026-07-26T10:00:00+00:00",
            "target_x_cm": "0.0",
            "target_y_cm": "0.0",
            "dev0_file": "dev0/divide_profiles_001",
            "dev1_file": "dev1/divide_profiles_001",
            "record_max_frames": str(_N_FRAMES),
        },
        {
            "recorded_at_utc": "",
            "target_x_cm": "",
            "target_y_cm": "",
            "dev0_file": "",
            "dev1_file": "",
            "record_max_frames": "",
        },
    ]
    _write_csv(tmp_path, rows)
    _write_binary_frames(tmp_path / "dev0/divide_profiles_001", _N_FRAMES, seed=1)
    _write_binary_frames(tmp_path / "dev1/divide_profiles_001", _N_FRAMES, seed=2)
    for suffix in ("002", "003"):
        (tmp_path / "dev0" / f"divide_profiles_{suffix}").touch()
        (tmp_path / "dev1" / f"divide_profiles_{suffix}").touch()
    return tmp_path


def test_prepare_dry_run_reports_orphans(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = _make_messy_experiment_dir(tmp_path)

    result = prepare_mod.prepare_cooperative_monostatic_dataset(
        parent,
        output_path=tmp_path / "dataset.h5",
        dry_run=True,
        vlen=_VLEN,
    )

    assert result["csv_rows_before"] == 2
    assert result["orphans_would_delete"] == 4
    assert result["files_would_harmonize"] == 0


def test_prepare_harmonizes_mismatched_frame_counts(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = tmp_path / "mismatch"
    rows = [
        {
            "recorded_at_utc": "2026-07-26T10:00:00+00:00",
            "target_x_cm": "0.0",
            "target_y_cm": "0.0",
            "dev0_file": "dev0/divide_profiles_001",
            "dev1_file": "dev1/divide_profiles_001",
            "record_max_frames": "3",
        },
    ]
    _write_csv(parent, rows)
    _write_binary_frames(parent / "dev0/divide_profiles_001", 3, seed=1)
    _write_binary_frames(parent / "dev1/divide_profiles_001", 2, seed=2)

    result = prepare_mod.prepare_cooperative_monostatic_dataset(
        parent,
        output_path=tmp_path / "dataset.h5",
        show_progress=False,
        vlen=_VLEN,
    )

    assert result["files_harmonized"] == 1
    with (parent / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        csv_rows = list(csv.DictReader(csv_f))
    assert csv_rows[0]["record_max_frames"] == "2"

    summary = summarize_cooperative_monostatic_h5(tmp_path / "dataset.h5")
    assert summary["num_sessions"] == 1
    assert summary["total_frames"] == 2


def test_prepare_sorts_prunes_and_builds_h5(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = _make_messy_experiment_dir(tmp_path)
    h5_path = tmp_path / "dataset.h5"

    result = prepare_mod.prepare_cooperative_monostatic_dataset(
        parent,
        output_path=h5_path,
        show_progress=False,
        vlen=_VLEN,
    )

    with (parent / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert len(rows) == 1
    assert result["csv_rows_after"] == 1
    assert result["orphans_deleted"] == 4
    assert not (parent / "dev0/divide_profiles_002").exists()
    assert h5_path.is_file()

    summary = summarize_cooperative_monostatic_h5(h5_path)
    assert summary["num_sessions"] == 1
    assert summary["total_frames"] == _N_FRAMES
