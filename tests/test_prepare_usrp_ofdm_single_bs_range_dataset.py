"""prepare_usrp_ofdm_single_bs_range_dataset 整理脚本测试。"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    summarize_usrp_ofdm_single_bs_range_h5,
)
from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    MONO_RANGE_TARGET_CSV_COLUMNS,
)

_VLEN = 8
_N_FRAMES = 2


def _load_prepare_module():
    prepare_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "prepare_usrp_ofdm_single_bs_range_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_usrp_ofdm_single_bs_range_dataset", prepare_path
    )
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
        writer = csv.DictWriter(csv_f, fieldnames=MONO_RANGE_TARGET_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_messy_experiment_dir(tmp_path: Path) -> Path:
    rows = [
        {
            "recorded_at_utc": "2026-08-20T06:48:10+00:00",
            "target_range_m": "0.5499999999999999",
            "data_file": "divide_profiles_002",
            "record_max_frames": str(_N_FRAMES),
        },
        {
            "recorded_at_utc": "2026-08-20T06:47:37+00:00",
            "target_range_m": "0.49999999999999994",
            "data_file": "divide_profiles_001",
            "record_max_frames": str(_N_FRAMES),
        },
        {
            "recorded_at_utc": "",
            "target_range_m": "",
            "data_file": "",
            "record_max_frames": "",
        },
    ]
    _write_csv(tmp_path, rows)
    _write_binary_frames(tmp_path / "divide_profiles_001", _N_FRAMES, seed=1)
    _write_binary_frames(tmp_path / "divide_profiles_002", _N_FRAMES, seed=2)
    (tmp_path / "divide_profiles_003").touch()
    (tmp_path / "Untitled").touch()
    return tmp_path


def test_prepare_dry_run_reports_orphans(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = _make_messy_experiment_dir(tmp_path)

    result = prepare_mod.prepare_usrp_ofdm_single_bs_range_dataset(
        parent,
        output_path=tmp_path / "dataset.h5",
        dry_run=True,
        vlen=_VLEN,
    )

    assert result["csv_rows_before"] == 3
    assert result["orphans_would_delete"] == 2
    assert result["files_would_harmonize"] == 0
    assert result["vlen"] == _VLEN


def test_prepare_harmonizes_mismatched_frame_counts(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = tmp_path / "mismatch"
    rows = [
        {
            "recorded_at_utc": "2026-08-20T06:47:37+00:00",
            "target_range_m": "0.50",
            "data_file": "divide_profiles_001",
            "record_max_frames": "2",
        },
    ]
    _write_csv(parent, rows)
    # File longer than declared max → truncate to record_max_frames.
    _write_binary_frames(parent / "divide_profiles_001", 4, seed=1)

    result = prepare_mod.prepare_usrp_ofdm_single_bs_range_dataset(
        parent,
        output_path=tmp_path / "dataset.h5",
        show_progress=False,
        vlen=_VLEN,
        max_frames=50,
    )

    assert result["files_harmonized"] == 1
    with (parent / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        csv_rows = list(csv.DictReader(csv_f))
    assert csv_rows[0]["record_max_frames"] == "2"
    assert csv_rows[0]["target_range_m"] == "0.50"

    summary = summarize_usrp_ofdm_single_bs_range_h5(tmp_path / "dataset.h5")
    assert summary["num_sessions"] == 1
    assert summary["total_frames"] == 2


def test_prepare_respects_max_frames(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = tmp_path / "max_frames"
    rows = [
        {
            "recorded_at_utc": "2026-08-20T06:47:37+00:00",
            "target_range_m": "0.50",
            "data_file": "divide_profiles_001",
            "record_max_frames": "4",
        },
    ]
    _write_csv(parent, rows)
    _write_binary_frames(parent / "divide_profiles_001", 4, seed=1)

    result = prepare_mod.prepare_usrp_ofdm_single_bs_range_dataset(
        parent,
        output_path=tmp_path / "dataset.h5",
        show_progress=False,
        vlen=_VLEN,
        max_frames=2,
    )

    assert result["files_harmonized"] == 1
    assert result["max_frames"] == 2
    with (parent / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        csv_rows = list(csv.DictReader(csv_f))
    assert csv_rows[0]["record_max_frames"] == "2"

    summary = summarize_usrp_ofdm_single_bs_range_h5(tmp_path / "dataset.h5")
    assert summary["num_sessions"] == 1
    assert summary["total_frames"] == 2
    assert summary["frames_per_session"] == 2


def test_prepare_sorts_rounds_prunes_and_builds_h5(tmp_path: Path) -> None:
    prepare_mod = _load_prepare_module()
    parent = _make_messy_experiment_dir(tmp_path)
    h5_path = tmp_path / "dataset.h5"

    result = prepare_mod.prepare_usrp_ofdm_single_bs_range_dataset(
        parent,
        output_path=h5_path,
        show_progress=False,
        vlen=_VLEN,
        max_frames=50,
    )

    with (parent / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert len(rows) == 2
    assert rows[0]["target_range_m"] == "0.50"
    assert rows[0]["data_file"] == "divide_profiles_001"
    assert rows[1]["target_range_m"] == "0.55"
    assert rows[1]["data_file"] == "divide_profiles_002"

    assert result["csv_rows_after"] == 2
    assert result["orphans_deleted"] == 2
    assert not (parent / "divide_profiles_003").exists()
    assert not (parent / "Untitled").exists()
    assert h5_path.is_file()

    summary = summarize_usrp_ofdm_single_bs_range_h5(h5_path)
    assert summary["num_sessions"] == 2
    assert summary["total_frames"] == 2 * _N_FRAMES
