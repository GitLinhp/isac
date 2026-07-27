"""truncate_cooperative_recordings 单元测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from isac_imp.record_target_metadata import (
    COOPERATIVE_TARGET_CSV,
    CSV_COLUMNS,
    truncate_cooperative_recordings,
)

_VLEN = 4
_N_FRAMES = 3
_FRAME_BYTES = _VLEN * 8


def _write_binary_frames(path: Path, n_frames: int, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for i in range(n_frames):
            f.write(bytes([(seed + i) % 256] * _FRAME_BYTES))


def _write_csv(parent: Path, rows: list[dict[str, str]]) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    csv_path = parent / COOPERATIVE_TARGET_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _make_row(dev_suffix: str, *, max_frames: str = "3") -> dict[str, str]:
    return {
        "recorded_at_utc": "2026-07-26T10:00:00+00:00",
        "target_x_cm": "10.0",
        "target_y_cm": "20.0",
        "dev0_file": f"dev0/divide_profiles_{dev_suffix}",
        "dev1_file": f"dev1/divide_profiles_{dev_suffix}",
        "record_max_frames": max_frames,
    }


def test_truncate_keeps_first_n_frames_and_updates_csv(tmp_path: Path) -> None:
    row = _make_row("001")
    _write_csv(tmp_path, [row])
    dev0 = tmp_path / row["dev0_file"]
    dev1 = tmp_path / row["dev1_file"]
    _write_binary_frames(dev0, _N_FRAMES, seed=1)
    _write_binary_frames(dev1, _N_FRAMES, seed=2)

    sessions, files = truncate_cooperative_recordings(
        tmp_path,
        max_frames=2,
        vlen=_VLEN,
        show_progress=False,
    )

    assert sessions == 1
    assert files == 2
    assert dev0.stat().st_size == 2 * _FRAME_BYTES
    assert dev1.stat().st_size == 2 * _FRAME_BYTES

    with (tmp_path / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert rows[0]["record_max_frames"] == "2"


def test_truncate_raises_when_not_enough_frames(tmp_path: Path) -> None:
    row = _make_row("001")
    _write_csv(tmp_path, [row])
    _write_binary_frames(tmp_path / row["dev0_file"], 1, seed=1)
    _write_binary_frames(tmp_path / row["dev1_file"], 1, seed=2)

    with pytest.raises(ValueError, match="insufficient frames"):
        truncate_cooperative_recordings(
            tmp_path,
            max_frames=2,
            vlen=_VLEN,
            show_progress=False,
        )


def test_truncate_dry_run_does_not_modify(tmp_path: Path) -> None:
    row = _make_row("001")
    _write_csv(tmp_path, [row])
    dev0 = tmp_path / row["dev0_file"]
    dev1 = tmp_path / row["dev1_file"]
    _write_binary_frames(dev0, _N_FRAMES, seed=1)
    _write_binary_frames(dev1, _N_FRAMES, seed=2)
    original_dev0_size = dev0.stat().st_size

    sessions, files = truncate_cooperative_recordings(
        tmp_path,
        max_frames=2,
        vlen=_VLEN,
        dry_run=True,
        show_progress=False,
    )

    assert sessions == 1
    assert files == 2
    assert dev0.stat().st_size == original_dev0_size
    with (tmp_path / COOPERATIVE_TARGET_CSV).open(encoding="utf-8") as csv_f:
        rows = list(csv.DictReader(csv_f))
    assert rows[0]["record_max_frames"] == "3"
