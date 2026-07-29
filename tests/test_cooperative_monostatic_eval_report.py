"""cooperative_monostatic_eval_report 单元测试。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "script" / "experiment"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from cooperative_monostatic_eval_report import (  # noqa: E402
    LOCALIZATION_CSV_COLUMNS,
    load_two_stage_eval_metrics,
    print_localization_rmse_summary,
    to_localization_row,
    write_localization_csv,
)


def test_write_localization_csv_seven_columns(tmp_path: Path) -> None:
    rows = [
        {
            "sample_idx": 0,
            "session_index": 1,
            "frame_index": 2,
            "true_x_m": 0.1,
            "true_y_m": -0.2,
            "est_x_m": 0.15,
            "est_y_m": -0.18,
            "rmse_xy_m": 0.05385,
            "true_subregion_id": 5,  # should be ignored in main CSV
        }
    ]
    path = tmp_path / "rmse.csv"
    write_localization_csv(path, rows)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == LOCALIZATION_CSV_COLUMNS
        got = list(reader)
    assert len(got) == 1
    assert "true_subregion_id" not in got[0]
    assert float(got[0]["rmse_xy_m"]) == 0.05385


def test_to_localization_row() -> None:
    full = {
        "sample_idx": 1,
        "session_index": 2,
        "frame_index": 3,
        "true_x_m": 0.0,
        "true_y_m": 0.0,
        "est_x_m": 0.1,
        "est_y_m": 0.2,
        "rmse_xy_m": 0.2236,
        "region_correct": 1,
    }
    loc = to_localization_row(full)
    assert set(loc) == set(LOCALIZATION_CSV_COLUMNS)
    assert "region_correct" not in loc


def test_print_summary_smoke(capsys) -> None:
    rows = [
        {
            "true_x_m": 0.1,
            "true_y_m": 0.1,
            "rmse_xy_m": 0.2,
        },
        {
            "true_x_m": 0.8,
            "true_y_m": 0.8,
            "rmse_xy_m": 0.5,
        },
    ]
    print_localization_rmse_summary(
        rows, title="Two-stage localization mean error summary"
    )
    out = capsys.readouterr().out
    assert "Two-stage localization mean error summary" in out
    assert "global" in out
    assert "inner" in out
    assert "outer" in out


def test_load_two_stage_eval_metrics_with_sidecar(tmp_path: Path) -> None:
    main_csv = tmp_path / "two_stage_rmse.csv"
    diag_csv = tmp_path / "two_stage_region_diagnostics.csv"
    write_localization_csv(
        main_csv,
        [
            {
                "sample_idx": 0,
                "session_index": 0,
                "frame_index": 0,
                "true_x_m": 0.1,
                "true_y_m": 0.1,
                "est_x_m": 0.2,
                "est_y_m": 0.1,
                "rmse_xy_m": 0.1,
            },
            {
                "sample_idx": 1,
                "session_index": 0,
                "frame_index": 1,
                "true_x_m": 0.9,
                "true_y_m": 0.9,
                "est_x_m": 0.7,
                "est_y_m": 0.9,
                "rmse_xy_m": 0.2,
            },
        ],
    )
    with diag_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "region_correct",
                "region_topk_hit",
                "rmse_xy_m",
                "rmse_xy_oracle_m",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_idx": 0,
                "region_correct": 1,
                "region_topk_hit": 1,
                "rmse_xy_m": 0.1,
                "rmse_xy_oracle_m": 0.05,
            }
        )
        writer.writerow(
            {
                "sample_idx": 1,
                "region_correct": 0,
                "region_topk_hit": 1,
                "rmse_xy_m": 0.2,
                "rmse_xy_oracle_m": 0.08,
            }
        )

    m = load_two_stage_eval_metrics(main_csv)
    assert m["n"] == 2
    assert abs(float(m["global_mean_err_m"]) - 0.15) < 1e-9
    assert float(m["region_top1_acc"]) == 0.5
    assert float(m["region_topk_hit"]) == 1.0
    assert abs(float(m["oracle_region_mean_err_m"]) - 0.065) < 1e-9
    assert np.isfinite(m["inner_mean_err_m"])
    assert np.isfinite(m["outer_mean_err_m"])
