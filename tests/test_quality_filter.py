"""data_collection.quality_filter 单元测试。"""

import argparse
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from isac.utils.data_collection.quality_filter import (
    MonteCarloSamplingParams,
    QualityFilterConfig,
    QualityFilterStats,
    assess_collection_sample,
    run_monte_carlo_with_quality_filter,
)
from isac.sensing.sample_quality import QualityFilterStats as ReexportedStats
from isac.sensing.sample_quality import SampleQualityConfig, SampleQualityResult


def test_quality_filter_stats_reexport_from_sample_quality():
    assert ReexportedStats is QualityFilterStats


def test_quality_filter_config_from_args_and_sample_quality_config():
    args = argparse.Namespace(
        quality_filter=True,
        require_los=False,
        min_los_ratio=0.5,
        min_peak_prominence_db=8.0,
        max_bin_offset=2,
        quality_max_trials_factor=30,
    )
    cfg = QualityFilterConfig.from_args(args)
    assert cfg.enabled is True
    assert cfg.require_los is False
    assert cfg.min_los_ratio == 0.5
    assert cfg.quality_max_trials_factor == 30

    sq = cfg.sample_quality_config()
    assert isinstance(sq, SampleQualityConfig)
    assert sq.require_los is False
    assert sq.min_los_ratio == 0.5
    assert sq.max_bin_offset == 2


@dataclass
class _MockGeom:
    range_tensor: object
    vel_tensor: object


def test_assess_collection_sample_delegates_to_evaluate_sample_quality():
    scene = MagicMock()
    scene.targets_states = object()
    scene.rx_states = object()
    scene.tx_states = object()
    scene.cfr_numpy.return_value = np.ones((2, 2), dtype=np.complex64)

    quality_cfg = QualityFilterConfig(enabled=True)
    expected = SampleQualityResult(passed=True, los_ratio=1.0)

    with patch(
        "isac.utils.data_collection.quality_filter.RxTargetTxGeometric.from_states",
        return_value=_MockGeom(
            range_tensor=np.array([[[40.0]]]),
            vel_tensor=np.array([[[2.0]]]),
        ),
    ), patch(
        "isac.utils.data_collection.quality_filter.evaluate_sample_quality",
        return_value=expected,
    ) as mock_eval:
        result = assess_collection_sample(
            scene,
            rg=MagicMock(),
            device="cpu",
            sensing_performance=MagicMock(),
            quality_cfg=quality_cfg,
        )

    assert result is expected
    mock_eval.assert_called_once()
    call_args = mock_eval.call_args
    assert call_args.args[2] == pytest.approx(40.0)
    assert call_args.args[3] == pytest.approx(2.0)
    assert isinstance(call_args.kwargs["cfg"], SampleQualityConfig)


def test_run_monte_carlo_with_quality_filter_accept_and_reject():
    pos_ok = np.array([1.0, 2.0, 0.0])
    vel_ok = np.array([0.0, 3.0, 0.0])
    pos_bad = np.array([3.0, 4.0, 0.0])
    vel_bad = np.array([0.0, 1.0, 0.0])

    pos_batches = [[pos_bad], [pos_ok]]
    vel_batches = [[vel_bad], [vel_ok]]
    assess_results = [
        SampleQualityResult(passed=False, reason="weak_los"),
        SampleQualityResult(passed=True, los_ratio=1.0),
    ]

    accepted: list[tuple[int, np.ndarray, np.ndarray]] = []

    def fake_update(_target, pos, vel):
        return None

    def fake_process(episode_idx, pos, vel):
        accepted.append((episode_idx, pos.copy(), vel.copy()))

    with patch(
        "isac.utils.data_collection.quality_filter.tg.generate_monte_carlo_points",
        side_effect=pos_batches,
    ), patch(
        "isac.utils.data_collection.quality_filter.tg.sample_monte_carlo_velocities",
        side_effect=vel_batches,
    ), patch(
        "isac.utils.data_collection.quality_filter.assess_collection_sample",
        side_effect=assess_results,
    ):
        stats = run_monte_carlo_with_quality_filter(
            scene=MagicMock(),
            target=MagicMock(),
            rg=MagicMock(),
            device="cpu",
            sensing_performance=MagicMock(),
            quality_cfg=QualityFilterConfig(
                enabled=True,
                quality_max_trials_factor=10,
            ),
            num_samples=1,
            seed=0,
            roi_box3d=((0, 1), (0, 1), (0, 0)),
            sampling=MonteCarloSamplingParams(
                sampling_mode="uniform",
                safe_margin=1.0,
                max_trials_factor=20,
                speed_range=(0.1, 1.0),
                velocity_sampling="sphere_uniform",
            ),
            update_target_pose=fake_update,
            process_accepted=fake_process,
        )

    assert stats.accepted == 1
    assert stats.rejected == 1
    assert stats.reject_counts["weak_los"] == 1
    assert len(accepted) == 1
    assert accepted[0][0] == 0
    np.testing.assert_array_equal(accepted[0][1], pos_ok)


def test_run_monte_carlo_with_quality_filter_raises_when_insufficient_samples():
    with patch(
        "isac.utils.data_collection.quality_filter.tg.generate_monte_carlo_points",
        return_value=[np.zeros(3)],
    ), patch(
        "isac.utils.data_collection.quality_filter.tg.sample_monte_carlo_velocities",
        return_value=[np.ones(3)],
    ), patch(
        "isac.utils.data_collection.quality_filter.assess_collection_sample",
        return_value=SampleQualityResult(passed=False, reason="weak_los"),
    ):
        with pytest.raises(RuntimeError, match="质量过滤后仅采集"):
            run_monte_carlo_with_quality_filter(
                scene=MagicMock(),
                target=MagicMock(),
                rg=MagicMock(),
                device="cpu",
                sensing_performance=MagicMock(),
                quality_cfg=QualityFilterConfig(
                    enabled=True,
                    quality_max_trials_factor=2,
                ),
                num_samples=2,
                seed=0,
                roi_box3d=((0, 1), (0, 1), (0, 0)),
                sampling=MonteCarloSamplingParams(
                    sampling_mode="uniform",
                    safe_margin=1.0,
                    max_trials_factor=20,
                    speed_range=(0.1, 1.0),
                    velocity_sampling="sphere_uniform",
                ),
                update_target_pose=lambda *_: None,
                process_accepted=lambda *_: None,
            )
