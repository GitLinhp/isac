"""训练前异常帧硬过滤 / 能量 MAD 软剔除单测。"""

from __future__ import annotations

import numpy as np
import pytest

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    cooperative_frame_cpi_energy,
    filter_cooperative_frames_energy_mad,
    filter_cooperative_frames_hard,
)


def test_filter_hard_drops_nan_label_oob_and_bad_cpi() -> None:
    target_xy = np.asarray(
        [
            [0.0, 0.0],  # keep
            [np.nan, 0.0],  # nan_label
            [1.5, 0.0],  # oob_xy
            [0.1, 0.1],  # nan_cpi
            [0.2, 0.2],  # near_zero on dev1
        ],
        dtype=np.float64,
    )
    vlen = 8
    profiles_dev0 = np.ones((5, vlen), dtype=np.complex64)
    profiles_dev1 = np.ones((5, vlen), dtype=np.complex64)
    profiles_dev0[3, 0] = np.nan + 0j
    profiles_dev1[4, :] = 0

    keep, counts = filter_cooperative_frames_hard(
        target_xy,
        profiles_dev0=profiles_dev0,
        profiles_dev1=profiles_dev1,
        xy_max_m=1.0,
        energy_eps=1e-8,
    )
    assert keep.tolist() == [0]
    assert counts["nan_label"] == 1
    assert counts["oob_xy"] == 1
    assert counts["nan_cpi"] == 1
    assert counts["near_zero"] == 1


def test_filter_hard_without_profiles_only_labels() -> None:
    target_xy = np.asarray([[0.0, 0.0], [2.0, 0.0], [np.inf, 0.0]], dtype=np.float64)
    keep, counts = filter_cooperative_frames_hard(target_xy, xy_max_m=1.0)
    assert keep.tolist() == [0]
    assert counts["oob_xy"] == 1
    assert counts["nan_label"] == 1
    assert counts["nan_cpi"] == 0
    assert counts["near_zero"] == 0


def test_energy_mad_drops_only_extreme_train_frame() -> None:
    # 同 session 内略有波动的能量 + 一帧极端值，使 MAD > 0
    rng = np.random.default_rng(0)
    session = np.zeros(20, dtype=np.int64)
    energy = 1.0 + 0.02 * rng.standard_normal(20)
    energy[5] = 10.0
    train_idx = np.arange(20, dtype=np.int64)
    kept, dropped = filter_cooperative_frames_energy_mad(
        train_idx, session, energy, z_thresh=5.0
    )
    assert dropped >= 1
    assert 5 not in set(kept.tolist())
    assert kept.size == 20 - dropped


def test_energy_mad_drops_extreme_frame_in_val_subset() -> None:
    """对 val 子集调用 soft 会丢掉极端能量帧。"""
    rng = np.random.default_rng(1)
    session = np.zeros(20, dtype=np.int64)
    energy = 1.0 + 0.02 * rng.standard_normal(20)
    energy[17] = 10.0  # val 子集中的极端帧
    train_idx = np.arange(12, dtype=np.int64)
    val_idx = np.arange(12, 20, dtype=np.int64)

    kept_train, dropped_train = filter_cooperative_frames_energy_mad(
        train_idx, session, energy, z_thresh=5.0
    )
    kept_val, dropped_val = filter_cooperative_frames_energy_mad(
        val_idx, session, energy, z_thresh=5.0
    )
    assert dropped_train == 0
    assert kept_train.tolist() == train_idx.tolist()
    assert dropped_val >= 1
    assert 17 not in set(kept_val.tolist())
    assert kept_val.size == val_idx.size - dropped_val


def test_cooperative_frame_cpi_energy_shape() -> None:
    d0 = np.ones((4, 16), dtype=np.complex64)
    d1 = 2 * np.ones((4, 16), dtype=np.complex64)
    e = cooperative_frame_cpi_energy(d0, d1)
    assert e.shape == (4,)
    assert np.all(np.isfinite(e))
    assert e[0] == pytest.approx(np.log1p(1.0 + 2.0))


def _load_cnn_rmse_eval_module():
    import importlib.util
    from pathlib import Path

    eval_path = (
        Path(__file__).resolve().parents[1]
        / "script"
        / "experiment"
        / "run_cooperative_monostatic_cnn_rmse.py"
    )
    spec = importlib.util.spec_from_file_location("cnn_rmse_eval_filter", eval_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_eval_outlier_filters_drops_oob_and_energy_outlier(tmp_path) -> None:
    """评估侧硬+软过滤：越界标签与 session 内极端能量帧被剔除。"""
    from isac_imp.data_collection.cooperative_monostatic_dataset import (
        DATASET_KEY_FEATURES,
        DATASET_KEY_FRAME_ENERGY,
        DATASET_KEY_FRAME_INDEX,
        DATASET_KEY_SESSION_INDEX,
        DATASET_KEY_TARGET_POSITION,
        META_KEY_DATA_KIND,
    )

    import h5py

    n = 20
    rng = np.random.default_rng(2)
    target = np.zeros((n, 3), dtype=np.float64)
    target[:, 0] = 0.1
    target[:, 1] = 0.1
    target[0, 0] = 2.0  # oob
    energy = 1.0 + 0.02 * rng.standard_normal(n)
    energy[7] = 12.0  # MAD outlier
    session = np.zeros(n, dtype=np.int64)
    frame_index = np.arange(n, dtype=np.int32)

    h5_path = tmp_path / "eval_features.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs[META_KEY_DATA_KIND] = "cooperative_monostatic_features"
        f.create_dataset(DATASET_KEY_TARGET_POSITION, data=target)
        f.create_dataset(DATASET_KEY_SESSION_INDEX, data=session)
        f.create_dataset(DATASET_KEY_FRAME_INDEX, data=frame_index)
        f.create_dataset(DATASET_KEY_FRAME_ENERGY, data=energy.astype(np.float64))
        f.create_dataset(
            DATASET_KEY_FEATURES, data=np.zeros((n, 4, 8), dtype=np.float32)
        )

    mod = _load_cnn_rmse_eval_module()
    cand = list(range(n))
    kept = mod._apply_eval_outlier_filters(
        h5_path,
        cand,
        xy_max_m=1.0,
        energy_eps=1e-8,
        energy_mad_z=5.0,
    )
    assert 0 not in kept
    assert 7 not in kept
    assert len(kept) <= n - 2
