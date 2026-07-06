"""RTDataset 序列索引协议测试。"""

import pytest
import torch

from isac import DEFAULT_DATASET_H5
from isac.collection import RTDataset

_H5_PATH = DEFAULT_DATASET_H5


@pytest.fixture
def loaded_dataset() -> RTDataset:
    if not _H5_PATH.is_file():
        pytest.skip(f"数据集不存在: {_H5_PATH}")
    try:
        return RTDataset.load(_H5_PATH)
    except ValueError as exc:
        if "旧 CFR 格式" in str(exc):
            pytest.skip("数据集为旧 CFR 格式，请重新采集 h_dd 数据集")
        raise


def test_len_matches_num_slots(loaded_dataset: RTDataset) -> None:
    assert len(loaded_dataset) == loaded_dataset.num_slots


def test_getitem_returns_training_dict(loaded_dataset: RTDataset) -> None:
    sample = loaded_dataset[0]
    assert set(sample.keys()) == {"features", "range_m", "velocity_mps", "slot"}
    assert sample["features"].ndim == 3
    assert sample["range_m"].dtype == torch.float32
    assert sample["velocity_mps"].dtype == torch.float32


def test_getitem_label_matches_kinematics(loaded_dataset: RTDataset) -> None:
    from isac.models import monostatic_labels_from_kinematics

    idx = 0
    sample = loaded_dataset[idx]
    range_m, vel_mps = monostatic_labels_from_kinematics(
        loaded_dataset.target_position[idx],
        loaded_dataset.target_velocity[idx],
        loaded_dataset.bs_pos,
    )
    assert sample["range_m"].item() == pytest.approx(range_m)
    assert sample["velocity_mps"].item() == pytest.approx(vel_mps)


def test_spectrum_tensor_shape(loaded_dataset: RTDataset) -> None:
    t = loaded_dataset.spectrum_tensor(0)
    assert t.shape == loaded_dataset.h_dd[0].shape
    assert t.dtype == torch.complex64


def test_getitem_out_of_range_raises(loaded_dataset: RTDataset) -> None:
    with pytest.raises(IndexError):
        _ = loaded_dataset[loaded_dataset.num_slots]
    with pytest.raises(IndexError):
        _ = loaded_dataset[-1]
