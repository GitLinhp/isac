"""Dataset 序列索引协议测试。"""

from pathlib import Path

import pytest

from isac import PROJECT_ROOT
from isac.datasets import Dataset

_H5_PATH = (
    PROJECT_ROOT / "out" / "dataset_collection" / "empty_room_mc_sionna_dataset.h5"
)


@pytest.fixture
def loaded_dataset() -> Dataset:
    if not _H5_PATH.is_file():
        pytest.skip(f"数据集不存在: {_H5_PATH}")
    return Dataset.load(_H5_PATH)


def test_len_matches_num_slots(loaded_dataset: Dataset) -> None:
    assert len(loaded_dataset) == loaded_dataset.num_slots


def test_getitem_returns_cfr_and_label(loaded_dataset: Dataset) -> None:
    cfr, label = loaded_dataset[0]
    assert cfr.shape == loaded_dataset.cfr[0].shape
    assert len(label) == 2
    assert len(label[0]) == 3
    assert len(label[1]) == 3


def test_getitem_label_matches_kinematics(loaded_dataset: Dataset) -> None:
    idx = 0
    _, label = loaded_dataset[idx]
    pos = loaded_dataset.target_position[idx]
    vel = loaded_dataset.target_velocity[idx]
    assert label[0] == pytest.approx(tuple(float(x) for x in pos))
    assert label[1] == pytest.approx(tuple(float(x) for x in vel))


def test_getitem_out_of_range_raises(loaded_dataset: Dataset) -> None:
    with pytest.raises(IndexError):
        _ = loaded_dataset[loaded_dataset.num_slots]
    with pytest.raises(IndexError):
        _ = loaded_dataset[-1]
