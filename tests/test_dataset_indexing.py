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
    return RTDataset.load(_H5_PATH)


def test_len_matches_h_dd_shape(loaded_dataset: RTDataset) -> None:
    assert len(loaded_dataset) == loaded_dataset.h_dd.shape[0]


def test_getitem_returns_raw_dict(loaded_dataset: RTDataset) -> None:
    sample = loaded_dataset[0]
    assert set(sample.keys()) == {
        "spectrum_tensor",
        "target_position",
        "target_velocity",
        "bs_pos",
        "slot",
    }
    assert sample["spectrum_tensor"].dtype == torch.complex64
    assert sample["target_position"].dtype == torch.float32
    assert sample["target_velocity"].dtype == torch.float32
    assert sample["bs_pos"].dtype == torch.float32


def test_getitem_kinematics_shape(loaded_dataset: RTDataset) -> None:
    sample = loaded_dataset[0]
    assert sample["target_position"].shape == (3,)
    assert sample["target_velocity"].shape == (3,)
    assert sample["bs_pos"].shape == (3,)


def test_spectrum_tensor_shape(loaded_dataset: RTDataset) -> None:
    t = loaded_dataset.spectrum_tensor(0)
    assert t.shape == loaded_dataset.h_dd[0].shape
    assert t.dtype == torch.complex64
    assert torch.equal(t, loaded_dataset[0]["spectrum_tensor"])


def test_getitem_out_of_range_raises(loaded_dataset: RTDataset) -> None:
    with pytest.raises(IndexError):
        _ = loaded_dataset[len(loaded_dataset)]
    with pytest.raises(IndexError):
        _ = loaded_dataset[-1]
