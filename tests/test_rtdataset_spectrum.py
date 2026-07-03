"""RTDataset h_dd 读取测试。"""

import pytest
import torch

from isac import DEFAULT_DATASET_H5
from isac.datasets import RTDataset

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


def test_h_dd_is_complex(loaded_dataset: RTDataset) -> None:
    assert loaded_dataset.h_dd.dtype == complex or loaded_dataset.h_dd.dtype.name.startswith("complex")


def test_spectrum_tensor_device(loaded_dataset: RTDataset) -> None:
    t = loaded_dataset.spectrum_tensor(0, device="cpu")
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 2
