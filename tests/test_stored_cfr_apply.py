"""存储 CFR 频域施加 helper 测试。"""

from pathlib import Path

import pytest
import torch

from isac import DEFAULT_DATASET_H5, PROJECT_ROOT
from isac.datasets import Dataset
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.utils.data_collection.channel_export import (
    apply_stored_cfr_frequency,
    cfr_numpy_to_h_freq,
)

_H5_PATH = DEFAULT_DATASET_H5
_CONFIG = PROJECT_ROOT / "config" / "data_collection" / "data_collection.toml"


@pytest.fixture
def system_and_cfr():
    if not _H5_PATH.is_file():
        pytest.skip(f"数据集不存在: {_H5_PATH}")
    set_random_seed(42)
    ds = Dataset.load(_H5_PATH)
    system = System(
        config=load_config(_CONFIG),
        batch_size=1,
        device="cpu",
    )
    cfr, _ = ds[0]
    return system, cfr


def test_cfr_numpy_to_h_freq_is_7d(system_and_cfr) -> None:
    _, cfr = system_and_cfr
    h = cfr_numpy_to_h_freq(cfr, device="cpu")
    assert h.ndim == 7
    assert h.dtype == torch.complex64


def test_apply_stored_cfr_frequency_shape(system_and_cfr) -> None:
    system, cfr = system_and_cfr
    _, x_rg, _ = system.transmit()
    y = apply_stored_cfr_frequency(
        x_rg, cfr, system.components.channel, snr_db=None
    )
    assert y.shape[-2:] == x_rg.shape[-2:]
    assert y.dtype == torch.complex64
    assert torch.any(y != 0)


def test_apply_stored_cfr_invalid_ndim_raises() -> None:
    with pytest.raises(ValueError, match="6D 或 7D"):
        cfr_numpy_to_h_freq(
            __import__("numpy").zeros((2, 3), dtype=complex),
            device="cpu",
        )
