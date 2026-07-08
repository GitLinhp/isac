"""RTDataset 序列索引协议测试。"""

import pytest
import torch

from isac import DEFAULT_DATASET_H5
from isac.collection import RTDataset
from isac.sensing.spectrum import SensingPerformance

_H5_PATH = DEFAULT_DATASET_H5


def _sensing_performance() -> SensingPerformance:
    from sionna.phy.ofdm import ResourceGrid

    rg = ResourceGrid(
        num_ofdm_symbols=512,
        fft_size=2048,
        subcarrier_spacing=30e3,
        cyclic_prefix_length=32,
        dc_null=False,
        device="cpu",
    )
    return SensingPerformance(rg, carrier_frequency=6e9)


@pytest.fixture
def loaded_dataset() -> RTDataset:
    if not _H5_PATH.is_file():
        pytest.skip(f"数据集不存在: {_H5_PATH}")
    return RTDataset.load(_H5_PATH, sensing_performance=_sensing_performance())


def test_len_matches_h_dd_shape(loaded_dataset: RTDataset) -> None:
    assert len(loaded_dataset) == loaded_dataset.h_dd.shape[0]


def test_getitem_returns_training_dict(loaded_dataset: RTDataset) -> None:
    sample = loaded_dataset[0]
    assert set(sample.keys()) == {
        "features",
        "peaks_delay",
        "peaks_doppler",
        "range_m",
        "velocity_mps",
        "slot",
    }
    assert sample["features"].ndim == 3
    assert sample["range_m"].dtype == torch.float32
    assert sample["velocity_mps"].dtype == torch.float32


def test_getitem_label_matches_kinematics(loaded_dataset: RTDataset) -> None:
    from isac.sensing.geometry import monostatic_range_velocity

    idx = 0
    sample = loaded_dataset[idx]
    range_m, vel_mps = monostatic_range_velocity(
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
        _ = loaded_dataset[len(loaded_dataset)]
    with pytest.raises(IndexError):
        _ = loaded_dataset[-1]
