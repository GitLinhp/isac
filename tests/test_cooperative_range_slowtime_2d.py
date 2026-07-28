"""range×slow-time 2D 预处理与 CooperativeMonostatic2DCNN 测试。"""

import numpy as np
import pytest
import torch

from isac.models import (
    CooperativeMonostatic2DCNN,
    cooperative_model_type,
    divide_cpi_dual_to_roi_range_slowtime_np,
    dual_slowtime_to_model_input,
)
from isac_imp.cooperative_monostatic_pipeline import (
    DEFAULT_RANGE_ROI,
    grc_cooperative_processing_params,
)


def _synthetic_divide_cpi() -> np.ndarray:
    params = grc_cooperative_processing_params()
    length = int(params["vlen_divide_cpi"])
    return np.random.randn(length).astype(np.complex64) + 1j * np.random.randn(length)


def test_divide_cpi_dual_to_roi_range_slowtime_shape():
    params = grc_cooperative_processing_params()
    dual = divide_cpi_dual_to_roi_range_slowtime_np(
        _synthetic_divide_cpi(),
        _synthetic_divide_cpi(),
        proc_params=params,
        range_roi=DEFAULT_RANGE_ROI,
    )
    assert dual.shape == (2, 4, dual.shape[2])
    assert dual.shape[2] > 0


def test_dual_slowtime_to_real_imag_features():
    dual = torch.randn(2, 4, 34, dtype=torch.complex64)
    feat = dual_slowtime_to_model_input(dual, mode="range_slowtime_2d")
    assert feat.shape == (4, 4, 34)
    assert feat.dtype == torch.float32


def test_cooperative_model_type_slowtime():
    assert cooperative_model_type("range_slowtime_2d") == "2d"
    assert cooperative_model_type("real_imag") == "1d"


def test_cooperative_monostatic_2d_cnn_forward():
    model = CooperativeMonostatic2DCNN(in_channels=4, base_channels=16, num_layers=2)
    x = torch.randn(2, 4, 4, 34)
    out = model(x)
    assert out.shape == (2, 2)


def test_cooperative_monostatic_2d_cnn_single_sample():
    model = CooperativeMonostatic2DCNN(in_channels=4, num_layers=2)
    x = torch.randn(4, 4, 34)
    out = model(x)
    assert out.shape == (1, 2)


def test_cooperative_monostatic_2d_cnn_bad_num_layers():
    with pytest.raises(ValueError, match="num_layers"):
        CooperativeMonostatic2DCNN(num_layers=0)
