"""空场景背景对消：``H - H_bg`` 单元测试。"""

from __future__ import annotations

import numpy as np
import torch

from isac.sensing.clutter import subtract_background_cfr


def test_subtract_background_cfr_torch() -> None:
    h = torch.tensor([[1 + 1j, 2 + 0j], [0.5 + 0.5j, 1 - 1j]])
    bg = torch.tensor([[0.5 + 0.5j, 1 + 0j], [0.25 + 0.25j, 0.5 - 0.5j]])
    out = subtract_background_cfr(h, bg)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, h - bg)


def test_subtract_background_cfr_numpy() -> None:
    h = np.array([[1 + 1j, 2 + 0j]], dtype=np.complex64)
    bg = np.array([[1 + 0j, 0 + 1j]], dtype=np.complex64)
    out = subtract_background_cfr(h, bg)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, h.astype(np.complex128) - bg.astype(np.complex128))


def test_subtract_background_cfr_shape_mismatch_raises() -> None:
    h = torch.zeros(2, 4, dtype=torch.complex64)
    bg = torch.zeros(2, 3, dtype=torch.complex64)
    try:
        subtract_background_cfr(h, bg)
    except ValueError as exc:
        assert "形状" in str(exc)
    else:
        raise AssertionError("expected ValueError")
