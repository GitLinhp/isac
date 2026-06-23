"""窗函数：仅 ``apply_window`` 对外接口。"""

import torch

from isac.utils.windows import apply_window


def test_apply_window_hamming_periodic_and_symmetric_differ():
    dev = torch.device("cpu")
    n = 64
    x = torch.ones(1, n, device=dev, dtype=torch.float64)
    out_periodic = apply_window(x, dim=-1, window="hamming", periodic=True)
    out_symmetric = apply_window(x, dim=-1, window="hamming", periodic=False)
    w_periodic = out_periodic[0]
    w_symmetric = out_symmetric[0]
    assert w_periodic.shape == (n,)
    assert w_symmetric.shape == (n,)
    assert torch.all(w_periodic >= 0)
    assert torch.all(w_symmetric >= 0)
    assert not torch.allclose(w_periodic, w_symmetric)


def test_apply_window_chebwin_tuple_and_dict():
    dev = torch.device("cpu")
    n = 32
    x = torch.ones(1, n, device=dev, dtype=torch.float64)
    out_tuple = apply_window(x, dim=-1, window=("chebwin", 60.0))
    out_dict = apply_window(
        x, dim=-1, window={"type": "chebwin", "at": 60.0}
    )
    assert torch.allclose(out_tuple[0], out_dict[0])
    assert torch.all(out_tuple[0] >= 0)


def test_apply_window_none_is_identity():
    x = torch.randn(4, 8, dtype=torch.complex64)
    out = apply_window(x, dim=-1, window=None)
    assert torch.equal(out, x)


def test_apply_window_matches_explicit_coefficients():
    dev = torch.device("cpu")
    x = torch.randn(3, 16, dtype=torch.complex64, device=dev)
    out = apply_window(x, dim=-1, window="hann", periodic=True)
    x_ones = torch.ones(1, 16, device=dev, dtype=torch.complex64)
    w_row = apply_window(x_ones, dim=-1, window="hann", periodic=True)[0]
    expected = x * w_row.reshape(1, 16)
    assert torch.allclose(out, expected)


def test_apply_window_hamming_from_config_dict():
    dev = torch.device("cpu")
    x = torch.ones(1, 8, device=dev, dtype=torch.float64)
    out = apply_window(x, dim=-1, window={"type": "hamming"})
    assert out.shape == (1, 8)
    assert torch.all(out[0] >= 0)
