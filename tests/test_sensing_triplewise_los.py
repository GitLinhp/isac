"""``RxTargetTxGeometric``：类型张量、路径长度与径向速度。"""

import numpy as np
import pytest
import torch

from isac.channel.rt.rx_target_tx_geometric import RxTargetTxGeometric
from isac.sensing.geometry import MONOSTATIC_TX_RX_EPS_M, compute_range


def _state(pos: list[float], vel: list[float] | None = None) -> list[np.ndarray]:
    v = vel if vel is not None else [0.0, 0.0, 0.0]
    return [
        np.asarray(pos, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
    ]


def test_triplewise_raises_when_any_states_empty():
    base_t = {"t0": _state([1.0, 0.0, 0.0])}
    base_r = {"r0": _state([0.0, 0.0, 0.0])}
    base_x = {"x0": _state([-1.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match="非空"):
        RxTargetTxGeometric.from_states({}, base_r, base_x)
    with pytest.raises(ValueError, match="非空"):
        RxTargetTxGeometric.from_states(base_t, {}, base_x)
    with pytest.raises(ValueError, match="非空"):
        RxTargetTxGeometric.from_states(base_t, base_r, {})


def test_type_tensor_monostatic_when_tx_rx_colocated():
    target_states = {"t0": _state([5.0, 0.0, 0.0])}
    rx_states = {"r0": _state([0.0, 0.0, 0.0])}
    tx_states = {"x0": _state([0.0, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    assert geom.type_tensor.shape == (1, 1, 1)
    assert not bool(geom.type_tensor[0, 0, 0].item())


def test_type_tensor_bistatic_when_tx_rx_separated():
    target_states = {"t0": _state([0.0, 0.0, 0.0])}
    rx_states = {"r0": _state([0.0, 0.0, 0.0])}
    tx_states = {"x0": _state([MONOSTATIC_TX_RX_EPS_M * 10, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    assert bool(geom.type_tensor[0, 0, 0].item())


def test_range_tensor_mono_vs_bi():
    target_states = {
        "ta": _state([10.0, 0.0, 0.0]),
        "tb": _state([0.0, 5.0, 1.0]),
    }
    rx_states = {"r1": _state([0.0, 0.0, 0.0])}
    tx_colocated = {"x0": _state([0.0, 0.0, 0.0])}
    geom_m = RxTargetTxGeometric.from_states(target_states, rx_states, tx_colocated)
    assert geom_m.type_tensor.shape == (1, 2, 1)
    for i, tn in enumerate(geom_m.target_names):
        ti = torch.tensor(target_states[tn][0], dtype=torch.float64).reshape(3)
        rj = torch.tensor(rx_states["r1"][0], dtype=torch.float64).reshape(3)
        expected_mono = torch.linalg.vector_norm(rj - ti)
        assert torch.allclose(geom_m.range_tensor[0, i, 0], expected_mono)

    tx_far = {"x0": _state([100.0, 0.0, 0.0])}
    geom_b = RxTargetTxGeometric.from_states(target_states, rx_states, tx_far)
    assert bool(geom_b.type_tensor[0, 0, 0].item())
    ti = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    xk = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64)
    rj = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    expected_bi = torch.linalg.vector_norm(ti - xk) + torch.linalg.vector_norm(rj - ti)
    assert torch.allclose(geom_b.range_tensor[0, 0, 0], expected_bi)


def test_vel_tensor_bistatic_matches_range_rate_formula():
    """双基地：``d/dt(||T-X||+||R-T||) = (v_T-v_X)·û_TX + (v_R-v_T)·û_RT``（共线几何手算）。"""
    target_states = {"t0": _state([50.0, 0.0, 0.0], vel=[3.0, 0.0, 0.0])}
    rx_states = {"r0": _state([100.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])}
    tx_states = {"x0": _state([0.0, 0.0, 0.0], vel=[1.0, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    assert bool(geom.type_tensor[0, 0, 0].item())
    expected = -1.0  # (3-1)*1 + (0-3)*1
    assert torch.allclose(
        geom.vel_tensor[0, 0, 0],
        torch.tensor(expected, dtype=torch.float64),
    )


def test_vel_tensor_matches_rx_radial_projection():
    """单基地格点：速度等于 RX–目标视线径向投影（多 TX 须均与 RX 共址，否则部分列为双基地公式）。"""
    target_states = {"t0": _state([1.0, 0.0, 0.0], vel=[1.0, 0.0, 0.0])}
    rx_states = {"r0": _state([0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])}
    tx_states = {"x0": _state([0.0, 0.0, 0.0]), "x1": _state([0.0, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    t_pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    t_vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    r_pos = torch.zeros(3, dtype=torch.float64)
    r_vel = torch.zeros(3, dtype=torch.float64)
    los = t_pos - r_pos
    dist = torch.linalg.vector_norm(los).clamp_min(1e-12)
    expected_v = ((t_vel - r_vel) * (los / dist)).sum()
    assert torch.allclose(geom.vel_tensor[0, 0, 0], expected_v)
    assert torch.allclose(geom.vel_tensor[0, 0, 1], expected_v)


def test_vel_tensor_second_tx_bistatic_differs_from_rx_radial():
    """第二个 TX 不与 RX 共址时，该列使用双基地距离变化率，一般不等于 RX 视线径向速度。"""
    target_states = {"t0": _state([1.0, 0.0, 0.0], vel=[1.0, 0.0, 0.0])}
    rx_states = {"r0": _state([0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0])}
    tx_states = {"x0": _state([0.0, 0.0, 0.0]), "x1": _state([5.0, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    assert torch.allclose(geom.vel_tensor[0, 0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(geom.vel_tensor[0, 0, 1], torch.tensor(0.0, dtype=torch.float64))


def test_compute_range_where_matches_piecewise():
    t_stack = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    r_stack = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    x_stack = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)
    is_bi = torch.tensor([[[True]]], dtype=torch.bool)
    out = compute_range(is_bi, t_stack, r_stack, x_stack)
    assert out.shape == (1, 1, 1)
    assert torch.allclose(out, torch.tensor([[[7.0 + 3.0]]], dtype=torch.float64))
    is_mo = torch.tensor([[[False]]], dtype=torch.bool)
    out_m = compute_range(is_mo, t_stack, r_stack, x_stack)
    assert torch.allclose(out_m, torch.tensor([[[3.0]]], dtype=torch.float64))


def test_triplewise_name_order_follows_dict_insertion_order():
    target_states = {"z_last": _state([0.0, 0.0, 1.0]), "a_first": _state([1.0, 0.0, 0.0])}
    rx_states = {"rz": _state([0.0, 1.0, 0.0])}
    tx_states = {"tx": _state([2.0, 0.0, 0.0])}
    geom = RxTargetTxGeometric.from_states(target_states, rx_states, tx_states)
    assert geom.target_names == ["z_last", "a_first"]


def test_display_includes_type_range_vel(capsys):
    geom = RxTargetTxGeometric.from_states(
        {"t0": _state([1.0, 0.0, 0.0])},
        {"r0": _state([0.0, 0.0, 0.0])},
        {"x0": _state([0.0, 0.0, 0.0])},
    )
    geom.display()
    text = capsys.readouterr().out
    assert "路径类型" in text and "路径长度_m" in text and "径向速度_mps" in text
    assert "monostatic" in text
    assert "t0" in text and "r0" in text and "x0" in text
