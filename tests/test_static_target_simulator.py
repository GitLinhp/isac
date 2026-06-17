"""static_target_simulator Torch 实现与 gr-radar 对照测试。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from isac.channel.static_target_simulator import (
    StaticTargetParams,
    static_target_params_from_grc,
    static_target_simulator,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gnuradio.radar") is None,
    reason="gnuradio.radar 未安装",
)

N_SAMPLES = 4096
SAMP_RATE = 30_720_000
CENTER_FREQ = 6e9


def _random_tx(rng: np.random.Generator, n: int = N_SAMPLES) -> np.ndarray:
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _run_gr_radar(
    tx: np.ndarray,
    *,
    range_m: float | list[float],
    velocity_mps: float | list[float],
    rcs: float | list[float],
    azimuth_deg: float | list[float] = 0.0,
    position_rx_m: list[float] | None = None,
    self_coupling_db: float = -10.0,
    rndm_phaseshift: bool = False,
    self_coupling: bool = True,
) -> np.ndarray:
    from gnuradio import blocks, gr, radar

    if position_rx_m is None:
        position_rx_m = [0.0]

    def _vec(x: float | list[float]) -> list[float]:
        return x if isinstance(x, list) else [float(x)]

    n = tx.size

    class TopBlock(gr.top_block):
        def __init__(self) -> None:
            super().__init__()
            self.src = blocks.vector_source_c(tx.tolist(), False)
            self.tag = blocks.stream_to_tagged_stream(
                gr.sizeof_gr_complex, 1, n, "packet_len"
            )
            self.sim = radar.static_target_simulator_cc(
                _vec(range_m),
                _vec(velocity_mps),
                _vec(rcs),
                _vec(azimuth_deg),
                position_rx_m,
                SAMP_RATE,
                CENTER_FREQ,
                self_coupling_db,
                rndm_phaseshift,
                self_coupling,
                "packet_len",
            )
            self.snk = blocks.vector_sink_c(1)
            self.connect(self.src, self.tag, self.sim, self.snk)

    tb = TopBlock()
    tb.run()
    return np.asarray(tb.snk.data(), dtype=np.complex64)


def _run_torch(
    tx: np.ndarray,
    params: StaticTargetParams,
) -> np.ndarray:
    out = static_target_simulator(torch.from_numpy(tx), params)
    return np.asarray(out.detach().cpu().numpy(), dtype=np.complex64)


def _assert_close_to_gr(y_torch: np.ndarray, y_gr: np.ndarray, *, rtol: float) -> None:
    denom = max(float(np.max(np.abs(y_gr))), 1e-12)
    rel = float(np.max(np.abs(y_torch - y_gr)) / denom)
    corr = float(
        abs(np.vdot(y_gr, y_torch))
        / (np.linalg.norm(y_gr) * np.linalg.norm(y_torch) + 1e-20)
    )
    assert rel <= rtol, f"rel max err={rel:.6f} > {rtol}"
    assert corr >= 1.0 - rtol, f"norm corr={corr:.6f}"


def test_self_coupling_only_matches_gr() -> None:
    rng = np.random.default_rng(1)
    tx = _random_tx(rng)
    params = static_target_params_from_grc(
        range_m=1e6,
        velocity_mps=0.0,
        rcs=1e-30,
        rndm_phaseshift=False,
        self_coupling=True,
    )
    y_gr = _run_gr_radar(
        tx,
        range_m=1e6,
        velocity_mps=0.0,
        rcs=1e-30,
        rndm_phaseshift=False,
        self_coupling=True,
    )
    y_torch = _run_torch(tx, params)
    np.testing.assert_allclose(y_torch, y_gr, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            range_m=100.0,
            velocity_mps=5.0,
            rcs=1e25,
            self_coupling=True,
        ),
        dict(
            range_m=100.0,
            velocity_mps=5.0,
            rcs=1e25,
            self_coupling=False,
        ),
        dict(
            range_m=[80.0, 150.0],
            velocity_mps=[3.0, -7.0],
            rcs=[1e25, 1e25],
            azimuth_deg=[0.0, 0.0],
            self_coupling=False,
        ),
    ],
)
def test_echo_matches_gr_within_fft_tolerance(kwargs: dict) -> None:
    """Torch FFT 与 gr-radar FFTW 存在 ~3% 峰值误差，形状相关 >0.999。"""
    rng = np.random.default_rng(2)
    tx = _random_tx(rng)
    params = static_target_params_from_grc(rndm_phaseshift=False, **kwargs)
    y_gr = _run_gr_radar(tx, rndm_phaseshift=False, **kwargs)
    y_torch = _run_torch(tx, params)
    _assert_close_to_gr(y_torch, y_gr, rtol=0.035)


def test_multi_rx_output_shape() -> None:
    rng = np.random.default_rng(3)
    tx = _random_tx(rng, n=512)
    params = static_target_params_from_grc(
        position_rx_m=(0.0, 0.5),
        rndm_phaseshift=False,
        self_coupling=False,
    )
    out = static_target_simulator(torch.from_numpy(tx), params)
    assert out.shape == (2, tx.size)


def test_invalid_target_vector_lengths() -> None:
    with pytest.raises(ValueError):
        StaticTargetParams(
            range_m=[100.0, 200.0],
            velocity_mps=[1.0],
            rcs=[1e25, 1e25],
            azimuth_deg=[0.0, 0.0],
            position_rx_m=[0.0],
            samp_rate=SAMP_RATE,
            center_freq=CENTER_FREQ,
        )
