"""gnuradio 薄封装：re-export ISAC Torch RCSChannel。"""
import sys
from pathlib import Path

_GRC = Path(__file__).resolve().parent
_REPO = _GRC.parent
_SRC = _REPO / "src"
for _p in (_GRC, str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch

from isac.channel import RCSScene, RCSTarget, RCSChannel

__all__ = [
    "RCSScene",
    "RCSTarget",
    "RCSChannel",
    "apply_grc_default_channel",
]


def apply_grc_default_channel(
    tx: np.ndarray | torch.Tensor,
    range_m: float = 100.0,
    velocity_mps: float = 5.0,
    *,
    rcs: float = 1e25,
    center_freq: float = 6e9,
    samp_rate: int = 30_720_000,
    self_coupling_db: float = -10.0,
    rndm_phaseshift: bool = True,
    self_coupling: bool = True,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """用 simulator_ofdm.grc 默认参数对时域 IQ 施加 static_target 信道，返回 numpy complex64。"""
    target = RCSTarget(
        range_m=range_m,
        velocity_mps=velocity_mps,
        rcs=rcs,
        azimuth_deg=0.0,
        position_rx_m=0.0,
    )
    scene = RCSScene(
        target=target,
        self_coupling_db=self_coupling_db,
        rndm_phaseshift=rndm_phaseshift,
        self_coupling=self_coupling,
    )
    sim = RCSChannel(
        rcs_scene=lambda: scene,
        center_freq=float(center_freq),
        samp_rate=float(samp_rate),
    )
    if isinstance(tx, np.ndarray):
        tx_t = torch.from_numpy(np.asarray(tx, dtype=np.complex64)).to(device)
    else:
        tx_t = tx.to(device)

    rx_t = sim(tx_t, generator=generator)
    return np.asarray(rx_t.detach().cpu().numpy(), dtype=np.complex64)
