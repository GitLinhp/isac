"""PyTorch Dataset：HDF5 CFR → 时延–多普勒特征 + 几何标签。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import tomli

from ..data_structures import SystemParams
from ..data_structures.components.ofdm_components import OFDMComponents
from ..datasets import Dataset as IsacDataset
from ..sensing.sensing_performance import SensingPerformance
from ..utils.config_loader import load_config
from .dd_spectrum import (
    compute_dd_spectrum,
    dd_spectrum_to_features,
    monostatic_labels_from_kinematics,
)


def _normalize_torch_device(device: torch.device | str | None) -> torch.device:
    """将 ``cuda`` 规范为 ``cuda:0``，与 Sionna / 项目其它脚本一致。"""
    dev = torch.device(device or "cpu")
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda:0")
    return dev


def _sionna_device_str(device: torch.device) -> str:
    """Sionna 仅接受 ``cuda:0`` 等形式，不接受裸 ``cuda``。"""
    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    return "cpu"


class MonostaticSensingTorchDataset(Dataset):
    """单基地感知训练集。

    从 ISAC HDF5 读取 CFR，在线计算时延–多普勒谱并裁剪 ROI；
    标签为几何斜距与 RX 视线径向速度。
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        config_file: str | Path,
        offset: int = 128,
        use_phase: bool = True,
        device: torch.device | None = None,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.offset = offset
        self.use_phase = use_phase
        self.device = _normalize_torch_device(device)

        self._isac_ds = IsacDataset.load(self.h5_path)
        n = self._isac_ds.num_slots
        self._indices = (
            np.arange(n, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

        params = _load_system_params(config_file)
        ofdm = OFDMComponents.build_from_params(params, device=_sionna_device_str(self.device))
        self._sensing_perf = SensingPerformance(
            ofdm.rg,
            carrier_frequency=self._isac_ds.carrier_frequency,
        )
        self.max_range_m = float(self._sensing_perf.max_range)
        self.max_velocity_mps = float(self._sensing_perf.max_velocity)

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        slot = int(self._indices[idx])
        cfr = self._isac_ds.cfr[slot]
        h_dd = compute_dd_spectrum(
            cfr,
            self._sensing_perf,
            device=self.device,
        )
        features = dd_spectrum_to_features(
            h_dd,
            offset=self.offset,
            use_phase=self.use_phase,
        )

        range_m, vel_mps = monostatic_labels_from_kinematics(
            self._isac_ds.target_position[slot],
            self._isac_ds.target_velocity[slot],
            self._isac_ds.bs_pos,
        )

        return {
            "features": features.to(dtype=torch.float32),
            "range_m": torch.tensor(range_m, dtype=torch.float32),
            "velocity_mps": torch.tensor(vel_mps, dtype=torch.float32),
            "slot": torch.tensor(slot, dtype=torch.int64),
        }


def _load_system_params(config_file: str | Path) -> SystemParams:
    """从 TOML 构建 ``SystemParams``；支持绝对路径或 ``config/`` 下相对名。"""
    path = Path(config_file)
    if path.is_absolute() and path.exists():
        with open(path, "rb") as f:
            cfg = tomli.load(f)
    else:
        cfg = load_config(path.name if path.suffix else str(path))
    return SystemParams.from_dict(cfg)
