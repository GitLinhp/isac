"""PyTorch Dataset：HDF5 CFR → System 感知链 h_dd → CNN 特征 + 几何标签。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from isac.datasets import Dataset as IsacDataset
from isac.system import System
from isac.utils import load_config
from isac.utils.data_collection.channel_export import cfr_numpy_to_h_freq

from .dd_spectrum import dd_spectrum_to_features, monostatic_labels_from_kinematics



class MonostaticSensingTorchDataset(Dataset):
    """单基地感知训练集。

    与 ``run_sensing_from_dataset.py`` 一致：逐样本注入存储 CFR，
    经发射 → 信道（含 AWGN）→ ``compute_sensing_spectrum`` 得到 ``h_dd``，
    再转为 CNN 特征；标签为几何斜距与径向速度。
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        config_file: str | Path,
        offset: int = 128,
        use_phase: bool = True,
        device: torch.device | str | None = None,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.offset = offset
        self.use_phase = use_phase
        self.device = device

        self._isac_ds = IsacDataset.load(self.h5_path)
        n = self._isac_ds.num_slots
        self._indices = (
            np.arange(n, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        config = load_config(config_path)
        self._system = System(
            config=config,
            batch_size=1,
            device=str(self.device),
        )

        comps = self._system.components
        if comps.rt_simulator is None:
            raise ValueError("训练数据集要求 channel.type='rt' 且已配置 [rt_simulator]")
        if comps.ls_channel_estimator is None:
            raise ValueError(
                "训练数据集要求已构建 ls_channel_estimator（与 compute_sensing_spectrum 一致）"
            )
        if comps.sensing_performance is None:
            raise ValueError("训练数据集要求已构建 sensing_performance 组件")

        self.range_resolution = float(comps.sensing_performance.range_resolution)
        self.velocity_resolution = float(
            comps.sensing_performance.velocity_resolution
        )
        self._snr_db = self._system.params.channel.snr_db

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        slot = int(self._indices[idx])
        cfr, _label = self._isac_ds[slot]

        system = self._system
        channel = system.components.channel
        _, x_rg, _ = system.transmit()
        channel.cfr = cfr_numpy_to_h_freq(cfr, device=x_rg.device)
        y_rg = channel(x_rg, domain="frequency", snr_db=self._snr_db)
        h_dd = system.compute_sensing_spectrum(x_rg, y_rg)

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
