"""小米单站测距 Torch Dataset（HDF5 距离谱 → 特征）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from isac.xiaomi_models.preprocess import (
    DEFAULT_RANGE_ROI,
    FeatureMode,
    default_range_bin_step,
    profile_to_features,
    profile_to_roi,
)
from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    DATASET_KEY_FRAME_INDEX,
    DATASET_KEY_PROFILES,
    DATASET_KEY_SESSION_INDEX,
    DATASET_KEY_TARGET_RANGE,
    META_KEY_FFT_LEN,
    META_KEY_VLEN,
    META_KEY_ZEROPADDING_FAC,
)


class SingleBsRangeTorchDataset(Dataset):
    """单站测距训练 Dataset：``features (C,L)`` + ``target_range``。"""

    def __init__(
        self,
        h5_path: str | Path,
        *,
        range_roi: tuple[float, float] = DEFAULT_RANGE_ROI,
        range_bin_step: float | None = None,
        feature_mode: FeatureMode = "real_imag",
        cache_features: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(self.h5_path)

        self.range_roi = (float(range_roi[0]), float(range_roi[1]))
        self.feature_mode: FeatureMode = feature_mode
        self.cache_features = bool(cache_features)

        with h5py.File(self.h5_path, "r") as f:
            profiles = np.asarray(f[DATASET_KEY_PROFILES][:], dtype=np.complex64)
            self.target_range = np.asarray(f[DATASET_KEY_TARGET_RANGE][:], dtype=np.float64)
            self.session_index = np.asarray(f[DATASET_KEY_SESSION_INDEX][:], dtype=np.int32)
            self.frame_index = np.asarray(f[DATASET_KEY_FRAME_INDEX][:], dtype=np.int32)
            attrs = dict(f.attrs)
            fft_len = int(attrs.get(META_KEY_FFT_LEN, 4096))
            zp = int(attrs.get(META_KEY_ZEROPADDING_FAC, 4))
            self.vlen = int(attrs.get(META_KEY_VLEN, profiles.shape[1]))

        if range_bin_step is None:
            self.range_bin_step = default_range_bin_step(fft_len=fft_len, zeropadding_fac=zp)
        else:
            self.range_bin_step = float(range_bin_step)

        self._features: torch.Tensor | None = None
        if self.cache_features:
            feats = [
                profile_to_features(
                    profile_to_roi(
                        profiles[i],
                        range_roi=self.range_roi,
                        range_bin_step=self.range_bin_step,
                    ),
                    mode=self.feature_mode,
                )
                for i in range(profiles.shape[0])
            ]
            self._features = torch.stack(feats, dim=0)
            self._profiles = None
        else:
            self._profiles = profiles

        self.attrs: dict[str, Any] = {
            "fft_len": fft_len,
            "zeropadding_fac": zp,
            "vlen": self.vlen,
            "range_roi": self.range_roi,
            "range_bin_step": self.range_bin_step,
            "feature_mode": self.feature_mode,
        }

    def __len__(self) -> int:
        return int(self.target_range.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._features is not None:
            features = self._features[index]
        else:
            assert self._profiles is not None
            features = profile_to_features(
                profile_to_roi(
                    self._profiles[index],
                    range_roi=self.range_roi,
                    range_bin_step=self.range_bin_step,
                ),
                mode=self.feature_mode,
            )
        return {
            "features": features,
            "target_range": torch.tensor(
                float(self.target_range[index]), dtype=torch.float32
            ),
            "session_index": torch.tensor(
                int(self.session_index[index]), dtype=torch.int64
            ),
            "frame_index": torch.tensor(int(self.frame_index[index]), dtype=torch.int64),
        }
