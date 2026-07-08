"""时延–多普勒谱 CNN 输入特征预处理。"""

from __future__ import annotations

import numpy as np
import torch


def dd_spectrum_to_features(
    h_dd: torch.Tensor,
    *,
    eps: float = 1e-12,
    use_phase: bool = True,
) -> torch.Tensor:
    """将 ROI 裁切后的复数时延–多普勒谱转为 CNN 输入特征 ``(C, H, W)``。

    - 通道 0：幅度 dB（逐样本零均值、单位方差）
    - 通道 1（可选）：相位，映射到 ``[-1, 1]``
    """
    mag = torch.abs(h_dd).clamp_min(eps)
    mag_db = 20.0 * torch.log10(mag)
    mag_db = (mag_db - mag_db.mean()) / (mag_db.std() + eps)

    channels = [mag_db]
    if use_phase:
        phase = torch.angle(h_dd) / np.pi
        channels.append(phase)

    return torch.stack(channels, dim=0)
