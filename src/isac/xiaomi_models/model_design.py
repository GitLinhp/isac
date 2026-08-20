"""小米单站测距 1D CNN：ROI 距离谱特征 → 估计距离 (m)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_REQUIRED_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "num_layers",
    "dropout",
)


class Conv1dResidualBlock(nn.Module):
    """两层 3×1 一维卷积残差块。"""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def _build_conv1d_backbone(
    in_channels: int,
    base_channels: int,
    num_layers: int,
) -> tuple[nn.Sequential, nn.ModuleList, int]:
    c = base_channels
    stem = nn.Sequential(
        nn.Conv1d(in_channels, c, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm1d(c),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(3, stride=2, padding=1),
    )
    layers: list[Conv1dResidualBlock] = []
    for i in range(num_layers):
        if i == 0:
            layers.append(Conv1dResidualBlock(c, c))
        else:
            in_ch = c * (2 ** (i - 1))
            out_ch = c * (2**i)
            layers.append(Conv1dResidualBlock(in_ch, out_ch, stride=2))
    final_ch = c * (2 ** (num_layers - 1))
    return stem, nn.ModuleList(layers), final_ch


class SingleBsRangeCNN(nn.Module):
    """单站 ROI 距离谱特征 → 标量距离回归。

    输入 ``(B, C, L)`` float，输出 ``(B,)`` 估计距离（米）。
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        base_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.stem, self.layers, final_ch = _build_conv1d_backbone(
            self.in_channels, self.base_channels, self.num_layers
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_ch, final_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(final_ch // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入须为 (B, C, L)，收到 {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"期望 in_channels={self.in_channels}，收到 {x.shape[1]}"
            )
        h = self.stem(x)
        for block in self.layers:
            h = block(h)
        return self.head(h).squeeze(-1)


def save_single_bs_range_cnn_checkpoint(
    path: str | Path,
    model: SingleBsRangeCNN,
    *,
    feature_mode: str = "real_imag",
    range_roi: tuple[float, float] = (0.0, 8.0),
    range_bin_step: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "base_channels": model.base_channels,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "feature_mode": feature_mode,
        "range_roi": (float(range_roi[0]), float(range_roi[1])),
        "range_bin_step": float(range_bin_step),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_single_bs_range_cnn_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[SingleBsRangeCNN, dict[str, Any]]:
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing = [k for k in _REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少键 {missing}: {path}")
    model = SingleBsRangeCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
