"""单基地时延–多普勒谱 CNN：回归 ROI 局部 bin（MusicPeaks 坐标系）。

数据流::

    输入 (B, 2, H, W) 时延–多普勒特征
      → stem + 4 级残差编码
      → 回归头（线性）
      → MusicPeaks(peaks_delay, peaks_doppler)
      → SensingEstimator → 距离/速度
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_structures.types import MusicPeaks

_REQUIRED_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "dropout",
    "range_resolution",
    "velocity_resolution",
    "max_range_m",
    "max_velocity_mps",
    "num_doppler_bins",
)


class ConvResidualBlock(nn.Module):
    """两层 3×3 卷积残差块。

    ``stride > 1`` 或输入/输出通道不一致时，使用 1×1 卷积捷径对齐形状。
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class MonostaticDelayDopplerCNN(nn.Module):
    """单基地自发自收感知 CNN。

    输入为时延–多普勒谱特征图 ``(B, C, H, W)``（默认 2 通道：dB 幅度 + 相位），
    ``forward_bins`` 输出 ROI 局部 ``[peaks_delay, peaks_doppler]``；
    ``forward`` 返回与 MUSIC 同格式的 :class:`~isac.data_structures.types.MusicPeaks`。
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        base_channels: int = 32,
        dropout: float = 0.2,
        range_resolution: float = 2.5,
        velocity_resolution: float = 0.5,
        max_range_m: float,
        max_velocity_mps: float,
        num_doppler_bins: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dropout = dropout
        self.range_resolution = range_resolution
        self.velocity_resolution = velocity_resolution
        self.max_range_m = max_range_m
        self.max_velocity_mps = max_velocity_mps
        self.num_doppler_bins = num_doppler_bins
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ConvResidualBlock(c, c)
        self.layer2 = ConvResidualBlock(c, c * 2, stride=2)
        self.layer3 = ConvResidualBlock(c * 2, c * 4, stride=2)
        self.layer4 = ConvResidualBlock(c * 4, c * 8, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(c * 4, 2),
        )

    @property
    def roi_max_range_m(self) -> float:
        return self.max_range_m

    @property
    def roi_max_velocity_mps(self) -> float:
        return self.max_velocity_mps

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward_bins(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ROI 局部 bin 预测 ``(B, 2)`` = ``[peaks_delay, peaks_doppler]``。"""
        return self.head(self._forward_features(x))

    def bins_to_music_peaks(
        self,
        bins: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> MusicPeaks:
        """将 ``(B, 2)`` 局部 bin 转为 ``MusicPeaks``（B=1 时用于推理）。"""
        if bins.ndim != 2 or bins.shape[-1] != 2:
            raise ValueError(f"bins 须为 (B, 2)，收到 {tuple(bins.shape)}")
        if bins.shape[0] != 1:
            raise ValueError("bins_to_music_peaks 推理路径仅支持 B=1")
        dev = device or bins.device
        return MusicPeaks.from_local_bins(
            bins[0, 0],
            bins[0, 1],
            device=dev,
        )

    def forward(self, x: torch.Tensor) -> MusicPeaks:
        """推理默认接口：返回 ``MusicPeaks``（B=1）。"""
        if x.shape[0] != 1:
            raise ValueError("forward MusicPeaks 仅支持 batch_size=1")
        return self.bins_to_music_peaks(self.forward_bins(x), device=x.device)


@dataclass(frozen=True)
class MonostaticCnnCheckpointMeta:
    """``run_train_monostatic_cnn.py`` checkpoint 中的训练/推理元数据。"""

    max_range_m: float
    max_velocity_mps: float
    range_resolution: float
    velocity_resolution: float
    num_doppler_bins: int
    dataset_h5: Path | None = None
    config_file: Path | None = None
    epoch: int | None = None


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).resolve()


def _validate_checkpoint_dict(ckpt: dict) -> None:
    missing = [key for key in _REQUIRED_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")


def _load_checkpoint_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_dict(ckpt)
    return ckpt


def _meta_from_checkpoint_dict(ckpt: dict) -> MonostaticCnnCheckpointMeta:
    epoch_raw = ckpt.get("epoch")
    return MonostaticCnnCheckpointMeta(
        max_range_m=float(ckpt["max_range_m"]),
        max_velocity_mps=float(ckpt["max_velocity_mps"]),
        range_resolution=float(ckpt["range_resolution"]),
        velocity_resolution=float(ckpt["velocity_resolution"]),
        num_doppler_bins=int(ckpt["num_doppler_bins"]),
        dataset_h5=_optional_path(ckpt.get("dataset_h5")),
        config_file=_optional_path(ckpt.get("config_file")),
        epoch=int(epoch_raw) if epoch_raw is not None else None,
    )


def read_monostatic_cnn_checkpoint_meta(
    path: str | Path,
) -> MonostaticCnnCheckpointMeta:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")
    return _meta_from_checkpoint_dict(_load_checkpoint_dict(ckpt_path))


def load_monostatic_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[MonostaticDelayDopplerCNN, MonostaticCnnCheckpointMeta]:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")

    ckpt = _load_checkpoint_dict(ckpt_path)
    meta = _meta_from_checkpoint_dict(ckpt)

    model = MonostaticDelayDopplerCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        dropout=float(ckpt["dropout"]),
        range_resolution=meta.range_resolution,
        velocity_resolution=meta.velocity_resolution,
        max_range_m=meta.max_range_m,
        max_velocity_mps=meta.max_velocity_mps,
        num_doppler_bins=meta.num_doppler_bins,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta
