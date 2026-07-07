"""单基地时延–多普勒谱 CNN：回归目标距离与径向速度。

数据流::

    输入 (B, 2, H, W) 时延–多普勒特征
      → stem + 4 级残差编码
      → 回归头（线性）
      → bins_to_physical → [range_m, radial_velocity_mps]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REQUIRED_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "dropout",
    "range_resolution",
    "velocity_resolution",
    "max_range_m",
    "max_velocity_mps",
)


def physical_to_bins(
    range_m: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    range_resolution: float,
    velocity_resolution: float,
) -> torch.Tensor:
    """物理量转为连续 bin 监督目标 ``(B, 2)``。

    - 第 0 维：距离 bin ``range_m / range_resolution``
    - 第 1 维：多普勒 bin（有符号）``velocity_mps / velocity_resolution``
    """
    range_bin = range_m / range_resolution
    vel_bin = velocity_mps / velocity_resolution
    return torch.stack([range_bin, vel_bin], dim=-1)


def bins_to_physical(
    bins: torch.Tensor,
    *,
    range_resolution: float,
    velocity_resolution: float,
) -> torch.Tensor:
    """连续 bin 还原为 ``(range_m, velocity_mps)``。"""
    range_m = bins[..., 0] * range_resolution
    velocity = bins[..., 1] * velocity_resolution
    return torch.stack([range_m, velocity], dim=-1)


class ConvResidualBlock(nn.Module):
    """两层 3×3 卷积残差块。

    ``stride > 1`` 或输入/输出通道不一致时，使用 1×1 卷积捷径对齐形状。
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        """参数:
        ----------
        - in_ch : 输入通道数
        - out_ch : 输出通道数
        - stride : 第一层卷积步长；``> 1`` 时同步下采样空间分辨率
        """
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
    输出为 ``(B, 2)``，依次为 ``[range_m, radial_velocity_mps]``。

    网络结构（默认 ``base_channels=32``）::

        stem: Conv7×7/2 + MaxPool
        layer1~4: 通道 c→c→2c→4c→8c，后三级 stride=2 下采样
        head: GAP + MLP(8c→4c→2) 线性输出

    训练时在分辨率 bin 空间优化；推理时由 ``bins_to_physical`` 还原物理量。
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
    ) -> None:
        """参数:
        ----------
        - in_channels : 输入特征通道数（幅度 dB + 相位时为 2）
        - base_channels : 编码器基础通道数 ``c``，末级特征通道为 ``8c``
        - dropout : 回归头 MLP 的 Dropout 比例
        - range_resolution : 距离分辨率 (m/bin)
        - velocity_resolution : 速度分辨率 (m/s/bin)
        - max_range_m : DD 谱 ROI 最大距离 (m)
        - max_velocity_mps : DD 谱 ROI 最大多普勒半幅速度 (m/s)
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dropout = dropout
        self.range_resolution = range_resolution
        self.velocity_resolution = velocity_resolution
        self.max_range_m = max_range_m
        self.max_velocity_mps = max_velocity_mps
        c = base_channels

        # --- 编码器入口 ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # --- 残差下采样：c → 2c → 4c → 8c ---
        self.layer1 = ConvResidualBlock(c, c)
        self.layer2 = ConvResidualBlock(c, c * 2, stride=2)
        self.layer3 = ConvResidualBlock(c * 2, c * 4, stride=2)
        self.layer4 = ConvResidualBlock(c * 4, c * 8, stride=2)

        # --- 回归头：GAP + MLP，线性输出 bin ---
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
        """ROI 时延轴最大距离 (m)，供日志参考。"""
        return self.max_range_m

    @property
    def roi_max_velocity_mps(self) -> float:
        """ROI 多普勒半幅速度 (m/s)，供日志参考。"""
        return self.max_velocity_mps

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """编码器前向：返回末级特征图，不含回归头。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward_bins(self, x: torch.Tensor) -> torch.Tensor:
        """返回连续 bin 预测 ``(B, 2)``，供训练损失使用。"""
        return self.head(self._forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推理默认接口：返回物理单位的距离与径向速度。"""
        return self.bins_to_physical(self.forward_bins(x))

    def bins_to_physical(self, bins: torch.Tensor) -> torch.Tensor:
        """将 bin 预测映射为物理单位。"""
        return bins_to_physical(
            bins,
            range_resolution=self.range_resolution,
            velocity_resolution=self.velocity_resolution,
        )

    @staticmethod
    def physical_to_bins(
        range_m: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        range_resolution: float,
        velocity_resolution: float,
    ) -> torch.Tensor:
        """将物理标签转为 ``(B, 2)`` bin 监督目标。"""
        return physical_to_bins(
            range_m,
            velocity_mps,
            range_resolution=range_resolution,
            velocity_resolution=velocity_resolution,
        )


@dataclass(frozen=True)
class MonostaticCnnCheckpointMeta:
    """``run_train_monostatic_cnn.py`` checkpoint 中的训练/推理元数据。"""

    max_range_m: float
    max_velocity_mps: float
    range_resolution: float
    velocity_resolution: float
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
    """读取 checkpoint 并校验必填字段。"""
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
        dataset_h5=_optional_path(ckpt.get("dataset_h5")),
        config_file=_optional_path(ckpt.get("config_file")),
        epoch=int(epoch_raw) if epoch_raw is not None else None,
    )


def read_monostatic_cnn_checkpoint_meta(
    path: str | Path,
) -> MonostaticCnnCheckpointMeta:
    """仅读取 checkpoint 元数据（不加载权重）。"""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")
    return _meta_from_checkpoint_dict(_load_checkpoint_dict(ckpt_path))


def load_monostatic_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[MonostaticDelayDopplerCNN, MonostaticCnnCheckpointMeta]:
    """加载 ``run_train_monostatic_cnn.py`` 保存的 checkpoint。

    返回 ``(model, meta)``；``model`` 已 ``eval()`` 并置于 ``device``。
    """
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta
