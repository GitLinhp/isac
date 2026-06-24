"""单基地时延–多普勒谱 CNN：回归目标距离与径向速度。"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MonostaticCNNConfig:
    """网络超参数。"""

    in_channels: int = 2
    base_channels: int = 32
    dropout: float = 0.2
    max_range_m: float = 5000.0
    max_velocity_mps: float = 50.0


class _ConvResidualBlock(nn.Module):
    """两层 3×3 卷积 + 可选下采样捷径。"""

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
    输出为 ``(B, 2)``，依次为 ``[range_m, radial_velocity_mps]``。

    训练时建议在 ``[0, 1]`` 归一化标签上优化；推理阶段由 ``denormalize`` 还原物理量。
    """

    def __init__(self, cfg: MonostaticCNNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MonostaticCNNConfig()
        c = self.cfg.base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _ConvResidualBlock(c, c)
        self.layer2 = _ConvResidualBlock(c, c * 2, stride=2)
        self.layer3 = _ConvResidualBlock(c * 2, c * 4, stride=2)
        self.layer4 = _ConvResidualBlock(c * 4, c * 8, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.cfg.dropout),
            nn.Linear(c * 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y_norm = self.head(x)
        return self.denormalize(y_norm)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``[0, 1]`` 归一化输出，供训练损失使用。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """将 Sigmoid 输出映射为物理单位。"""
        scale = y_norm.new_tensor(
            [self.cfg.max_range_m, self.cfg.max_velocity_mps],
        )
        # 速度轴以 0 为中心：将 [0,1] 线性映射到 [-Vmax, +Vmax]
        y = y_norm * scale
        y = y.clone()
        y[..., 1] = (y_norm[..., 1] * 2.0 - 1.0) * self.cfg.max_velocity_mps
        return y

    @staticmethod
    def normalize_labels(
        range_m: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        max_range_m: float,
        max_velocity_mps: float,
    ) -> torch.Tensor:
        """将物理标签转为 ``[0, 1]`` 监督目标 ``(B, 2)``。"""
        r_norm = range_m / max_range_m
        v_norm = velocity_mps / (2.0 * max_velocity_mps) + 0.5
        return torch.stack([r_norm, v_norm], dim=-1)
