"""单基地时延–多普勒谱 CNN：回归 ROI 局部 bin（MusicPeaks 坐标系）。

数据流::

    输入 spectrum_tensor（复数裁切谱）
      → spectrum_tensor_to_features
      → stem + 4 级残差编码
      → 回归头（线性）
      → (B, 2) 局部 bin 张量 [peaks_delay, peaks_doppler]

感知 ROI / 分辨率由 TOML ``System`` 提供，不保存在本模型或 checkpoint 中。

网络结构
--------
stem（7×7 conv + pool）→ layer1–4（残差块，通道 32→64→128→256）→
全局池化 + MLP 回归头 → ``(B, 2)`` 局部 bin。

checkpoint
----------
仅保存 ``model_state_dict`` 与 ``in_channels`` / ``base_channels`` / ``dropout``；
见 ``_REQUIRED_CKPT_KEYS`` 与 ``load_monostatic_cnn_checkpoint``。

调用方
------
- 训练：``run_train_monostatic_cnn.py`` 写入 checkpoint；标签由 ``kinematics_to_target_bins`` 生成
- 推理：``run_sensing_from_dataset.py`` 加载权重；bin → ``MusicPeaks`` → ``SensingEstimator`` 换算物理量
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import spectrum_tensor_to_features

# checkpoint 必填键（感知 ROI/分辨率不写入，由 TOML System 提供）
_REQUIRED_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "dropout",
)


class ConvResidualBlock(nn.Module):
    """两层 3×3 卷积残差块。

    ``stride > 1`` 或输入/输出通道不一致时，使用 1×1 卷积捷径对齐形状。
    ``stride=1`` 时输出空间尺寸与输入相同，否则按 stride 下采样。
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
        """两层卷积 + BN → ReLU → 与 shortcut 残差相加。"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class MonostaticDelayDopplerCNN(nn.Module):
    """单基地自发自收感知 CNN。

    输入 ROI 裁切复数 ``spectrum_tensor``，``forward`` 返回可微
    ``(B, 2)`` 局部 bin 张量 ``[peaks_delay, peaks_doppler]``。
    输出 bin 坐标系与 :class:`~isac.sensing.metric.SpectrumMetric` /
    :class:`~isac.data_structures.types.MusicPeaks` 一致。
    推理侧由调用方将单条 bin 转为 ``MusicPeaks``，再经 ``SensingEstimator`` 换算物理量。

    参数
    ----
    in_channels : int
        特征通道数，默认 2（幅度 dB + 相位，见 ``spectrum_tensor_to_features``）
    base_channels : int
        stem 与 layer1 的基础通道数，后续残差层逐层加倍
    dropout : float
        回归头中的 dropout 概率
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        base_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dropout = dropout
        c = base_channels

        # stem：下采样特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # layer1–4：残差编码，逐层加倍通道
        self.layer1 = ConvResidualBlock(c, c)
        self.layer2 = ConvResidualBlock(c, c * 2, stride=2)
        self.layer3 = ConvResidualBlock(c * 2, c * 4, stride=2)
        self.layer4 = ConvResidualBlock(c * 4, c * 8, stride=2)
        # head：全局池化 + 线性回归至 2 维 bin
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(c * 4, 2),
        )

    def forward(self, spectrum_tensor: torch.Tensor) -> torch.Tensor:
        """复数裁切谱 → ROI 局部 bin 预测 ``(B, 2)``。

        输入 ``(H, W)`` 或 ``(B, H, W)`` complex64 裁切谱；经
        ``spectrum_tensor_to_features`` 转为 float 特征后再进卷积栈。
        输出 ``[peaks_delay, peaks_doppler]`` 局部 bin 坐标。
        """
        features = spectrum_tensor_to_features(spectrum_tensor)
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {features.shape[1]} 与模型 in_channels={self.in_channels} 不一致"
            )

        x = self.stem(features)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)

        return x


def load_monostatic_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> MonostaticDelayDopplerCNN:
    """从 checkpoint 加载 CNN 并置 ``eval()`` 模式。

    必填字段见 ``_REQUIRED_CKPT_KEYS``。感知参数（ROI、分辨率等）须由调用方
    经 ``data_collection.toml`` / ``System`` 单独提供。

    Raises
    ------
    FileNotFoundError
        路径不存在
    KeyError
        checkpoint 缺少必填键
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")

    # 先在 CPU 加载，再校验必填键
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in _REQUIRED_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")

    model = MonostaticDelayDopplerCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model
