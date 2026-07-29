"""单基地时延–多普勒谱 CNN：回归 ROI 局部 bin（MusicPeaks 坐标系）。

数据流::

    输入 spectrum_tensor（复数裁切谱）
      → spectrum_tensor_to_features
      → stem + num_layers 级残差编码（默认 4）
      → 回归头（线性）
      → (B, 2) 局部 bin 张量 [peaks_delay, peaks_doppler]

感知 ROI / 分辨率由 TOML ``System`` 提供，不保存在本模型或 checkpoint 中。

网络结构
--------
stem（7×7 conv + pool）→ 残差块 × ``num_layers``（默认通道 32→64→128→256）→
全局池化 + MLP 回归头 → ``(B, 2)`` 局部 bin。

checkpoint
----------
保存 ``model_state_dict`` 与 ``in_channels`` / ``base_channels`` / ``num_layers`` /
``dropout``；见 ``_REQUIRED_CKPT_KEYS`` 与 ``load_sensing_cnn_checkpoint``。

调用方
------
- 训练：``run_train_sensing_cnn.py`` 写入 checkpoint；标签由 ``kinematics_to_target_bins`` 生成
- 推理：``run_sensing_from_dataset.py`` 加载权重；bin → ``MusicPeaks`` → ``SensingEstimator`` 换算物理量
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import (
    dual_range_profiles_to_features,
    spectrum_tensor_to_features,
)

# checkpoint 必填键（感知 ROI/分辨率不写入，由 TOML System 提供）
_REQUIRED_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "num_layers",
    "dropout",
)
# 旧版 checkpoint 可能缺少 num_layers，加载时默认 4
_CORE_CKPT_KEYS = (
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


class SensingCNN(nn.Module):
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
        stem 与首层残差的基础通道数，后续残差层逐层加倍
    num_layers : int
        残差编码块数量，默认 4；首块 stride=1，其余 stride=2 下采样
    dropout : float
        回归头中的 dropout 概率
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        base_channels: int = 32,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.dropout = dropout
        c = base_channels

        # stem：下采样特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 残差编码：首层同分辨率，后续逐层加倍通道并下采样
        layers: list[ConvResidualBlock] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvResidualBlock(c, c))
            else:
                in_ch = c * (2 ** (i - 1))
                out_ch = c * (2**i)
                layers.append(ConvResidualBlock(in_ch, out_ch, stride=2))
        self.layers = nn.ModuleList(layers)

        final_ch = c * (2 ** (num_layers - 1))
        # head：全局池化 + 线性回归至 2 维 bin
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_ch, final_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(final_ch // 2, 2),
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
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)

        return x


def load_sensing_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> SensingCNN:
    """从 checkpoint 加载 CNN 并置 ``eval()`` 模式。

    必填字段见 ``_CORE_CKPT_KEYS``；``num_layers`` 缺省时按 4 处理以兼容旧 checkpoint。
    感知参数（ROI、分辨率等）须由调用方经 ``data_collection.toml`` / ``System`` 单独提供。

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
    missing = [key for key in _CORE_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")

    model = SensingCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        num_layers=int(ckpt.get("num_layers", 4)),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


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
        out = F.relu(out + self.shortcut(x))
        return out


# 1D 定位 CNN 的 range 池化方式（S1：保留距离维几何信息）
COOPERATIVE_POOL_MODES = (
    "gap",
    "attention",
    "multiscale",
    "gap_gmp",
    "soft_argmax",
)
CooperativePoolMode = str  # one of COOPERATIVE_POOL_MODES

# 双站融合方式（S2：晚融合 / 分站几何）
COOPERATIVE_FUSION_MODES = (
    "early",
    "late",
)
CooperativeFusionMode = str  # one of COOPERATIVE_FUSION_MODES


class RangeAttentionPool1d(nn.Module):
    """对 range 维做可学习注意力加权求和，保留峰位置线索。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, C)
        attn = torch.softmax(self.score(x), dim=-1)
        return (x * attn).sum(dim=-1)


class SoftArgmaxRangeCue(nn.Module):
    """将特征投影为 range 能量，再 soft-argmax 得到归一化距离估计。"""

    def __init__(
        self,
        channels: int,
        *,
        num_ranges: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_ranges < 1:
            raise ValueError(f"num_ranges 须 >= 1，收到 {num_ranges}")
        if temperature <= 0:
            raise ValueError(f"temperature 须 > 0，收到 {temperature}")
        self.num_ranges = int(num_ranges)
        self.proj = nn.Conv1d(channels, self.num_ranges, kernel_size=1, bias=True)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, num_ranges) 归一化 soft-argmax 位置 ∈ [0, 1]
        logits = self.proj(x) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        length = x.shape[-1]
        if length <= 1:
            return weights.sum(dim=-1) * 0.0
        coords = torch.linspace(
            0.0,
            1.0,
            length,
            device=x.device,
            dtype=x.dtype,
        )
        return (weights * coords).sum(dim=-1)


class CooperativeRangePool(nn.Module):
    """保留距离维信息的 range 池化（无 MLP）。

    ``gap`` / ``attention`` / ``multiscale`` / ``gap_gmp`` / ``soft_argmax``
    含义同 :class:`CooperativeRangePoolHead`；``soft_argmax`` 的线索维数由
    ``soft_argmax_ranges`` 控制（早融合默认 2，晚融合单站为 1）。
    """

    def __init__(
        self,
        channels: int,
        *,
        pool_mode: str = "gap",
        multiscale_bins: int = 8,
        soft_argmax_temp: float = 1.0,
        soft_argmax_ranges: int = 2,
    ) -> None:
        super().__init__()
        mode = str(pool_mode)
        if mode not in COOPERATIVE_POOL_MODES:
            raise ValueError(
                f"pool_mode 须为 {COOPERATIVE_POOL_MODES}，收到 {mode!r}"
            )
        if multiscale_bins < 1:
            raise ValueError(f"multiscale_bins 须 >= 1，收到 {multiscale_bins}")

        self.pool_mode = mode
        self.multiscale_bins = int(multiscale_bins)
        self.channels = int(channels)
        self.soft_argmax_ranges = int(soft_argmax_ranges)

        self.attn_pool: RangeAttentionPool1d | None = None
        self.soft_argmax: SoftArgmaxRangeCue | None = None
        self.multi_pool: nn.AdaptiveAvgPool1d | None = None

        if mode == "gap":
            self.out_features = channels
        elif mode == "attention":
            self.attn_pool = RangeAttentionPool1d(channels)
            self.out_features = channels
        elif mode == "multiscale":
            self.multi_pool = nn.AdaptiveAvgPool1d(self.multiscale_bins)
            self.out_features = channels * self.multiscale_bins
        elif mode == "gap_gmp":
            self.out_features = channels * 2
        else:  # soft_argmax
            self.soft_argmax = SoftArgmaxRangeCue(
                channels,
                num_ranges=self.soft_argmax_ranges,
                temperature=soft_argmax_temp,
            )
            self.out_features = channels + self.soft_argmax_ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.pool_mode
        if mode == "gap":
            return F.adaptive_avg_pool1d(x, 1).flatten(1)
        if mode == "attention":
            assert self.attn_pool is not None
            return self.attn_pool(x)
        if mode == "multiscale":
            assert self.multi_pool is not None
            return self.multi_pool(x).flatten(1)
        if mode == "gap_gmp":
            gap = F.adaptive_avg_pool1d(x, 1).flatten(1)
            gmp = F.adaptive_max_pool1d(x, 1).flatten(1)
            return torch.cat([gap, gmp], dim=1)
        assert self.soft_argmax is not None
        gap = F.adaptive_avg_pool1d(x, 1).flatten(1)
        range_cue = self.soft_argmax(x)
        return torch.cat([gap, range_cue], dim=1)


class CooperativeRangePoolHead(CooperativeRangePool):
    """保留距离维信息的池化 + MLP 回归头（早融合；池化属性平铺以兼容 S1 checkpoint 键）。"""

    def __init__(
        self,
        channels: int,
        *,
        dropout: float = 0.2,
        pool_mode: str = "gap",
        multiscale_bins: int = 8,
        soft_argmax_temp: float = 1.0,
        soft_argmax_ranges: int = 2,
        out_dim: int = 2,
    ) -> None:
        super().__init__(
            channels,
            pool_mode=pool_mode,
            multiscale_bins=multiscale_bins,
            soft_argmax_temp=soft_argmax_temp,
            soft_argmax_ranges=soft_argmax_ranges,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.out_features, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(channels // 2, out_dim),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return CooperativeRangePool.forward(self, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._pool(x))


def _build_conv1d_backbone(
    in_channels: int,
    base_channels: int,
    num_layers: int,
) -> tuple[nn.Sequential, nn.ModuleList, int]:
    """stem + ResBlock×num_layers；返回 ``(stem, layers, final_ch)``。"""
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


def _mlp_xy_head(in_features: int, hidden: int, *, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, 2),
    )


def _mlp_class_head(
    in_features: int,
    hidden: int,
    num_classes: int,
    *,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, num_classes),
    )


class BidirectionalStationCrossAttention(nn.Module):
    """晚融合：双站池化特征的一层双向交叉注意力。

    本站为 query、对站为 key/value。单 key 时 softmax 平凡，故用 query–key
    点积经 sigmoid 得到门控，使融合依赖 query；残差 + LayerNorm。
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim 须 > 0，收到 {dim}")
        heads = int(num_heads)
        if heads < 1:
            raise ValueError(f"num_heads 须 >= 1，收到 {num_heads}")
        while heads > 1 and dim % heads != 0:
            heads //= 2
        self.dim = int(dim)
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _attend(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        # query/key_value: (B, D)
        batch = query.shape[0]
        q = self.q_proj(query).view(batch, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch, self.num_heads, self.head_dim)
        score = (q * k).sum(dim=-1) * self.scale
        gate = torch.sigmoid(score).unsqueeze(-1)
        mixed = (gate * v).reshape(batch, self.dim)
        return self.out_proj(mixed)

    def forward(
        self, f0: torch.Tensor, f1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0_out = self.norm(f0 + self.dropout(self._attend(f0, f1)))
        f1_out = self.norm(f1 + self.dropout(self._attend(f1, f0)))
        return f0_out, f1_out


def localize_xy_two_monostatic_ranges_torch(
    r0: torch.Tensor,
    r1: torch.Tensor,
    *,
    pos0_xy: torch.Tensor,
    pos1_xy: torch.Tensor,
    y_hint: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """可微双圆交会：两单基地斜距 → 目标 ``(B, 2)`` xy。

    解析交点公式与 ``isac.sensing.localization.localize_xy_two_monostatic_ranges``
    一致；无实交点时 ``h`` 置 0（落在根轴上），仍可反传。镜像歧义按 ``y_hint``
    （默认偏好 ``|y|`` 更小）硬选择一支。
    """
    r0_b = r0.reshape(-1)
    r1_b = r1.reshape(-1)
    if r0_b.shape != r1_b.shape:
        raise ValueError(
            f"r0/r1 批大小须一致，收到 {tuple(r0.shape)} 与 {tuple(r1.shape)}"
        )
    c0 = pos0_xy.reshape(2).to(dtype=r0_b.dtype, device=r0_b.device)
    c1 = pos1_xy.reshape(2).to(dtype=r0_b.dtype, device=r0_b.device)
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    d = torch.sqrt(dx * dx + dy * dy + eps)
    r0_sq = r0_b * r0_b
    r1_sq = r1_b * r1_b
    a = (r0_sq - r1_sq + d * d) / (2.0 * d)
    h_sq = r0_sq - a * a
    h = torch.sqrt(torch.clamp(h_sq, min=0.0) + eps)
    xm = c0[0] + a * dx / d
    ym = c0[1] + a * dy / d
    rx = -dy / d
    ry = dx / d
    p_plus = torch.stack([xm + h * rx, ym + h * ry], dim=-1)
    p_minus = torch.stack([xm - h * rx, ym - h * ry], dim=-1)
    hint = 0.0 if y_hint is None else float(y_hint)
    score_plus = torch.abs(p_plus[:, 1] - hint)
    score_minus = torch.abs(p_minus[:, 1] - hint)
    use_plus = score_plus <= score_minus
    return torch.where(use_plus.unsqueeze(-1), p_plus, p_minus)


class CooperativeMonostaticCNN(nn.Module):
    """双站 cooperative monostatic ROI 距离谱 → 目标 (x, y) 回归 CNN。

    ``fusion_mode``
        ``early``：四通道从 stem 起早期混合（默认）
        ``late``：两站共享权重骨干 → 池化特征拼接 → xy 头
    ``aux_range``
        额外预测 ``(r0, r1)`` 单站几何距离（训练用辅助头；推理仍只取 xy）
    ``geom_residual``
        仅 ``late``：共享 range 头 → 可微双圆交会 ``xy_geom``，``xy_head`` 预测残差
        ``Δxy``，输出 ``xy = xy_geom + Δxy``
    ``geom_only``
        仅 ``late``：只学 ``(r0, r1)``，推理 ``xy = xy_geom``（无 Δxy 头）；
        与 ``geom_residual`` 互斥
    ``cross_attn``
        仅 ``late``：池化后、xy/残差头前插入一层双向交叉注意力；可与
        ``geom_residual`` 叠加（range 仍用池化后、交叉注意力前的局部分站特征）
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.3,
        pool_mode: str = "attention",
        multiscale_bins: int = 8,
        soft_argmax_temp: float = 1.0,
        fusion_mode: str = "late",
        aux_range: bool = False,
        geom_residual: bool = False,
        geom_only: bool = False,
        stopgrad_geom: bool = False,
        cross_attn: bool = False,
        cross_attn_heads: int = 4,
        dev0_xy: tuple[float, float] = (0.0, -2.0),
        dev1_xy: tuple[float, float] = (-2.0, 0.0),
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_mode = str(pool_mode)
        if self.pool_mode not in COOPERATIVE_POOL_MODES:
            raise ValueError(
                f"pool_mode 须为 {COOPERATIVE_POOL_MODES}，收到 {self.pool_mode!r}"
            )
        self.multiscale_bins = int(multiscale_bins)
        self.soft_argmax_temp = float(soft_argmax_temp)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in COOPERATIVE_FUSION_MODES:
            raise ValueError(
                f"fusion_mode 须为 {COOPERATIVE_FUSION_MODES}，收到 {self.fusion_mode!r}"
            )
        self.aux_range = bool(aux_range)
        self.geom_residual = bool(geom_residual)
        self.geom_only = bool(geom_only)
        self.stopgrad_geom = bool(stopgrad_geom)
        self.cross_attn = bool(cross_attn)
        self.cross_attn_heads = int(cross_attn_heads)
        if self.geom_residual and self.geom_only:
            raise ValueError("geom_residual 与 geom_only 互斥")
        if self.geom_residual and self.fusion_mode != "late":
            raise ValueError("geom_residual 仅支持 fusion_mode='late'")
        if self.geom_only and self.fusion_mode != "late":
            raise ValueError("geom_only 仅支持 fusion_mode='late'")
        if self.cross_attn and self.fusion_mode != "late":
            raise ValueError("cross_attn 仅支持 fusion_mode='late'")
        if self.cross_attn and self.geom_only:
            raise ValueError("geom_only 不使用 xy 头，不支持 cross_attn")
        if self.stopgrad_geom and not (self.geom_residual or self.geom_only):
            raise ValueError("stopgrad_geom 须与 geom_residual 或 geom_only 同时使用")

        if self.fusion_mode == "late":
            if in_channels % 2 != 0:
                raise ValueError(
                    f"late fusion 要求 in_channels 为偶数（每站各半），收到 {in_channels}"
                )
            self.station_channels = in_channels // 2
        else:
            self.station_channels = in_channels

        # early + gap + 无 aux：保留旧 Sequential head，兼容已有 checkpoint 键名
        self._legacy_gap_head = (
            self.fusion_mode == "early"
            and self.pool_mode == "gap"
            and not self.aux_range
        )

        stem_in = (
            self.station_channels if self.fusion_mode == "late" else in_channels
        )
        self.stem, self.layers, final_ch = _build_conv1d_backbone(
            stem_in, base_channels, num_layers
        )
        self.final_ch = final_ch

        soft_ranges = 1 if self.fusion_mode == "late" else 2
        # 各路径只注册一套子模块，避免 state_dict 重复键
        self.pool: CooperativeRangePool | None = None
        self.xy_head: nn.Module | None = None
        self.head: nn.Module | None = None
        self.station_cross_attn: BidirectionalStationCrossAttention | None = None

        if self._legacy_gap_head:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(final_ch, final_ch // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(final_ch // 2, 2),
            )
            pool_dim = final_ch
        elif self.fusion_mode == "early":
            # 早融合非 gap：CooperativeRangePoolHead 作为 self.head
            head = CooperativeRangePoolHead(
                final_ch,
                dropout=dropout,
                pool_mode=self.pool_mode,
                multiscale_bins=self.multiscale_bins,
                soft_argmax_temp=self.soft_argmax_temp,
                soft_argmax_ranges=soft_ranges,
            )
            self.head = head
            pool_dim = head.out_features
        else:
            self.pool = CooperativeRangePool(
                final_ch,
                pool_mode=self.pool_mode,
                multiscale_bins=self.multiscale_bins,
                soft_argmax_temp=self.soft_argmax_temp,
                soft_argmax_ranges=soft_ranges,
            )
            pool_dim = self.pool.out_features
            if not self.geom_only:
                self.xy_head = _mlp_xy_head(
                    pool_dim * 2, final_ch // 2, dropout=dropout
                )
            if self.cross_attn:
                self.station_cross_attn = BidirectionalStationCrossAttention(
                    pool_dim,
                    num_heads=self.cross_attn_heads,
                    dropout=dropout,
                )

        self.range_head: nn.Module | None = None
        if self.aux_range or self.geom_residual or self.geom_only:
            # 晚融合：共享单站 head；早融合：联合特征一次出 (r0, r1)
            out_r = 1 if self.fusion_mode == "late" else 2
            self.range_head = nn.Sequential(
                nn.Linear(pool_dim, max(pool_dim // 2, 8)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(max(pool_dim // 2, 8), out_r),
            )

        if self.geom_residual or self.geom_only:
            self.register_buffer(
                "dev0_xy",
                torch.tensor(
                    [float(dev0_xy[0]), float(dev0_xy[1])], dtype=torch.float32
                ),
            )
            self.register_buffer(
                "dev1_xy",
                torch.tensor(
                    [float(dev1_xy[0]), float(dev1_xy[1])], dtype=torch.float32
                ),
            )
        else:
            self.dev0_xy = None
            self.dev1_xy = None

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        for layer in self.layers:
            x = layer(x)
        return x

    def _pool_features(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            return self.pool(encoded)
        if isinstance(self.head, CooperativeRangePoolHead):
            return self.head._pool(encoded)
        return F.adaptive_avg_pool1d(encoded, 1).flatten(1)

    def _features_from_input(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        if dual_profiles.is_complex():
            features = dual_range_profiles_to_features(dual_profiles)
        else:
            features = dual_profiles
            if features.ndim == 2:
                features = features.unsqueeze(0)
            elif features.ndim != 3:
                raise ValueError(
                    "CooperativeMonostaticCNN float 输入须为 (C, L) 或 (B, C, L)，"
                    f"收到 {tuple(features.shape)}"
                )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {features.shape[1]} 与 in_channels={self.in_channels} 不一致"
            )
        return features

    def _station_ranges_from_pooled(
        self, f0: torch.Tensor, f1: torch.Tensor
    ) -> torch.Tensor:
        """共享 ``range_head`` → ``(B, 2)``；geom 路径 softplus 保证正距。"""
        assert self.range_head is not None
        r0 = self.range_head(f0)
        r1 = self.range_head(f1)
        ranges = torch.cat([r0, r1], dim=1)
        if self.geom_residual or self.geom_only:
            ranges = F.softplus(ranges) + 1e-3
        return ranges

    def forward_with_aux(
        self, dual_profiles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回 ``(pred_xy, pred_ranges)``；``pred_ranges`` 为 ``(B, 2)`` 或 ``None``。"""
        features = self._features_from_input(dual_profiles)

        if self.fusion_mode == "late":
            c = self.station_channels
            h0 = self._encode(features[:, :c])
            h1 = self._encode(features[:, c:])
            f0 = self._pool_features(h0)
            f1 = self._pool_features(h1)
            # range 用局部分站特征；xy/残差头可用交叉注意力后的融合特征
            pred_r: torch.Tensor | None = None
            if self.range_head is not None:
                pred_r = self._station_ranges_from_pooled(f0, f1)
            if self.geom_only:
                assert pred_r is not None
                assert self.dev0_xy is not None and self.dev1_xy is not None
                xy_geom = localize_xy_two_monostatic_ranges_torch(
                    pred_r[:, 0],
                    pred_r[:, 1],
                    pos0_xy=self.dev0_xy,
                    pos1_xy=self.dev1_xy,
                )
                if self.stopgrad_geom:
                    xy_geom = xy_geom.detach()
                return xy_geom, pred_r
            if self.station_cross_attn is not None:
                f0, f1 = self.station_cross_attn(f0, f1)
            assert self.xy_head is not None
            delta_or_xy = self.xy_head(torch.cat([f0, f1], dim=1))
            if self.geom_residual:
                assert pred_r is not None
                assert self.dev0_xy is not None and self.dev1_xy is not None
                xy_geom = localize_xy_two_monostatic_ranges_torch(
                    pred_r[:, 0],
                    pred_r[:, 1],
                    pos0_xy=self.dev0_xy,
                    pos1_xy=self.dev1_xy,
                )
                if self.stopgrad_geom:
                    xy_geom = xy_geom.detach()
                return xy_geom + delta_or_xy, pred_r
            return delta_or_xy, pred_r

        # early fusion
        encoded = self._encode(features)
        if self._legacy_gap_head:
            assert self.head is not None
            return self.head(encoded), None
        assert isinstance(self.head, CooperativeRangePoolHead)
        if not self.aux_range:
            return self.head(encoded), None
        pooled = self.head._pool(encoded)
        xy = self.head.mlp(pooled)
        assert self.range_head is not None
        return xy, self.range_head(pooled)

    def forward(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        """ROI 双设备距离谱 → 目标位置 ``(B, 2)`` = ``[x_m, y_m]``。

        输入 ``(2, L)`` 或 ``(B, 2, L)`` complex，或已提取的 ``(B, C, L)`` float 特征。
        """
        xy, _ = self.forward_with_aux(dual_profiles)
        return xy


class CooperativeMonostaticTransformer(nn.Module):
    """双站 late-fusion 极轻量 range-bin Transformer → 目标 (x, y)。

    每站：``Conv1d`` 投影 → 可学习位置编码 → ``TransformerEncoder`` →
    attention 池化；双站特征拼接后 MLP 回归 xy。两站共享编码器权重。
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        max_range_bins: int = 512,
    ) -> None:
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"late fusion Transformer 要求 in_channels 为偶数，收到 {in_channels}"
            )
        if d_model < 1:
            raise ValueError(f"d_model 须 >= 1，收到 {d_model}")
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")
        if nhead < 1 or d_model % nhead != 0:
            raise ValueError(
                f"nhead 须整除 d_model，收到 d_model={d_model} nhead={nhead}"
            )
        if max_range_bins < 1:
            raise ValueError(f"max_range_bins 须 >= 1，收到 {max_range_bins}")

        self.in_channels = int(in_channels)
        self.station_channels = in_channels // 2
        self.d_model = int(d_model)
        # checkpoint / 训练脚本兼容字段
        self.base_channels = self.d_model
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.max_range_bins = int(max_range_bins)

        self.input_proj = nn.Conv1d(
            self.station_channels, self.d_model, kernel_size=1, bias=True
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_range_bins, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.pool = RangeAttentionPool1d(self.d_model)
        self.xy_head = _mlp_xy_head(
            self.d_model * 2, max(self.d_model // 2, 8), dropout=self.dropout
        )

    def _features_from_input(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        if dual_profiles.is_complex():
            features = dual_range_profiles_to_features(dual_profiles)
        else:
            features = dual_profiles
            if features.ndim == 2:
                features = features.unsqueeze(0)
            elif features.ndim != 3:
                raise ValueError(
                    "CooperativeMonostaticTransformer float 输入须为 (C, L) 或 "
                    f"(B, C, L)，收到 {tuple(features.shape)}"
                )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {features.shape[1]} 与 in_channels={self.in_channels} 不一致"
            )
        return features

    def _encode_station(self, station_features: torch.Tensor) -> torch.Tensor:
        """``(B, C_st, L)`` → ``(B, d_model)``。"""
        h = self.input_proj(station_features)
        tokens = h.transpose(1, 2)
        length = tokens.shape[1]
        if length > self.max_range_bins:
            raise ValueError(
                f"range bins={length} 超过 max_range_bins={self.max_range_bins}"
            )
        tokens = tokens + self.pos_embed[:, :length, :]
        tokens = self.encoder(tokens)
        return self.pool(tokens.transpose(1, 2))

    def forward(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        features = self._features_from_input(dual_profiles)
        c = self.station_channels
        f0 = self._encode_station(features[:, :c])
        f1 = self._encode_station(features[:, c:])
        return self.xy_head(torch.cat([f0, f1], dim=1))


class CooperativeMonostatic2DCNN(nn.Module):
    """双站 cooperative monostatic range×slow-time ROI → 目标 (x, y) 回归 2D CNN。"""

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.dropout = dropout
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        layers: list[ConvResidualBlock] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvResidualBlock(c, c, stride=1))
            else:
                in_ch = c * (2 ** (i - 1))
                out_ch = c * (2**i)
                layers.append(ConvResidualBlock(in_ch, out_ch, stride=(1, 2)))
        self.layers = nn.ModuleList(layers)

        final_ch = c * (2 ** (num_layers - 1))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_ch, final_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(final_ch // 2, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(C, T, L)`` 或 ``(B, C, T, L)`` float → ``(B, 2)`` xy (m)。"""
        x = features
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(
                "CooperativeMonostatic2DCNN 输入须为 (C, T, L) 或 (B, C, T, L)，"
                f"收到 {tuple(features.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {x.shape[1]} 与 in_channels={self.in_channels} 不一致"
            )
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _build_cooperative_localization_model_from_ckpt(
    ckpt: dict[str, Any],
) -> nn.Module:
    model_type = str(ckpt.get("model_type", "1d"))
    in_channels = int(ckpt["in_channels"])
    base_channels = int(ckpt["base_channels"])
    num_layers = int(ckpt.get("num_layers", 3 if model_type == "1d" else 2))
    dropout = float(ckpt["dropout"])
    if model_type == "transformer":
        d_model = int(ckpt.get("d_model", base_channels))
        return CooperativeMonostaticTransformer(
            in_channels=in_channels,
            d_model=d_model,
            nhead=int(ckpt.get("nhead", 4)),
            num_layers=num_layers,
            dim_feedforward=int(ckpt.get("dim_feedforward", max(d_model * 2, 128))),
            dropout=dropout,
            max_range_bins=int(ckpt.get("max_range_bins", 512)),
        )
    if model_type == "2d":
        return CooperativeMonostatic2DCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    geom_residual = bool(ckpt.get("geom_residual", False))
    geom_only = bool(ckpt.get("geom_only", False))
    cross_attn = bool(ckpt.get("cross_attn", False))
    dev0 = ckpt.get("dev0_xy", (0.0, -2.0))
    dev1 = ckpt.get("dev1_xy", (-2.0, 0.0))
    return CooperativeMonostaticCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        dropout=dropout,
        pool_mode=str(ckpt.get("pool_mode", "gap")),
        multiscale_bins=int(ckpt.get("multiscale_bins", 8)),
        soft_argmax_temp=float(ckpt.get("soft_argmax_temp", 1.0)),
        fusion_mode=str(ckpt.get("fusion_mode", "early")),
        aux_range=bool(ckpt.get("aux_range", False)),
        geom_residual=geom_residual,
        geom_only=geom_only,
        stopgrad_geom=bool(ckpt.get("stopgrad_geom", False)),
        cross_attn=cross_attn,
        cross_attn_heads=int(ckpt.get("cross_attn_heads", 4)),
        dev0_xy=(float(dev0[0]), float(dev0[1])),
        dev1_xy=(float(dev1[0]), float(dev1[1])),
    )


def load_cooperative_monostatic_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> CooperativeMonostaticCNN | CooperativeMonostatic2DCNN | CooperativeMonostaticTransformer:
    """从 checkpoint 加载 Cooperative Monostatic 定位模型（1D CNN / 2D CNN / Transformer）。"""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in _CORE_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")

    model = _build_cooperative_localization_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


class CooperativeMonostaticRegionCNN(nn.Module):
    """双站 late-fusion ROI 距离谱 → 16 子区域分类 logits。

    骨干与 ``CooperativeMonostaticCNN`` late 路径一致；输出头为 ``num_classes`` 维。
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        pool_mode: str = "attention",
        multiscale_bins: int = 8,
        soft_argmax_temp: float = 1.0,
        num_classes: int = 16,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")
        if in_channels % 2 != 0:
            raise ValueError(
                f"late fusion 要求 in_channels 为偶数，收到 {in_channels}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes 须 >= 2，收到 {num_classes}")
        if pool_mode not in COOPERATIVE_POOL_MODES:
            raise ValueError(
                f"pool_mode 须为 {COOPERATIVE_POOL_MODES}，收到 {pool_mode!r}"
            )

        self.in_channels = int(in_channels)
        self.station_channels = in_channels // 2
        self.base_channels = int(base_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.pool_mode = str(pool_mode)
        self.multiscale_bins = int(multiscale_bins)
        self.soft_argmax_temp = float(soft_argmax_temp)
        self.num_classes = int(num_classes)
        self.fusion_mode = "late"

        self.stem, self.layers, final_ch = _build_conv1d_backbone(
            self.station_channels, base_channels, num_layers
        )
        self.final_ch = final_ch
        self.pool = CooperativeRangePool(
            final_ch,
            pool_mode=self.pool_mode,
            multiscale_bins=self.multiscale_bins,
            soft_argmax_temp=self.soft_argmax_temp,
            soft_argmax_ranges=1,
        )
        pool_dim = self.pool.out_features
        self.class_head = _mlp_class_head(
            pool_dim * 2,
            final_ch // 2,
            self.num_classes,
            dropout=dropout,
        )

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        for layer in self.layers:
            x = layer(x)
        return x

    def _features_from_input(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        if dual_profiles.is_complex():
            features = dual_range_profiles_to_features(dual_profiles)
        else:
            features = dual_profiles
            if features.ndim == 2:
                features = features.unsqueeze(0)
            elif features.ndim != 3:
                raise ValueError(
                    "CooperativeMonostaticRegionCNN float 输入须为 (C, L) 或 "
                    f"(B, C, L)，收到 {tuple(features.shape)}"
                )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {features.shape[1]} 与 in_channels={self.in_channels} 不一致"
            )
        return features

    def forward(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        """``(B, C, L)`` → ``(B, num_classes)`` logits。"""
        features = self._features_from_input(dual_profiles)
        c = self.station_channels
        f0 = self.pool(self._encode(features[:, :c]))
        f1 = self.pool(self._encode(features[:, c:]))
        return self.class_head(torch.cat([f0, f1], dim=1))


def subregion_id_to_one_hot(
    subregion_id: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """``(B,)`` 子区域 id → ``(B, num_classes)`` one-hot（oracle / ablation）。"""
    if subregion_id.ndim != 1:
        subregion_id = subregion_id.reshape(-1)
    return F.one_hot(subregion_id.long(), num_classes=int(num_classes)).to(
        dtype=torch.float32
    )


class CooperativeMonostaticFineCNN(nn.Module):
    """双站 late-fusion + Region 概率条件化 → 全局 ``(x, y)``。

    主输入与单阶段 CNN 相同（``dual_profiles``）；串联条件为上游 Region 的
    ``softmax`` 概率 ``(B, num_classes)``。正式推理须经
    :class:`CooperativeMonostaticTwoStageCNN`，勿绕过 Region。
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        pool_mode: str = "attention",
        multiscale_bins: int = 8,
        soft_argmax_temp: float = 1.0,
        num_classes: int = 16,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers 须 >= 1，收到 {num_layers}")
        if in_channels % 2 != 0:
            raise ValueError(
                f"late fusion 要求 in_channels 为偶数，收到 {in_channels}"
            )
        if num_classes < 2:
            raise ValueError(f"num_classes 须 >= 2，收到 {num_classes}")
        if pool_mode not in COOPERATIVE_POOL_MODES:
            raise ValueError(
                f"pool_mode 须为 {COOPERATIVE_POOL_MODES}，收到 {pool_mode!r}"
            )

        self.in_channels = int(in_channels)
        self.station_channels = in_channels // 2
        self.base_channels = int(base_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.pool_mode = str(pool_mode)
        self.multiscale_bins = int(multiscale_bins)
        self.soft_argmax_temp = float(soft_argmax_temp)
        self.num_classes = int(num_classes)
        self.fusion_mode = "late"

        self.stem, self.layers, final_ch = _build_conv1d_backbone(
            self.station_channels, base_channels, num_layers
        )
        self.final_ch = final_ch
        self.pool = CooperativeRangePool(
            final_ch,
            pool_mode=self.pool_mode,
            multiscale_bins=self.multiscale_bins,
            soft_argmax_temp=self.soft_argmax_temp,
            soft_argmax_ranges=1,
        )
        pool_dim = self.pool.out_features
        self.xy_head = _mlp_xy_head(
            pool_dim * 2 + self.num_classes,
            final_ch // 2,
            dropout=dropout,
        )

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(features)
        for layer in self.layers:
            x = layer(x)
        return x

    def _features_from_input(self, dual_profiles: torch.Tensor) -> torch.Tensor:
        if dual_profiles.is_complex():
            features = dual_range_profiles_to_features(dual_profiles)
        else:
            features = dual_profiles
            if features.ndim == 2:
                features = features.unsqueeze(0)
            elif features.ndim != 3:
                raise ValueError(
                    "CooperativeMonostaticFineCNN float 输入须为 (C, L) 或 "
                    f"(B, C, L)，收到 {tuple(features.shape)}"
                )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"特征通道数 {features.shape[1]} 与 in_channels={self.in_channels} 不一致"
            )
        return features

    def forward(
        self,
        dual_profiles: torch.Tensor,
        region_probs: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, C, L)`` + ``(B, num_classes)`` region_probs → ``(B, 2)`` 全局坐标。"""
        features = self._features_from_input(dual_profiles)
        c = self.station_channels
        f0 = self.pool(self._encode(features[:, :c]))
        f1 = self.pool(self._encode(features[:, c:]))
        if region_probs.ndim != 2:
            raise ValueError(
                f"region_probs 须为 (B, num_classes)，收到 {tuple(region_probs.shape)}"
            )
        if region_probs.shape[0] != f0.shape[0]:
            raise ValueError(
                "region_probs batch 须与特征对齐，"
                f"收到 {tuple(region_probs.shape)} vs {f0.shape[0]}"
            )
        if region_probs.shape[1] != self.num_classes:
            raise ValueError(
                f"region_probs 类数 {region_probs.shape[1]} 与 "
                f"num_classes={self.num_classes} 不一致"
            )
        probs = region_probs.to(dtype=f0.dtype)
        return self.xy_head(torch.cat([f0, f1, probs], dim=1))


class CooperativeMonostaticTwoStageCNN(nn.Module):
    """RegionCNN → softmax → FineCNN 串联定位。

    正式推理 / 联合训练 / 评估统一入口。``region_probs_override`` 仅用于
    oracle ablation（例如真值 one-hot），非部署路径。
    """

    def __init__(
        self,
        region_model: CooperativeMonostaticRegionCNN,
        fine_model: CooperativeMonostaticFineCNN,
    ) -> None:
        super().__init__()
        if int(region_model.num_classes) != int(fine_model.num_classes):
            raise ValueError(
                "Region 与 Fine 的 num_classes 不一致："
                f"{region_model.num_classes} vs {fine_model.num_classes}"
            )
        self.region_model = region_model
        self.fine_model = fine_model
        self.num_classes = int(fine_model.num_classes)

    def forward(
        self,
        dual_profiles: torch.Tensor,
        *,
        region_probs_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(pred_xy, region_logits)``；``pred_xy`` 为全局坐标 ``(B, 2)``。"""
        logits = self.region_model(dual_profiles)
        if region_probs_override is not None:
            probs = region_probs_override
        else:
            probs = F.softmax(logits, dim=-1)
        xy = self.fine_model(dual_profiles, probs)
        return xy, logits


_REGION_FINE_CKPT_KEYS = (
    "model_state_dict",
    "in_channels",
    "base_channels",
    "num_layers",
    "dropout",
    "model_kind",
)


def load_cooperative_monostatic_region_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> CooperativeMonostaticRegionCNN:
    """从 checkpoint 加载子区域分类 CNN。"""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in _REGION_FINE_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")
    if str(ckpt.get("model_kind")) != "region":
        raise ValueError(
            f"期望 model_kind='region'，收到 {ckpt.get('model_kind')!r}"
        )
    model = CooperativeMonostaticRegionCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        pool_mode=str(ckpt.get("pool_mode", "attention")),
        multiscale_bins=int(ckpt.get("multiscale_bins", 8)),
        soft_argmax_temp=float(ckpt.get("soft_argmax_temp", 1.0)),
        num_classes=int(ckpt.get("num_classes", 16)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_cooperative_monostatic_fine_cnn_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> CooperativeMonostaticFineCNN:
    """从 checkpoint 加载概率条件化 Fine CNN（全局 xy）。

    旧版含 ``region_embed`` 的 checkpoint 不兼容，会抛出明确错误。
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in _REGION_FINE_CKPT_KEYS if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint 缺少必填字段: {', '.join(missing)}")
    if str(ckpt.get("model_kind")) != "fine":
        raise ValueError(
            f"期望 model_kind='fine'，收到 {ckpt.get('model_kind')!r}"
        )
    state = ckpt["model_state_dict"]
    if any(str(k).startswith("region_embed") for k in state):
        raise ValueError(
            "checkpoint 含旧版 region_embed 权重，与当前概率条件化 FineCNN "
            "不兼容，请使用新架构重新训练"
        )
    model = CooperativeMonostaticFineCNN(
        in_channels=int(ckpt["in_channels"]),
        base_channels=int(ckpt["base_channels"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
        pool_mode=str(ckpt.get("pool_mode", "attention")),
        multiscale_bins=int(ckpt.get("multiscale_bins", 8)),
        soft_argmax_temp=float(ckpt.get("soft_argmax_temp", 1.0)),
        num_classes=int(ckpt.get("num_classes", 16)),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_cooperative_monostatic_two_stage_checkpoints(
    region_path: str | Path,
    fine_path: str | Path,
    device: torch.device | str,
) -> CooperativeMonostaticTwoStageCNN:
    """加载 Region + Fine checkpoint 并组装串联模块。"""
    region = load_cooperative_monostatic_region_cnn_checkpoint(region_path, device)
    fine = load_cooperative_monostatic_fine_cnn_checkpoint(fine_path, device)
    model = CooperativeMonostaticTwoStageCNN(region, fine)
    model.to(device)
    model.eval()
    return model
