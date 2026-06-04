"""ZC（Zadoff-Chu）序列源，接口对齐 ``sionna.phy.mapping.BinarySource``。"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch
from sionna.phy.block import Block
from sionna.phy.config import Precision


class ZCSource(Block):
    """生成给定形状的 Zadoff-Chu 序列（最后一维为序列长度 ``N``，其余维广播复制）。

    :param root_index: 根索引 :math:`u`，须满足 ``gcd(u, N) == 1``。
    :param normalize: 若为 ``True``，对长度为 ``N`` 的一维基序列做单位能量归一化（除以 :math:`\\sqrt{N}`）。
    :param precision: 内部计算与输出的精度；``None`` 时使用全局 Config。
    :param device: 计算设备；``None`` 时使用全局 Config。

    :input shape: 与 ``BinarySource`` 相同，``list`` / ``tuple`` / ``torch.Size`` 的整型尺寸；
        最后一维为 ZC 长度 ``N``。

    :output zc: ``shape``，复数 dtype（``self.cdtype``）。
    """

    def __init__(
        self,
        root_index: int = 1,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._root_index = int(root_index)
        self._normalize = bool(normalize)

    def call(
        self,
        inputs: Union[List[int], Tuple[int, ...], torch.Size],
    ) -> torch.Tensor:
        shape = list(inputs)
        if len(shape) < 1:
            raise ValueError("shape must have at least one dimension")
        n = int(shape[-1])
        if n < 1:
            raise ValueError(f"last dimension (ZC length N) must be >= 1, got {n}")

        u = self._root_index
        if math.gcd(u, n) != 1:
            raise ValueError(
                f"ZC root_index u={u} must be coprime to sequence length N={n} (require gcd(u, N)=1)"
            )

        idx = torch.arange(n, device=self.device, dtype=self.dtype)
        phase = -math.pi * u * idx * (idx + 1.0) / n
        z = torch.complex(torch.cos(phase), torch.sin(phase)).to(self.cdtype)

        if self._normalize:
            z = z / (n**0.5)

        if len(shape) == 1:
            return z

        view_shape = [1] * (len(shape) - 1) + [n]
        return z.reshape(view_shape).expand(tuple(shape)).contiguous()
