"""
数值计算工具模块

提供数值计算相关的工具函数，包括：
- 数学运算
- 角度转换
- 分位数计算
"""

import numpy as np
import torch


def next_pow2(n: int) -> int:
    """
    计算大于等于n的最小2的幂次

    该函数用于找到大于等于输入整数n的最小2的幂次方数。
    常用于FFT计算、内存对齐等需要2的幂次方长度的场景。

    参数:
    ----------
        n : int
            输入整数，必须大于0

    返回:
    ----------
        int
            大于等于n的最小2的幂次方数
            - 当n <= 1时，返回1
            - 当n > 1时，返回2^k，其中k是满足2^k >= n的最小整数
    """
    return 1 << (int(np.log2(n - 1)) + 1) if n > 1 else 1


def approx_quantile(
    tensor: torch.Tensor, q: float, sample_ratio: float = 0.02
) -> torch.Tensor:
    """
    随机采样计算近似分位数（用于大数据集加速）

    该函数通过随机采样来快速计算大张量的近似分位数，
    避免对全部数据进行排序，显著提高计算效率。
    适用于数据量很大且对精度要求不是特别严格的场景。

    参数:
    ----------
        tensor : torch.Tensor
            输入张量，任意形状
        q : float
            分位数，取值范围[0, 1]
            - 0.5表示中位数
            - 0.25表示第一四分位数
            - 0.75表示第三四分位数
        sample_ratio : float, 可选
            采样比例，取值范围(0, 1]，默认0.02
            - 值越小计算越快但精度越低
            - 值越大精度越高但计算越慢

    返回:
    ----------
        torch.Tensor
            近似分位数值，标量张量

    示例:
    ----------
        >>> tensor = torch.randn(10000)
        >>> approx_quantile(tensor, 0.5)
        tensor(0.0123)
        >>> approx_quantile(tensor, 0.95, sample_ratio=0.05)
        tensor(1.6456)
    """
    num_samples = int(tensor.numel() * sample_ratio)
    sampled = tensor.view(-1)[torch.randperm(tensor.numel())[:num_samples]]
    return torch.quantile(sampled, q)


# =============================================================================
# 角度转换工具函数
# =============================================================================
def degree_to_radian(degree: float | np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    将角度转换为弧度

    该函数将角度值转换为弧度值，支持标量、numpy数组和PyTorch张量。
    转换公式：弧度 = 角度 × π / 180

    参数:
    ----------
        degree : float | np.ndarray | torch.Tensor
            输入角度值，支持：
            - 标量：单个角度值
            - numpy数组：多个角度值
            - PyTorch张量：任意形状的角度张量

    返回:
    ----------
        torch.Tensor
            转换后的弧度值，数据类型为torch.float32
            输出形状与输入相同

    示例:
    ----------
        >>> degree_to_radian(90)
        tensor(1.5708)
        >>> degree_to_radian([0, 90, 180, 270])
        tensor([0.0000, 1.5708, 3.1416, 4.7124])
    """
    if not isinstance(degree, torch.Tensor):
        degree = torch.tensor(degree, dtype=torch.float32)
    else:
        degree = degree.to(torch.float32)
    return degree * np.pi / 180


def radian_to_degree(radian: float | np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    将弧度转换为角度

    该函数将弧度值转换为角度值，支持标量、numpy数组和PyTorch张量。
    转换公式：角度 = 弧度 × 180 / π

    参数:
    ----------
        radian : float | np.ndarray | torch.Tensor
            输入弧度值，支持：
            - 标量：单个弧度值
            - numpy数组：多个弧度值
            - PyTorch张量：任意形状的弧度张量

    返回:
    ----------
        torch.Tensor
            转换后的角度值，数据类型为torch.float32
            输出形状与输入相同

    示例:
    ----------
        >>> radian_to_degree(3.14159)
        tensor(180.0000)
        >>> radian_to_degree([0, 1.5708, 3.1416, 4.7124])
        tensor([0.0000, 90.0000, 180.0000, 270.0000])
    """
    if not isinstance(radian, torch.Tensor):
        radian = torch.tensor(radian, dtype=torch.float32)
    else:
        radian = radian.to(torch.float32)
    return radian * 180 / np.pi


#  线性尺度转换为 dB 尺度工具函数
def linear_to_db(
    linear_tensor: torch.Tensor | np.ndarray | float,
    is_power: bool = False,
) -> torch.Tensor | np.ndarray | float:
    """将线性尺度转换为 dB 尺度。

    参数:
    - linear_tensor : torch.Tensor | np.ndarray | float
        输入张量或标量，支持 ``torch.Tensor``、``numpy.ndarray`` 或浮点数。
    - is_power : bool
        为 ``True`` 时按照功率(20log10)，否则按照幅度(10log10)。

    返回:
        与输入类型匹配的 dB 值。
    """

    factor = 20.0 if is_power else 10.0
    min_value = 1e-20

    if isinstance(linear_tensor, torch.Tensor):
        safe_tensor = torch.clamp(linear_tensor, min=min_value)
        return factor * torch.log10(safe_tensor)
    if isinstance(linear_tensor, np.ndarray):
        safe_tensor = np.clip(linear_tensor, min_value, None)
        return factor * np.log10(safe_tensor)

    safe_value = max(float(linear_tensor), min_value)
    return float(factor * np.log10(safe_value))


def db_to_linear(
    db_tensor: torch.Tensor | np.ndarray | float,
    is_power: bool = False,
) -> torch.Tensor | np.ndarray | float:
    """将 dB 尺度转换为线性尺度。

    参数:
    -------
    - db_tensor : torch.Tensor | np.ndarray | float
        输入张量或标量，支持 ``torch.Tensor``、``numpy.ndarray`` 或浮点数。
    - is_power : bool
        为 ``True`` 时按照功率(20log10)，否则按照幅度(10log10)。

    返回:
    -------
    - 与输入类型匹配的线性值。
    """
    factor = 10.0 if is_power else 20.0

    min_value = 1e-20
    if isinstance(db_tensor, torch.Tensor):
        safe_tensor = torch.clamp(db_tensor, min=min_value)
        return torch.pow(10.0, safe_tensor / factor)
    if isinstance(db_tensor, np.ndarray):
        safe_tensor = np.clip(db_tensor, min_value, None)
        return np.power(10.0, safe_tensor / factor)
    safe_value = max(float(db_tensor), min_value)
    return np.power(10.0, safe_value / factor)


# 量化解量化
def quantize(
    x: torch.Tensor, num_bits: int, value_range: tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    将连续值做均匀量化，并输出扁平化后的比特矩阵（bit-plane flatten）。

    量化范围默认固定为 ``[-1, 1]``，输入会先裁剪到该范围内，保证
    只依赖 (x, num_bits) 就能与反量化互逆使用。

    支持任意前导维度：对最后一维的每个标量独立量化，仅最后一维由 ``N`` 变为 ``N * num_bits``。

    参数:
    ----------
    x:
        连续值张量，形状 ``(..., N)``，至少一维；例如 ``[N]``、``[B, N]``、``[B, C, H, W]``（此时 ``N=W``）。
    num_bits:
        量化位数。
    value_range:
        量化映射区间 (min_val, max_val)。

    返回:
    ----------
    torch.Tensor:
        比特张量，形状 ``(..., N * num_bits)``，
        dtype 为 ``torch.uint8``，比特顺序为从高位到低位（MSB -> LSB）。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("quantize: x 必须是 torch.Tensor")
    if x.dim() < 1:
        raise ValueError(f"quantize: x 至少需要一维 (... , N)，当前为 {x.shape}")
    if not isinstance(num_bits, int) or num_bits <= 0:
        raise ValueError(f"quantize: num_bits 必须为正整数，当前为 {num_bits}")

    min_val, max_val = float(value_range[0]), float(value_range[1])
    if max_val <= min_val:
        raise ValueError(f"quantize: value_range 无效: {value_range}")

    x = x.to(torch.float32)
    x = torch.clamp(x, min=min_val, max=max_val)

    levels = 1 << num_bits  # 2^num_bits
    # 线性映射到整数码本: [0, levels-1]
    scale = (levels - 1) / (max_val - min_val)
    codes = torch.round((x - min_val) * scale).to(torch.int64)  # 与 x 同形状
    codes = torch.clamp(codes, 0, levels - 1)

    # MSB -> LSB
    # 生成位移序列: [num_bits-1, ..., 0]，用于 MSB -> LSB
    shifts = torch.arange(num_bits - 1, -1, -1, device=codes.device, dtype=torch.int64)
    bits = ((codes.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    # 扁平化：将最后两维 [N, num_bits] 或 [B, N, num_bits] 展平成 [N*num_bits] / [B, N*num_bits]
    bits_flat = bits.reshape(*x.shape[:-1], x.shape[-1] * num_bits)

    return bits_flat


def dequantize(
    bits: torch.Tensor,
    num_bits: int,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    将 `quantize()` 的扁平化输出比特张量反量化回连续值。

    与 ``quantize`` 对称：形状 ``(..., N * num_bits)`` 反量化为 ``(..., N)``。

    参数:
    ----------
    bits:
        比特张量，形状 ``(..., N * num_bits)``，至少一维。
        支持 dtype 为 ``uint8/int/float``（会按 0/1 处理）。
    num_bits:
        量化位数（与 quantize 保持一致）。
    value_range:
        量化映射区间 (min_val, max_val)。

    返回:
    ----------
    torch.Tensor:
        反量化后的连续值，形状 ``(..., N)``，dtype 为 ``torch.float32``。
    """
    if not isinstance(bits, torch.Tensor):
        raise TypeError("dequantize: bits 必须是 torch.Tensor")
    if bits.dim() < 1:
        raise ValueError(
            f"dequantize: bits 至少需要一维 (... , N * num_bits)，当前为 {bits.shape}"
        )
    if not isinstance(num_bits, int) or num_bits <= 0:
        raise ValueError(f"dequantize: num_bits 必须为正整数，当前为 {num_bits}")

    total_bits = int(bits.size(-1))
    if total_bits % num_bits != 0:
        raise ValueError(
            f"dequantize: bits.size(-1)={total_bits} 不能被 num_bits={num_bits} 整除，无法恢复 N"
        )
    n = total_bits // num_bits

    min_val, max_val = float(value_range[0]), float(value_range[1])
    if max_val <= min_val:
        raise ValueError(f"dequantize: value_range 无效: {value_range}")

    # reshape back to [..., N, num_bits] for MSB->LSB decoding
    bits_reshaped = bits.reshape(*bits.shape[:-1], n, num_bits)
    bits_int = bits_reshaped.to(torch.int64) & 1  # 强制为 0/1
    levels = 1 << num_bits

    # 生成位移序列: [num_bits-1, ..., 0]，与 quantize 保持一致
    shifts = torch.arange(
        num_bits - 1, -1, -1, device=bits_int.device, dtype=torch.int64
    )
    codes = torch.sum(bits_int * (1 << shifts), dim=-1)  # [..., N]
    codes = torch.clamp(codes, 0, levels - 1).to(torch.float32)

    # codes -> [min_val, max_val]
    x = min_val + codes * (max_val - min_val) / (levels - 1)

    return x
