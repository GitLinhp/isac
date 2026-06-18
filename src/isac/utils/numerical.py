"""
数值计算工具模块

提供数值计算相关的工具函数，包括：
- 数学运算
- 角度转换
- 分位数计算
"""

import numpy as np
import torch

from .type_converter import convert


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


def cartesian_direction_to_yaw_pitch_roll(
    direction: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """将笛卡尔方向向量转换为 ``[yaw, pitch, roll]``（弧度）。

    参数:
    -------
    - direction : torch.Tensor | np.ndarray
        方向向量，形状应可展平为长度 3。

    返回:
    -------
    - np.ndarray
        长度为 3 的 ``[yaw, pitch, roll]``，单位为弧度。
    """
    vec = np.asarray(convert(direction, "numpy"), dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"direction 必须为长度 3 的向量，当前长度为 {vec.size}")

    x, y, z = vec
    r = float(np.linalg.norm(vec))
    if r < 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    theta = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
    phi = float(np.arctan2(y, x))

    yaw = phi
    pitch = (np.pi / 2.0) - theta
    roll = 0.0

    return np.array([yaw, pitch, roll], dtype=np.float64)


#  线性尺度转换为 dB 尺度工具函数
def linear_to_db(
    linear_tensor: torch.Tensor | np.ndarray | float,
    is_power: bool = False,
    min_value: float = 1e-20,
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
    min_value: float = 1e-20,
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

    if isinstance(db_tensor, torch.Tensor):
        safe_tensor = torch.clamp(db_tensor, min=min_value)
        return torch.pow(10.0, safe_tensor / factor)
    if isinstance(db_tensor, np.ndarray):
        safe_tensor = np.clip(db_tensor, min_value, None)
        return np.power(10.0, safe_tensor / factor)
    safe_value = max(float(db_tensor), min_value)

    return np.power(10.0, safe_value / factor)
