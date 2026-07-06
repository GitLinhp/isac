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
        方向向量，支持：
        - 单条：形状 ``(3,)``
        - 批量：形状 ``(N, 3)``

    返回:
    -------
    - np.ndarray
        单条输入返回形状 ``(3,)``，批量输入返回 ``(N, 3)``，单位为弧度。
        零向量对应 ``[0, 0, 0]``。
    """
    arr = np.asarray(convert(direction, "numpy"), dtype=np.float64)
    squeeze = False
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(
                f"direction 必须为形状 (3,) 或 (N, 3)，当前一维长度为 {arr.size}"
            )
        arr = arr.reshape(1, 3)
        squeeze = True
    elif arr.ndim == 2:
        if arr.shape[-1] != 3:
            raise ValueError(
                f"direction 必须为形状 (3,) 或 (N, 3)，当前末维为 {arr.shape[-1]}"
            )
    else:
        raise ValueError(f"direction 必须为形状 (3,) 或 (N, 3)，当前维度为 {arr.ndim}")

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    r = np.linalg.norm(arr, axis=1)
    result = np.zeros((arr.shape[0], 3), dtype=np.float64)
    valid = r >= 1e-12
    if np.any(valid):
        rv = r[valid]
        xv, yv, zv = x[valid], y[valid], z[valid]
        theta = np.arccos(np.clip(zv / rv, -1.0, 1.0))
        phi = np.arctan2(yv, xv)
        result[valid, 0] = phi
        result[valid, 1] = (np.pi / 2.0) - theta
        result[valid, 2] = 0.0

    return result[0] if squeeze else result


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
