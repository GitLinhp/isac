"""
杂项工具函数模块

提供各种实用工具函数，包括：
- 随机种子设置
- 字符串和类型转换
- 组合生成
- 单位转换
- 文件操作
- 进度条创建
"""

import torch
import numpy as np
import sionna

from .type_converter import convert


# ============================================================================
# 随机种子设置
# ============================================================================
def set_random_seed(seed: int) -> None:
    """统一设置所有随机数生成器的种子

    同时设置 NumPy、TensorFlow 和 PyTorch 的随机种子，确保实验的可重复性。

    参数:
    -------
    - seed : int
        随机种子值，应为非负整数
    """
    # 设置 NumPy 随机种子
    np.random.seed(seed)

    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)

    # 如果 CUDA 可用，设置所有 GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性（如果支持）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    sionna.phy.config.seed = seed


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
