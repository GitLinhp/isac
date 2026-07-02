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


def csv_float2_scalar(value: object) -> str:
    """将标量格式化为保留两位小数的 CSV 字段字符串。"""
    return f"{convert(value, 'float'):.2f}"


def csv_vec3(vec: np.ndarray | object) -> str:
    """三维向量 → CSV 单元格 ``[x, y, z]``。"""
    row = np.asarray(vec, dtype=np.float64).reshape(-1)
    parts = ", ".join(csv_float2_scalar(row[i]) for i in range(3))
    return f"[{parts}]"
