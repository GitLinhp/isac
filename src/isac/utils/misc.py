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
from pathlib import Path
import sionna
from tqdm import tqdm

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


def write_txt(arr_in: torch.Tensor | np.ndarray, path: Path | str) -> None:
    """将数组写入文本文件。

    参数:
    -------
    - arr_in : torch.Tensor | np.ndarray
        输入数组，支持 ``torch.Tensor`` 或 ``numpy.ndarray``。
    - path : Path | str
        文件路径，支持 ``Path`` 或 ``str``。
    """
    if isinstance(path, str):
        path = Path(path)
    if path.suffix != ".txt":
        path = path.with_suffix(".txt")

    array = convert(arr_in, "numpy")
    if array.ndim == 1:
        array = array.reshape(1, -1)

    with open(path, "w", encoding="utf-8") as fp:
        rows, cols = array.shape
        for i in range(rows):
            fp.write("  ".join(str(array[i, j]) for j in range(cols)))
            if i < rows - 1:
                fp.write("\n")


# ============================================================================
# 进度条工具
# ============================================================================
def create_progress_bar(
    total: int, desc: str, unit: str = "image", ncols: int = 100
) -> tqdm:
    """
    创建标准化的进度条

    参数:
    ----------
    - total : int
        总任务数
    - desc : str
        进度条描述
    - unit : str, 可选
        单位，默认"image"
    - ncols : int, 可选
        进度条宽度，默认100

    返回:
    ----------
    - tqdm
        配置好的进度条对象
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        ncols=ncols,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


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
