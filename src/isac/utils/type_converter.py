"""
类型转换模块

该模块提供了以下功能：
- 类型转换：将输入值转换为指定的目标类型
- 字符串转换为布尔值：将字符串转换为布尔值
- 转换为元组：将输入值转换为元组
- 图像与比特的相互转换：将图像转换为比特序列，或将比特序列转换为图像
"""

from typing import Literal, Union, overload

import numpy as np
import torch

_ConvertValue = Union[float, int, bool, np.ndarray, torch.Tensor, list]
_ConvertReturn = Union[float, int, list, tuple, np.ndarray, torch.Tensor]

# 支持的转换类型映射
TYPE_ALIASES = {
    "numpy": "numpy",
    "np": "numpy",
    "torch": "torch",
    "tensor": "torch",
    "pytorch": "torch",
    "float": "float",
    "int": "int",
    "list": "list",
    "tuple": "tuple",
}


# ============================================================================
# 类型转换
# ============================================================================
@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["torch", "tensor", "pytorch"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> torch.Tensor: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["numpy", "np"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> np.ndarray: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["float"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> float: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["int"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> int: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["bool"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> bool: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["list"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> list: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: Literal["tuple"],
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> tuple: ...


@overload
def convert(
    value: _ConvertValue,
    target_type: str,
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> _ConvertReturn: ...


def convert(
    value: _ConvertValue,
    target_type: str,
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> _ConvertReturn:
    """将输入值转换为指定的目标类型

    参数:
    ----------
        value : float | int | bool | np.ndarray | torch.Tensor | list
            输入值，支持以下类型：
            - Python原生类型：float, int, bool
            - NumPy数组：numpy.ndarray
            - PyTorch张量：torch.Tensor
            - list: Python列表（数值列表）
        target_type : str
            目标类型，支持以下值：
            - "numpy" / "np": NumPy数组
            - "torch" / "tensor" / "pytorch": PyTorch张量
            - Python原生类型（需显式指定）:
              - "int" / "float" / "bool"
              - "list" / "tuple"
        dtype : torch.dtype, 可选
            目标数据类型
            - 目标类型为 torch 时：用于指定输出 torch.Tensor 的 dtype
        device : torch.device, 可选
            目标设备（仅当目标类型为 torch 时有效）
            默认自动选择cuda或cpu（仅当目标类型为torch时）

    返回:
    ----------
        Union[float, int, list, tuple, np.ndarray, torch.Tensor]
            转换后的值，类型由target_type决定

    异常:
    ----------
        ValueError
            当target_type不是支持的类型时抛出
        TypeError
            当输入类型不支持时抛出
    """
    # 使用统一的类型别名映射
    target_type = TYPE_ALIASES.get(target_type.lower(), target_type.lower())

    # 根据目标类型进行转换
    if target_type in {"int", "float", "bool"}:
        arr = _to_numpy(value)
        if arr.size != 1:
            raise TypeError(f"无法将多元素数据转换为 {target_type}")
        item = arr.reshape(-1)[0].item()
        if target_type == "int":
            return int(item)
        if target_type == "float":
            return float(item)
        return bool(item)

    elif target_type in {"list", "tuple"}:
        arr = _to_numpy(value)
        lst = [arr.item()] if (arr.ndim == 0 or arr.size == 1) else arr.tolist()
        return lst if target_type == "list" else tuple(lst)

    elif target_type == "numpy" or target_type == "np":
        return _to_numpy(value)
    elif target_type == "torch" or target_type == "tensor" or target_type == "pytorch":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _to_torch(value, dtype=dtype, device=device)
    else:
        raise ValueError(
            f"不支持的目标类型: {target_type}。"
            f"支持的类型: numpy, torch, int, float, bool, list, tuple"
        )


def _to_numpy(
    value: Union[
        float,
        int,
        bool,
        np.ndarray,
        torch.Tensor,
        list,
    ],
) -> np.ndarray:
    """转换为NumPy数组

    参数:
    ----------
        value : 任意支持的类型
            支持的类型包括：
            - np.ndarray: NumPy数组
            - torch.Tensor: PyTorch张量
            - float, int, bool: Python标量
            - list: Python列表（数值列表）

    返回:
    ----------
        np.ndarray
            转换后的NumPy数组
    """
    if isinstance(value, np.ndarray):
        return value

    elif isinstance(value, torch.Tensor):
        if value.is_cuda:
            value = value.cpu()
        return value.detach().numpy()

    elif isinstance(value, (float, int, bool)):
        return np.array(value)

    elif isinstance(value, list):
        return np.array(value)

    else:
        raise TypeError(f"不支持的输入类型: {type(value).__name__}")


def _to_torch(
    value: Union[
        float,
        int,
        bool,
        np.ndarray,
        torch.Tensor,
        list,
    ],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """转换为PyTorch张量

    参数:
    ----------
        value : 任意支持的类型
            支持的类型包括：
            - torch.Tensor: PyTorch张量
            - np.ndarray: NumPy数组
            - float, int, bool: Python标量
            - list: Python列表（数值列表）
        dtype : torch.dtype, 可选
            目标数据类型
        device : torch.device, 可选
            目标设备

    返回:
    ----------
        torch.Tensor
            转换后的PyTorch张量
    """
    if isinstance(value, torch.Tensor):
        if dtype is not None or (device is not None and value.device != device):
            return value.to(device=device, dtype=dtype)
        return value

    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    elif isinstance(value, (float, int, bool)):
        tensor = torch.tensor(value, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    elif isinstance(value, list):
        tensor = torch.tensor(value, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    else:
        raise TypeError(f"不支持的输入类型: {type(value).__name__}")
