"""
类型转换模块

该模块提供了以下功能：
- 类型转换：将输入值转换为指定的目标类型
- 字符串转换为布尔值：将字符串转换为布尔值
- 转换为元组：将输入值转换为元组
- 字节与比特序列的相互转换：将字节序列转换为比特序列，或将比特序列转换为字节序列
- 图像与比特的相互转换：将图像转换为比特序列，或将比特序列转换为图像
"""

from typing import Union
import torch
import numpy as np
import io
from pathlib import Path
from PIL import Image
from torchvision import transforms
import builtins
import argparse

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
    "bytes": "bytes",
    "bytearray": "bytearray",
    "bytesarray": "bytearray",
    "memoryview": "memoryview",
}


# ============================================================================
# 类型转换
# ============================================================================
def convert(
    value: Union[
        float,
        int,
        bool,
        np.ndarray,
        torch.Tensor,
        builtins.bytes,
        builtins.bytearray,
        builtins.memoryview,
    ],
    target_type: str,
    dtype: Union[torch.dtype, None] = None,
    device: torch.device = None,
) -> Union[float, int, np.ndarray, torch.Tensor]:
    """将输入值转换为指定的目标类型

    参数:
    ----------
        value : float | int | bool | np.ndarray | torch.Tensor | tf.Tensor | bytes | bytearray | memoryview
            输入值，支持以下类型：
            - Python原生类型：float, int, bool
            - NumPy数组：numpy.ndarray
            - PyTorch张量：torch.Tensor
            - Python字节类型：bytes, bytearray, memoryview（按uint8字节序列处理）
        target_type : str
            目标类型，支持以下值：
            - "numpy" / "np": NumPy数组
            - "torch" / "tensor" / "pytorch": PyTorch张量
            - Python原生类型（需显式指定）:
              - "int" / "float" / "bool"
              - "list" / "tuple"
              - "bytes" / "bytearray" / "memoryview"
        dtype : torch.dtype, 可选
            目标数据类型
            - 目标类型为 torch 时：用于指定输出 torch.Tensor 的 dtype
        device : torch.device, 可选
            目标设备（仅当目标类型为 torch 时有效）
            默认自动选择cuda或cpu（仅当目标类型为torch时）

    返回:
    ----------
        Union[float, int, list, np.ndarray, torch.Tensor]
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

    elif target_type in {"bytes", "bytearray", "memoryview"}:
        raw = _to_bytes(value)
        if target_type == "bytes":
            return builtins.bytes(raw)
        if target_type == "bytearray":
            return builtins.bytearray(raw)
        return builtins.memoryview(raw)

    elif target_type == "numpy":
        return _to_numpy(value)
    elif target_type == "torch":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _to_torch(value, dtype=dtype, device=device)
    else:
        raise ValueError(
            f"不支持的目标类型: {target_type}。"
            f"支持的类型: numpy, torch, int, float, bool, list, tuple, bytes, bytearray, memoryview"
        )


def _to_bytes(
    value: Union[
        np.ndarray,
        torch.Tensor,
        list,
        builtins.bytes,
        builtins.bytearray,
        builtins.memoryview,
        int,
        float,
        bool,
    ],
) -> builtins.bytes:
    """将输入转换为原始字节序列（bytes）.

    - bytes/bytearray/memoryview: 直接返回 bytes(value)
    - torch/np/tf: 期望为 1D 的 uint8 序列（或可无损转换到 uint8）
    - list: 视为数值列表，转换到 uint8 后再转 bytes
    - 标量: 转为单字节（uint8）后再转 bytes
    """
    if isinstance(value, (builtins.bytes, builtins.bytearray, builtins.memoryview)):
        return builtins.bytes(value)

    if isinstance(value, torch.Tensor):
        t = value.detach().to("cpu")
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t.flatten().numpy().tobytes()

    if isinstance(value, np.ndarray):
        arr = value
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr).flatten().tobytes()

    if isinstance(value, list):
        return np.asarray(value, dtype=np.uint8).tobytes()

    if isinstance(value, (int, float, bool)):
        return np.asarray([value], dtype=np.uint8).tobytes()

        raise TypeError(f"不支持的输入类型: {type(value).__name__}")


def _to_numpy(
    value: Union[
        float,
        int,
        bool,
        np.ndarray,
        torch.Tensor,
        list,
        builtins.bytes,
        builtins.bytearray,
        builtins.memoryview,
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

    elif isinstance(value, (builtins.bytes, builtins.bytearray, builtins.memoryview)):
        # bytes-like -> uint8 1D numpy array (zero-copy when possible)
        return np.frombuffer(value, dtype=np.uint8)

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
        builtins.bytes,
        builtins.bytearray,
        builtins.memoryview,
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

    elif isinstance(value, (builtins.bytes, builtins.bytearray, builtins.memoryview)):
        # bytes-like -> uint8 tensor
        # np.frombuffer 得到的数组通常不可写，torch.from_numpy 会给出告警；这里 copy 一次避免该告警
        np_arr = np.frombuffer(value, dtype=np.uint8).copy()
        tensor = torch.from_numpy(np_arr)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if dtype is not None:
            tensor = tensor.to(dtype)

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


# ============================================================================
# 字符串转换为布尔值
# ============================================================================
def str_to_bool(value: str) -> bool:
    """
    将字符串转换为布尔值

    该函数支持多种常见的布尔值字符串表示形式，不区分大小写。
    通常用于命令行参数解析或配置文件读取时的布尔值转换。

    参数:
    ----------
        value : str
            输入字符串，支持以下格式（不区分大小写）：
            - True: "yes", "true", "t", "y", "1"
            - False: "no", "false", "f", "n", "0"

    返回:
    ----------
        bool
            转换后的布尔值：
            - True: 当输入为 "yes", "true", "t", "y", "1" 时
            - False: 当输入为 "no", "false", "f", "n", "0" 时

    异常:
    ----------
        argparse.ArgumentTypeError
            当输入字符串不是有效的布尔值表示时抛出

    示例:
    ----------
        >>> str_to_bool("yes")
        True
        >>> str_to_bool("NO")
        False
        >>> str_to_bool("1")
        True
        >>> str_to_bool("false")
        False
        >>> str_to_bool("invalid")
        argparse.ArgumentTypeError: Boolean value expected.
    """
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# ============================================================================
# 转换为元组
# ============================================================================
def to_tuple(
    value: tuple | list | int | np.ndarray | torch.Tensor,
) -> tuple:
    """将输入转换为元组"""
    # 将输入转换为元组
    if isinstance(value, tuple):
        return value
    elif isinstance(value, np.ndarray):
        return tuple(value.tolist())
    elif isinstance(value, torch.Tensor):
        return tuple(value.tolist())
    elif isinstance(value, int):
        return (value,)
    else:
        raise ValueError(
            f"不支持的类型: {type(value)}。支持的类型: tuple, list, int, np.ndarray, torch.Tensor"
        )


# ============================================================================
# 字节与比特序列的相互转换
# ============================================================================
def bytes_to_bits(
    bytes_data: (
        list | np.ndarray | torch.Tensor | builtins.bytes | builtins.bytearray | builtins.memoryview
    ),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_type: str = "torch",
    batch: bool = False,
) -> Union[torch.Tensor, list]:
    """
    字节转比特序列（小端序）

    该函数将字节数组转换为比特序列张量，使用小端序（LSB优先）。
    每个字节被展开为8个比特，通过并行位移和掩码操作实现高效转换。
    支持批量处理多个字节数组。

    参数:
    ----------
        bytes_data : list | np.ndarray | torch.Tensor | bytes | bytearray | memoryview
            输入的字节数组
            - list: Python列表（单个字节数组或批量字节数组）
            - np.ndarray: NumPy数组
            - torch.Tensor: PyTorch张量
            - bytes/bytearray/memoryview: Python字节类型
        device : torch.device, 可选
            计算设备，默认自动选择cuda或cpu
            - cuda: 使用GPU加速计算
            - cpu: 使用CPU计算
        output_type : str, 可选
            输出类型，默认"torch"
            - "torch": PyTorch张量
            - "numpy": NumPy数组
            - 其他支持的类型
        batch : bool, 可选
            是否批量处理，默认False
            - False: 单个输入，返回单个输出
            - True: 输入为列表时，批量处理每个元素，返回列表

    返回:
    ----------
        Union[torch.Tensor, list]
            - 单个模式：比特序列张量
                - 形状：[N*8]，其中N是输入字节数
                - 数据类型：torch.uint8
                - 每个元素为0或1
            - 批量模式：比特序列张量列表
                - 每个元素形状：[Ni*8]，其中Ni是第i个输入的字节数

    示例:
    ----------
        >>> # 单个输入
        >>> bytes_data = b"\\x01\\x02"
        >>> bits = bytes_to_bits(bytes_data)
        >>> # 返回: tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        >>> # 批量输入
        >>> bytes_list = [b"\\x01", b"\\x02", b"\\x03"]
        >>> bits_list = bytes_to_bits(bytes_list, batch=True)
        >>> # 返回: [tensor([1, 0, ...]), tensor([0, 1, ...]), tensor([1, 1, ...])]
    """
    # 检查是否为批量处理模式
    if batch and isinstance(bytes_data, list) and len(bytes_data) > 0:
        # 检查列表元素是否为字节类型（批量模式）
        first_elem = bytes_data[0]
        if isinstance(
            first_elem,
            (
                builtins.bytes,
                builtins.bytearray,
                builtins.memoryview,
                np.ndarray,
                torch.Tensor,
            ),
        ):
            # 批量处理：对每个元素递归调用
            return [
                bytes_to_bits(item, device=device, output_type=output_type, batch=False)
                for item in bytes_data
            ]

    # 单个处理模式
    # 输入类型统一转换为 torch.uint8 张量（并放到指定 device）
    bytes_data = convert(bytes_data, target_type="torch", dtype=torch.uint8, device=device)

    # 确保是2D张量 (N, 1) 以便广播
    bytes_data = bytes_data.view(-1, 1)

    # 创建位移掩码 (小端序: 0-7位)
    shift = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=device)

    # 并行执行位移和掩码操作
    bits = (bytes_data >> shift) & 1

    # 展平为1D比特序列
    bits = bits.view(-1).to(torch.uint8)

    # 输出类型转换
    output_type_norm = TYPE_ALIASES.get(output_type.lower(), output_type.lower())

    return convert(bits, target_type=output_type_norm)


def bits_to_bytes(
    bits: Union[torch.Tensor, np.ndarray, list],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    output_type: str = "torch",
    batch: bool = False,
) -> Union[torch.Tensor, list]:
    """
    比特序列转字节（小端序）

    该函数将比特序列转换回字节数据，使用小端序（LSB优先）。
    每8个比特组合成1个字节，自动处理末尾不足8位的情况（用0填充）。
    支持批量处理多个比特序列。

    参数:
    ----------
        bits : Union[torch.Tensor, np.ndarray, list]
            输入的比特序列
            - torch.Tensor: PyTorch张量
            - np.ndarray: NumPy数组
            - list: Python列表（单个比特序列或批量比特序列）
            - 形状：任意维度，会被展平为1D
            - 数据类型：torch.uint8
            - 每个元素为0或1
        device : torch.device, 可选
            计算设备，默认自动选择cuda或cpu
            - cuda: 使用GPU加速计算
            - cpu: 使用CPU计算
        output_type : str, 可选
            输出类型，默认"torch"
            - "torch": PyTorch张量
            - "numpy": NumPy数组
            - "bytes": Python bytes对象
            - 其他支持的类型
        batch : bool, 可选
            是否批量处理，默认False
            - False: 单个输入，返回单个输出
            - True: 输入为列表时，批量处理每个元素，返回列表

    返回:
    ----------
        Union[torch.Tensor, list]
            - 单个模式：字节数据张量
                - 形状：[N//8]，其中N是输入比特数
                - 数据类型：torch.uint8
            - 批量模式：字节数据列表
                - 每个元素形状：[Ni//8]，其中Ni是第i个输入的比特数

    示例:
    ----------
        >>> # 单个输入
        >>> bits = tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        >>> bytes_data = bits_to_bytes(bits)
        >>> # 返回: tensor([1, 2])

        >>> # 批量输入
        >>> bits_list = [tensor([1, 0, ...]), tensor([0, 1, ...]), tensor([1, 1, ...])]
        >>> bytes_list = bits_to_bytes(bits_list, batch=True)
        >>> # 返回: [tensor([1]), tensor([2]), tensor([3])]
    """
    # 检查是否为批量处理模式
    if batch and isinstance(bits, list) and len(bits) > 0:
        # 检查列表元素是否为张量/数组类型（批量模式）
        first_elem = bits[0]
        if isinstance(first_elem, (torch.Tensor, np.ndarray, list)):
            # 批量处理：对每个元素递归调用
            return [
                bits_to_bytes(item, device=device, output_type=output_type, batch=False)
                for item in bits
            ]

    # 单个处理模式
    # 输入类型统一转换为 torch.uint8 张量（并放到指定 device）
    bits = convert(bits, target_type="torch", dtype=torch.uint8, device=device)

    # 确保输入为一维张量
    if bits.dim() > 1:
        bits = bits.view(-1)

    # 计算字节数 (自动处理末尾不足8位的情况)
    bit_count = bits.shape[0]
    byte_count = (bit_count + 7) // 8  # 向上取整

    if byte_count == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)

    # 处理填充
    pad_length = byte_count * 8 - bit_count
    if pad_length > 0:
        # 用0填充不足的位
        padded = torch.cat([bits, torch.zeros(pad_length, dtype=torch.uint8, device=device)])
        reshaped = padded.view(byte_count, 8)
    else:
        reshaped = bits.view(byte_count, 8)

    # 创建权重张量 (小端序: 2^0, 2^1, ..., 2^7)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)

    # 并行计算所有字节
    bytes_result = (reshaped * powers).sum(dim=1)

    bytes_result = bytes_result.to(torch.uint8).to(device)

    # 输出类型转换
    output_type_norm = TYPE_ALIASES.get(output_type.lower(), output_type.lower())

    return convert(bytes_result, target_type=output_type_norm)


# ============================================================================
# 图像与比特的相互转换
# ============================================================================
def image_to_bits(
    image_path: Path | str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """
    图像文件转比特序列

    该函数读取图像文件（JPEG/PNG等），将文件字节转换为比特序列张量。
    保留完整的文件格式信息，支持可逆转换。

    参数:
    ----------
        image_path : Path | str
            图像文件路径
            - Path: 直接使用Path对象
            - str: 转换为Path对象
        device : torch.device, 可选
            计算设备，默认自动选择cuda或cpu
            - cuda: 使用GPU加速计算
            - cpu: 使用CPU计算

    返回:
    ----------
        torch.Tensor
            图像比特序列张量
            - 形状：[N]，其中N是文件字节数*8
            - 数据类型：torch.uint8
            - 每个元素为0或1

    异常:
    ----------
        FileNotFoundError
            当图像文件不存在时抛出
    """
    # 检查并转换路径
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取图像文件为字节
    with open(image_path, "rb") as f:
        # 正确转换bytes为torch tensor
        bytes_data = bytearray(f.read())
        image_bytes = torch.frombuffer(bytes_data, dtype=torch.uint8).to(device)
        # 转换为比特张量
        image_bits = bytes_to_bits(image_bytes, device)

    return image_bits


def bits_to_image(
    image_bits: torch.Tensor,
    image_path: Path | str | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor | None:
    """
    比特序列转图像

    该函数将比特序列转换回图像字节数据，可选地保存为图像文件或返回张量。
    这是 image2bits 的逆操作，完全可逆。

    参数:
    ----------
        image_bits : torch.Tensor
            输入的比特序列张量
            - 形状：[N]，其中N是比特数
            - 数据类型：torch.uint8
            - 每个元素为0或1
        image_path : Path | str, 可选
            可选的保存路径，默认None
            - 提供路径：保存文件并返回None
            - 不提供：返回torch.Tensor图像张量
        device : torch.device, 可选
            计算设备，默认自动选择cuda或cpu
            - cuda: 使用GPU加速计算
            - cpu: 使用CPU计算

    返回:
    ----------
        torch.Tensor | None
            - 如果未指定image_path：返回torch.Tensor图像张量 (C, H, W)
            - 如果指定了image_path：返回None（文件已保存）
    """
    # 1. 将比特转换为字节
    image_bytes = bits_to_bytes(image_bits, device)

    byte_data = image_bytes.cpu().numpy().tobytes()
    image_buffer = io.BytesIO(byte_data)

    pil_image = Image.open(image_buffer).convert("RGB")
    pil_image_tensor = transforms.ToTensor()(pil_image).to(device)

    if image_path is not None:
        # 如果指定了保存路径，则保存到文件
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        with open(image_path, "wb") as f:
            f.write(byte_data)
        return None

    return pil_image_tensor
