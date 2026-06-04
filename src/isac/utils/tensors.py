"""
张量工具函数模块

提供张量维度操作和归一化等功能。
"""

from typing import Tuple
import torch
import torch.nn.functional as F

import numpy as np
from typing import Optional, Union

from .type_converter import convert


def is_bits_sequence(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """
    判断两个张量是否为比特序列

    该函数检查两个张量是否都是1维uint8类型的张量，
    用于判断输入是否为比特序列数据。

    参数:
    ----------
    - tensor1 : torch.Tensor
            第一个张量
    - tensor2 : torch.Tensor
            第二个张量

    返回:
    ----------
    - bool
        - True: 两个张量都是1维uint8张量（比特序列）
        - False: 其他情况

    示例:
    ----------
        >>> bits1 = torch.tensor([0, 1, 0, 1], dtype=torch.uint8)
        >>> bits2 = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
        >>> is_bits_sequence(bits1, bits2)
        True
        >>> img = torch.randn(3, 224, 224)
        >>> is_bits_sequence(bits1, img)
        False
    """
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        return False

    return (
        tensor1.dim() == 1
        and tensor2.dim() == 1
        and tensor1.dtype == torch.uint8
        and tensor2.dtype == torch.uint8
    )


def serial_to_parallel(
    serial_tensor: Union[torch.Tensor, np.ndarray],
    batch_size: int,
    segment_length: int,
    pad_mode: str = "random_bits",
    output_type: str = "torch",
    dtype: Optional[Union[torch.dtype]] = torch.float32,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    将一维张量重塑为并行批处理格式

    该函数将一维张量重塑为三维张量，用于并行批处理。
    如果原始数据长度不是批次大小的整数倍，会自动填充。
    常用于深度学习中的批处理数据准备。
    支持自动类型转换（NumPy、PyTorch）。

    参数:
    ----------
    - serial_tensor : torch.Tensor | np.ndarray
        输入张量，支持PyTorch或NumPy张量
        如果不是一维，会自动展平
    - batch_size : int
        每批次的样本数量，必须大于0
    - segment_length : int
        每个样本的长度，必须大于0
    - pad_mode : str, 可选
        填充模式，默认"random_bits"
        支持的模式：
        - "random_bits": 随机比特填充
        - "constant": 常量填充
        - "random_uniform": 均匀分布随机填充
        - "random_normal": 正态分布随机填充
    - output_type : str, 可选
        输出数据类型，默认"torch"
        - "torch": 返回PyTorch张量
    - dtype : torch.dtype, 可选
        输出数据类型，默认None（保持输入类型或使用默认类型）

    返回:
    ----------
    - tuple[torch.Tensor, int, torch.Tensor | None]
        包含三个元素的元组：
        - 重塑后的三维张量，形状为[num_batches, batch_size, segment_length]
        - 填充比特数
        - 填充mask，形状为[num_batches, batch_size, segment_length]
        mask中True表示原始数据，False表示填充数据
    """
    serial_tensor = convert(serial_tensor, "torch")

    # 将多维张量展平为一维张量
    if len(serial_tensor.shape) != 1:
        serial_tensor = torch.reshape(serial_tensor, [-1])
    else:
        serial_tensor = serial_tensor

    # 计算需要的填充长度
    total_length = serial_tensor.size(0)
    batch_length = batch_size * segment_length
    num_batches = int(np.ceil(total_length / batch_length))
    padded_length = num_batches * batch_length

    # 计算填充比特数
    padding_count = padded_length - total_length

    # 创建填充mask（在填充之前）
    mask_1d = torch.zeros(padded_length, dtype=torch.bool)
    mask_1d[:total_length] = True  # 原始数据位置为True
    mask_3d = torch.reshape(mask_1d, [num_batches, batch_size, segment_length])

    # 填充张量
    padded_tensor = pad_to_length(serial_tensor, padded_length, pad_mode=pad_mode)
    parallel_tensor = torch.reshape(
        padded_tensor, [num_batches, batch_size, segment_length]
    )

    # 根据参数组合决定返回值
    return convert(parallel_tensor, output_type, dtype=dtype), padding_count, mask_3d


def parallel_to_serial(
    parallel_tensor: Union[torch.Tensor, np.ndarray],
    serial_lens: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    从并行批处理格式恢复为原始一维张量

    该函数将三维并行批处理张量恢复为一维张量，
    根据每个样本的实际长度去除填充部分。
    与serial_to_parallel函数配合使用。
    支持自动类型转换（NumPy、PyTorch）。

    参数:
    ----------
    - parallel_tensor : torch.Tensor | np.ndarray
        并行处理的张量，形状为[num_batches, batch_size, segment_length]
        支持PyTorch或NumPy张量，会自动转换为PyTorch
    - serial_lens : int
        原始一维张量的长度
        - 长度应该等于batch_size * num_batches * segment_length
    - device : torch.device, 可选
        目标设备，如果提供，会将结果张量移到指定设备
        默认None，保持输入张量的设备

    返回:
    ----------
    - torch.Tensor
        恢复后的原始一维张量
        长度等于serial_lens
    """
    # 转换为PyTorch张量
    if not isinstance(parallel_tensor, torch.Tensor):
        parallel_tensor = convert(parallel_tensor, "torch", device=device)
    elif device is not None:
        # 如果已经是PyTorch张量但需要移动到指定设备
        parallel_tensor = parallel_tensor.to(device)

    serial_tensor = torch.reshape(parallel_tensor, [-1])[:serial_lens]

    return serial_tensor


def pad_to_length(
    tensor: torch.Tensor,
    target_length: int,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    填充张量到目标长度

    该函数将张量在指定维度填充到目标长度，支持多种填充模式。
    常用于数据预处理、批处理对齐等场景。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量，支持任意维度
    - target_length : int
        目标长度，必须大于等于当前张量长度
    - pad_mode : str, 可选
        填充模式，默认"constant"
        - "constant": 常量填充，使用pad_value填充
        - "random_bits": 随机比特填充，生成0或1
        - "random_uniform": 均匀分布随机填充[0,1)
        - "random_normal": 正态分布随机填充(均值0,方差1)
    - pad_value : float, 可选
        当pad_mode为"constant"时的填充值，默认0.0
    - dim : int, 可选
        填充维度，默认 -1（最后一维）

    返回:
    ----------
    - torch.Tensor
        填充后的张量，指定维度长度为 target_length
    """
    if tensor.ndim == 0:
        raise ValueError("pad_to_length 不支持标量张量，请至少提供 1 维张量。")

    # 规范化维度索引
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"dim={dim} 越界，张量维度为 {tensor.ndim}")

    current_length = tensor.size(dim)
    if current_length > target_length:
        raise ValueError(
            f"当前张量第 {dim} 维长度 {current_length} 已经大于目标长度 {target_length}，无需填充。"
        )

    pad_len = target_length - current_length
    if pad_len == 0:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    pad_shape = tuple(pad_shape)

    if pad_mode == "constant":
        # 常量填充（补零或其他固定值），沿指定维度拼接
        fill_value: bool | float
        if tensor.dtype == torch.bool:
            fill_value = bool(pad_value)
        else:
            fill_value = float(pad_value)
        pad_tensor = torch.full(
            pad_shape, fill_value=fill_value, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, pad_tensor], dim=dim)

    elif pad_mode == "random_bits":
        # 随机比特填充（0或1），沿指定维度拼接
        if tensor.dtype == torch.bool:
            # 对于布尔张量，生成随机布尔值
            random_pad = torch.randint(
                0, 2, pad_shape, dtype=torch.bool, device=tensor.device
            )
        else:
            # 对于数值张量，生成0或1
            random_pad = torch.randint(
                0, 2, pad_shape, dtype=tensor.dtype, device=tensor.device
            )
        return torch.cat([tensor, random_pad], dim=dim)

    elif pad_mode == "random_uniform":
        # 均匀分布随机填充（指定维度）
        if tensor.dtype == torch.bool:
            raise ValueError(
                "pad_mode='random_uniform' 不支持 bool 张量，请使用 random_bits。"
            )
        random_pad = torch.rand(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, random_pad], dim=dim)

    elif pad_mode == "random_normal":
        # 正态分布随机填充（指定维度）
        if tensor.dtype == torch.bool:
            raise ValueError(
                "pad_mode='random_normal' 不支持 bool 张量，请使用 random_bits。"
            )
        random_pad = torch.randn(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, random_pad], dim=dim)

    else:
        raise ValueError(
            f"不支持的填充模式: {pad_mode}。支持的模式: constant, random_bits, random_uniform, random_normal"
        )


def insert_dims(tensor: torch.Tensor, num_dims: int, axis: int = -1) -> torch.Tensor:
    """向张量添加多个长度为1的维度

    该操作是 PyTorch `unsqueeze` 函数的扩展。
    它在张量的 `axis` 维度处插入 `num_dims` 个长度为1的维度。
    维度索引遵循 Python 索引规则，即从零开始，负索引从末尾开始计数。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量

    - num_dims : int
        要添加的维度数量

    - axis : int, 可选
        在哪个维度索引处扩展张量的形状。给定一个 `D` 维张量，
        `axis` 必须在范围 `[-(D+1), D]` 内（包含边界）。

    返回:
    ----------
    - torch.Tensor
        与输入张量数据相同的张量，在 `axis` 指定的索引处插入了 `num_dims` 个额外维度
    """
    if num_dims < 0:
        raise ValueError("`num_dims` 必须为非负数。")

    rank = tensor.ndim
    if axis > rank or axis < -(rank + 1):
        raise ValueError(f"`axis` 超出范围 `[-(D+1), D]`，其中 D={rank}")

    # 处理负索引
    if axis < 0:
        axis = rank + axis + 1

    # 在指定轴处插入维度
    for _ in range(num_dims):
        tensor = tensor.unsqueeze(axis)

    return tensor


def expand_to_rank(
    tensor: torch.Tensor, target_rank: int, axis: int = -1
) -> torch.Tensor:
    """将张量扩展到目标秩

    该操作在指定轴处插入维度，使输出张量的秩达到 `target_rank`。
    维度索引遵循 Python 索引规则，即从零开始，负索引从末尾开始计数。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量

    - target_rank : int
        输出张量的目标秩。
        如果 `target_rank` 小于输入张量的秩，函数不执行任何操作。

    - axis : int, 可选
        在哪个维度索引处扩展张量的形状。给定一个 `D` 维张量，
        `axis` 必须在范围 `[-(D+1), D]` 内（包含边界）。

    返回:
    ----------
    - torch.Tensor
        与输入张量数据相同的张量，在 `axis` 指定的索引处插入了
        `target_rank` - rank(`tensor`) 个额外维度。
        如果 `target_rank` <= rank(`tensor`)，则返回原始张量。
    """
    current_rank = tensor.ndim
    num_dims = max(target_rank - current_rank, 0)

    if num_dims == 0:
        return tensor

    # 处理负索引
    if axis < 0:
        axis = current_rank + axis + 1

    # 在指定轴处插入维度
    for _ in range(num_dims):
        tensor = tensor.unsqueeze(axis)

    return tensor


def expand_to_dimension(
    tensor: torch.Tensor, target_dim: int
) -> Tuple[torch.Tensor, bool]:
    """
    将张量扩展到指定维度

    如果输入维度低于目标维度，则在前面添加维度；如果输入维度等于目标维度，则保持不变；
    如果输入维度超过目标维度，则抛出错误。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量
    - target_dim : int
        目标维度，必须是正整数

    返回:
    ----------
    - Tuple[torch.Tensor, bool]
        包含以下元素的元组：
        - 扩展后的张量（维度为 target_dim）
        - is_expanded: 是否为扩展的（原始维度低于目标维度）
    """
    if target_dim < 1:
        raise ValueError(f"目标维度必须为正整数，得到: {target_dim}")

    current_dim = tensor.ndim

    if current_dim > target_dim:
        raise ValueError(
            f"输入张量维度 ({current_dim}D) 超过目标维度 ({target_dim}D)。"
            f"请确保输入张量维度不超过 {target_dim}D"
        )

    if current_dim == target_dim:
        # 维度已匹配，无需扩展
        return tensor, False

    if current_dim < target_dim:
        # 需要扩展维度
        dims_to_add = target_dim - current_dim
        # 在前面添加维度：例如 1D -> 2D: [N] -> [1, N]
        expanded = tensor
        for _ in range(dims_to_add):
            expanded = expanded.unsqueeze(0)
        return expanded, True

    # 理论上不会到达这里
    raise RuntimeError(f"意外的维度情况: current={current_dim}, target={target_dim}")


def pad_last_dimension(
    tensor: torch.Tensor,
    divisor: int,
) -> Tuple[torch.Tensor, int]:
    """
    对最后一个维度进行补零填充，使其长度能被指定数值整除

    该函数用于确保张量的最后一个维度长度能被指定的除数整除。
    如果已经满足整除要求，则不进行填充；否则在最后一个维度的右侧补零。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量，任意维度
    - divisor : int
        除数，必须是正整数。最后一个维度的长度需要能被该值整除

    返回:
    ----------
    - Tuple[torch.Tensor, int]
        包含以下元素的元组：
        - 填充后的张量（最后一个维度长度能被 divisor 整除）
        - padding_size: 填充的零的个数（如果未填充则为0）

    异常:
    ----------
    - ValueError
        如果 divisor 不是正整数
    """
    if divisor <= 0:
        raise ValueError(f"除数必须是正整数，得到: {divisor}")

    total_length = tensor.shape[-1]
    remainder = total_length % divisor

    if remainder == 0:
        # 已经满足整除要求，无需填充
        return tensor, 0

    # 计算需要填充的零的个数
    padding_size = divisor - remainder

    # 在最后一个维度进行补零填充
    # 创建一个零张量用于拼接
    padding_shape = list(tensor.shape)
    padding_shape[-1] = padding_size
    padding_zeros = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)

    # 在最后一个维度拼接
    padded_tensor = torch.cat([tensor, padding_zeros], dim=-1)

    return padded_tensor, padding_size


def normalize_energy(
    tensor: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    对张量进行能量归一化

    该函数计算张量的平均功率，然后将张量归一化，使得归一化后的平均功率为1。
    对于复数张量，计算其幅度平方的平均值作为功率。

    参数:
    ----------
    - tensor : torch.Tensor
        输入张量，支持实数和复数张量，任意形状
    - epsilon : float, 可选
        数值稳定性参数，防止除零错误，默认为 1e-10

    返回:
    ----------
    - torch.Tensor
        归一化后的张量，保持原始形状和数据类型
    """
    # 计算平均功率
    if tensor.is_complex():
        # 对于复数张量，计算 |tensor|^2 的平均值
        mean_power = torch.mean(torch.abs(tensor) ** 2)
    else:
        # 对于实数张量，计算 tensor^2 的平均值
        mean_power = torch.mean(tensor**2)

    # 计算归一化因子（能量归一化：使平均功率为1）
    normalization_factor = torch.sqrt(mean_power + epsilon)

    # 执行归一化
    normalized_tensor = tensor / normalization_factor

    return normalized_tensor


def last_dim_real_to_complex(real_tensor: torch.Tensor) -> torch.Tensor:
    """
    将最后一维按 (实部, 虚部) 成对合并为复数张量

    仅在最后一维上解读：相邻两个浮点数为一对 ``(real, imag)``，合并为
    一个复数元素。使用 ``torch.view_as_complex``，结果为视图（在可整除
    步长时），否则可能产生连续副本。

    参数:
    ----------
    - real_tensor : torch.Tensor
        浮点张量。最后一维须为偶数：
        - ``[..., 2]`` → 复数形状 ``[...]``（最后一维消失）
        - ``[..., 2*K]``（K≥1）→ 先视为 ``[..., K, 2]`` 再合并 → ``[..., K]`` 复数
        - ``[..., K, 2]`` → ``[..., K]`` 复数

    返回:
    ----------
    - torch.Tensor
        ``dtype`` 为 ``complex64`` / ``complex128``（由输入浮点类型决定）

    异常:
    ----------
    - TypeError
        输入非浮点类型
    - ValueError
        最后一维长度为奇数
    """
    if not real_tensor.dtype.is_floating_point:
        raise TypeError(
            f"last_dim_real_to_complex 需要浮点张量，当前 dtype={real_tensor.dtype}"
        )
    n_last = real_tensor.shape[-1]
    if n_last % 2 != 0:
        raise ValueError(f"最后一维长度须为偶数（real/imag 成对），得到 {n_last}")
    x = real_tensor.contiguous()
    if n_last != 2:
        x = x.reshape(*x.shape[:-1], -1, 2)
    return torch.view_as_complex(x)


def last_dim_complex_to_real(complex_tensor: torch.Tensor) -> torch.Tensor:
    """
    将复数张量按最后一维展开为 (实部, 虚部) 浮点张量

    与 :func:`last_dim_real_to_complex` 互逆（在成对约定一致时）：最后一维上
    每个复数标量变为长度 2 的实数对 ``[Re, Im]``。

    参数:
    ----------
    - complex_tensor : torch.Tensor
        ``is_complex()`` 为真的张量，任意前置形状 ``[...]``

    返回:
    ----------
    - torch.Tensor
        形状 ``[..., 2]``，dtype 为 ``float32`` / ``float64`` 等与复数 dtype 对应

    异常:
    ----------
    - TypeError
        输入非复数张量
    """
    if not complex_tensor.is_complex():
        raise TypeError(
            f"last_dim_complex_to_real 需要复数张量，当前 dtype={complex_tensor.dtype}"
        )
    return torch.view_as_real(complex_tensor.contiguous())
