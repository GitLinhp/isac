"""
张量工具函数模块

提供张量维度操作和归一化等功能。
"""

import torch


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
