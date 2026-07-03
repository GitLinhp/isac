"""
CFAR 检测（CA / OS，一维 / 二维，Torch 实现）

与历史 ``processing`` 模块函数行为一致；统一运行时入口为 ``CFARDetector.__call__``。
"""

from typing import List, Literal, Optional, Union
from warnings import warn
import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from isac.utils import convert


# -------------------------- 辅助函数 --------------------------
def _same_type_output(data: torch.Tensor, like: Union[NDArray, torch.Tensor]):
    if isinstance(like, torch.Tensor):
        return data
    return convert(data, "numpy")


def _is_complex(data: Union[NDArray, torch.Tensor]) -> bool:
    if isinstance(data, torch.Tensor):
        return bool(torch.is_complex(data))
    return bool(np.iscomplexobj(data))


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1.0)


def _conv1d_same(data_1d: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    x = data_1d.unsqueeze(0).unsqueeze(0)
    k = kernel_1d.flip(0).unsqueeze(0).unsqueeze(0)
    pad = kernel_1d.numel() // 2
    y = F.conv1d(F.pad(x, (pad, pad), mode="constant", value=0), k)
    return y.squeeze(0).squeeze(0)


def _guard_trailing_1d(
    guard: Union[int, List[int]], trailing: Union[int, List[int]]
) -> tuple[int, int]:
    ga = np.array(guard, dtype=int).ravel()
    ta = np.array(trailing, dtype=int).ravel()
    return int(ga[0]), int(ta[0])


def cfar_ca_1d(
    data: Union[NDArray, torch.Tensor],
    guard: int,
    trailing: int,
    pfa: float = 1e-5,
    axis: int = 0,
    detector: str = "squarelaw",
    offset: Optional[float] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    一维 CA-CFAR（Cell Averaging）。

    参数:
    -------
    - data:
        幅度/功率数据。``linear`` 使用幅度，``squarelaw`` 使用功率
    - guard:
        单侧保护单元数（总计 ``2*guard``）
    - trailing:
        单侧参考单元数（总计 ``2*trailing``）
    - pfa:
        虚警率
    - axis:
        计算轴，支持 0 或 1
    - detector:
        ``linear`` 或 ``squarelaw``
    - offset:
        阈值缩放因子。若为 None 则按 pfa 自动计算

    返回:
    -------
    - cfar:
        与输入同形状的 CFAR 阈值
    """
    if _is_complex(data):
        raise ValueError("Input data should not be complex.")

    x = convert(data, "torch", dtype=torch.float64, device=torch.device("cpu"))
    cfar = torch.zeros_like(x)

    if offset is None:
        if detector == "squarelaw":
            a = trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1)
        elif detector == "linear":
            a = math.sqrt(trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = torch.ones((guard + trailing) * 2 + 1, dtype=x.dtype)
    cfar_win[trailing : (trailing + guard * 2 + 1)] = 0
    cfar_win = cfar_win / torch.sum(cfar_win)

    if axis == 0:
        if x.ndim == 1:
            cfar = a * _conv1d_same(x, cfar_win)
        elif x.ndim == 2:
            for idx in range(0, x.shape[1]):
                cfar[:, idx] = a * _conv1d_same(x[:, idx], cfar_win)
    elif axis == 1:
        for idx in range(0, x.shape[0]):
            cfar[idx, :] = a * _conv1d_same(x[idx, :], cfar_win)

    cfar = cfar.to(dtype=torch.float32 if x.dtype == torch.float32 else torch.float64)
    return _same_type_output(cfar, data)


def cfar_ca_2d(
    data: Union[NDArray, torch.Tensor],
    guard: Union[int, List[int]],
    trailing: Union[int, List[int]],
    pfa: float = 1e-5,
    detector: str = "squarelaw",
    offset: Optional[float] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    二维 CA-CFAR（Cell Averaging）。

    参数:
    -------
    - data:
        幅度/功率数据（二维）
    - guard:
        保护单元，可为标量或 ``[axis0, axis1]``
    - trailing:
        参考单元，可为标量或 ``[axis0, axis1]``
    - pfa:
        虚警率
    - detector:
        ``linear`` 或 ``squarelaw``
    - offset:
        阈值缩放因子。若为 None 则按 pfa 自动计算

    返回:
    -------
    - cfar:
        与输入同形状的 CFAR 阈值
    """
    if _is_complex(data):
        raise ValueError("Input data should not be complex.")

    x = convert(data, "torch", dtype=torch.float64, device=torch.device("cpu"))
    guard_arr = np.array(guard, dtype=int)
    if guard_arr.size == 1:
        guard_arr = np.tile(guard_arr, 2)
    trailing_arr = np.array(trailing, dtype=int)
    if trailing_arr.size == 1:
        trailing_arr = np.tile(trailing_arr, 2)

    if offset is None:
        tg_sum = trailing_arr + guard_arr
        t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
        g_num = (2 * guard_arr[0] + 1) * (2 * guard_arr[1] + 1)
        if t_num == g_num:
            raise ValueError("No trailing bins!")
        if detector == "squarelaw":
            a = (t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1)
        elif detector == "linear":
            a = math.sqrt((t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = torch.ones(
        int((guard_arr[0] + trailing_arr[0]) * 2 + 1),
        int((guard_arr[1] + trailing_arr[1]) * 2 + 1),
        dtype=x.dtype,
    )
    cfar_win[
        trailing_arr[0] : (trailing_arr[0] + guard_arr[0] * 2 + 1),
        trailing_arr[1] : (trailing_arr[1] + guard_arr[1] * 2 + 1),
    ] = 0
    cfar_win = cfar_win / torch.sum(cfar_win)

    x4d = x.unsqueeze(0).unsqueeze(0)
    k4d = cfar_win.flip(0, 1).unsqueeze(0).unsqueeze(0)
    pad_h = cfar_win.shape[0] // 2
    pad_w = cfar_win.shape[1] // 2
    y = F.conv2d(
        F.pad(x4d, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0), k4d
    )
    out = (a * y.squeeze(0).squeeze(0)).to(dtype=torch.float64)

    return _same_type_output(out, data)


def os_cfar_threshold(k: int, n: int, pfa: float) -> float:
    """
    用割线法计算 OS-CFAR 阈值缩放因子。

    参考：
    Rohling, 1983.

    参数:
    -------
    - k:
        统计量索引
    - n:
        窗口大小
    - pfa:
        虚警率

    返回:
    -------
    - threshold:
        阈值缩放因子
    """

    def fun(k_: int, n_: int, t_os: float, pfa_: float) -> float:
        arr = np.arange(n_, n_ - k_, -1, dtype=float) + t_os
        return (
            _log_factorial(n_)
            - _log_factorial(n_ - k_)
            - float(np.sum(np.log(arr)))
            - math.log(pfa_)
        )

    max_iter = 10000
    t_max = 1e32
    t_min = 1.0

    for _ in range(0, max_iter):
        f_min = fun(k, n, t_min, pfa)
        f_max = fun(k, n, t_max, pfa)
        m_n = t_max - f_max * (t_min - t_max) / (f_min - f_max)
        f_m_n = fun(k, n, m_n, pfa)
        if f_m_n == 0 or abs(f_m_n) < 0.0001:
            return m_n

        if f_max * f_m_n < 0:
            t_min = m_n
        elif f_min * f_m_n < 0:
            t_max = m_n
        else:
            break
    return None


def cfar_os_1d(
    data: Union[NDArray, torch.Tensor],
    guard: int,
    trailing: int,
    k: int,
    pfa: float = 1e-5,
    axis: int = 0,
    detector: str = "squarelaw",
    offset: Optional[float] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    一维 OS-CFAR（Ordered Statistic）。

    边界单元使用循环索引（rollover）补齐窗口。

    参数:
    -------
    - data:
        幅度/功率数据
    - guard:
        保护单元数
    - trailing:
        参考单元数
    - k:
        统计量索引
    - pfa:
        虚警率
    - detector:
        ``linear`` 或 ``squarelaw``
    - offset:
        阈值缩放因子。若为 None 则按 pfa 自动计算

    返回:
    -------
    - cfar:
        与输入同形状的 CFAR 阈值
    """
    if _is_complex(data):
        raise ValueError("Input data should not be complex.")

    x = convert(data, "torch", dtype=torch.float64, device=torch.device("cpu"))
    cfar = torch.zeros_like(x)
    leading = trailing

    if offset is None:
        if detector == "squarelaw":
            a = os_cfar_threshold(k, trailing * 2, pfa)
        elif detector == "linear":
            a = math.sqrt(os_cfar_threshold(k, trailing * 2, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < trailing or k > trailing * 2:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            f"(N = {trailing * 2})``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    if axis == 0:
        for idx in range(0, x.shape[0]):
            left = torch.arange(idx - leading - guard, idx - guard, 1, dtype=torch.long)
            right = torch.arange(
                idx + 1 + guard, idx + 1 + trailing + guard, 1, dtype=torch.long
            )
            win_idx = torch.remainder(torch.cat([left, right]), x.shape[0]).long()
            if x.ndim == 1:
                samples, _ = torch.sort(x[win_idx])
                cfar[idx] = a * samples[k]
            elif x.ndim == 2:
                samples, _ = torch.sort(x[win_idx, :], dim=0)
                cfar[idx, :] = a * samples[k, :]
    elif axis == 1:
        for idx in range(0, x.shape[1]):
            left = torch.arange(idx - leading - guard, idx - guard, 1, dtype=torch.long)
            right = torch.arange(
                idx + 1 + guard, idx + 1 + trailing + guard, 1, dtype=torch.long
            )
            win_idx = torch.remainder(torch.cat([left, right]), x.shape[1]).long()
            samples, _ = torch.sort(x[:, win_idx], dim=1)
            cfar[:, idx] = a * samples[:, k]

    return _same_type_output(cfar, data)


def cfar_os_2d(
    data: Union[NDArray, torch.Tensor],
    guard: Union[int, List[int]],
    trailing: Union[int, List[int]],
    k: int,
    pfa: float = 1e-5,
    detector: str = "squarelaw",
    offset: Optional[float] = None,
) -> Union[NDArray, torch.Tensor]:
    """
    二维 OS-CFAR（Ordered Statistic）。

    边界单元使用循环索引（rollover）补齐窗口。

    参数:
    -------
    - data:
        幅度/功率数据
    - guard:
        保护单元数
    - trailing:
        参考单元数
    - k:
        统计量索引
    - pfa:
        虚警率
    - detector:
        ``linear`` 或 ``squarelaw``
    - offset:
        阈值缩放因子。若为 None 则按 pfa 自动计算

    返回:
    -------
    - cfar:
        与输入同形状的 CFAR 阈值
    """
    if _is_complex(data):
        raise ValueError("Input data should not be complex.")

    x = convert(data, "torch", dtype=torch.float64, device=torch.device("cpu"))
    cfar = torch.zeros_like(x)

    guard_arr = np.array(guard, dtype=int)
    if guard_arr.size == 1:
        guard_arr = np.tile(guard_arr, 2)
    trailing_arr = np.array(trailing, dtype=int)
    if trailing_arr.size == 1:
        trailing_arr = np.tile(trailing_arr, 2)

    tg_sum = trailing_arr + guard_arr
    t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
    g_num = (2 * guard_arr[0] + 1) * (2 * guard_arr[1] + 1)

    if offset is None:
        if t_num == g_num:
            raise ValueError("No trailing bins!")
        if detector == "squarelaw":
            a = os_cfar_threshold(k, t_num - g_num, pfa)
        elif detector == "linear":
            a = math.sqrt(os_cfar_threshold(k, t_num - g_num, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < (t_num - g_num) / 2 or k > t_num - g_num:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            f"(N = {t_num - g_num})``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    cfar_win = torch.ones(
        int(tg_sum[0] * 2 + 1), int(tg_sum[1] * 2 + 1), dtype=torch.bool
    )
    cfar_win[
        trailing_arr[0] : (trailing_arr[0] + guard_arr[0] * 2 + 1),
        trailing_arr[1] : (trailing_arr[1] + guard_arr[1] * 2 + 1),
    ] = False

    for idx_0 in range(0, x.shape[0]):
        for idx_1 in range(0, x.shape[1]):
            win_idx_0 = torch.remainder(
                torch.arange(
                    idx_0 - tg_sum[0], idx_0 + 1 + tg_sum[0], 1, dtype=torch.long
                ),
                x.shape[0],
            )
            win_idx_1 = torch.remainder(
                torch.arange(
                    idx_1 - tg_sum[1], idx_1 + 1 + tg_sum[1], 1, dtype=torch.long
                ),
                x.shape[1],
            )
            grid0, grid1 = torch.meshgrid(win_idx_0, win_idx_1, indexing="ij")
            sample_cube = x[grid0, grid1]
            samples, _ = torch.sort(sample_cube[cfar_win].flatten())
            cfar[idx_0, idx_1] = a * samples[k]

    return _same_type_output(cfar, data)


class CFARDetector:
    """封装 CA/OS 与 1D/2D CFAR 参数；通过 ``__call__(data, mode=...)`` 计算阈值面。"""

    __slots__ = ("cfar_type", "guard", "trailing", "pfa", "detector", "offset", "k")

    def __init__(
        self,
        cfar_type: str,
        guard: Union[int, List[int]],
        trailing: Union[int, List[int]],
        pfa: float = 1e-4,
        detector: str = "linear",
        offset: Optional[float] = None,
        k: Optional[int] = None,
    ) -> None:
        t = cfar_type.strip().lower()
        if t not in ("ca", "os"):
            raise ValueError("cfar_type must be 'ca' or 'os'")
        if t == "os" and k is None:
            raise ValueError("cfar_type 'os' requires integer k")
        self.cfar_type = t
        self.guard = guard
        self.trailing = trailing
        self.pfa = pfa
        self.detector = detector
        self.offset = offset
        self.k = k

    def __call__(
        self,
        data: Union[NDArray, torch.Tensor],
        *,
        mode: Literal["1d", "2d"] = "2d",
        axis: int = 0,
    ) -> Union[NDArray, torch.Tensor]:
        if mode == "2d":
            if self.cfar_type == "ca":
                return cfar_ca_2d(
                    data,
                    guard=self.guard,
                    trailing=self.trailing,
                    pfa=self.pfa,
                    detector=self.detector,
                    offset=self.offset,
                )
            assert self.k is not None
            return cfar_os_2d(
                data,
                guard=self.guard,
                trailing=self.trailing,
                k=self.k,
                pfa=self.pfa,
                detector=self.detector,
                offset=self.offset,
            )

        g1, t1 = _guard_trailing_1d(self.guard, self.trailing)
        if self.cfar_type == "ca":
            return cfar_ca_1d(
                data,
                guard=g1,
                trailing=t1,
                pfa=self.pfa,
                axis=axis,
                detector=self.detector,
                offset=self.offset,
            )
        assert self.k is not None
        return cfar_os_1d(
            data,
            guard=g1,
            trailing=t1,
            k=self.k,
            pfa=self.pfa,
            axis=axis,
            detector=self.detector,
            offset=self.offset,
        )
