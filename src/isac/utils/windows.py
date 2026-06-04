"""窗函数与 PyTorch 张量之间的封装。

- **优先**在可用且 ``dtype`` 为 ``float32``/``float64`` 时，用 ``torch.signal.windows``
  在目标 ``device`` 上直接生成 1D 实值窗（与 MTD 等模块对齐）。
- **否则**用 ``scipy.signal.windows.get_window`` 在 CPU 上计算，再 ``torch.as_tensor``
  到目标 ``device``/``dtype``，避免与 CUDA 张量混乘时的设备不一致。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal.windows import get_window

HAS_TORCH_SIGNAL_WINDOWS = hasattr(torch, "signal") and hasattr(torch.signal, "windows")

WindowSpec = Union[str, Tuple[Any, ...]]

WindowConfig = Union[str, Dict[str, Any]]

OptionalWindowInput = Optional[Union[WindowSpec, WindowConfig]]


def window_spec_from_config(spec: WindowConfig) -> WindowSpec:
    """将 TOML/配置中的窗描述转为 ``get_window`` 可用的 ``WindowSpec``。

    - 字符串：无参窗名（如 ``\"hamming\"``）。
    - 字典：须含 ``type`` 或 ``name``；带参窗见各分支必填键。
    """
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"window spec must be str or dict, got {type(spec)!r}")
    raw_kind = spec.get("type") or spec.get("name")
    if raw_kind is None or not isinstance(raw_kind, str):
        raise ValueError("window config dict must include a string 'type' or 'name'")
    kind = raw_kind.lower()
    extras = {k: v for k, v in spec.items() if k not in ("type", "name")}

    if kind == "chebwin":
        if "at" not in extras:
            raise ValueError("chebwin window requires numeric 'at' (sidelobe attenuation dB)")
        return ("chebwin", float(extras["at"]))
    if kind == "kaiser":
        if "beta" not in extras:
            raise ValueError("kaiser window requires numeric 'beta'")
        return ("kaiser", float(extras["beta"]))
    if kind == "tukey":
        if "alpha" not in extras:
            raise ValueError("tukey window requires numeric 'alpha'")
        return ("tukey", float(extras["alpha"]))
    if kind == "general_hamming":
        if "alpha" not in extras:
            raise ValueError("general_hamming window requires numeric 'alpha'")
        return ("general_hamming", float(extras["alpha"]))

    if extras:
        raise ValueError(
            f"unknown or unsupported window type {raw_kind!r} with extra keys {set(extras)!r}; "
            "use a no-parameter name only, or a supported parameterized window."
        )
    return kind


def numpy_window_to_torch(
    w: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """将实值窗系数 ``ndarray`` 转为指定设备与 dtype 的一维 ``Tensor``。"""
    return torch.as_tensor(np.asarray(w, dtype=np.float64), device=device, dtype=dtype)


def get_named_window_tensor_1d(
    window_type: str,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sym: bool = True,
) -> torch.Tensor:
    """具名 1D 实值窗：优先 ``torch.signal.windows``，否则 ``get_window_tensor``（SciPy）。

    与 MTD 常用约定一致：``sym=True`` 为对称窗（对应 ``torch.hann_window(..., periodic=False)``）；
    ``sym=False`` 为周期窗，对应 SciPy ``get_window(..., fftbins=True)``。

    参数
    -----
    window_type
        不区分大小写；内置优先走 Torch 的分支为 ``hann`` / ``hamming`` / ``blackman``。
        回退 SciPy 时，任何 ``get_window`` 支持的字符串或元组规格均可扩展传入（当前 MTD 仅传三者）。
    """
    name = window_type.strip().lower()
    use_torch = (
        HAS_TORCH_SIGNAL_WINDOWS
        and dtype in (torch.float32, torch.float64)
        and name in ("hann", "hamming", "blackman")
    )
    if use_torch:
        tw = torch.signal.windows
        kw: Dict[str, Any] = dict(
            sym=sym,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        if name == "hann":
            return tw.hann(nx, **kw)
        if name == "hamming":
            return tw.hamming(nx, **kw)
        return tw.blackman(nx, **kw)

    return get_window_tensor(name, nx, device=device, dtype=dtype, fftbins=not sym)


def get_window_tensor(
    window: WindowSpec,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fftbins: bool = True,
) -> torch.Tensor:
    """等价于 ``scipy.signal.windows.get_window``，返回 1D ``torch.Tensor``。

    ``window`` 与 SciPy 约定一致，例如 ``\"hamming\"``、``(\"kaiser\", beta)``、
    ``(\"chebwin\", at_db)`` 等。
    """
    w = get_window(window, nx, fftbins=fftbins)
    return numpy_window_to_torch(w, device=device, dtype=dtype)


def apply_window(
    x: torch.Tensor,
    dim: int,
    window: OptionalWindowInput,
    *,
    fftbins: bool = True,
) -> torch.Tensor:
    """沿 ``dim`` 对 ``x`` 施加 SciPy 窗（与 ``get_window`` 的 ``window`` 参数约定一致）。

    - ``window is None``：不加窗，直接返回 ``x``。
    - ``window`` 为 ``dict``：按 ``window_spec_from_config`` 解析后再加窗。
    - ``window`` 为 ``str`` 或 ``tuple``：视为已是 ``get_window`` 可用的窗规格。

    在 ``x.device`` 上用 ``x.real.dtype`` 生成可广播的一维窗并与 ``x`` 相乘。
    """
    if x.ndim < 1:
        raise ValueError("apply_window expects x.ndim >= 1")
    if window is None:
        return x
    if isinstance(window, dict):
        spec: WindowSpec = window_spec_from_config(window)
    else:
        spec = window
    dim = dim % x.ndim
    nx = x.shape[dim]
    w = get_window_tensor(
        spec, nx, device=x.device, dtype=x.real.dtype, fftbins=fftbins
    )
    view_shape = [1] * x.ndim
    view_shape[dim] = nx
    return x * w.reshape(view_shape)


def window_callable_to_tensor(
    fn: Callable[..., np.ndarray],
    *args: Any,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs: Any,
) -> torch.Tensor:
    """对任意 ``scipy.signal.windows`` 可调用对象求值后转为 ``Tensor``。

    用于 ``get_window`` 不便表达的额外关键字参数（如 ``tukey`` 的 ``alpha``）。
    """
    w = fn(*args, **kwargs)
    return numpy_window_to_torch(w, device=device, dtype=dtype)
