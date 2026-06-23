"""窗函数与 PyTorch 张量之间的封装。

唯一公开入口：``apply_window``。
``periodic=True`` 对应 SciPy ``get_window(..., fftbins=True)`` / 周期窗（DD 谱默认）；
``periodic=False`` 对应对称窗（MTD 默认）。
"""

from typing import Any

import numpy as np
import torch
from scipy.signal.windows import get_window

__all__ = ["apply_window"]

HAS_TORCH_SIGNAL_WINDOWS = hasattr(torch, "signal") and hasattr(torch.signal, "windows")

WindowSpec = str | tuple[Any, ...]

WindowConfig = str | dict[str, Any]

OptionalWindowInput = WindowSpec | WindowConfig | None

_TORCH_NAMED_WINDOWS = frozenset({"hann", "hamming", "blackman"})


def _window_spec_from_config(spec: WindowConfig) -> WindowSpec:
    """将 TOML/配置中的窗描述转为 ``get_window`` 可用的 ``WindowSpec``。"""
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
            raise ValueError(
                "chebwin window requires numeric 'at' (sidelobe attenuation dB)"
            )
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


def _numpy_window_to_torch(
    w: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """将实值窗系数 ``ndarray`` 转为指定设备与 dtype 的一维 ``Tensor``。"""
    return torch.as_tensor(np.asarray(w, dtype=np.float64), device=device, dtype=dtype)


def _resolve_window_spec(window: WindowSpec | WindowConfig) -> WindowSpec:
    if isinstance(window, dict):
        return _window_spec_from_config(window)
    return window


def _make_window_tensor(
    window: WindowSpec | WindowConfig,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    periodic: bool = True,
) -> torch.Tensor:
    """模块内：生成长度为 ``nx`` 的 1D 实值窗系数。"""
    if nx < 1:
        raise ValueError(f"nx 须 >= 1，收到 {nx}")
    spec = _resolve_window_spec(window)
    name = spec.strip().lower() if isinstance(spec, str) else None
    use_torch = (
        HAS_TORCH_SIGNAL_WINDOWS
        and dtype in (torch.float32, torch.float64)
        and name in _TORCH_NAMED_WINDOWS
    )
    if use_torch:
        tw = torch.signal.windows
        kw: dict[str, Any] = dict(
            sym=not periodic,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        if name == "hann":
            return tw.hann(nx, **kw)
        if name == "hamming":
            return tw.hamming(nx, **kw)
        return tw.blackman(nx, **kw)

    w = get_window(spec, nx, fftbins=periodic)
    return _numpy_window_to_torch(w, device=device, dtype=dtype)


def apply_window(
    x: torch.Tensor,
    dim: int,
    window: OptionalWindowInput,
    *,
    periodic: bool = True,
) -> torch.Tensor:
    """沿 ``dim`` 对 ``x`` 施加窗系数并与 ``x`` 相乘。

    ``window`` 可为：

    - ``None``：不加窗，直接返回 ``x``。
    - ``str`` / ``tuple``：SciPy ``get_window`` 窗规格（如 ``\"hamming\"``、``(\"chebwin\", 60)``）。
    - ``dict``：TOML 窗配置（含 ``type``/``name`` 及窗参数，如 ``{\"type\": \"chebwin\", \"at\": 60}``）。

    ``periodic=True``：周期窗（DD 谱默认）；``periodic=False``：对称窗（MTD 默认）。
    """
    if x.ndim < 1:
        raise ValueError("apply_window expects x.ndim >= 1")
    if window is None:
        return x
    dim = dim % x.ndim
    nx = x.shape[dim]
    w = _make_window_tensor(
        window,
        nx,
        device=x.device,
        dtype=x.real.dtype,
        periodic=periodic,
    )
    view_shape = [1] * x.ndim
    view_shape[dim] = nx
    return x * w.reshape(view_shape)
