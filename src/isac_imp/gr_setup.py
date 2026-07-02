"""GRC 共享 System 配置：对齐 run_sensing_monostatic，不复用 gnuradio/core/gr_system。

职责：
  - 将 GRC 变量（OFDM 参数等）合并进 TOML 配置后构建 ``System``
  - 通过进程内 registry 让多个 epy_block 共享同一 ``System`` 实例

设计背景：
  - 刻意不复用 ``gnuradio/core/gr_system.py``，与脚本侧 ``System`` + ``load_config`` 路径一致
  - TX/RX 为独立 epy 块，无法共享 ``self._system``；相同 ``make_cache_key`` 命中 registry
  - 发端 ``transmit()`` 后 ``set_last_x_rg()`` 写入参考频域网格，供收端 ``sensing()`` 读取

数据流（简化）::

  GRC OFDM 变量 + TOML → apply_grc_ofdm_overrides → _build_system → _SYSTEM_REGISTRY
  TX epy_block → create_system / set_last_x_rg
  RX epy_block → create_system / get_last_x_rg
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import sionna.phy
import torch

from isac.system import System
from isac.utils import load_config, set_random_seed

# GRC 传入的 [ofdm] 覆盖字典；键名与 apply_grc_ofdm_overrides / epy_block 一致
_OfdmOverrides = Mapping[str, Any] | None


def apply_grc_ofdm_overrides(
    raw: dict,
    *,
    num_symbols: int,
    fft_size: int,
    subcarrier_spacing: float,
    cp_len: int,
) -> dict:
    """将 GRC 变量合并进 TOML 字典的 [ofdm] 段。

    Args:
    - raw: ``load_config`` 返回的原始配置字典（不会被原地修改）
    - num_symbols: OFDM 符号数
    - fft_size: FFT 点数
    - subcarrier_spacing: 子载波间隔 (Hz)
    - cp_len: 循环前缀长度（采样点）
    - log: 是否在覆盖项与 TOML 不一致时打印日志

    Returns:
    - 合并后的新配置字典；GRC 四参数优先于 TOML [ofdm] 同名项
    """
    merged = dict(raw)
    toml_ofdm = dict(merged.get("ofdm") or {})
    ofdm = dict(toml_ofdm)
    ofdm["num_symbols"] = int(num_symbols)
    ofdm["fft_size"] = int(fft_size)
    ofdm["subcarrier_spacing"] = float(subcarrier_spacing)
    ofdm["cyclic_prefix_length"] = int(cp_len)
    merged["ofdm"] = ofdm
    return merged


def make_cache_key(
    config_file: str,
    device: str,
    seed: int,
    ofdm_overrides: _OfdmOverrides = None,
) -> tuple:
    """收发端共享 System 的 registry 键。

    Returns:
        (config_file, device, seed, num_symbols, fft_size, subcarrier_spacing, cp_len)
    """
    if ofdm_overrides is None:
        n_sym, fft_size, scs, cp = 0, 0, 0.0, 0
    else:
        n_sym = int(ofdm_overrides["num_symbols"])
        fft_size = int(ofdm_overrides["fft_size"])
        scs = float(ofdm_overrides["subcarrier_spacing"])
        cp = int(ofdm_overrides["cyclic_prefix_length"])
    return (str(config_file), str(device), int(seed), n_sym, fft_size, scs, cp)


@dataclass
class SystemSession:
    """registry 中缓存的一条 System 会话。

    Attributes:
        system: 已构建的 ``System`` 实例
        cache_key: 对应的 ``make_cache_key`` 返回值
        last_x_rg: 发端 ``transmit()`` 后的参考频域网格（供收端 sensing 使用）
    """

    system: System
    cache_key: tuple
    last_x_rg: Any = None


# 进程内单例表：cache_key → SystemSession；供多 epy_block 共享同一 System
_SYSTEM_REGISTRY: dict[tuple, SystemSession] = {}


def invalidate_system_cache(cache_key: tuple | None = None) -> None:
    """按 key 清除 registry 条目；省略 key 时清空全部。

    可在 GRC 中修改 seed / device / OFDM 参数且需强制重建 System 时调用。
    """
    if cache_key is None:
        _SYSTEM_REGISTRY.clear()
        return
    _SYSTEM_REGISTRY.pop(cache_key, None)


def set_last_x_rg(system: System, x_rg) -> None:
    """TX ``transmit()`` 后将参考频域网格写入对应 session。"""
    for session in _SYSTEM_REGISTRY.values():
        if session.system is system:
            session.last_x_rg = x_rg
            return


def get_last_x_rg(system: System):
    """读取与 ``system`` 对象关联 session 中的 ``last_x_rg``；未写入时返回 None。"""
    for session in _SYSTEM_REGISTRY.values():
        if session.system is system:
            return session.last_x_rg
    return None


def _build_system(
    config_file: str,
    *,
    device: str,
    seed: int,
    batch_size: int,
    ofdm_overrides: _OfdmOverrides,
) -> System:
    """加载配置并构建 ``System``。

    流程：设随机种子与 Sionna 设备 → ``load_config`` → 可选剔除信道段
    → 可选 GRC OFDM 覆盖 → ``System(config=...)``。

    ``include_channel=False`` 时移除 ``[channel]``，用于 USRP 空口路径：
    不经 RT 仿真信道，避免空场景初始化失败。
    """
    set_random_seed(seed)
    sionna.phy.config.device = device
    torch.set_num_threads(1)

    raw = load_config(config_file)
    if ofdm_overrides is not None:
        raw = apply_grc_ofdm_overrides(
            raw,
            num_symbols=int(ofdm_overrides["num_symbols"]),
            fft_size=int(ofdm_overrides["fft_size"]),
            subcarrier_spacing=float(ofdm_overrides["subcarrier_spacing"]),
            cp_len=int(ofdm_overrides["cyclic_prefix_length"]),
        )

    system = System(
        config=raw,
        batch_size=int(batch_size),
        device=str(device),
    )
    return system


def create_system(
    config_file: str,
    *,
    device: str = "cpu",
    seed: int = 42,
    batch_size: int = 1,
    ofdm_overrides: _OfdmOverrides = None,
    use_cache: bool = True,
) -> System:
    """构建或与 registry 共享 ``System`` 实例（对齐 run_sensing_monostatic 配置路径）。

    相同 ``cache_key`` 下 TX/RX epy_block 得到同一 ``System`` 对象。

    参数:
    ----------
        - config_file: TOML 相对路径（经 ``load_config`` 解析）
        - device: Sionna/Torch 计算设备
        - seed: 随机种子
        - batch_size: 批大小，传入 ``System.batch_size``
        - ofdm_overrides: GRC OFDM 四参数覆盖；None 表示不覆盖 TOML [ofdm]
        - use_cache: True 时命中 registry 直接返回已有实例

    返回:
    ----------
        - 新建或缓存的 ``System`` 实例
    """
    # 生成 cache_key
    cache_key = make_cache_key(config_file, device, seed, ofdm_overrides)
    if use_cache and cache_key in _SYSTEM_REGISTRY:
        return _SYSTEM_REGISTRY[cache_key].system

    # 构建 System
    system = _build_system(
        config_file,
        device=device,
        seed=seed,
        batch_size=batch_size,
        ofdm_overrides=ofdm_overrides,
    )

    # 注册 System
    if use_cache:
        # 注册供后续 epy 块（相同 cache_key）命中同一实例
        _SYSTEM_REGISTRY[cache_key] = SystemSession(system=system, cache_key=cache_key)

    return system
