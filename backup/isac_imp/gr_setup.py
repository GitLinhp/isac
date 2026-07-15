"""GRC 共享 System 配置：对齐 run_sensing_monostatic，不复用 gnuradio/core/gr_system。

职责：
  - 仅读配置的 ``resolve_*``：``load_config`` + ``SystemParams.from_dict``（不建 System）
  - 将 GRC 变量（OFDM 参数等）合并进 TOML 配置后构建 ``System``（``create_system``）
  - 通过进程内 registry 让多个 epy_block 共享同一 ``System`` 实例

设计背景：
  - 刻意不复用 ``gnuradio/core/gr_system.py``，与脚本侧 ``System`` + ``load_config`` 路径一致
  - TX/RX 为独立 epy 块，无法共享 ``self._system``；相同 ``make_cache_key`` 命中 registry
  - 发端 ``transmit()`` 后 ``set_last_x_rg()`` / ``set_last_x_time()`` 写入参考网格，
    供收端相关同步与 ``sensing()`` 读取
  - GNU Radio 多线程下用 RLock 保护 create/set/get，避免双检竞态产生孤儿 System

数据流（简化）::

  TOML → load_system_params → resolve_*（samp_rate / burst_len / cache / dd_vlen）
  GRC OFDM 变量 + TOML → apply_grc_ofdm_overrides → _build_system → _SYSTEM_REGISTRY
  TX epy_block → create_system / set_last_x_rg / set_last_x_time
  RX epy_block → create_system / get_last_x_rg / get_last_x_time

下文按功能分区组织（类型与状态 → OFDM 覆盖 → registry → 参考波形 →
System 构建 → 仅读配置 resolve_* → create_system 入口）。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import sionna.phy
import torch
from scipy.constants import speed_of_light as _C

from isac.data_structures import SystemParams
from isac.system import System
from isac.utils import load_config, set_random_seed

# ---------------------------------------------------------------------------
# 1. 类型别名与模块状态
# ---------------------------------------------------------------------------

# GRC 传入的 [ofdm] 覆盖字典；键名与 apply_grc_ofdm_overrides / epy_block 一致
_OfdmOverrides = Mapping[str, Any] | None

_LOG_PREFIX = "[isac_imp.gr_setup]"
_registry_lock = threading.RLock()
_warned_fallback_x_time = False
_warned_fallback_x_rg = False


# ---------------------------------------------------------------------------
# 2. GRC OFDM 覆盖与 cache key
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. System 会话与 registry
# ---------------------------------------------------------------------------


@dataclass
class SystemSession:
    """registry 中缓存的一条 System 会话。

    Attributes:
        system: 已构建的 ``System`` 实例
        cache_key: 对应的 ``make_cache_key`` 返回值
        last_x_rg: 发端 ``transmit()`` 后的参考频域网格（供收端 sensing 使用）
        last_x_time: 发端 ``transmit()`` 后的参考时域波形（供收端相关同步，未乘 tx_amp）
    """

    system: System
    cache_key: tuple
    last_x_rg: Any = None
    last_x_time: Any = None


# 进程内单例表：cache_key → SystemSession；供多 epy_block 共享同一 System
_SYSTEM_REGISTRY: dict[tuple, SystemSession] = {}
# 旁路：id(system) → cache_key，便于孤儿 System 仍能按 key 找回 session
_SYSTEM_ID_TO_KEY: dict[int, tuple] = {}
# 仅读配置：config_file → SystemParams（不建 System / 不入 System registry）
_PARAMS_CACHE: dict[str, SystemParams] = {}
_params_lock = threading.RLock()


def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", file=sys.stderr, flush=True)


def invalidate_system_cache(cache_key: tuple | None = None) -> None:
    """按 key 清除 registry 条目；省略 key 时清空全部。

    可在 GRC 中修改 seed / device / OFDM 参数且需强制重建 System 时调用。
    """
    with _registry_lock:
        if cache_key is None:
            _SYSTEM_REGISTRY.clear()
            _SYSTEM_ID_TO_KEY.clear()
            return
        session = _SYSTEM_REGISTRY.pop(cache_key, None)
        if session is not None:
            _SYSTEM_ID_TO_KEY.pop(id(session.system), None)


def registry_status() -> list[dict[str, Any]]:
    """返回 registry 摘要（不含波形），供 RX 调试打印。"""
    with _registry_lock:
        out: list[dict[str, Any]] = []
        for key, session in _SYSTEM_REGISTRY.items():
            out.append(
                {
                    "cache_key": key,
                    "system_id": id(session.system),
                    "has_last_x_time": session.last_x_time is not None,
                    "has_last_x_rg": session.last_x_rg is not None,
                }
            )
        return out


def _find_session(system: System) -> SystemSession | None:
    """先按对象身份，再按 id→key 旁路查找 session。调用方须已持锁。"""
    for session in _SYSTEM_REGISTRY.values():
        if session.system is system:
            return session
    key = _SYSTEM_ID_TO_KEY.get(id(system))
    if key is not None:
        return _SYSTEM_REGISTRY.get(key)
    return None


def _single_nonempty(attr: str):
    """若 registry 中恰好一个 session 的 attr 非空，返回该值；否则 None。调用方须已持锁。"""
    hits = [getattr(s, attr) for s in _SYSTEM_REGISTRY.values() if getattr(s, attr) is not None]
    if len(hits) == 1:
        return hits[0]
    return None


# ---------------------------------------------------------------------------
# 4. TX/RX 参考波形（last_x_rg / last_x_time）
# ---------------------------------------------------------------------------


def set_last_x_rg(system: System, x_rg) -> bool:
    """TX ``transmit()`` 后将参考频域网格写入对应 session。成功返回 True。"""
    with _registry_lock:
        session = _find_session(system)
        if session is None:
            _log(f"set_last_x_rg failed: system id={id(system)} not in registry")
            return False
        session.last_x_rg = x_rg
        return True


def get_last_x_rg(system: System):
    """读取与 ``system`` 关联 session 中的 ``last_x_rg``；未写入时返回 None。

    身份/key 未命中或本 session 为空时，若 registry 中恰好只有一个非空
    ``last_x_rg``，则回退返回该值（并 stderr 警告一次）。
    """
    global _warned_fallback_x_rg
    with _registry_lock:
        session = _find_session(system)
        if session is not None and session.last_x_rg is not None:
            return session.last_x_rg
        fallback = _single_nonempty("last_x_rg")
        if fallback is not None:
            if not _warned_fallback_x_rg:
                _warned_fallback_x_rg = True
                _log(
                    "get_last_x_rg: identity/key miss or empty; using sole "
                    f"non-empty last_x_rg in registry (n={len(_SYSTEM_REGISTRY)})"
                )
            return fallback
        return None


def set_last_x_time(system: System, x_time) -> bool:
    """TX ``transmit()`` 后将参考时域波形写入对应 session（未乘 tx_amp）。成功返回 True。"""
    with _registry_lock:
        session = _find_session(system)
        if session is None:
            _log(f"set_last_x_time failed: system id={id(system)} not in registry")
            return False
        session.last_x_time = x_time
        return True


def get_last_x_time(system: System):
    """读取与 ``system`` 关联 session 中的 ``last_x_time``；未写入时返回 None。

    身份/key 未命中或本 session 为空时，若 registry 中恰好只有一个非空
    ``last_x_time``，则回退返回该值（并 stderr 警告一次）。
    """
    global _warned_fallback_x_time
    with _registry_lock:
        session = _find_session(system)
        if session is not None and session.last_x_time is not None:
            return session.last_x_time
        fallback = _single_nonempty("last_x_time")
        if fallback is not None:
            if not _warned_fallback_x_time:
                _warned_fallback_x_time = True
                _log(
                    "get_last_x_time: identity/key miss or empty; using sole "
                    f"non-empty last_x_time in registry (n={len(_SYSTEM_REGISTRY)})"
                )
            return fallback
        return None


def set_last_x_rg_by_key(cache_key: tuple, x_rg) -> bool:
    """按 cache_key 写入 last_x_rg。"""
    with _registry_lock:
        session = _SYSTEM_REGISTRY.get(cache_key)
        if session is None:
            return False
        session.last_x_rg = x_rg
        return True


def set_last_x_time_by_key(cache_key: tuple, x_time) -> bool:
    """按 cache_key 写入 last_x_time。"""
    with _registry_lock:
        session = _SYSTEM_REGISTRY.get(cache_key)
        if session is None:
            return False
        session.last_x_time = x_time
        return True


def get_last_x_rg_by_key(cache_key: tuple):
    """按 cache_key 读取 last_x_rg。"""
    with _registry_lock:
        session = _SYSTEM_REGISTRY.get(cache_key)
        return None if session is None else session.last_x_rg


def get_last_x_time_by_key(cache_key: tuple):
    """按 cache_key 读取 last_x_time。"""
    with _registry_lock:
        session = _SYSTEM_REGISTRY.get(cache_key)
        return None if session is None else session.last_x_time


# ---------------------------------------------------------------------------
# 5. System 内部构建
# ---------------------------------------------------------------------------


def _build_system(
    config_file: str,
    *,
    device: str,
    seed: int,
    ofdm_overrides: _OfdmOverrides,
) -> System:
    """加载配置并构建 ``System``。

    流程：设随机种子与 Sionna 设备 → ``load_config`` → 可选 GRC OFDM 覆盖
    → ``System`` / ``System.from_dict``。
    """
    set_random_seed(seed)
    sionna.phy.config.device = device
    torch.set_num_threads(1)

    if ofdm_overrides is None:
        return System(config_file, device=str(device))

    raw = load_config(config_file)
    raw = apply_grc_ofdm_overrides(
        raw,
        num_symbols=int(ofdm_overrides["num_symbols"]),
        fft_size=int(ofdm_overrides["fft_size"]),
        subcarrier_spacing=float(ofdm_overrides["subcarrier_spacing"]),
        cp_len=int(ofdm_overrides["cyclic_prefix_length"]),
    )
    return System.from_dict(raw, device=str(device), config_file=str(config_file))


# ---------------------------------------------------------------------------
# 6. 仅读配置：SystemParams 与 resolve_*
# ---------------------------------------------------------------------------


def load_system_params(config_file: str) -> SystemParams:
    """``load_config`` + ``SystemParams.from_dict``；按 config_file 进程内缓存。

    仅解析结构化参数，不构建 ``System`` / ``SystemComponents``。
    """
    key = str(config_file)
    with _params_lock:
        hit = _PARAMS_CACHE.get(key)
        if hit is not None:
            return hit
    params = SystemParams.from_dict(load_config(config_file))
    with _params_lock:
        # 双检：解析期间另一线程可能已写入
        hit = _PARAMS_CACHE.get(key)
        if hit is not None:
            return hit
        _PARAMS_CACHE[key] = params
    return params


def resolve_ofdm_samp_rate(config_file: str) -> int:
    """从 TOML [ofdm] 解析采样率 ``fft_size * subcarrier_spacing``。"""
    ofdm = load_system_params(config_file).ofdm
    if ofdm is None:
        raise ValueError(f"TOML 缺少 [ofdm]: {config_file}")
    return int(ofdm.samp_rate)


def resolve_ofdm_burst_len(config_file: str) -> int:
    """从 TOML [ofdm] 解析载荷突发样点数 ``num_symbols * (fft_size + cp)``。

    不含 SC 前导；前导长度见 ``ofdm_sc_preamble.preamble_len``。
    """
    ofdm = load_system_params(config_file).ofdm
    if ofdm is None:
        raise ValueError(f"TOML 缺少 [ofdm]: {config_file}")
    return int(ofdm.num_symbols * (ofdm.fft_size + ofdm.cyclic_prefix_length))


def resolve_ofdm_fft_cp(config_file: str) -> tuple[int, int]:
    """从 TOML [ofdm] 解析 ``(fft_size, cyclic_prefix_length)``。"""
    ofdm = load_system_params(config_file).ofdm
    if ofdm is None:
        raise ValueError(f"TOML 缺少 [ofdm]: {config_file}")
    return int(ofdm.fft_size), int(ofdm.cyclic_prefix_length)


def resolve_ofdm_num_symbols(config_file: str) -> int:
    """从 TOML [ofdm] 解析 OFDM 载荷符号数 ``num_symbols``。"""
    ofdm = load_system_params(config_file).ofdm
    if ofdm is None:
        raise ValueError(f"TOML 缺少 [ofdm]: {config_file}")
    return int(ofdm.num_symbols)


def resolve_preamble_len(config_file: str) -> int:
    """SC 前导时域样点数 ``2 * (fft_size + cp)``（不含载荷）。"""
    fft_size, cp_len = resolve_ofdm_fft_cp(config_file)
    return 2 * (fft_size + cp_len)


def resolve_gr_carrier_tuples(
    config_file: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """解析 GR ``occupied_carriers`` / ``pilot_carriers``（与 ofdm_sc_preamble 同构）。

    ``n_carriers = fft_size - 2``（全带除 DC）；导频放在占用带外缘，供
    ``digital.ofdm_txrx._make_sync_word*`` 使用。
    """
    fft_size, _cp = resolve_ofdm_fft_cp(config_file)
    n = int(fft_size) - 2
    if n < 2 or n % 2:
        raise ValueError(f"fft_size={fft_size} 无法构成偶长度 occupied 集合")
    occupied = (
        tuple(range(-n // 2, 0)) + tuple(range(1, n // 2 + 1)),
    )
    pilots = (
        tuple(range(-n // 2 - 2, -n // 2))
        + tuple(range(n // 2 + 1, n // 2 + 3)),
    )
    return occupied, pilots


def preamble_time_from_sync_words(
    fft_size: int,
    cp_len: int,
    sync_word1: np.ndarray,
    sync_word2: np.ndarray,
) -> np.ndarray:
    """频域 sync word（fftshift）→ 时域前导 ``sync1|sync2``（含 CP）。"""
    from isac_imp.ofdm_sc_preamble import _symbol_time  # noqa: PLC0415

    s1 = _symbol_time(sync_word1, cp_len)
    s2 = _symbol_time(sync_word2, cp_len)
    return np.concatenate([s1, s2]).astype(np.complex64, copy=False)


def resolve_gr_preamble_time(config_file: str) -> np.ndarray:
    """用 GR ``ofdm_txrx`` sync word 生成与 RX 链一致的时域前导。"""
    from gnuradio import digital  # noqa: PLC0415

    fft_size, cp_len = resolve_ofdm_fft_cp(config_file)
    occupied, pilots = resolve_gr_carrier_tuples(config_file)
    sw1 = digital.ofdm_txrx._make_sync_word1(fft_size, occupied, pilots)
    sw2 = digital.ofdm_txrx._make_sync_word2(fft_size, occupied, pilots)
    return preamble_time_from_sync_words(fft_size, cp_len, sw1, sw2)


def resolve_source_cache_file(config_file: str) -> str:
    """从 TOML ``source.cache_file`` 解析发射缓存目录路径。"""
    src = load_system_params(config_file).source
    if src is None or not src.cache_file:
        raise ValueError(f"TOML 未配置 source.cache_file: {config_file}")
    return str(src.cache_file)


def resolve_dd_output_vlen(config_file: str) -> int:
    """解析 DD 谱图/RX 输出向量长度（时延维列数）。

    有 ``[dd_spectrum_roi]`` 时返回与 ``DelayDopplerRoi.delay_bin_count(monostatic)``
    等价的纯公式：``max(1, int(max_range_m / (c / (2 B))) + 1)``，
    ``B = fft_size * subcarrier_spacing``；无 ROI 时回退 ``fft_size``。

    不构建 ``System``。
    """
    params = load_system_params(config_file)
    ofdm = params.ofdm
    if ofdm is None:
        raise ValueError(f"TOML 缺少 [ofdm]: {config_file}")
    roi = params.dd_spectrum_roi
    if roi is None:
        return int(ofdm.fft_size)
    bandwidth = float(ofdm.fft_size) * float(ofdm.subcarrier_spacing)
    range_res = _C / (2.0 * bandwidth)  # monostatic: c * (1/B) / 2
    return max(1, int(roi.max_range_m / range_res) + 1)


# ---------------------------------------------------------------------------
# 7. System 公开入口（create_system）
# ---------------------------------------------------------------------------


def create_system(
    config_file: str,
    *,
    device: str = "cpu",
    seed: int = 42,
    ofdm_overrides: _OfdmOverrides = None,
    use_cache: bool = True,
) -> System:
    """构建或与 registry 共享 ``System`` 实例（对齐 run_sensing_monostatic 配置路径）。

    相同 ``cache_key`` 下 TX/RX epy_block 得到同一 ``System`` 对象。
    构建在锁外进行；注册前二次检查，避免多线程双实例覆盖。

    参数:
    ----------
        - config_file: TOML 相对路径（经 ``load_config`` 解析）
        - device: Sionna/Torch 计算设备
        - seed: 随机种子
        - ofdm_overrides: GRC OFDM 四参数覆盖；None 表示不覆盖 TOML [ofdm]
        - use_cache: True 时命中 registry 直接返回已有实例

    返回:
    ----------
        - 新建或缓存的 ``System`` 实例
    """
    cache_key = make_cache_key(config_file, device, seed, ofdm_overrides)

    if use_cache:
        with _registry_lock:
            hit = _SYSTEM_REGISTRY.get(cache_key)
            if hit is not None:
                return hit.system

    system = _build_system(
        config_file,
        device=device,
        seed=seed,
        ofdm_overrides=ofdm_overrides,
    )

    if use_cache:
        with _registry_lock:
            # 双检：构建期间另一线程可能已注册
            hit = _SYSTEM_REGISTRY.get(cache_key)
            if hit is not None:
                return hit.system
            _SYSTEM_REGISTRY[cache_key] = SystemSession(system=system, cache_key=cache_key)
            _SYSTEM_ID_TO_KEY[id(system)] = cache_key

    return system
