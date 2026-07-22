"""协作单基地感知评估：空场景背景对消 + 可选短抽头 SIC → 零多普勒距离谱。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/simulation/sensing/rt/run_sensing_cooperative_monostatic.py

默认配置
--------
``config/simulation/sensing/sensing_cooperative_monostatic.toml``：
共址 TX/RX（``bs1``，``rx_position_offset`` 略偏移以避免完全共址）+ 1 金属球目标。

与 ``run_sensing_monostatic.py`` 的区别
--------------------------------------
- 本脚本不走 ``system.sensing`` / MUSIC，而是复现 GNU Radio 零多普勒距离谱流程
  （参考 ``gnuradio/tests/data_collection/usrp_ofdm_echotimer_dd_data_collection_test``）。
- 通过 **空场景背景对消** 抑制墙面等静态散射与强直射；再可选 **短抽头 SIC** 消除
  TX–RX 残余自干扰（共址间距由 ``rx_position_offset`` 决定）。

感知链路
--------
1. ``transmit`` 得到 ``x_rg`` / ``x_time``（背景与有目标共用同一发射波形）。
2. 临时移出 RT 目标 → 无噪 LS 估计背景 ``H_bg`` → 恢复目标。
3. 有目标 ``channel`` + LS 得 ``H``（含 AWGN）。
4. ``H_clean = H - H_bg``（频域相减，消除静态杂波）。
5. 可选 ``cancel_short_tap_si``：IFFT 域减去前 ``num_taps`` 近零时延抽头。
6. 逐 OFDM 符号：子载波 ``fftshift`` → 零填充 × ``ZEROPADDING_FAC`` → Blackman-Harris
   → IFFT → |·|² → 跨符号非相干累加 → dB。
7. 按 ``dd_spectrum_roi.max_range_m`` 裁剪 ROI，``argmax`` 得估计距离，与
   ``rx_target_tx_geometric`` 几何真值对比。

设计约束
--------
- 配置保持 ``path_solver.los = true``：直射由数字域 SIC/背景对消处理，不关闭 LOS。
- 不使用 MTI：静止目标零多普勒，MTI 会将其与静态杂波一并滤除。

CLI 开关
--------
- ``--no-background-cancel``：跳过空场景 ``H_bg`` 估计，仅做有目标 LS。
- ``--no-sic``：背景对消后不再做短抽头自干扰消除。
- ``--sic-taps N``：手动指定 SIC 抽头数；缺省由 TX–RX 间距与 ``delay_resolution`` 自动估计。

输出目录
--------
``out/sensing_cooperative_monostatic/``：
  - ``sensing_cooperative_monostatic_scene.png`` — RT 场景渲染
  - ``sensing_cooperative_monostatic_range_spectrum.png`` — ROI 距离谱（真值/估计竖线）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import speed_of_light as c
from scipy.signal.windows import blackmanharris

from isac import PROJECT_ROOT
from isac.sensing.clutter import (
    cancel_short_tap_si,
    remove_targets_from_scene,
    restore_targets_to_scene,
    subtract_background_cfr,
    suggest_si_num_taps,
)
from isac.system import System
from isac.utils import set_random_seed

# 本脚本专用输出目录（场景 PNG + 距离谱 PNG）
SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "sensing_cooperative_monostatic"
# 距离向零填充倍数，与 GRC echotimer 距离谱一致，提高距离栅格分辨率
ZEROPADDING_FAC = 2


def argument_parser() -> argparse.Namespace:
    """解析 CLI。"""
    parser = argparse.ArgumentParser(
        description="ISAC 协作单基地距离谱（空场景背景对消 + 可选 SIC）"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_cooperative_monostatic.toml",
        help="配置文件路径（相对 config/ 或绝对路径）",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="计算设备",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（ZC / AWGN 等）",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="信道施加域：frequency 或 time",
    )
    parser.add_argument(
        "--no-background-cancel",
        action="store_true",
        help="关闭空场景背景对消（默认开启）",
    )
    parser.add_argument(
        "--no-sic",
        action="store_true",
        help="关闭短抽头 SIC（默认在背景对消后仍开启）",
    )
    parser.add_argument(
        "--sic-taps",
        type=int,
        default=None,
        help="SIC 近零时延抽头数；缺省由 TX–RX 间距自动估计",
    )

    return parser.parse_args()


def range_axis_from_ofdm(
    *,
    fft_size: int,
    subcarrier_spacing: float,
    zeropadding_fac: int = ZEROPADDING_FAC,
) -> tuple[np.ndarray, float, float]:
    """构造与 GRC 一致的距离轴 ``(range_m, range_bin_step, R_max)``。

    采样率 ``f_s = N_fft · Δf``，无模糊最大距离 ``R_max = c·N_fft / (2·f_s)``；
    零填充后栅格数 ``vlen = N_fft · zeropadding_fac``，步进 ``ΔR = R_max / vlen``。
    """
    samp_rate = float(fft_size) * float(subcarrier_spacing)
    r_max = c / (2.0 * samp_rate) * float(fft_size)
    vlen = int(fft_size) * int(zeropadding_fac)
    range_bin_step = r_max / float(vlen)
    range_m = np.arange(vlen, dtype=np.float64) * range_bin_step
    return range_m, range_bin_step, r_max


def compute_range_profile_db_from_h_freq(
    h_freq: torch.Tensor | np.ndarray,
    *,
    zeropadding_fac: int = ZEROPADDING_FAC,
    n_db: float = 10.0,
    eps: float = 1e-30,
) -> np.ndarray:
    """LS / 清理后的频域 CFR → 零多普勒距离谱（dB）。

    参数:
        h_freq: 形状 ``(S, F)``，``S`` 为 OFDM 符号数，``F`` 为 FFT 大小。
        zeropadding_fac: 时延维零填充倍数。
        n_db: dB 换算底数（10 → 功率 dB，20 → 幅度 dB）。
        eps: ``log10`` 下限，避免 ``log(0)``。

    返回:
        长度 ``F · zeropadding_fac`` 的一维距离功率谱（dB），符号间非相干积分。

    时延约定与短抽头 SIC / DD 谱一致：子载波维 ``fftshift`` 后再 ``ifft``，
    使正时延落在低编号抽头（近距），而非裸 ``fft`` 卷绕到远端。
    """
    h = (
        h_freq.detach().cpu().numpy()
        if isinstance(h_freq, torch.Tensor)
        else np.asarray(h_freq)
    )
    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"h_freq 须为 2D (S, F)，收到 shape={h.shape}")
    n_sym, fft_size = h.shape
    vlen_out = fft_size * int(zeropadding_fac)
    bh = np.asarray(blackmanharris(vlen_out), dtype=np.float64)

    power_sum = np.zeros(vlen_out, dtype=np.float64)
    for s in range(n_sym):
        h_pad = np.zeros(vlen_out, dtype=np.complex128)
        # fftshift：DC/零时延居中 → 正时延映射到低编号 IFFT 抽头（近距在左）
        h_pad[:fft_size] = np.fft.fftshift(h[s].astype(np.complex128, copy=False))
        rd = np.fft.ifft(h_pad * bh)  # Blackman-Harris 抑制时延旁瓣
        power_sum += np.abs(rd) ** 2  # 符号间非相干积分

    return (n_db * np.log10(np.maximum(power_sum, eps))).astype(np.float64)


def slice_range_roi(
    profile_db: np.ndarray,
    range_m: np.ndarray,
    *,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """裁剪 ``[0, max_range_m]`` 距离 ROI，返回 ``(range_roi, profile_roi, n_bins)``。"""
    if max_range_m <= 0:
        raise ValueError(f"max_range_m 须为正，收到 {max_range_m}")
    step = float(range_m[1] - range_m[0]) if range_m.size > 1 else float(range_m[0])
    n_bins = int(round(max_range_m / step)) + 1
    n_bins = max(1, min(n_bins, profile_db.size))
    return range_m[:n_bins], profile_db[:n_bins], n_bins


def _tx_rx_separation_m(rt_simulator) -> float:
    """第一对 TX/RX 天线中心的欧氏间距（米）。

    用于自动估计 SIC 抽头数：共址偏移 ``rx_position_offset`` 决定直射时延落在
    前若干 IFFT 抽头内的宽度。
    """
    tx_pos = next(iter(rt_simulator.tx_states.values()))[0]
    rx_pos = next(iter(rt_simulator.rx_states.values()))[0]
    return float(np.linalg.norm(np.asarray(rx_pos) - np.asarray(tx_pos)))


def estimate_h_background(
    comps,
    x_rg: torch.Tensor,
    x_time: torch.Tensor,
    *,
    domain: str,
) -> torch.Tensor:
    """临时移出 RT 目标，估计无噪背景 LS 信道 ``H_bg``，并恢复目标位姿。

    背景估计不加 AWGN（``snr_db=None``），以便 ``H - H_bg`` 主要保留目标回波与
    残余 SI，而非差分噪声放大。``finally`` 块保证目标一定被还原，避免污染后续仿真。
    """
    rt = comps.rt_simulator
    snapshots = remove_targets_from_scene(rt)
    try:
        rt.paths(update=True)
        y_bg = comps.channel(x_rg, x_time, domain=domain, snr_db=None)
        h_bg = comps.ls_channel_estimator(x_rg, y_bg)
    finally:
        restore_targets_to_scene(rt, snapshots)
        rt.paths(update=True)
    return h_bg


def save_range_spectrum_plot(
    range_roi_m: np.ndarray,
    profile_roi_db: np.ndarray,
    file_name: Path,
    *,
    est_range_m: float | None = None,
    true_range_m: float | None = None,
) -> None:
    """保存 ROI 零多普勒距离谱 PNG（可选标注几何真值与 ``argmax`` 估计）。"""
    file_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        range_roi_m, profile_roi_db, color="C0", linewidth=1.2, label="Range profile"
    )
    if true_range_m is not None:
        ax.axvline(
            true_range_m,
            color="C2",
            linestyle="--",
            linewidth=1.0,
            label=f"True {true_range_m:.2f} m",
        )
    if est_range_m is not None:
        ax.axvline(
            est_range_m,
            color="C3",
            linestyle=":",
            linewidth=1.2,
            label=f"Est {est_range_m:.2f} m",
        )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Zero-Doppler Range Spectrum (background cancel + SIC)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    plt.close(fig)
    print(f"距离谱已保存: {file_name}")


def main() -> None:
    """端到端：背景对消 → 可选 SIC → 距离谱峰检测与误差打印。"""
    # --- 参数与随机种子 ---
    args = argument_parser()
    set_random_seed(args.seed)

    # --- 构建系统（RT 场景 + OFDM + 信道 + LS 估计器）---
    system = System(args.config_file, device=args.device)
    comps = system.components
    ofdm = system.params.ofdm
    if ofdm is None:
        raise RuntimeError("配置须包含 [ofdm]")

    # --- 渲染 RT 场景（便于核对目标/收发机几何）---
    comps.rt_simulator.render_to_file(
        filename=str(SCRIPT_OUT_DIR / "sensing_cooperative_monostatic_scene.png")
    )

    # --- 发射：背景与有目标两次信道共用同一 x_rg / x_time ---
    _, x_rg, x_time = system.transmit()
    # 打印时延分辨率等感知参数，供 SIC 抽头自动估计使用
    comps.sensing_performance()

    # --- 空场景背景 LS（无 AWGN）：H_bg ≈ 静态杂波 + 强直射 ---
    h_bg = None
    if not args.no_background_cancel:
        print("空场景背景对消：估计 H_bg …")
        h_bg = estimate_h_background(
            comps, x_rg, x_time, domain=args.domain
        )

    # --- 有目标信道 + LS：H 含目标回波与 AWGN ---
    y_rg = comps.channel(
        x_rg, x_time, domain=args.domain, snr_db=system.params.channel.snr_db
    )
    h_freq = comps.ls_channel_estimator(x_rg, y_rg)
    # 对消前谱，仅用于诊断 0 m bin 的直射/杂波强度
    profile_raw = compute_range_profile_db_from_h_freq(h_freq)

    # --- 频域背景对消：H_work = H - H_bg ---
    h_work = h_freq
    if h_bg is not None:
        h_work = subtract_background_cfr(h_freq, h_bg)
        profile_bg = compute_range_profile_db_from_h_freq(h_work)
        print(
            f"背景对消 0 m bin: 前={profile_raw[0]:.2f} dB, "
            f"后={profile_bg[0]:.2f} dB, "
            f"抑制={profile_raw[0] - profile_bg[0]:.2f} dB"
        )

    # --- 可选短抽头 SIC：消除 TX–RX 残余近零时延自干扰 ---
    if not args.no_sic:
        delay_res = float(comps.sensing_performance.delay_resolution)
        if args.sic_taps is not None:
            sic_taps = int(args.sic_taps)
        else:
            sep_m = _tx_rx_separation_m(comps.rt_simulator)
            sic_taps = suggest_si_num_taps(sep_m, delay_resolution_s=delay_res)
            print(
                f"SIC 自动抽头: TX–RX 间距={sep_m:.4f} m → num_taps={sic_taps} "
                f"(Δτ={delay_res * 1e9:.2f} ns)"
            )
        profile_pre_sic = compute_range_profile_db_from_h_freq(h_work)
        h_work = cancel_short_tap_si(h_work, num_taps=sic_taps)
        profile_post_sic = compute_range_profile_db_from_h_freq(h_work)
        print(
            f"SIC 0 m bin: 前={profile_pre_sic[0]:.2f} dB, "
            f"后={profile_post_sic[0]:.2f} dB, "
            f"抑制={profile_pre_sic[0] - profile_post_sic[0]:.2f} dB (taps={sic_taps})"
        )

    # --- 最终距离谱与物理距离轴 ---
    profile_db = compute_range_profile_db_from_h_freq(h_work)
    range_m, _, _ = range_axis_from_ofdm(
        fft_size=int(ofdm.fft_size),
        subcarrier_spacing=float(ofdm.subcarrier_spacing),
    )

    # ROI 上限优先读 TOML [dd_spectrum_roi]，否则默认 30 m
    max_range_m = 30.0
    if system.params.dd_spectrum_roi is not None:
        max_range_m = float(system.params.dd_spectrum_roi.max_range_m)
    range_roi, profile_roi, _ = slice_range_roi(
        profile_db, range_m, max_range_m=max_range_m
    )

    # --- 几何真值：单基地 RX–目标–TX 折叠路径长 ---
    geom = comps.rt_simulator.rx_target_tx_geometric
    true_range_m = float(geom.range_tensor.reshape(-1)[0].item())
    true_bin = int(round(true_range_m / float(range_roi[1] - range_roi[0])))
    true_bin = max(0, min(true_bin, len(profile_roi) - 1))
    print(
        f"真值邻域: r≈{range_roi[true_bin]:.2f} m, "
        f"幅度={profile_roi[true_bin]:.2f} dB (bin={true_bin})"
    )

    # --- 简单峰值测距：ROI 内全局 argmax（零多普勒单目标场景）---
    peak_idx = int(np.argmax(profile_roi))
    est_range_m = float(range_roi[peak_idx])
    range_err_m = est_range_m - true_range_m

    save_range_spectrum_plot(
        range_roi,
        profile_roi,
        SCRIPT_OUT_DIR / "sensing_cooperative_monostatic_range_spectrum.png",
        est_range_m=est_range_m,
        true_range_m=true_range_m,
    )

    print(
        f"协作感知单基地 — 距离谱峰: bin={peak_idx}, "
        f"估计={est_range_m:.2f} m, 真值={true_range_m:.2f} m, "
        f"误差={range_err_m:.2f} m, |误差|={abs(range_err_m):.2f} m"
    )


if __name__ == "__main__":
    main()
