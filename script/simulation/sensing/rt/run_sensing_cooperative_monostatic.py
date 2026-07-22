"""协作单基地感知评估：LS 后走 GRC 对齐的零多普勒距离谱链路。

参考流图：``gnuradio/tests/data_collection/usrp_ofdm_echotimer_dd_data_collection_test``。

感知链路（相对 GRC）：
  LS ``h_freq ≈ Y/X`` 对应 ``radar_ofdm_divide_vcvc``；
  其后：零填充 ×``ZEROPADDING_FAC`` → Blackman-Harris → FFT → |·|²
  → 全 CPI 符号非相干积分 → ``10·log10``（对应 ``nlog10_ff``）；
  ROI ``[0, max_range_m]`` 内 ``argmax`` 测距，并与 ``rx_target_tx_geometric`` 对比。

输出：
  - ``out/.../sensing_cooperative_monostatic_scene.png``
  - ``out/.../sensing_cooperative_monostatic_range_spectrum.png``
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
from isac.system import System
from isac.utils import set_random_seed

SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "sensing_cooperative_monostatic"
# 与 GRC 变量 zeropadding_fac 一致：距离 FFT 长度 = fft_size * ZEROPADDING_FAC
ZEROPADDING_FAC = 2


def argument_parser() -> argparse.Namespace:
    """解析 CLI：配置路径、设备、随机种子、信道施加域。"""
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 协作感知单基地距离谱评估（GRC 对齐）"
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

    return parser.parse_args()


def range_axis_from_ofdm(
    *,
    fft_size: int,
    subcarrier_spacing: float,
    zeropadding_fac: int = ZEROPADDING_FAC,
) -> tuple[np.ndarray, float, float]:
    """构造与 GRC 一致的距离轴。

    GRC 标定：
      ``samp_rate = fft_len * subcarrier_spacing``
      ``R_max = c/2/samp_rate * fft_len``（= ``c / (2 Δf)``）
      ``range_bin_step = R_max / (fft_len * zeropadding_fac)``

    返回:
        ``(range_m, range_bin_step, R_max)``，``range_m[k] = k * range_bin_step``。
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
    """由 LS 频域信道计算单 CPI 距离谱（dB）。

    对应 GRC 块序（divide 之后）：
      每符号 ``h`` → 高索引零填充到 ``F·zp`` → Blackman-Harris
      → ``fft``（前向）→ ``|·|²`` → 对全部符号非相干求和 → ``n_db * log10``。

    参数:
        h_freq: LS 估计，形状 ``(S, F)``（多余维会 squeeze）。
        zeropadding_fac: 距离维零填充倍数（GRC ``zeropadding_fac``）。
        n_db: 对数系数，GRC ``nlog10_ff(n=10)`` 对应功率 dB。
        eps: 避免 ``log10(0)``。

    返回:
        形状 ``(F * zeropadding_fac,)`` 的 float64 距离谱（dB）。
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
    # 与 GRC fft_vxx + window.blackmanharris(fft_len * zeropadding_fac) 一致
    bh = np.asarray(blackmanharris(vlen_out), dtype=np.float64)

    power_sum = np.zeros(vlen_out, dtype=np.float64)
    for s in range(n_sym):
        h_pad = np.zeros(vlen_out, dtype=np.complex128)
        # ofdm_divide 零填充：仅占用前 fft_len 点，后半段为 0（提高距离采样密度）
        h_pad[:fft_size] = h[s].astype(np.complex128, copy=False)
        rd = np.fft.fft(h_pad * bh)
        power_sum += np.abs(rd) ** 2  # 非相干积累（GRC integrate_ff）

    return (n_db * np.log10(np.maximum(power_sum, eps))).astype(np.float64)


def slice_range_roi(
    profile_db: np.ndarray,
    range_m: np.ndarray,
    *,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """按物理上界裁剪距离谱 ROI ``[0, max_range_m]``。

    ``n_bins = round(max_range_m / ΔR) + 1``（含 0 m bin），并钳位到全谱长度。
    默认上界来自 TOML ``[dd_spectrum_roi].max_range_m``（与 GRC ``range_roi`` 上限一致）。

    返回:
        ``(range_roi_m, profile_roi_db, n_bins)``。
    """
    if max_range_m <= 0:
        raise ValueError(f"max_range_m 须为正，收到 {max_range_m}")
    step = float(range_m[1] - range_m[0]) if range_m.size > 1 else float(range_m[0])
    n_bins = int(round(max_range_m / step)) + 1
    n_bins = max(1, min(n_bins, profile_db.size))
    return range_m[:n_bins], profile_db[:n_bins], n_bins


def save_range_spectrum_plot(
    range_roi_m: np.ndarray,
    profile_roi_db: np.ndarray,
    file_name: Path,
    *,
    est_range_m: float | None = None,
    true_range_m: float | None = None,
) -> None:
    """将 ROI 距离谱保存为 PNG；可选叠加真值/估计竖线。"""
    file_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range_roi_m, profile_roi_db, color="C0", linewidth=1.2, label="Range profile")
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
    ax.set_title("Zero-Doppler Range Spectrum (GRC-aligned)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    plt.close(fig)
    print(f"距离谱已保存: {file_name}")


def main() -> None:
    # --- CLI / 系统 ---
    args = argument_parser()
    set_random_seed(args.seed)

    system = System(args.config_file, device=args.device)
    comps = system.components
    ofdm = system.params.ofdm
    if ofdm is None:
        raise RuntimeError("配置须包含 [ofdm]")

    # --- 场景渲染 ---
    comps.rt_simulator.render_to_file(
        filename=str(SCRIPT_OUT_DIR / "sensing_cooperative_monostatic_scene.png")
    )

    # --- 发射与信道 ---
    _, x_rg, x_time = system.transmit()
    y_rg = comps.channel(
        x_rg, x_time, domain=args.domain, snr_db=system.params.channel.snr_db
    )

    # --- LS 信道估计（对应 GRC ofdm_divide 的 Y/X）---
    comps.sensing_performance()
    h_freq = comps.ls_channel_estimator(x_rg, y_rg)

    # --- 距离谱（GRC：zeropad → BH → FFT → |·|² → integrate → nlog10）---
    profile_db = compute_range_profile_db_from_h_freq(h_freq)
    range_m, _, _ = range_axis_from_ofdm(
        fft_size=int(ofdm.fft_size),
        subcarrier_spacing=float(ofdm.subcarrier_spacing),
    )

    # ROI：优先读 [dd_spectrum_roi].max_range_m，缺省 30 m（与 GRC range_roi 一致）
    max_range_m = 30.0
    if system.params.dd_spectrum_roi is not None:
        max_range_m = float(system.params.dd_spectrum_roi.max_range_m)
    range_roi, profile_roi, _ = slice_range_roi(
        profile_db, range_m, max_range_m=max_range_m
    )

    # --- 谱峰测距与几何真值对比 ---
    peak_idx = int(np.argmax(profile_roi))
    est_range_m = float(range_roi[peak_idx])

    geom = comps.rt_simulator.rx_target_tx_geometric
    true_range_m = float(geom.range_tensor.reshape(-1)[0].item())
    range_err_m = est_range_m - true_range_m
    range_rmse_m = abs(range_err_m)  # 单峰单真值时 |误差| 即 RMSE

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
        f"误差={range_err_m:.2f} m, |误差|/RMSE={range_rmse_m:.2f} m"
    )


if __name__ == "__main__":
    main()
