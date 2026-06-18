"""协同 ISAC 感知评估：2 发射机 + 1 接收机，按 TX 路径分离 CFR 后各自算谱与 MUSIC。

管线概要
--------
1. 从 RT ``Paths.cfr`` 按发射机切片得到各 TX→RX 频域信道 ``h_tx``（路径分离，非叠加谱剥峰）。
2. 对各 ``h_tx`` 施加 MTI，分别计算时延–多普勒谱并出图（每 TX 一张 + 多 TX 总览）。
3. 各 TX 谱线上 MUSIC（``sens_mode`` 由几何单基地/双基地决定），记录距离/速度 RMSE。
4. 在 ``z=0`` 且单基地 TX 与 RX 共址时，融合单基地斜距与双基地折叠路径长求 ``(x,y)``。
"""

from __future__ import annotations

import argparse
import re

import torch

from isac import PROJECT_ROOT
from isac.sensing.localization import (
    localize_xy_z0_colocated_tx_mono_bistatic,
    position_rmse_xy,
)
from isac.system import System
from isac.utils import match_peaks_and_compute_radial_rmse, set_random_seed


def _slug_tx_name(tx_name: str) -> str:
    """文件名安全片段。"""
    return re.sub(r"[^\w.-]+", "_", tx_name).strip("_") or "tx"


def argument_parser() -> argparse.Namespace:
    """解析协作感知评估所需的设备、随机种子、信道域与速度反演模型。"""
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 协作感知评估 (2 TX + 1 RX)"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="sensing_cooperative.toml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cpu"],
        help="计算设备类型",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="保留 CLI；路径分离使用 RT CFR，与 domain 无关",
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="range_velocity",
        choices=["delay_doppler", "range_velocity"],
        help="谱图与 MUSIC 日志 metric：delay_doppler 用时延 (ns) / 多普勒 (Hz)；range_velocity 用距离 (m) / 速度 (m/s)",
    )

    return parser.parse_args()


def main() -> None:
    """构建系统、按 TX 分离信道并跑协作感知链。"""
    args = argument_parser()
    set_random_seed(args.seed)
    system = System(args)

    scene = system.components.rt_scene
    channel = system.components.channel
    dd = system.components.delay_doppler_spectrum

    script_out_dir = PROJECT_ROOT / "out" / "sensing_cooperative"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    sensing_perf = system.components.sensing_performance
    sensing_perf.display_performance()

    scene.render_to_file(
        filename=script_out_dir / "sensing_cooperative_scene.png"
    )

    device = torch.device(args.device)
    h_per_tx = channel.cfr_per_tx(scene, device=device)

    geom = scene.rx_target_tx_geometric
    geom.display()

    if len(geom.rx_names) != 1:
        raise ValueError(f"协作感知假定 1 个接收机，收到 rx={geom.rx_names!r}")
    if len(geom.tx_names) != 2:
        raise ValueError(f"协作感知假定 2 个发射机，收到 tx={geom.tx_names!r}")
    if len(geom.target_names) != 1:
        raise ValueError(f"协作感知假定单目标，收到 target={geom.target_names!r}")

    missing = set(geom.tx_names) - set(h_per_tx.keys())
    if missing:
        raise ValueError(f"CFR 按 TX 分离缺少: {sorted(missing)!r}")

    rx_idx = 0
    target_idx = 0
    rx_name = geom.rx_names[rx_idx]
    target_name = geom.target_names[target_idx]

    tx_order = sorted(
        geom.tx_names,
        key=lambda name: bool(
            geom.type_tensor[rx_idx, target_idx, geom.tx_names.index(name)].item()
        ),
    )

    spectra_stack: list[torch.Tensor] = []
    panel_labels: list[str] = []

    mono_tx_name: str | None = None
    bistatic_tx_name: str | None = None
    est_r_mono: torch.Tensor | None = None
    est_r_bistatic: torch.Tensor | None = None
    true_r_mono: torch.Tensor | None = None
    true_r_bistatic: torch.Tensor | None = None

    for tx_name in tx_order:
        tx_idx = geom.tx_names.index(tx_name)
        is_bistatic = bool(geom.type_tensor[rx_idx, target_idx, tx_idx].item())
        sens_mode = "bistatic" if is_bistatic else "monostatic"
        path_label = "双基地" if is_bistatic else "单基地"

        h_tx = h_per_tx[tx_name]
        h_tx = system.components.moving_target_indication(h_tx, axis=-2)
        h_dd_tx = dd(h_tx)

        slug = _slug_tx_name(tx_name)
        dd.h_delay_doppler = h_dd_tx
        dd.visualize(
            offset=100,
            file_name=script_out_dir / f"sensing_cooperative_delay_doppler_{slug}.png",
            to_db=False,
            metric_mode=args.metric_mode,
            backend="matplotlib",
        )
        print(f"协同感知 — 已保存 {tx_name} 时延–多普勒谱")

        spectra_stack.append(h_dd_tx)
        panel_labels.append(tx_name)

        est_ranges, est_velocities, _ = system.components.music_estimator(
            spectrum_tensor=h_dd_tx,
            metric_mode=args.metric_mode,
            sens_mode=sens_mode,
            num_sources=1,
        )

        true_range = geom.range_tensor[rx_idx, target_idx, tx_idx]
        true_velocity = geom.vel_tensor[rx_idx, target_idx, tx_idx]

        if is_bistatic:
            distance_label = "LoS路径长度"
            velocity_label = "几何距离变化率"
        else:
            distance_label = "径向距离"
            velocity_label = "径向速度"

        _, _, est_range_m, est_velocity_mps, _ = match_peaks_and_compute_radial_rmse(
            est_ranges=est_ranges,
            est_velocities=est_velocities,
            true_ranges=true_range,
            true_velocities=true_velocity,
            label=(
                f"协同感知 — RX {rx_name} / TX {tx_name} "
                f"({path_label}, sens_mode={sens_mode})"
            ),
            distance_axis_label=distance_label,
            velocity_axis_label=velocity_label,
        )

        if is_bistatic:
            bistatic_tx_name = tx_name
            est_r_bistatic = est_range_m
            true_r_bistatic = true_range.reshape(-1)[0]
        else:
            mono_tx_name = tx_name
            est_r_mono = est_range_m
            true_r_mono = true_range.reshape(-1)[0]

    h_dd_all = torch.stack(spectra_stack, dim=0)
    dd.h_delay_doppler = h_dd_all
    dd.visualize(
        offset=100,
        file_name=script_out_dir / "sensing_cooperative_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
        panel_labels=panel_labels,
    )

    h_combined = sum(h_per_tx[name] for name in geom.tx_names)
    h_combined = system.components.moving_target_indication(h_combined, axis=-2)
    h_dd_combined = dd(h_combined)
    dd.h_delay_doppler = h_dd_combined
    dd.visualize(
        offset=100,
        file_name=script_out_dir / "sensing_cooperative_delay_doppler_combined.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
    )

    if mono_tx_name is None or bistatic_tx_name is None:
        print(
            "协同感知定位跳过：需要各 1 条单基地与双基地距离量测，"
            f"当前 mono_tx={mono_tx_name!r}, bistatic_tx={bistatic_tx_name!r}"
        )
        return

    rx_states = scene.rx_states
    tx_states = scene.tx_states
    true_pos = scene.targets_states[target_name]["pos"]
    true_xy = (float(true_pos[0]), float(true_pos[1]))
    y_hint = float(true_pos[1])

    mono_pos = rx_states[rx_name]["pos"]
    bi_pos = tx_states[bistatic_tx_name]["pos"]
    mono_tx_pos = tx_states[mono_tx_name]["pos"]

    z_target = 0.0
    r_m_true = float(true_r_mono.item())
    r_b_true = float(true_r_bistatic.item())
    r_m_est = float(est_r_mono.item())
    r_b_est = float(est_r_bistatic.item())

    xy_truth_ranges = localize_xy_z0_colocated_tx_mono_bistatic(
        mono_pos,
        bi_pos,
        r_mono_slant_m=r_m_true,
        r_bistatic_sum_m=r_b_true,
        z_target_m=z_target,
        y_hint=y_hint,
        tx_pos=mono_tx_pos,
    )
    rmse_truth_ranges = position_rmse_xy(xy_truth_ranges, true_xy)
    print(
        f"协同感知 — z=0 平面定位 (真值距离 r_m={r_m_true:.2f} m, r_b={r_b_true:.2f} m) — "
        f"估计 (x,y)=({xy_truth_ranges[0]:.2f}, {xy_truth_ranges[1]:.2f}) m, "
        f"真值 ({true_xy[0]:.2f}, {true_xy[1]:.2f}) m, 位置 RMSE: {rmse_truth_ranges:.2f} m"
    )

    r_leg2_est = r_b_est - r_m_est
    if r_leg2_est <= 0.0:
        print(
            f"协同感知 — MUSIC 定位跳过：双基地折叠路径长 {r_b_est:.2f} m "
            f"不大于单基地斜距 {r_m_est:.2f} m"
        )
        return

    try:
        xy_music = localize_xy_z0_colocated_tx_mono_bistatic(
            mono_pos,
            bi_pos,
            r_mono_slant_m=r_m_est,
            r_bistatic_sum_m=r_b_est,
            z_target_m=z_target,
            y_hint=y_hint,
            tx_pos=mono_tx_pos,
        )
    except ValueError as exc:
        print(f"协同感知 — MUSIC 定位跳过：{exc}")
        return

    rmse_music = position_rmse_xy(xy_music, true_xy)
    print(
        f"协同感知 — z=0 平面定位 (MUSIC 距离 r_m={r_m_est:.2f} m, r_b={r_b_est:.2f} m) — "
        f"估计 (x,y)=({xy_music[0]:.2f}, {xy_music[1]:.2f}) m, "
        f"真值 ({true_xy[0]:.2f}, {true_xy[1]:.2f}) m, 位置 RMSE: {rmse_music:.2f} m"
    )


if __name__ == "__main__":
    main()
