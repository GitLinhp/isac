"""协同 ISAC 感知评估：2 发射机 + 1 接收机，按 TX 路径分离 CFR 后各自算谱与 MUSIC。

管线概要
--------
1. 从 RT ``Paths.cfr`` 按发射机切片得到各 TX→RX 频域信道 ``h_tx``（路径分离，非叠加谱剥峰）。
2. 对各 ``h_tx`` 施加 MTI，分别计算时延–多普勒谱并出图（每 TX 一张 + 多 TX 总览）。
3. 各 TX 谱线上 MUSIC（``sens_mode`` 由几何单基地/双基地决定），记录距离/速度 RMSE。
4. 在 ``z=0`` 且单基地 TX 与 RX 共址时，融合单基地斜距与双基地折叠路径长求 ``(x,y)``。
"""

import argparse
import re

import numpy as np
import torch
from tabulate import tabulate

from isac import PROJECT_ROOT
from isac.sensing import match_peaks_and_compute_radial_rmse
from isac.sensing.localization import (
    localize_xy_z0_colocated_tx_mono_bistatic,
    position_rmse_xy,
)
from isac.system import System
from isac.utils import load_config, set_random_seed
from isac.channel import RTChannel


def _slug_tx_name(tx_name: str) -> str:
    """文件名安全片段。"""
    return re.sub(r"[^\w.-]+", "_", tx_name).strip("_") or "tx"


def argument_parser() -> argparse.Namespace:
    """解析协作感知评估所需的设备、随机种子、信道域与速度反演模型。"""
    parser = argparse.ArgumentParser(
        description="ISAC 系统仿真 — 协作感知评估 (2 TX + 1 RX)"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="simulation/sensing/sensing_cooperative.toml",
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
        default="rv",
        choices=["dd", "rv"],
        help="谱图与 MUSIC 日志 metric：dd 用时延 (ns) / 多普勒 (Hz)；rv 用距离 (m) / 速度 (m/s)",
    )

    return parser.parse_args()


def main() -> None:
    """构建系统、按 TX 分离信道并跑协作感知链。"""
    args = argument_parser()
    set_random_seed(args.seed)
    config = load_config(args.config_file)
    system = System(
        config=config,
        device=args.device,
    )

    scene = system.components.rt_simulator
    comps = system.components
    dd = comps.delay_doppler_spectrum

    script_out_dir = PROJECT_ROOT / "out" / "sensing_cooperative"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    comps.sensing_performance()

    scene.render_to_file(filename=script_out_dir / "sensing_cooperative_scene.png")

    # 发射参考波形（协作感知信道仍由 RT CFR 按 TX 分离）
    _, x_rg, x_time = system.transmit()

    device = torch.device(args.device)
    rt_channel = comps.channel
    if not isinstance(rt_channel, RTChannel):
        raise TypeError("协作感知脚本需要 channel.type='rt'")
    if scene is None or comps.rg is None:
        raise ValueError("协作感知需要 rt_simulator 与 OFDM 资源网格 rg")
    rg = comps.rg
    h_pairs = rt_channel.cfr_split(
        rg.num_ofdm_symbols,
        1 / rg.ofdm_symbol_duration,
    )

    geom = scene.rx_target_tx_geometric
    geom.display()

    if len(geom.rx_names) != 1:
        raise ValueError(f"协作感知假定 1 个接收机，收到 rx={geom.rx_names!r}")
    if len(geom.tx_names) != 2:
        raise ValueError(f"协作感知假定 2 个发射机，收到 tx={geom.tx_names!r}")
    if len(geom.target_names) != 1:
        raise ValueError(f"协作感知假定单目标，收到 target={geom.target_names!r}")

    rx_idx = 0
    target_idx = 0
    rx_name = geom.rx_names[rx_idx]
    target_name = geom.target_names[target_idx]

    expected_pair_keys = {
        RTChannel.cfr_pair_key(rx_name, tx_name) for tx_name in geom.tx_names
    }
    missing = expected_pair_keys - set(h_pairs.keys())
    if missing:
        raise ValueError(f"CFR 按收发机对分离缺少: {sorted(missing)!r}")

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
    link_result_rows: list[list[object]] = []

    for tx_name in tx_order:
        tx_idx = geom.tx_names.index(tx_name)
        is_bistatic = bool(geom.type_tensor[rx_idx, target_idx, tx_idx].item())
        sens_mode = "bistatic" if is_bistatic else "monostatic"
        path_label = "双基地" if is_bistatic else "单基地"

        h_tx = h_pairs[RTChannel.cfr_pair_key(rx_name, tx_name)].to(device)
        h_tx = system.components.moving_target_indication(h_tx, axis=-2)
        h_dd_tx = dd(h_tx)

        slug = _slug_tx_name(tx_name)
        dd.h_delay_doppler = h_dd_tx
        dd.visualize(
            file_name=script_out_dir / f"sensing_cooperative_delay_doppler_{slug}.png",
            to_db=False,
            metric_mode=args.metric_mode,
            backend="matplotlib",
            sens_mode=sens_mode,
        )

        spectra_stack.append(h_dd_tx)
        panel_labels.append(tx_name)

        true_range = geom.range_tensor[rx_idx, target_idx, tx_idx]
        true_velocity = geom.vel_tensor[rx_idx, target_idx, tx_idx]

        if is_bistatic:
            distance_label = "LoS路径长度"
            velocity_label = "几何距离变化率"
        else:
            distance_label = "径向距离"
            velocity_label = "径向速度"

        peaks = system.components.music_estimator(
            h_dd_tx,
            num_sources=1,
        )

        estimate = system.components.sensing_estimator(
            peaks,
            metric_mode=args.metric_mode,
            sens_mode=sens_mode,
            log_peaks=False,
        )

        rmse_range, rmse_velocity, est_range_m, est_velocity_mps, _ = (
            match_peaks_and_compute_radial_rmse(
                est_ranges=estimate.est_ranges,
                est_velocities=estimate.est_velocities,
                true_ranges=true_range,
                true_velocities=true_velocity,
                label=(
                    f"协同感知 — RX {rx_name} / TX {tx_name} "
                    f"({path_label}, sens_mode={sens_mode})"
                ),
                distance_axis_label=distance_label,
                velocity_axis_label=velocity_label,
                verbose=False,
            )
        )
        rmse_range_m = rmse_range
        rmse_velocity_mps = rmse_velocity

        true_r = float(true_range.reshape(-1)[0].item())
        true_v = float(true_velocity.reshape(-1)[0].item())
        link_result_rows.append(
            [
                rx_name,
                tx_name,
                path_label,
                true_r,
                float(est_range_m.item()),
                float(rmse_range.item()),
                true_v,
                float(est_velocity_mps.item()),
                float(rmse_velocity.item()),
            ]
        )

        if is_bistatic:
            bistatic_tx_name = tx_name
            est_r_bistatic = est_range_m
            true_r_bistatic = true_range.reshape(-1)[0]
        else:
            mono_tx_name = tx_name
            est_r_mono = est_range_m
            true_r_mono = true_range.reshape(-1)[0]

    print("各链路感知结果:")
    print(
        tabulate(
            link_result_rows,
            headers=[
                "RX",
                "TX",
                "路径",
                "距离真值(m)",
                "距离估计(m)",
                "距离RMSE(m)",
                "速度真值(m/s)",
                "速度估计(m/s)",
                "速度RMSE(m/s)",
            ],
            tablefmt="simple_grid",
            floatfmt=".2f",
        )
    )

    h_dd_all = torch.stack(spectra_stack, dim=0)
    dd.h_delay_doppler = h_dd_all
    dd.visualize(
        file_name=script_out_dir / "sensing_cooperative_delay_doppler_spectrum.png",
        to_db=False,
        metric_mode=args.metric_mode,
        backend="matplotlib",
        panel_labels=panel_labels,
    )

    h_combined = sum(
        h_pairs[RTChannel.cfr_pair_key(rx_name, name)].to(device)
        for name in geom.tx_names
    )
    h_combined = system.components.moving_target_indication(h_combined, axis=-2)
    h_dd_combined = dd(h_combined)
    dd.h_delay_doppler = h_dd_combined
    dd.visualize(
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
    true_pos = scene.targets_states[target_name][0]
    pos = np.asarray(true_pos, dtype=np.float64).reshape(-1)
    true_xy = (pos[0].item(), pos[1].item())
    y_hint = pos[1].item()

    mono_pos = rx_states[rx_name][0]
    bi_pos = tx_states[bistatic_tx_name][0]
    mono_tx_pos = tx_states[mono_tx_name][0]

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

    localization_rows: list[list[object]] = [
        [
            "真值距离",
            r_m_true,
            r_b_true,
            xy_truth_ranges[0],
            xy_truth_ranges[1],
            true_xy[0],
            true_xy[1],
            rmse_truth_ranges,
        ]
    ]

    r_leg2_est = r_b_est - r_m_est
    if r_leg2_est <= 0.0:
        print(
            f"协同感知 — MUSIC 定位跳过：双基地折叠路径长 {r_b_est:.2f} m "
            f"不大于单基地斜距 {r_m_est:.2f} m"
        )
    else:
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
        else:
            rmse_music = position_rmse_xy(xy_music, true_xy)
            localization_rows.append(
                [
                    "MUSIC距离",
                    r_m_est,
                    r_b_est,
                    xy_music[0],
                    xy_music[1],
                    true_xy[0],
                    true_xy[1],
                    rmse_music,
                ]
            )

    print("z=0 平面定位:")
    print(
        tabulate(
            localization_rows,
            headers=[
                "来源",
                "r_m (m)",
                "r_b (m)",
                "x_est (m)",
                "y_est (m)",
                "x_true (m)",
                "y_true (m)",
                "位置RMSE (m)",
            ],
            tablefmt="simple_grid",
            floatfmt=".2f",
        )
    )


if __name__ == "__main__":
    main()
