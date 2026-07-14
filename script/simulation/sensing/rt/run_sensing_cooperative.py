"""协同 ISAC 感知评估：2 发射机 + 1 接收机，按 TX 注入 CFR 后走标准 channel → sensing 链。

须在 **ISAC conda 环境**中、从仓库根目录运行::

    python script/simulation/sensing/rt/run_sensing_cooperative.py

默认配置
--------
``config/simulation/sensing/sensing_cooperative.toml``：
2 TX（``bs1`` 共址 TX/RX + ``bs2`` 仅 TX）+ 1 RX（``bs1_rx``）+ 1 目标。

拓扑前提
--------
- 几何假定 1 RX、2 TX、1 目标（与默认 TOML 一致）。
- 定位闭式解要求单基地 TX 与 RX 共址，见
  ``localize_xy_z0_colocated_tx_mono_bistatic``。

管线概要
--------
1. ``transmit`` 得到 ``x_rg`` / ``x_time``；获取完整 RT CFR，并校验各 RX–TX 对键齐全。
2. 逐 TX：注入仅保留该 TX–RX 对的 CFR → ``channel`` →（可选解调）→
   ``system.sensing``（与单/双基地脚本同构：LS → DD → MUSIC）。
3. 各 TX 谱 stack 为多 panel 总览；``cfr=None`` 再跑一遍作为未分离叠加基线。
4. 在 ``z=0`` 且单基地 TX 与 RX 共址时，融合单基地斜距与双基地折叠路径长求 ``(x,y)``。

与 ``--domain`` 的关系
----------------------
多 TX 路径分离依赖 ``RTChannel.cfr`` 频域注入；``--domain time`` 的时域 CIR 不受
CFR 注入影响，协作脚本仅支持 ``frequency``（默认），传入 ``time`` 将显式报错。

输出目录
--------
``out/sensing_cooperative/``：场景 PNG、各 TX / 多 TX 总览 / 叠加对比 DD 谱图，
以及控制台链路 RMSE 表与 ``z=0`` 定位表。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

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
from isac.utils import set_random_seed
from isac.channel import RTChannel

SCRIPT_OUT_DIR = PROJECT_ROOT / "out" / "sensing_cooperative"


def _slug_tx_name(tx_name: str) -> str:
    """将 TX 名称转为输出 PNG 文件名可用的安全片段。"""
    return re.sub(r"[^\w.-]+", "_", tx_name).strip("_") or "tx"


def _cfr_isolate_pair(full_cfr: torch.Tensor, rx_i: int, tx_i: int) -> torch.Tensor:
    """从完整 CFR 中只保留指定 (RX, TX) 切片，其余置零（兼容 6D / 7D）。"""
    out = torch.zeros_like(full_cfr)
    if full_cfr.ndim == 7:
        out[0, rx_i, 0, tx_i, 0] = full_cfr[0, rx_i, 0, tx_i, 0]
    elif full_cfr.ndim == 6:
        out[0, rx_i, tx_i, 0] = full_cfr[0, rx_i, tx_i, 0]
    else:
        raise ValueError(
            "cfr 须为 6D (batch, num_rx, num_tx, num_rx_ant, S, F) 或 "
            "7D (batch, num_rx, num_rx_ant, num_tx, num_tx_ant, S, F)，"
            f"收到 ndim={full_cfr.ndim}, shape={tuple(full_cfr.shape)}"
        )
    return out


def _save_dd_spectrum(
    dd,
    h: torch.Tensor,
    file_name: Path,
    metric_mode: str,
    *,
    sens_mode: str | None = None,
    panel_labels: Sequence[str] | None = None,
) -> None:
    """写出时延–多普勒谱图（多 panel 总览等 ``system.sensing`` 未覆盖的出图）。"""
    dd.h_delay_doppler = h
    kwargs: dict = {
        "file_name": file_name,
        "to_db": False,
        "metric_mode": metric_mode,
        "backend": "matplotlib",
    }
    if sens_mode is not None:
        kwargs["sens_mode"] = sens_mode
    if panel_labels is not None:
        kwargs["panel_labels"] = panel_labels
    dd.visualize(**kwargs)


def _localization_row(
    source: str,
    mono_pos,
    bi_pos,
    r_m: float,
    r_b: float,
    true_xy: tuple[float, float],
    y_hint: float,
    mono_tx_pos,
    z_target_m: float = 0.0,
) -> list[object]:
    """由单基地斜距与双基地折叠路径长求 ``(x,y)``，返回定位表的一行。"""
    xy = localize_xy_z0_colocated_tx_mono_bistatic(
        mono_pos,
        bi_pos,
        r_mono_slant_m=r_m,
        r_bistatic_sum_m=r_b,
        z_target_m=z_target_m,
        y_hint=y_hint,
        tx_pos=mono_tx_pos,
    )
    return [
        source,
        r_m,
        r_b,
        xy[0],
        xy[1],
        true_xy[0],
        true_xy[1],
        position_rmse_xy(xy, true_xy),
    ]


def argument_parser() -> argparse.Namespace:
    """解析协作感知评估所需的设备、随机种子、信道域与 metric 模式。

    ``--metric_mode`` 同时决定 DD 谱图轴标签与 RMSE 表单位
    （``dd``：时延/多普勒；``rv``：距离/速度）。
    """
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
        help="信道施加域；协作多 TX 路径分离仅支持 frequency",
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
    """构建系统、按 TX 注入 CFR 并跑与单/双基地同构的协作感知链。

    阶段见模块 docstring「管线概要」：CFR 注入分离 → 逐 TX sensing → 谱图汇总 → z=0 定位。
    """
    # --- 参数解析 ---
    args = argument_parser()
    set_random_seed(args.seed)

    # --- 构建系统 ---
    system = System(
        args.config_file,
        device=args.device,
    )
    comps = system.components

    # --- 渲染场景 ---
    comps.rt_simulator.render_to_file(
        filename=str(SCRIPT_OUT_DIR / "sensing_cooperative_scene.png")
    )

    # --- 发射 ---
    _, x_rg, x_time = system.transmit()
    # 在 x_rg 第 0 维前新增一个维度，然后复制一次
    print(x_rg.shape)
    x_rg = x_rg.unsqueeze(0).repeat(2, *[1 for _ in range(x_rg.dim())])
    print(x_rg.shape)

    # --- 应用信道 ---
    snr_db = system.params.channel.snr_db

    y_rg = comps.channel(x_rg, x_time, domain="frequency", snr_db=snr_db)

    print(y_rg.shape)

    h_freq = comps.ls_channel_estimator(x_rg, y_rg)

    # 时延多普勒谱
    h_dd = comps.delay_doppler_spectrum(h_freq, sens_mode="monostatic")

    comps.delay_doppler_spectrum.visualize(
        file_name=SCRIPT_OUT_DIR / "sensing_cooperative_delay_doppler_spectrum.png",
        metric_mode=args.metric_mode,
        sens_mode="monostatic",
        to_db=False,
    )

    # # --- 获取完整 CFR，校验 RX–TX 对齐全 ---
    # full_cfr = rt_channel.get_cfr(
    #     num_time_steps=rg.num_ofdm_symbols,
    #     sampling_frequency=1 / rg.ofdm_symbol_duration,
    # )
    # h_pairs = rt_channel.cfr_split(
    #     rg.num_ofdm_symbols,
    #     1 / rg.ofdm_symbol_duration,
    # )

    # # --- 几何与 TX 排序 ---
    # geom = scene.rx_target_tx_geometric
    # geom.display()

    # rx_idx = 0
    # target_idx = 0
    # rx_name = geom.rx_names[rx_idx]
    # target_name = geom.target_names[target_idx]

    # expected_pair_keys = {
    #     RTChannel.cfr_pair_key(rx_name, tx_name) for tx_name in geom.tx_names
    # }
    # missing = expected_pair_keys - set(h_pairs.keys())
    # if missing:
    #     raise ValueError(f"CFR 按收发机对分离缺少: {sorted(missing)!r}")

    # # 单基地 TX 优先，使后续定位变量赋值顺序稳定（双基地在后）
    # tx_order = sorted(
    #     geom.tx_names,
    #     key=lambda name: bool(
    #         geom.type_tensor[rx_idx, target_idx, geom.tx_names.index(name)].item()
    #     ),
    # )

    # spectra_stack: list[torch.Tensor] = []
    # panel_labels: list[str] = []
    # # sens_mode -> (tx_name, true_r_m, est_r_m)
    # link_ranges: dict[str, tuple[str, float, float]] = {}
    # link_result_rows: list[list[object]] = []

    # # --- 逐 TX：CFR 注入 → channel → sensing（与 mono/bistatic 同构）---
    # for tx_name in tx_order:
    #     tx_idx = geom.tx_names.index(tx_name)
    #     # True → 双基地折叠路径；False → 单基地斜距；驱动 MUSIC 距离/速度反演模型
    #     is_bistatic = bool(geom.type_tensor[rx_idx, target_idx, tx_idx].item())
    #     sens_mode = "bistatic" if is_bistatic else "monostatic"
    #     path_label = "双基地" if is_bistatic else "单基地"

    #     rt_channel.cfr = _cfr_isolate_pair(full_cfr, rx_idx, tx_idx)

    #     # --- 应用信道 ---
    #     y_out = comps.channel(x_rg, x_time, domain=domain, snr_db=snr_db)
    #     if domain == "time":
    #         y_rg = comps.demodulator(y_out)
    #     else:
    #         y_rg = y_out

    #     # --- 感知 ---
    #     slug = _slug_tx_name(tx_name)
    #     h_dd, estimate = system.sensing(
    #         x_rg,
    #         y_rg,
    #         metric_mode=args.metric_mode,
    #         sens_mode=sens_mode,
    #         visualize_file=SCRIPT_OUT_DIR
    #         / f"sensing_cooperative_delay_doppler_{slug}.png",
    #         to_db=False,
    #     )

    #     spectra_stack.append(h_dd)
    #     panel_labels.append(tx_name)

    #     true_range = geom.range_tensor[rx_idx, target_idx, tx_idx]
    #     true_velocity = geom.vel_tensor[rx_idx, target_idx, tx_idx]

    #     # 轴标签与几何真值语义一致，供 RMSE 匹配日志使用
    #     if is_bistatic:
    #         distance_label = "LoS路径长度"
    #         velocity_label = "几何距离变化率"
    #     else:
    #         distance_label = "径向距离"
    #         velocity_label = "径向速度"

    #     rmse_range, rmse_velocity, est_range_m, est_velocity_mps, _ = (
    #         match_peaks_and_compute_radial_rmse(
    #             est_ranges=estimate.est_ranges,
    #             est_velocities=estimate.est_velocities,
    #             true_ranges=true_range,
    #             true_velocities=true_velocity,
    #             label=(
    #                 f"协同感知 — RX {rx_name} / TX {tx_name} "
    #                 f"({path_label}, sens_mode={sens_mode})"
    #             ),
    #             distance_axis_label=distance_label,
    #             velocity_axis_label=velocity_label,
    #             verbose=False,
    #         )
    #     )

    #     true_r = float(true_range.reshape(-1)[0].item())
    #     est_r = float(est_range_m.item())
    #     link_result_rows.append(
    #         [
    #             rx_name,
    #             tx_name,
    #             path_label,
    #             true_r,
    #             est_r,
    #             float(rmse_range.item()),
    #             float(true_velocity.reshape(-1)[0].item()),
    #             float(est_velocity_mps.item()),
    #             float(rmse_velocity.item()),
    #         ]
    #     )
    #     link_ranges[sens_mode] = (tx_name, true_r, est_r)

    # rt_channel.cfr = None  # 恢复 live CFR

    # print("各链路感知结果:")
    # print(
    #     tabulate(
    #         link_result_rows,
    #         headers=[
    #             "RX",
    #             "TX",
    #             "路径",
    #             "距离真值(m)",
    #             "距离估计(m)",
    #             "距离RMSE(m)",
    #             "速度真值(m/s)",
    #             "速度估计(m/s)",
    #             "速度RMSE(m/s)",
    #         ],
    #         tablefmt="simple_grid",
    #         floatfmt=".2f",
    #     )
    # )

    # # --- 谱图汇总与叠加对比 ---
    # # 各 TX 分离谱 stack 为多 panel 总览
    # _save_dd_spectrum(
    #     dd,
    #     torch.stack(spectra_stack, dim=0),
    #     SCRIPT_OUT_DIR / "sensing_cooperative_delay_doppler_spectrum.png",
    #     args.metric_mode,
    #     panel_labels=panel_labels,
    # )

    # # 未分离基线：live 全 TX CFR → channel → sensing，展示多径叠加谱峰混叠
    # y_out = comps.channel(x_rg, x_time, domain=domain, snr_db=snr_db)
    # y_rg = comps.demodulator(y_out) if domain == "time" else y_out
    # system.sensing(
    #     x_rg,
    #     y_rg,
    #     metric_mode=args.metric_mode,
    #     sens_mode="monostatic",
    #     visualize_file=SCRIPT_OUT_DIR
    #     / "sensing_cooperative_delay_doppler_combined.png",
    #     to_db=False,
    # )

    # # --- z=0 平面定位 ---
    # # 须各有一条单基地斜距与双基地折叠路径长量测
    # mono = link_ranges.get("monostatic")
    # bistatic = link_ranges.get("bistatic")
    # if mono is None or bistatic is None:
    #     print(
    #         "协同感知定位跳过：需要各 1 条单基地与双基地距离量测，"
    #         f"当前 mono_tx={(mono[0] if mono else None)!r}, "
    #         f"bistatic_tx={(bistatic[0] if bistatic else None)!r}"
    #     )
    #     return

    # mono_tx_name, r_m_true, r_m_est = mono
    # bistatic_tx_name, r_b_true, r_b_est = bistatic

    # true_pos = scene.targets_states[target_name][0]
    # pos = np.asarray(true_pos, dtype=np.float64).reshape(-1)
    # true_xy = (pos[0].item(), pos[1].item())
    # y_hint = pos[1].item()

    # # mono_pos：单基地 RX；bi_pos：双基地 TX；mono_tx_pos：共址 TX（定位侧共址校验）
    # mono_pos = scene.rx_states[rx_name][0]
    # bi_pos = scene.tx_states[bistatic_tx_name][0]
    # mono_tx_pos = scene.tx_states[mono_tx_name][0]

    # # 几何真值距离定位，作为参考上界
    # localization_rows: list[list[object]] = [
    #     _localization_row(
    #         "真值距离",
    #         mono_pos,
    #         bi_pos,
    #         r_m_true,
    #         r_b_true,
    #         true_xy,
    #         y_hint,
    #         mono_tx_pos,
    #     )
    # ]

    # # 双基地第二段距离须 > 0，否则 localize_* 抛 ValueError
    # try:
    #     localization_rows.append(
    #         _localization_row(
    #             "MUSIC距离",
    #             mono_pos,
    #             bi_pos,
    #             r_m_est,
    #             r_b_est,
    #             true_xy,
    #             y_hint,
    #             mono_tx_pos,
    #         )
    #     )
    # except ValueError as exc:
    #     print(f"协同感知 — MUSIC 定位跳过：{exc}")

    # # --- 定位结果打印 ---
    # print("z=0 平面定位:")
    # print(
    #     tabulate(
    #         localization_rows,
    #         headers=[
    #             "来源",
    #             "r_m (m)",
    #             "r_b (m)",
    #             "x_est (m)",
    #             "y_est (m)",
    #             "x_true (m)",
    #             "y_true (m)",
    #             "位置RMSE (m)",
    #         ],
    #         tablefmt="simple_grid",
    #         floatfmt=".2f",
    #     )
    # )


if __name__ == "__main__":
    main()
