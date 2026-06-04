"""ISAC 数据集采集入口：按轨迹或蒙特卡洛生成目标位姿序列，可选每步单站感知，并写出 CSV / HDF5 / GIF。

流程概要
--------
1. 解析参数，设置种子；按 ``--source`` 与 ``--roi`` 等准备采样区域与 episode 数。
2. 构建 ``System``，在 ``out/dataset_collection/`` 下落盘（与 ``--config_file`` 解耦的脚本级输出根目录）。
3. 主循环：对每条 episode 更新 RT 目标位姿 → 记真值径向几何 →（可选）``monostatic_eval`` 或 ``bistatic_eval``（``--sensing_layout``）→ 累计 CFR/CIR/CSV 行/可选 GIF 帧。
4. 循环结束后按 ``--csv_mode``、``--save_h5``、``--save_gif`` 写出产物；HDF5 文件名随 ``source`` 与 ``--run_sensing`` 组合变化。

与 ``run_sensing_monostatic.py`` 的差异：本脚本将感知嵌在数据采集循环内，并承担批量 episode 的 I/O。
"""
import argparse
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import c
from tqdm import tqdm

from isac import PROJECT_ROOT
from isac.datasets import Dataset
from isac.data_structures.rx_target_tx_geometric import RxTargetTxGeometric
from isac.sensing.utils import doppler_to_velocity
from isac.system import System, _csv_float2_scalar
from isac.utils import (
    compute_rmse,
    get_logger,
    images_to_gif,
    paths_cfr_numpy,
    paths_cir_numpy,
    scene_slug_from_rt_scene,
    select_peak_and_log_radial_rmse,
    set_random_seed,
    stack_ragged_cir_samples,
)
from isac.utils import target_generation as tg

warnings.filterwarnings(
    "ignore",
    message=r"The AST-transforming decorator @drjit\.syntax was called more than 1000 times.*",
    category=RuntimeWarning,
    module=r"drjit\.ast",
)

logger = get_logger(__name__)


def _paths_doppler_hz_los(
    rt_scene: object,
    tau_true_s: float | torch.Tensor,
    *,
    rx_idx: int = 0,
    tx_idx: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """从 Sionna ``Paths`` 中取与几何 LoS 总时延最接近的路径的多普勒真值（Hz）。脚本内副本，与旧 ``isac.sensing.utils.paths_doppler_hz_los`` 等价。"""
    paths = rt_scene.paths
    tau_np = np.asarray(paths.tau, dtype=np.float64)
    valid_np = np.asarray(paths.valid, dtype=bool)
    doppler_np = np.asarray(paths.doppler, dtype=np.float64)
    if tau_np.ndim != 3:
        raise ValueError(
            "_paths_doppler_hz_los 当前仅支持形状 [num_rx,num_tx,max_paths]（与 path_solver.synthetic_array=true 一致）；"
            f"当前 ndim={tau_np.ndim}, shape={tau_np.shape}。"
        )
    if rx_idx >= tau_np.shape[0] or tx_idx >= tau_np.shape[1]:
        raise IndexError(
            f"_paths_doppler_hz_los: rx_idx={rx_idx} 或 tx_idx={tx_idx} 越界，形状 {tau_np.shape}"
        )

    tau_slice = tau_np[rx_idx, tx_idx, :]
    valid_slice = np.asarray(valid_np[rx_idx, tx_idx, :], dtype=bool)
    dop_slice = doppler_np[rx_idx, tx_idx, :]

    if isinstance(tau_true_s, torch.Tensor):
        t0 = float(tau_true_s.detach().cpu().to(dtype=torch.float64).reshape(()).item())
    else:
        t0 = float(tau_true_s)

    candidates = np.flatnonzero(valid_slice & (tau_slice >= 0.0))
    if candidates.size == 0:
        candidates = np.flatnonzero(valid_slice)
    if candidates.size == 0:
        candidates = np.arange(tau_slice.size, dtype=np.int64)

    err = np.abs(tau_slice[candidates] - t0)
    k = int(candidates[int(np.argmin(err))])
    fd = float(dop_slice[k])

    dev = device if device is not None else torch.device("cpu")
    return torch.tensor(fd, dtype=torch.float64, device=dev)


def argument_parser() -> argparse.Namespace:
    """构造数据集采集脚本的全部 CLI 参数（轨迹 / 蒙特卡洛、感知、导出格式）。"""
    parser = argparse.ArgumentParser(description="ISAC 系统仿真 — 数据集采集主流程")

    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument(
        "--config_file",
        type=str,
        default="data_collection.toml",
        help="配置文件路径（须含非空 [rt_scene]，默认使用仓库 config/data_collection.toml）",
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
        help="随机种子（蒙特卡洛位置/速度采样）",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="monte_carlo",
        choices=["monte_carlo", "trajectory"],
        help="目标点来源",
    )
    parser.add_argument("--run_sensing", action="store_true", help="每步/每样本执行单站感知")
    parser.add_argument("--no-save-h5", dest="save_h5", action="store_false", help="不写 HDF5")
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false", help="不写 CSV")
    parser.add_argument("--save_gif", action="store_true", help="导出场景 GIF")
    parser.set_defaults(save_h5=True, save_csv=True)

    parser.add_argument(
        "--csv_mode",
        type=str,
        default="legacy",
        choices=["unified", "legacy"],
        help="CSV 模式",
    )
    parser.add_argument(
        "--sensing_domain",
        type=str,
        default="frequency",
        choices=["frequency", "time"],
        help="感知链路 domain（与 run_sensing 配合）",
    )
    parser.add_argument(
        "--velocity_model",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic", "bistatic_rx_radial"],
        help="MUSIC 多普勒→标量速度换算；双基地布局下真值速度来自 paths.doppler，经同一模型换算",
    )
    parser.add_argument(
        "--sensing_layout",
        type=str,
        default="monostatic",
        choices=["monostatic", "bistatic"],
        help="run_sensing 时：monostatic=RX 径向真值与估计；bistatic=LoS 总路径 + paths.doppler 速度（建议 csv_mode=unified）",
    )
    parser.add_argument(
        "--log_per_step_sensing",
        action="store_true",
        help="每步打印感知一行日志",
    )

    parser.add_argument("--time_delta", type=float, default=None, help="轨迹模式：步长时间间隔")
    parser.add_argument("--steps", type=int, default=None, help="轨迹模式：最大步数")

    parser.add_argument(
        "--roi",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="蒙特卡洛：ROI 六元组",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="蒙特卡洛：样本数")
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="蒙特卡洛：目标位置在 ROI 内的采样分布（均匀或高斯）",
    )
    parser.add_argument(
        "--safe_margin",
        type=float,
        default=2.0,
        help="蒙特卡洛：障碍物包围盒额外安全距离（米），位置合法性校验用",
    )
    parser.add_argument(
        "--max_trials_factor",
        type=int,
        default=20,
        help="蒙特卡洛：拒绝采样最大尝试次数 = num_samples × 该因子（防死循环）",
    )
    parser.add_argument(
        "--speed_range",
        nargs=2,
        type=float,
        default=[0.1, 50.0],
        metavar=("MIN", "MAX"),
        help="蒙特卡洛速度幅值范围（未提供 velocities 数组时）",
    )
    parser.add_argument(
        "--velocity_sampling",
        type=str,
        default="sphere_uniform",
        choices=["sphere_uniform", "axis_box"],
        help="蒙特卡洛：速度方向/分量采样方式（球面均匀或各轴独立盒式）",
    )

    return parser.parse_args()


def main() -> None:
    """跑通「生成 episode → 可选感知 → 写出 CSV/HDF5/GIF」的完整管线。"""
    args = argument_parser()
    set_random_seed(args.seed)

    # 蒙特卡洛用 ROI：显式 --roi 为 ((xmin,xmax),(ymin,ymax),(zmin,zmax))；未指定时采用与历史脚本一致的默认平面区域
    roi_tuple = None
    if args.roi is not None:
        r = args.roi
        roi_tuple = ((r[0], r[1]), (r[2], r[3]), (r[4], r[5]))
    elif args.source == "monte_carlo":
        roi_tuple = ((0.0, 80.0), (-40.0, 40.0), (0.0, 0.0))

    system = System(args)

    script_out_dir = PROJECT_ROOT / "out" / "dataset_collection"
    script_out_dir.mkdir(parents=True, exist_ok=True)

    source: Literal["monte_carlo", "trajectory"] = args.source
    run_sensing = args.run_sensing
    save_h5 = args.save_h5
    save_csv = args.save_csv
    save_gif = args.save_gif
    csv_mode: Literal["unified", "legacy"] = args.csv_mode
    sensing_domain = args.sensing_domain
    velocity_model = args.velocity_model
    sensing_layout: Literal["monostatic", "bistatic"] = args.sensing_layout
    log_per_step_sensing_line = args.log_per_step_sensing

    if run_sensing and sensing_layout == "bistatic" and csv_mode == "legacy":
        logger.warning(
            "双基地感知 CSV 列与 legacy 固定表头不一致，已改用 csv_mode=unified"
        )
        csv_mode = "unified"

    def monostatic_eval(
        domain: str,
        vel_model: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对**当前** RT 场景状态跑一条单站感知链，并写谱图、打日志。

        与独立评估脚本逻辑对齐：OFDM 链 → 信道估计 → 时延–多普勒谱 → MUSIC →
        峰转径向距离/速度 → 联合误差选峰 → RMSE 日志。返回值供 CSV 写入与本步诊断。
        """
        x_rg = system.tx_symbols_to_resource_grid()

        if domain == "frequency":
            y_rg = system.apply_channel(x_rg, domain=domain)
        elif domain == "time":
            x_time = system.components.modulator(x_rg)
            y_time = system.apply_channel(x_time, domain=domain)
            y_rg = system.components.demodulator(y_time)
        else:
            raise ValueError(f"不支持的域: {domain}")

        h = system.estimate_channel(x_rg, y_rg)

        h_delay_doppler = system.components.delay_doppler_spectrum(h)

        # 覆盖写入固定文件名：循环中每步刷新同一张图，便于查看最新一步谱图
        system.components.delay_doppler_spectrum.visualize(
            offset=200,
            file_name=script_out_dir / "sensing_monostatic_delay_doppler_spectrum.png",
            to_db=False,
            metric_mode="delay_doppler",
            backend="matplotlib",
        )

        est_ranges_t, est_velocities_t, _ = system.components.music_estimator(
            spectrum_tensor=h_delay_doppler,
            metric_mode="delay_doppler",
            sens_mode="monostatic",
        )
        if est_ranges_t.numel() == 0:
            raise RuntimeError("单基地感知评估：MUSIC 未检测到谱峰，无法估计距离/速度")

        scen = system.components.rt_scene

        target_states = scen.get_targets_states
        rx_states = scen.get_rx_states
        tx_states = scen.get_tx_states

        los_geom = RxTargetTxGeometric.from_states(
            target_states,
            rx_states,
            tx_states,
            device=system.device,
        )
        true_radial_range = los_geom.range_tensor[0, 0, 0]
        true_radial_velocity = los_geom.vel_tensor[0, 0, 0]

        _, _, est_range_t, est_velocity_t, est_power_db_t = select_peak_and_log_radial_rmse(
            logger,
            est_ranges=est_ranges_t,
            est_velocities=est_velocities_t,
            true_ranges=true_radial_range,
            true_velocities=true_radial_velocity,
            log_prefix="单基地感知",
        )
        return est_range_t, est_velocity_t, est_power_db_t

    def bistatic_eval(
        domain: str,
        vel_model: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """双基地：LoS 路径长度 \(c\tau\) + ``paths.doppler`` 速度真值与联合选峰。"""
        x_rg = system.tx_symbols_to_resource_grid()

        if domain == "frequency":
            y_rg = system.apply_channel(x_rg, domain=domain)
        elif domain == "time":
            x_time = system.components.modulator(x_rg)
            y_time = system.apply_channel(x_time, domain=domain)
            y_rg = system.components.demodulator(y_time)
        else:
            raise ValueError(f"不支持的域: {domain}")

        h = system.estimate_channel(x_rg, y_rg)
        h_delay_doppler = system.components.delay_doppler_spectrum(h)

        system.components.delay_doppler_spectrum.visualize(
            offset=200,
            file_name=script_out_dir / "sensing_monostatic_delay_doppler_spectrum.png",
            to_db=False,
            metric_mode="delay_doppler",
            backend="matplotlib",
        )

        est_paths_t, est_velocities_t, _ = system.components.music_estimator(
            spectrum_tensor=h_delay_doppler,
            metric_mode="delay_doppler",
            sens_mode="bistatic",
        )
        if est_paths_t.numel() == 0:
            raise RuntimeError("双基地感知评估：MUSIC 未检测到谱峰")

        scen = system.components.rt_scene
        target_states = scen.get_targets_states
        rx_states = scen.get_rx_states
        tx_states = scen.get_tx_states

        los_geom = RxTargetTxGeometric.from_states(
            target_states,
            rx_states,
            tx_states,
            device=system.device,
        )
        true_path_m = los_geom.range_tensor[0, 0, 0]
        tau_true_s = true_path_m / c
        fd_true_t = _paths_doppler_hz_los(
            scen, tau_true_s, rx_idx=0, tx_idx=0, device=system.device
        )
        true_velocity_paths = doppler_to_velocity(
            fd_true_t,
            float(system.params.carrier_frequency),
            "bistatic",
        )

        _, _, est_path_t, est_vel_t, est_power_db_t = select_peak_and_log_radial_rmse(
            logger,
            est_ranges=est_paths_t,
            est_velocities=est_velocities_t,
            true_ranges=true_path_m,
            true_velocities=true_velocity_paths,
            log_prefix="双基地数据集感知（LoS路径+paths.doppler）",
            distance_axis_label="LoS路径长度",
            velocity_axis_label="标量速度",
        )
        return est_path_t, est_vel_t, est_power_db_t, true_path_m, true_velocity_paths

    # --- 场景与 episode 序列：单脚本假设至少一个 RT 目标，取字典第一个参与采样 ---
    scene = system.components.rt_scene
    if not scene.rt_targets:
        raise RuntimeError("当前场景中没有可用的 RT 目标（scene.rt_targets 为空）")
    _, target = next(iter(scene.rt_targets.items()))
    scene_slug = scene_slug_from_rt_scene(scene)

    pos_arr, vel_arr = tg.generate_target_episodes(
        system.components.rt_scene,
        source=source,
        time_delta=args.time_delta,
        steps=args.steps,
        roi=roi_tuple,
        num_samples=args.num_samples if source == "monte_carlo" else None,
        sampling_mode=args.sampling_mode,
        safe_margin=args.safe_margin,
        max_trials_factor=args.max_trials_factor,
        velocities=None,
        speed_range=(float(args.speed_range[0]), float(args.speed_range[1])),
        velocity_sampling=args.velocity_sampling,
        velocity_roi_vx=None,
        velocity_roi_vy=None,
        velocity_roi_vz=None,
        seed=args.seed,
        rng=None,
    )
    n_ep = int(pos_arr.shape[0])
    if n_ep == 0:
        logger.info("无有效 Episode，结束")
        return

    # CSV 索引列名：轨迹用 step，蒙特卡洛用 sample_idx（与 legacy 分裂表约定一致）
    index_key: Literal["step", "sample_idx"] = "step" if source == "trajectory" else "sample_idx"
    desc = "轨迹数据集" if source == "trajectory" else "MC 数据集"
    unit = "step" if source == "trajectory" else "sample"

    h_freq_list: list[np.ndarray] = []
    cir_a_list: list[np.ndarray] = []
    cir_tau_list: list[np.ndarray] = []
    target_pos_list: list[np.ndarray] = []
    target_vel_list: list[np.ndarray] = []
    scene_frames: list[np.ndarray] = []
    csv_rows: list[dict[str, str | int]] = []

    # --- 主循环：更新目标 → 真值几何 → 可选感知 → 累计 HDF5 / GIF 原材料 ---
    for i in tqdm(range(n_ep), desc=desc, unit=unit):
        pos = pos_arr[i]
        vel = vel_arr[i]
        system._update_rt_target_pose_from_velocity(target, pos, vel)

        pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
        vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)

        if save_h5:
            target_pos_list.append(pos_row.copy())
            target_vel_list.append(vel_row.copy())

        _tg = scene.get_targets_states
        _rg = scene.get_rx_states
        _xg = scene.get_tx_states
        los_geom_sens = RxTargetTxGeometric.from_states(
            _tg,
            _rg,
            _xg,
            device=system.device,
        )
        true_r = los_geom_sens.range_tensor[0, 0, 0]
        true_v = los_geom_sens.vel_tensor[0, 0, 0]

        row: dict[str, str | int] = {
            index_key: i,
            "pos_x_m": _csv_float2_scalar(pos_row[0]),
            "pos_y_m": _csv_float2_scalar(pos_row[1]),
            "pos_z_m": _csv_float2_scalar(pos_row[2]),
            "vel_x_mps": _csv_float2_scalar(vel_row[0]),
            "vel_y_mps": _csv_float2_scalar(vel_row[1]),
            "vel_z_mps": _csv_float2_scalar(vel_row[2]),
            "true_range_m": _csv_float2_scalar(true_r),
            "true_radial_velocity_mps": _csv_float2_scalar(true_v),
        }

        if run_sensing:
            if sensing_layout == "bistatic":
                est_p, est_vel, est_db, true_p, true_vp = bistatic_eval(
                    sensing_domain,
                    velocity_model,
                )
                if log_per_step_sensing_line:
                    logger.info(
                        "%s=%03d 双基地感知: LoS_path=%.3f m, velocity=%.3f m/s, MUSIC_peak=%.3f dB",
                        index_key,
                        i,
                        float(est_p.item()),
                        float(est_vel.item()),
                        float(est_db.item()),
                    )
                    logger.info("")
                rmse_p = compute_rmse(est_p.reshape(1), true_p.reshape(1))
                rmse_v = compute_rmse(est_vel.reshape(1), true_vp.reshape(1))
                row["true_los_path_length_m"] = _csv_float2_scalar(true_p)
                row["true_velocity_paths_doppler_mps"] = _csv_float2_scalar(true_vp)
                row["est_los_path_length_m"] = _csv_float2_scalar(est_p)
                row["est_velocity_paths_doppler_mps"] = _csv_float2_scalar(est_vel)
                row["rmse_los_path_m"] = _csv_float2_scalar(rmse_p)
                row["rmse_velocity_paths_doppler_mps"] = _csv_float2_scalar(rmse_v)
            else:
                est_range_t, est_velocity_t, est_power_db_t = monostatic_eval(
                    sensing_domain,
                    velocity_model,
                )
                if log_per_step_sensing_line:
                    logger.info(
                        "%s=%03d 感知: range=%.3f m, velocity=%.3f m/s, MUSIC_peak=%.3f dB",
                        index_key,
                        i,
                        float(est_range_t.item()),
                        float(est_velocity_t.item()),
                        float(est_power_db_t.item()),
                    )
                    logger.info("")
                rmse_range_t = compute_rmse(
                    est_range_t.reshape(1),
                    true_r.reshape(1),
                )
                rmse_vel_t = compute_rmse(
                    est_velocity_t.reshape(1),
                    true_v.reshape(1),
                )
                row["est_range_m"] = _csv_float2_scalar(est_range_t)
                row["rmse_range_m"] = _csv_float2_scalar(rmse_range_t)
                row["est_radial_velocity_mps"] = _csv_float2_scalar(est_velocity_t)
                row["rmse_radial_velocity_mps"] = _csv_float2_scalar(rmse_vel_t)

        if save_csv:
            csv_rows.append(row)

        if save_h5:
            h_freq_list.append(paths_cfr_numpy(system.components.rg, system.components.rt_scene))
            ca, ct = paths_cir_numpy(system.components.rg, system.components.rt_scene)
            cir_a_list.append(ca)
            cir_tau_list.append(ct)

        if save_gif:
            scene_image = scene.render()
            # Matplotlib Figure 与 ndarray 图像两种返回路径
            if hasattr(scene_image, "canvas"):
                scene_image.canvas.draw()
                frame = np.asarray(scene_image.canvas.buffer_rgba())[..., :3].copy()
                scene_frames.append(frame)
                plt.close(scene_image)
            else:
                scene_frames.append(scene_image)

    # --- 后处理：Episode CSV、HDF5（CFR+CIR+kinematics）、GIF ---
    if save_csv:
        system.save_episodes_csv(
            scene_slug=scene_slug,
            source=source,
            rows=csv_rows,
            run_sensing=run_sensing,
            csv_mode=csv_mode,
            output_root=script_out_dir,
        )

    if save_h5 and len(h_freq_list) > 0:
        cir_a_arr, cir_tau_arr = stack_ragged_cir_samples(cir_a_list, cir_tau_list)
        if source == "trajectory":
            if run_sensing:
                h5_path = script_out_dir / f"{scene_slug}_trajectory_monostatic_sensing.h5"
                desc_h5 = (
                    f"Trajectory + monostatic sensing ({len(h_freq_list)} steps) in {scene_slug}"
                )
                scene_name = scene_slug
            else:
                h5_path = script_out_dir / f"{scene_slug}_sionna_dataset.h5"
                desc_h5 = None
                scene_name = scene_slug
        else:
            mc_slug = f"{scene_slug}_mc"
            if run_sensing:
                h5_path = script_out_dir / f"{scene_slug}_mc_monostatic_sensing.h5"
                desc_h5 = f"Monte Carlo + monostatic sensing ({n_ep} samples) in {scene_slug}"
                scene_name = mc_slug
            else:
                h5_path = script_out_dir / f"{scene_slug}_mc_sionna_dataset.h5"
                desc_h5 = (
                    f"Sionna generated ISAC Monte Carlo dataset ({n_ep} samples) in {scene_slug}"
                )
                scene_name = mc_slug
        Dataset.from_export_arrays(
            np.array(h_freq_list),
            cir_a_arr,
            cir_tau_arr,
            np.array(target_pos_list),
            np.array(target_vel_list),
            np.array(scene.transceivers["bs1"].position),
            system.params.carrier_frequency,
            system.params.ofdm.subcarrier_spacing,
            system.params.ofdm.num_subcarriers,
            len(h_freq_list),
            scene_name,
            description=desc_h5,
        ).save(h5_path)
    elif save_h5:
        logger.warning("未采集 CFR/CIR，跳过 HDF5")

    if save_gif:
        if not scene_frames:
            logger.warning("无场景帧，跳过 GIF 导出")
        else:
            gif_path = (
                script_out_dir / "scene_image_mc.gif"
                if source == "monte_carlo"
                else (
                    script_out_dir / "scene_image_trajectory_sensing.gif"
                    if run_sensing
                    else script_out_dir / "scene_image.gif"
                )
            )
            images_to_gif(
                filepath=gif_path,
                images=scene_frames,
                time_slot=1,
                speed=5,
            )


if __name__ == "__main__":
    main()
