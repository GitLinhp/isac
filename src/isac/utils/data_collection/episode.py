"""Episode 级 RT 采集：目标位姿更新与单条 episode 缓冲写入。"""

from __future__ import annotations

import numpy as np
import torch

from isac.channel.rt import RTTarget, RTSimulator, RxTargetTxGeometric
from isac.datasets import EpisodeBuffers, Hdf5CollectionWriter
from isac.system import System

from ..misc import csv_float2_scalar, csv_vec3
from .channel_export import paths_cfr_numpy




def los_truth_from_kinematics(
    pos: np.ndarray,
    vel: np.ndarray,
    rt_simulator: RTSimulator,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由目标运动学计算默认三元组 ``(range, radial_velocity)``，形状均为标量张量。"""
    target_name = next(iter(rt_simulator.rt_targets.keys()))
    target_states = {
        target_name: [
            np.asarray(pos, dtype=np.float64),
            np.asarray(vel, dtype=np.float64),
        ],
    }
    geom = RxTargetTxGeometric.from_states(
        target_states,
        rt_simulator.rx_states,
        rt_simulator.tx_states,
        device=device,
    )
    return geom.range_tensor[0, 0, 0], geom.vel_tensor[0, 0, 0]


def los_truth_at_first_triple(
    rt_simulator: RTSimulator,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回默认三元组 ``(range, radial_velocity)``，形状均为标量张量。"""
    target_name = next(iter(rt_simulator.rt_targets.keys()))
    pos, vel = rt_simulator.targets_states[target_name]
    return los_truth_from_kinematics(pos, vel, rt_simulator, device)


def kinematics_row(
    episode_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    true_range: torch.Tensor,
    true_velocity: torch.Tensor,
) -> dict[str, str | int]:
    """构造每条 episode 共有的 kinematics + 几何真值列。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)
    return {
        "sample_idx": episode_idx,
        "position": csv_vec3(pos_row),
        "velocity": csv_vec3(vel_row),
        "true_range_m": csv_float2_scalar(true_range),
        "true_radial_velocity_mps": csv_float2_scalar(true_velocity),
    }


def process_episode(
    *,
    system: System,
    rt_simulator: RTSimulator,
    episode_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    buffers: EpisodeBuffers,
    h5_writer: Hdf5CollectionWriter | None = None,
) -> None:
    """单条 episode：几何真值 / CFR / CSV 缓冲写入。"""
    pos_row = np.asarray(pos, dtype=np.float64).reshape(-1)
    vel_row = np.asarray(vel, dtype=np.float64).reshape(-1)

    buffers.target_pos_list.append(pos_row.copy())
    buffers.target_vel_list.append(vel_row.copy())
    true_range, true_velocity = los_truth_at_first_triple(rt_simulator, system.device)
    row = kinematics_row(episode_idx, pos_row, vel_row, true_range, true_velocity)

    buffers.csv_rows.append(row)
    cfr = paths_cfr_numpy(system.components.rg, rt_simulator)
    if h5_writer is not None:
        h5_writer.append_episode(cfr, pos_row, vel_row)
    else:
        buffers.h_freq_list.append(cfr)
