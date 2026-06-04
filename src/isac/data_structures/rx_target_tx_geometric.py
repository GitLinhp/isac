"""接收机–目标–发射机三元组几何（直射几何假设下的路径类型、长度与 RX 视线径向速度）。

张量形状为 ``(n_rx, n_target, n_tx)``，由 ``compute_path_type`` / ``compute_range`` / ``compute_vel`` 直接生成。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tabulate import tabulate

from ..sensing.utils import (
    MONOSTATIC_TX_RX_EPS_M,
    stack_state_field,
    compute_path_type,
    compute_range,
    compute_vel,
)


@dataclass(frozen=True)
class RxTargetTxGeometric:
    """接收机–目标–发射机（Rx–Target–Tx）三元组几何：路径类型、几何路径长度与距离变化率。

    ``type_tensor`` / ``range_tensor`` / ``vel_tensor`` 形状均为 ``(n_rx, n_target, n_tx)``。
    ``vel_tensor`` 在单基地格点为相对 RX 的视线径向速度；在双基地格点为 ``||T-X||+||R-T||`` 对时间的导数。
    """

    target_names: list[str]
    rx_names: list[str]
    tx_names: list[str]
    type_tensor: torch.Tensor
    range_tensor: torch.Tensor
    vel_tensor: torch.Tensor

    def format_table(
        self,
        *,
        tablefmt: str = "simple_grid",
        floatfmt: str = ".6f",
    ) -> str:
        """表格含路径类型、路径长度、几何距离变化率 (m/s)。"""
        tt = self.type_tensor.detach().cpu()
        rr = self.range_tensor.detach().cpu().to(dtype=torch.float64)
        vv = self.vel_tensor.detach().cpu().to(dtype=torch.float64)
        rows: list[list[object]] = []
        for j, rname in enumerate(self.rx_names):
            for i, tname in enumerate(self.target_names):
                for k, xname in enumerate(self.tx_names):
                    path_mode = "bistatic" if bool(tt[j, i, k].item()) else "monostatic"
                    rows.append(
                        [
                            rname,
                            tname,
                            xname,
                            path_mode,
                            float(rr[j, i, k].item()),
                            float(vv[j, i, k].item()),
                        ]
                    )
        headers = ["接收机", "目标", "发射机", "路径类型", "路径长度_m", "径向速度_mps"]
        return tabulate(rows, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt)

    def display(
        self,
        *,
        tablefmt: str = "simple_grid",
        floatfmt: str = ".2f",
    ) -> None:
        """打印路径类型、路径长度、径向速度（数值默认保留两位小数）。"""
        print("接收机–目标–发射机三元组几何：")
        print(self.format_table(tablefmt=tablefmt, floatfmt=floatfmt))

    @classmethod
    def from_states(
        cls,
        target_states: dict[str, dict[str, np.ndarray]],
        rx_states: dict[str, dict[str, np.ndarray]],
        tx_states: dict[str, dict[str, np.ndarray]],
        *,
        device: torch.device | None = None,
        tx_rx_colocated_eps_m: float = MONOSTATIC_TX_RX_EPS_M,
    ) -> RxTargetTxGeometric:
        """由场景状态构造三元组几何（类型 → 长度 → 速度），张量布局为 ``(n_rx, n_target, n_tx)``。"""
        if not target_states or not rx_states or not tx_states:
            raise ValueError("target_states、rx_states 与 tx_states 须均为非空字典。")

        dev = device if device is not None else torch.device("cpu")
        target_names = list(target_states.keys())
        rx_names = list(rx_states.keys())
        tx_names = list(tx_states.keys())
        n_t = len(target_names)

        t_stack = stack_state_field(target_states, target_names, "pos", dev)
        t_vel = stack_state_field(target_states, target_names, "vel", dev)
        r_stack = stack_state_field(rx_states, rx_names, "pos", dev)
        r_vel = stack_state_field(rx_states, rx_names, "vel", dev)
        x_stack = stack_state_field(tx_states, tx_names, "pos", dev)
        x_vel = stack_state_field(tx_states, tx_names, "vel", dev)

        type_tensor = compute_path_type(r_stack, x_stack, n_t, eps_m=tx_rx_colocated_eps_m)
        range_tensor = compute_range(type_tensor, t_stack, r_stack, x_stack)
        vel_tensor = compute_vel(type_tensor, t_stack, t_vel, r_stack, r_vel, x_stack, x_vel)

        return cls(
            target_names=target_names,
            rx_names=rx_names,
            tx_names=tx_names,
            type_tensor=type_tensor,
            range_tensor=range_tensor,
            vel_tensor=vel_tensor,
        )
