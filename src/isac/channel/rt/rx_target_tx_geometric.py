"""接收机–目标–发射机 (Rx–Target–Tx) 三元组几何。

在直射几何假设下，对场景中每个 ``(rx, target, tx)`` 组合计算：

- 路径类型（单基地 / 双基地，由 TX 与 RX 间距判定）；
- 几何路径长度 ``range_tensor`` (m)；
- 与路径定义配套的距离变化率 ``vel_tensor`` (m/s)。

符号约定：

- ``T``：目标位置/速度；``R``：接收机；``X``：发射机。
- 单基地（``type_tensor=False``）：``range = ||R−T||``，
  ``vel`` 为 ``(v_T−v_R)`` 在 ``T−R`` 方向上的投影。
- 双基地（``type_tensor=True``）：``range = ||T−X|| + ||R−T||``（折叠路径），
  ``vel = d/dt range``，TX/RX/目标均可运动。

张量索引顺序为 ``[j, i, k]`` ↔ ``(rx_names[j], target_names[i], tx_names[k])``，
形状均为 ``(n_rx, n_target, n_tx)``，dtype 为 ``float64`` / ``bool``。
底层公式见 ``compute_path_type`` / ``compute_range`` / ``compute_vel``。

典型入口与用法：

- ``RTScene.rx_target_tx_geometric`` 根据当前 ``targets_states``、``rx_states``、
  ``tx_states`` 调用 ``from_states`` 构造；属性每次读取时按最新状态重算。
- 单链路仿真/数据集采集常以 ``geom.range_tensor[0, 0, 0]``、
  ``geom.vel_tensor[0, 0, 0]`` 作为 MUSIC 匹配真值（见 ``run_dataset_collection``）。
- ``System.run_sensing(..., compute_rmse=True)`` 可将完整 ``range_tensor`` /
  ``vel_tensor`` 传入 ``match_peaks_and_compute_radial_rmse`` 做匈牙利多峰匹配。
"""

from dataclasses import dataclass

import numpy as np
import torch
from tabulate import tabulate

# 判定 TX/RX 是否视为共址（monostatic）：间距小于等于该阈值 (m)
MONOSTATIC_TX_RX_EPS_M = 1e-3


def stack_state_field(
    states: dict[str, dict[str, np.ndarray]],
    names: list[str],
    field: str,
    device: torch.device,
) -> torch.Tensor:
    """将 ``states[name][field]`` 按 ``names`` 顺序堆成 ``(len(names), 3)`` 的 float64 张量。"""
    return torch.stack(
        [
            torch.as_tensor(
                states[n][field], dtype=torch.float64, device=device
            ).reshape(3)
            for n in names
        ],
        dim=0,
    )


def compute_path_type(
    r_stack: torch.Tensor,
    x_stack: torch.Tensor,
    n_targets: int,
    *,
    eps_m: float = MONOSTATIC_TX_RX_EPS_M,
) -> torch.Tensor:
    """``True``=bistatic，``False``=monostatic；形状 ``(n_rx, n_target, n_tx)``。"""
    sep = torch.linalg.vector_norm(
        r_stack[:, None, :] - x_stack[None, :, :],
        dim=-1,
    )
    is_bistatic = sep > eps_m
    return is_bistatic.unsqueeze(1).expand(-1, n_targets, -1).clone()


def compute_range(
    is_bistatic: torch.Tensor,
    t_stack: torch.Tensor,
    r_stack: torch.Tensor,
    x_stack: torch.Tensor,
) -> torch.Tensor:
    """按路径类型计算几何路径长度 (m)。

    - **bistatic**：双基地「折叠」路径长 ``||T-X|| + ||R-T||``（直射几何，不经遮挡判定）。
    - **monostatic**：TX/RX 共址近似下采用单基地量程 ``||R-T||``，与 ``||T-X||+||R-T||`` 在 ``X≈R`` 时一致。

    张量形状均为 ``(n_rx, n_target, n_tx)``，与 ``compute_path_type`` 一致。
    """
    n_tx = x_stack.shape[0]
    d_tx = torch.linalg.vector_norm(
        t_stack[None, :, None, :] - x_stack[None, None, :, :],
        dim=-1,
    )
    d_rx = torch.linalg.vector_norm(
        r_stack[:, None, None, :] - t_stack[None, :, None, :],
        dim=-1,
    )
    range_bi = d_tx + d_rx
    d_mono = torch.linalg.vector_norm(
        r_stack[:, None, None, :] - t_stack[None, :, None, :],
        dim=-1,
    )
    range_mono = d_mono.expand(-1, -1, n_tx)
    return torch.where(is_bistatic, range_bi, range_mono)


def compute_vel(
    is_bistatic: torch.Tensor,
    t_pos: torch.Tensor,
    t_vel: torch.Tensor,
    r_pos: torch.Tensor,
    r_vel: torch.Tensor,
    x_stack: torch.Tensor,
    x_vel: torch.Tensor,
) -> torch.Tensor:
    """几何距离变化率 (m/s)，与 ``compute_range`` 的路径定义配套。

    - **monostatic**：目标相对 RX 在 ``T-R`` 连线方向上的径向速度投影（与原 ``RX 视线径向速度`` 一致）。
    - **bistatic**：双基地路径长 ``||T-X||+||R-T||`` 对时间的一阶导数，
      ``(v_T-v_X)·(T-X)/||T-X|| + (v_R-v_T)·(R-T)/||R-T||``（TX/RX/目标均可运动）。

    返回形状 ``(n_rx, n_target, n_tx)``；``x_vel`` 与 ``x_stack`` 同为 ``(n_tx, 3)``。
    """
    n_tx = x_stack.shape[0]
    n_rx = r_pos.shape[0]
    eps = torch.tensor(1e-12, dtype=t_pos.dtype, device=t_pos.device)

    # --- monostatic：沿用 RX–目标视线投影 ---
    los_tr = t_pos[None, :, :] - r_pos[:, None, :]
    dist_tr = torch.linalg.vector_norm(los_tr, dim=-1, keepdim=True).clamp_min(eps)
    u_tr = los_tr / dist_tr
    vel_mono_2d = ((t_vel[None, :, :] - r_vel[:, None, :]) * u_tr).sum(dim=-1)
    vel_mono = vel_mono_2d.unsqueeze(-1).expand(-1, -1, max(n_tx, 1)).clone()

    # --- bistatic：TX 腿 + RX 腿 ---
    diff_tx = t_pos[:, None, :] - x_stack[None, :, :]  # (n_target, n_tx, 3)
    L_tx = torch.linalg.vector_norm(diff_tx, dim=-1, keepdim=True).clamp_min(eps)
    u_tx = diff_tx / L_tx
    rate_tx = ((t_vel[:, None, :] - x_vel[None, :, :]) * u_tx).sum(
        dim=-1
    )  # (n_target, n_tx)
    rate_tx = rate_tx.unsqueeze(0).expand(n_rx, -1, -1)

    diff_rt = (
        r_pos[:, None, :] - t_pos[None, :, :]
    )  # (n_rx, n_target, 3)，与 range 中 ‖R-T‖ 一致
    L_rx = torch.linalg.vector_norm(diff_rt, dim=-1, keepdim=True).clamp_min(eps)
    u_rt = diff_rt / L_rx
    rate_rx = ((r_vel[:, None, :] - t_vel[None, :, :]) * u_rt).sum(
        dim=-1
    )  # (n_rx, n_target)
    rate_rx = rate_rx.unsqueeze(-1).expand(-1, -1, n_tx)

    vel_bi = rate_tx + rate_rx
    return torch.where(is_bistatic, vel_bi, vel_mono)


@dataclass(frozen=True)
class RxTargetTxGeometric:
    """Rx–Target–Tx 三元组几何快照（不可变，便于作为感知真值在模块间传递）。

    ``frozen=True`` 保证构造后张量引用与名称列表不被意外改写；若场景状态更新，
    应通过 ``RTScene.rx_target_tx_geometric`` 或重新调用 ``from_states`` 获取新快照。

    Attributes
    ----------
    target_names, rx_names, tx_names
        与传入 ``from_states`` 的 ``*_states`` 字典键顺序一致，决定张量各轴上的实体名称。
    type_tensor
        ``bool``，形状 ``(n_rx, n_target, n_tx)``。
        ``False``：``||R−X|| ≤ tx_rx_colocated_eps_m``，视为单基地；
        ``True``：双基地。与目标位置无关，同一 ``(rx, tx)`` 列上各目标共享类型。
    range_tensor
        ``float64``，几何路径长度 (m)。单基地为 ``||R−T||``；双基地为
        ``||T−X|| + ||R−T||``（折叠路径，不经遮挡判定）。
    vel_tensor
        ``float64``，与 ``range_tensor`` 路径定义配套的距离变化率 (m/s)。
        单基地为 RX 视线径向速度；双基地为路径长对时间的一阶导数。
        注意：双基地 ``vel_tensor`` 与 ``paths.doppler`` 的物理含义不同，
        数据集脚本中双基地速度真值可能改用 ``paths.doppler``（见 ``bistatic_sensing_eval``）。
    """

    target_names: list[str]  # 长度 n_target，张量轴 i
    rx_names: list[str]  # 长度 n_rx，张量轴 j
    tx_names: list[str]  # 长度 n_tx，张量轴 k
    type_tensor: torch.Tensor  # (n_rx, n_target, n_tx), bool
    range_tensor: torch.Tensor  # (n_rx, n_target, n_tx), float64, 单位 m
    vel_tensor: torch.Tensor  # (n_rx, n_target, n_tx), float64, 单位 m/s

    def display(
        self,
        *,
        tablefmt: str = "simple_grid",
        floatfmt: str = ".2f",
    ) -> None:
        """将各三元组的路径类型、路径长度与径向速度打印为表格（调试 / CLI 用）。

        遍历顺序与张量轴 ``(j, i, k) = (rx, target, tx)`` 一致，与 ``display`` 表头
        「接收机 / 目标 / 发射机」列顺序对应。

        Parameters
        ----------
        tablefmt
            传给 ``tabulate`` 的表格样式，默认 ``simple_grid``。
        floatfmt
            路径长度与速度的浮点格式，默认保留两位小数。
        """
        tt = self.type_tensor.detach().cpu()
        rr = self.range_tensor.detach().cpu().to(dtype=torch.float64)
        vv = self.vel_tensor.detach().cpu().to(dtype=torch.float64)
        rows: list[list[object]] = []
        # 遍历顺序与张量轴 (j, i, k) = (rx, target, tx) 一致
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
        headers = [
            "接收机",
            "目标",
            "发射机",
            "路径类型",
            "路径长度(m)",
            "径向速度(m/s)",
        ]
        table = tabulate(rows, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt)
        print("接收机–目标–发射机三元组几何：")
        print(table)

    @classmethod
    def from_states(
        cls,
        target_states: dict[str, dict[str, np.ndarray]],
        rx_states: dict[str, dict[str, np.ndarray]],
        tx_states: dict[str, dict[str, np.ndarray]],
        *,
        device: torch.device | None = None,
        tx_rx_colocated_eps_m: float = MONOSTATIC_TX_RX_EPS_M,
    ) -> "RxTargetTxGeometric":
        """由场景实体状态字典构造三元组几何。

        每个 ``states[name]`` 须含 ``pos``、``vel`` 键，值为形状 ``(3,)`` 的
        ``numpy`` 向量（单位：m、m/s）。计算链：

        1. ``stack_state_field`` → ``t_stack, r_stack, x_stack`` 及各 ``*_vel``，
           形状 ``(n_entity, 3)``；
        2. ``compute_path_type(r_stack, x_stack)`` → ``(n_rx, n_target, n_tx)``；
        3. ``compute_range`` / ``compute_vel`` 按 ``type_tensor`` 分支广播到全三元组网格。

        Parameters
        ----------
        target_states, rx_states, tx_states
            目标、接收机、发射机的状态字典；三者均须非空。
        device
            输出张量所在设备；``None`` 时使用 CPU（与 ``RTScene.rx_target_tx_geometric`` 一致）。
        tx_rx_colocated_eps_m
            判定单基地的 TX–RX 共址阈值 (m)，默认 ``MONOSTATIC_TX_RX_EPS_M``（1 mm）。

        Returns
        -------
        RxTargetTxGeometric
            三个几何张量形状均为 ``(n_rx, n_target, n_tx)``。

        Raises
        ------
        ValueError
            任一状态字典为空，或 ``stack_state_field`` 缺少 ``pos``/``vel`` 时抛出。

        See Also
        --------
        RTScene.rx_target_tx_geometric : 按当前场景状态构造并缓存的便捷属性。
        """
        if not target_states or not rx_states or not tx_states:
            raise ValueError("target_states、rx_states 与 tx_states 须均为非空字典。")

        dev = device if device is not None else torch.device("cpu")
        target_names = list(target_states.keys())
        rx_names = list(rx_states.keys())
        tx_names = list(tx_states.keys())
        n_t = len(target_names)

        # 堆叠为 (n_entity, 3)；names 列表顺序即张量第 0 维
        t_stack = stack_state_field(target_states, target_names, "pos", dev)
        t_vel = stack_state_field(target_states, target_names, "vel", dev)
        r_stack = stack_state_field(rx_states, rx_names, "pos", dev)
        r_vel = stack_state_field(rx_states, rx_names, "vel", dev)
        x_stack = stack_state_field(tx_states, tx_names, "pos", dev)
        x_vel = stack_state_field(tx_states, tx_names, "vel", dev)

        # (n_rx, n_target, n_tx)：先按 (rx, tx) 判单/双基地，再 expand 到所有目标
        type_tensor = compute_path_type(
            r_stack, x_stack, n_t, eps_m=tx_rx_colocated_eps_m
        )
        # 按 type_tensor 分支：单基地 ||R−T||，双基地 ||T−X||+||R−T||
        range_tensor = compute_range(type_tensor, t_stack, r_stack, x_stack)
        # 与 range 定义配套的 d(range)/dt
        vel_tensor = compute_vel(
            type_tensor, t_stack, t_vel, r_stack, r_vel, x_stack, x_vel
        )

        return cls(
            target_names=target_names,
            rx_names=rx_names,
            tx_names=tx_names,
            type_tensor=type_tensor,
            range_tensor=range_tensor,
            vel_tensor=vel_tensor,
        )
