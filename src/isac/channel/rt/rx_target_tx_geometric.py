"""接收机–目标–发射机 (Rx–Target–Tx) 三元组几何。

在直射几何假设下，对场景中每个 ``(rx, target, tx)`` 组合计算：

- 路径类型（单基地 / 双基地，由 TX 与 RX 间距判定）；
- 几何路径长度 ``range_tensor`` (m)；
- 与路径定义配套的距离变化率 ``vel_tensor`` (m/s)。

符号约定：

- ``T``：目标位置/速度；``R``：接收机；``X``：发射机。
- 单基地（``type_tensor=False``）：``range = ||R-T||``，
  ``vel`` 为 ``(v_T-v_R)`` 在 ``T-R`` 方向上的投影。
- 双基地（``type_tensor=True``）：``range = ||T-X|| + ||R-T||``（折叠路径），
  ``vel = d/dt range``，TX/RX/目标均可运动。

张量索引顺序为 ``[j, i, k]`` ↔ ``(rx_names[j], target_names[i], tx_names[k])``，
形状均为 ``(n_rx, n_target, n_tx)``，dtype 为 ``float64`` / ``bool``。
底层公式见 ``compute_path_type`` / ``compute_range`` / ``compute_vel``。

典型入口与用法：

- ``RTSimulator.rx_target_tx_geometric`` 根据当前 ``targets_states``、``rx_states``、
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

# TX/RX 共址判定阈值 (m)，默认 1 mm：||R−X|| ≤ eps 时 type_tensor=False（单基地）
MONOSTATIC_TX_RX_EPS_M = 1e-3


# ==================== 状态堆叠 ====================
_STATE_FIELD_IDX = {"pos": 0, "vel": 1}


def stack_state_field(
    states: dict[str, list[np.ndarray]],
    names: list[str],
    field: str,
    device: torch.device,
) -> torch.Tensor:
    """将 ``states[name][field]`` 按 ``names`` 顺序堆叠为 float64 张量。

    参数:
    -------
    states: dict[str, list[np.ndarray]]
        实体状态字典；内层为 ``[pos, vel]`` 二元列表。
    names: list[str]
        堆叠顺序，与 ``states`` 键一致时即张量第 0 维实体顺序。
    field: str
        待读取字段名，通常为 ``"pos"`` 或 ``"vel"``。
    device: torch.device | None
        输出张量设备；``None`` 时由 ``torch.as_tensor`` 落在 CPU。

    返回:
    -------
    torch.Tensor
        形状 ``(len(names), 3)``，dtype ``float64``。
    """
    idx = _STATE_FIELD_IDX[field]
    return torch.stack(
        [
            torch.as_tensor(
                states[n][idx], dtype=torch.float64, device=device
            ).reshape(3)
            for n in names
        ],
        dim=0,
    )


# ==================== 几何计算（纯函数） ====================
def compute_path_type(
    r_stack: torch.Tensor,
    x_stack: torch.Tensor,
    n_targets: int,
    *,
    eps_m: float = MONOSTATIC_TX_RX_EPS_M,
) -> torch.Tensor:
    """判定各 ``(rx, tx)`` 对为单基地或双基地，并广播到全目标轴。

    ``False``=monostatic（``||R−X|| ≤ eps_m``），``True``=bistatic；与目标位置无关，
    同一 ``(rx, tx)`` 列上各目标共享类型。

    参数:
    -------
    r_stack: torch.Tensor
        接收机位置，形状 ``(n_rx, 3)``。
    x_stack: torch.Tensor
        发射机位置，形状 ``(n_tx, 3)``。
    n_targets: int
        目标数量，用于 expand 到 ``(n_rx, n_target, n_tx)``。
    eps_m: float
        TX/RX 共址阈值 (m)，默认 ``MONOSTATIC_TX_RX_EPS_M``。

    返回:
    -------
    torch.Tensor
        ``bool`` 张量，形状 ``(n_rx, n_target, n_tx)``。
    """
    # (n_rx, n_tx)：RX 与 TX 间距
    sep = torch.linalg.vector_norm(
        r_stack[:, None, :] - x_stack[None, :, :],
        dim=-1,
    )
    is_bistatic = sep > eps_m
    # expand 到 n_target 轴 → (n_rx, n_target, n_tx)
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

    参数:
    -------
    is_bistatic: torch.Tensor
        路径类型掩码，形状 ``(n_rx, n_target, n_tx)``。
    t_stack: torch.Tensor
        目标位置，形状 ``(n_target, 3)``。
    r_stack: torch.Tensor
        接收机位置，形状 ``(n_rx, 3)``。
    x_stack: torch.Tensor
        发射机位置，形状 ``(n_tx, 3)``。

    返回:
    -------
    torch.Tensor
        几何路径长度 (m)，形状 ``(n_rx, n_target, n_tx)``，dtype 与输入一致。
    """
    n_tx = x_stack.shape[0]
    # (n_rx, n_target, n_tx)：TX 腿 ‖T−X‖
    d_tx = torch.linalg.vector_norm(
        t_stack[None, :, None, :] - x_stack[None, None, :, :],
        dim=-1,
    )
    # (n_rx, n_target, n_tx)：RX 腿 ‖R−T‖
    d_rx = torch.linalg.vector_norm(
        r_stack[:, None, None, :] - t_stack[None, :, None, :],
        dim=-1,
    )
    range_bi = d_tx + d_rx
    # (n_rx, n_target, 1)：单基地量程 ‖R−T‖
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

    参数:
    -------
    is_bistatic: torch.Tensor
        路径类型掩码，形状 ``(n_rx, n_target, n_tx)``。
    t_pos, t_vel: torch.Tensor
        目标位置/速度，形状 ``(n_target, 3)``。
    r_pos, r_vel: torch.Tensor
        接收机位置/速度，形状 ``(n_rx, 3)``。
    x_stack, x_vel: torch.Tensor
        发射机位置/速度，形状 ``(n_tx, 3)``。

    返回:
    -------
    torch.Tensor
        距离变化率 (m/s)，形状 ``(n_rx, n_target, n_tx)``。
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


# ==================== 三元组快照 ====================
@dataclass(frozen=True)
class RxTargetTxGeometric:
    """Rx–Target–Tx 三元组几何快照（不可变，便于作为感知真值在模块间传递）。

    ``frozen=True`` 保证构造后张量引用与名称列表不被意外改写；若场景状态更新，
    应通过 ``RTSimulator.rx_target_tx_geometric`` 或重新调用 ``from_states`` 获取新快照。

    属性:
    -----
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
        """将各三元组的路径类型、路径长度与径向速度打印到 stdout（调试 / CLI 用）。

        遍历顺序与张量轴 ``(j, i, k) = (rx, target, tx)`` 一致，与表头
        「接收机 / 目标 / 发射机」列顺序对应。

        参数:
        -------
        tablefmt: str
            传给 ``tabulate`` 的表格样式，默认 ``simple_grid``。
        floatfmt: str
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
        target_states: dict[str, list[np.ndarray]],
        rx_states: dict[str, list[np.ndarray]],
        tx_states: dict[str, list[np.ndarray]],
        *,
        device: torch.device = torch.device("cpu"),
        tx_rx_colocated_eps_m: float = MONOSTATIC_TX_RX_EPS_M,
    ) -> "RxTargetTxGeometric":
        """由场景实体状态字典构造三元组几何。

        每个 ``states[name]`` 须为 ``[pos, vel]`` 二元列表，元素为形状 ``(3,)`` 的
        ``numpy`` 向量（单位：m、m/s）。计算链：

        1. ``stack_state_field`` → ``t_stack, r_stack, x_stack`` 及各 ``*_vel``，
           形状 ``(n_entity, 3)``；
        2. ``compute_path_type(r_stack, x_stack)`` → ``(n_rx, n_target, n_tx)``；
        3. ``compute_range`` / ``compute_vel`` 按 ``type_tensor`` 分支广播到全三元组网格。

        参数:
        -------
        target_states, rx_states, tx_states
            目标、接收机、发射机的状态字典；三者均须非空。
        device: torch.device | None
            输出张量所在设备；``None`` 时使用 CPU（与 ``RTSimulator.rx_target_tx_geometric`` 一致）。
        tx_rx_colocated_eps_m: float
            判定单基地的 TX–RX 共址阈值 (m)，默认 ``MONOSTATIC_TX_RX_EPS_M``（1 mm）。

        返回:
        -------
        RxTargetTxGeometric
            三个几何张量形状均为 ``(n_rx, n_target, n_tx)``。

        异常:
        ------
        ValueError
            任一状态字典为空，或 ``stack_state_field`` 字段索引无效时抛出。

        参见:
        ----
        RTSimulator.rx_target_tx_geometric
            按当前场景状态构造并缓存的便捷属性。
        """
        if not target_states or not rx_states or not tx_states:
            raise ValueError("target_states、rx_states 与 tx_states 须均为非空字典。")

        target_names = list(target_states.keys())
        rx_names = list(rx_states.keys())
        tx_names = list(tx_states.keys())
        n_t = len(target_names)

        # 堆叠为 (n_entity, 3)；names 列表顺序即张量第 0 维
        t_stack = stack_state_field(target_states, target_names, "pos", device)
        t_vel = stack_state_field(target_states, target_names, "vel", device)
        r_stack = stack_state_field(rx_states, rx_names, "pos", device)
        r_vel = stack_state_field(rx_states, rx_names, "vel", device)
        x_stack = stack_state_field(tx_states, tx_names, "pos", device)
        x_vel = stack_state_field(tx_states, tx_names, "vel", device)

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
