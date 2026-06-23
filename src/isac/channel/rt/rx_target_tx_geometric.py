"""接收机–目标–发射机 (Rx–Target–Tx) 三元组几何。

在直射几何假设下，对场景中每个 ``(rx, target, tx)`` 组合计算：

- 路径类型（单基地 / 双基地，由 TX 与 RX 间距判定）；
- 几何路径长度 ``range_tensor`` (m)；
- 与路径定义配套的距离变化率 ``vel_tensor`` (m/s)。

张量索引顺序为 ``[j, i, k]`` ↔ ``(rx_names[j], target_names[i], tx_names[k])``，
形状均为 ``(n_rx, n_target, n_tx)``。底层公式见 ``isac.sensing.utils`` 中的
``compute_path_type`` / ``compute_range`` / ``compute_vel``。

典型入口：``RTScene.rx_target_tx_geometric`` 根据当前 ``targets_states``、
``rx_states``、``tx_states`` 调用 ``from_states`` 构造；感知脚本与数据集采集
常以 ``[0, 0, 0]`` 切片作为单链路真值。
"""

from dataclasses import dataclass

import numpy as np
import torch
from tabulate import tabulate

from ...sensing.utils import (
    MONOSTATIC_TX_RX_EPS_M,
    stack_state_field,
    compute_path_type,
    compute_range,
    compute_vel,
)


@dataclass(frozen=True)
class RxTargetTxGeometric:
    """Rx–Target–Tx 三元组几何快照（不可变，便于作为感知真值传递）。

    Attributes
    ----------
    target_names, rx_names, tx_names
        与 ``*_states`` 字典键顺序一致，决定张量各轴上的实体名称。
    type_tensor
        ``bool``，形状 ``(n_rx, n_target, n_tx)``。
        ``False``：TX/RX 间距 ≤ ``tx_rx_colocated_eps_m``，视为单基地；
        ``True``：双基地。
    range_tensor
        几何路径长度 (m)。单基地为 ``||R−T||``；双基地为 ``||T−X|| + ||R−T||``
        （折叠路径，不经遮挡判定）。
    vel_tensor
        与 ``range_tensor`` 配套的距离变化率 (m/s)。
        单基地为目标相对 RX 在 ``T−R`` 方向上的径向速度；
        双基地为 ``d/dt (||T−X|| + ||R−T||)``。
    """

    target_names: list[str]
    rx_names: list[str]
    tx_names: list[str]
    type_tensor: torch.Tensor
    range_tensor: torch.Tensor
    vel_tensor: torch.Tensor

    def display(
        self,
        *,
        tablefmt: str = "simple_grid",
        floatfmt: str = ".2f",
    ) -> None:
        """将各三元组的路径类型、路径长度与径向速度打印为表格。

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
        ``numpy`` 向量（单位：m、m/s）。计算链：堆叠位置/速度 → 路径类型 →
        路径长度 → 距离变化率。

        Parameters
        ----------
        target_states, rx_states, tx_states
            目标、接收机、发射机的状态字典；三者均须非空。
        device
            输出张量所在设备；``None`` 时使用 CPU。
        tx_rx_colocated_eps_m
            判定单基地的 TX–RX 共址阈值 (m)，默认 ``MONOSTATIC_TX_RX_EPS_M``。

        Returns
        -------
        RxTargetTxGeometric
            三个几何张量形状均为 ``(n_rx, n_target, n_tx)``。

        Raises
        ------
        ValueError
            任一状态字典为空时抛出。
        """
        if not target_states or not rx_states or not tx_states:
            raise ValueError("target_states、rx_states 与 tx_states 须均为非空字典。")

        dev = device if device is not None else torch.device("cpu")
        target_names = list(target_states.keys())
        rx_names = list(rx_states.keys())
        tx_names = list(tx_states.keys())
        n_t = len(target_names)

        # (n_entity, 3) 位置与速度张量，轴顺序与 names 列表对齐
        t_stack = stack_state_field(target_states, target_names, "pos", dev)
        t_vel = stack_state_field(target_states, target_names, "vel", dev)
        r_stack = stack_state_field(rx_states, rx_names, "pos", dev)
        r_vel = stack_state_field(rx_states, rx_names, "vel", dev)
        x_stack = stack_state_field(tx_states, tx_names, "pos", dev)
        x_vel = stack_state_field(tx_states, tx_names, "vel", dev)

        type_tensor = compute_path_type(
            r_stack, x_stack, n_t, eps_m=tx_rx_colocated_eps_m
        )
        range_tensor = compute_range(type_tensor, t_stack, r_stack, x_stack)
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
