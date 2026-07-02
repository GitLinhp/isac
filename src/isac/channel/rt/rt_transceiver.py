"""射线追踪收发器封装：配置 Sionna ``Transmitter`` / ``Receiver``。

典型由 ``RTSimulator._init_transceivers`` 按 TOML 创建，校验通过后 ``scene.add`` 加入场景。
``transceiver_type`` 决定 ``self.tx`` / ``self.rx`` 是否为 ``None``（可仅 TX、仅 RX 或二者兼有）。
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from sionna.rt import Transmitter, Receiver

# 场景可视化默认颜色（RGB，各分量区间 [0, 1]）；仅在创建对应 TX/RX 时使用
DEFAULT_TX_COLOR = (0, 0, 1)  # 发射机：蓝色
DEFAULT_RX_COLOR = (0.4, 0.8, 0.4)  # 接收机：绿色


class RTTransceiver:
    """收发器配置容器：按类型创建 Sionna 发射机/接收机实例。

    - 逻辑名 ``name`` 对应 Sionna 实例名 ``{name}_tx`` / ``{name}_rx``
    - ``orientation``（欧拉角）与 ``look_at``（观察点）遵循 Sionna 惯例，通常择一配置
    - 构造与 property setter 均经 ``_validate_*`` 校验后再写入内部状态
    """

    def __init__(
        self,
        name: str,
        position: Optional[Union[List[float], Tuple[float, ...]]],
        orientation: Optional[Union[List[float], Tuple[float, ...]]] = None,
        look_at: Optional[Union[List[float], Tuple[float, ...]]] = None,
        transceiver_type: Union[str, List[str], Tuple[str, ...]] = ("tx", "rx"),
        tx_color: Optional[Tuple[float, float, float]] = None,
        rx_color: Optional[Tuple[float, float, float]] = None,
        power_dbm: Optional[float] = 44.0,
    ):
        """初始化收发器并创建 Sionna ``Transmitter`` / ``Receiver``（按 ``transceiver_type``）。

        参数:
        -------
        - name: str
            收发器逻辑名；Sionna 实例名为 ``{name}_tx`` / ``{name}_rx``。
        - position: list[float] | tuple[float, ...]
            世界坐标 ``[x, y, z]``（米），构造时必填。
        - orientation: list[float] | tuple[float, ...] | None
            欧拉角 ``[roll, pitch, yaw]``（弧度），与 ``look_at`` 通常择一。
        - look_at: list[float] | tuple[float, ...] | None
            观察点世界坐标 ``[x, y, z]``（米）；``None`` 表示不设置。
        - transceiver_type: str | list[str] | tuple[str, ...]
            ``"tx"``、``"rx"`` 或二者组合；至少含一项。
        - tx_color: tuple[float, float, float] | None
            发射机可视化 RGB ``[0, 1]``；默认 ``DEFAULT_TX_COLOR``。
        - rx_color: tuple[float, float, float] | None
            接收机可视化 RGB ``[0, 1]``；默认 ``DEFAULT_RX_COLOR``。
        - power_dbm: float | None
            发射功率（dBm），仅传入 ``Transmitter``；``None`` 时用 Sionna 库默认。
        """
        self._name = self._validate_name(name)
        self._position = self._validate_position(position)
        self._orientation = self._validate_orientation(orientation)
        self._look_at = self._validate_look_at(look_at)
        type_list = self._validate_type(transceiver_type)

        # 按 type 创建 Sionna Transmitter（name_tx）
        if "tx" in type_list:
            self.tx = Transmitter(
                name=name + "_tx",
                position=position,
                orientation=orientation,
                look_at=look_at,
                color=tx_color or DEFAULT_TX_COLOR,
                power_dbm=power_dbm,
            )
        else:
            self.tx = None

        # 按 type 创建 Sionna Receiver（name_rx）
        if "rx" in type_list:
            self.rx = Receiver(
                name=name + "_rx",
                position=position,
                orientation=orientation,
                look_at=look_at,
                color=rx_color or DEFAULT_RX_COLOR,
            )
        else:
            self.rx = None

    # ==================== 基本属性 ====================
    @property
    def name(self) -> str:
        """收发器逻辑名。"""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """设置逻辑名（经 ``_validate_name`` 校验后写入）。"""
        self._name = self._validate_name(name)

    @property
    def position(self) -> list[float]:
        """世界坐标 ``[x, y, z]``（米）；返回副本，避免外部修改内部状态。"""
        return self._position.copy()

    @position.setter
    def position(self, position: list[float]) -> None:
        """设置位置（经 ``_validate_position`` 校验后写入）。"""
        self._position = self._validate_position(position)

    @property
    def orientation(self) -> list[float]:
        """欧拉角 ``[roll, pitch, yaw]``（弧度）；返回副本，避免外部修改内部状态。"""
        return self._orientation.copy()

    @orientation.setter
    def orientation(self, orientation: list[float]) -> None:
        """设置方向（经 ``_validate_orientation`` 校验后写入）。"""
        self._orientation = self._validate_orientation(orientation)

    @property
    def look_at(self) -> list[float]:
        """观察点世界坐标 ``[x, y, z]``（米）；返回副本，避免外部修改内部状态。"""
        return self._look_at.copy()

    @look_at.setter
    def look_at(self, look_at: list[float]) -> None:
        """设置观察点（经 ``_validate_look_at`` 校验后写入）。"""
        self._look_at = self._validate_look_at(look_at)

    # ==================== 验证函数 ====================
    def _validate_name(self, name: str) -> str:
        """名称校验：非空字符串。

        参数:
        -------
        name: str
            待校验名称。

        返回:
        -------
        str
            规范化后的名称（与输入相同）。
        """
        if not isinstance(name, str):
            raise TypeError("名称必须是字符串类型")

        if len(name) == 0:
            raise ValueError("名称不能为空")

        return name

    def _validate_position(
        self, position: Optional[Union[list[float], tuple[float, ...]]]
    ) -> list[float]:
        """位置校验：三维世界坐标 ``[x, y, z]``（米）。

        构造时 ``position`` 必填；类型注解允许 ``Optional`` 仅为与其他校验函数签名一致。

        参数:
        -------
        position: list[float] | tuple[float, ...]
            待校验位置。

        返回:
        -------
        list[float]
            规范化后的位置列表。
        """
        if not isinstance(position, (list, tuple)):
            raise TypeError("位置必须是列表或元组类型")

        if len(position) != 3:
            raise ValueError("位置必须包含3个坐标值 (x, y, z)")

        for i, coord in enumerate(position):
            if not isinstance(coord, (int, float)):
                raise TypeError(f"位置坐标[{i}]必须是数值类型，当前类型: {type(coord)}")

        return list(position)

    def _validate_orientation(
        self, orientation: Optional[Union[list[float], tuple[float, ...]]]
    ) -> Optional[list[float]]:
        """方向校验：欧拉角 ``[roll, pitch, yaw]``（弧度，范围 ``[-π, π]``）。

        参数:
        -------
        orientation: list[float] | tuple[float, ...] | None
            待校验方向；``None`` 表示不设置。

        返回:
        -------
        list[float] | None
            规范化后的方向列表，或 ``None``。
        """
        if orientation is None:
            return None
        else:
            if not isinstance(orientation, (list, tuple)):
                raise TypeError("方向必须是列表或元组类型")

            if len(orientation) != 3:
                raise ValueError("方向必须包含3个角度值 (roll, pitch, yaw)")

            for i, angle in enumerate(orientation):
                if not isinstance(angle, (int, float)):
                    raise TypeError(
                        f"方向角度[{i}]必须是数值类型，当前类型: {type(angle)}"
                    )

            # 检查角度范围 [-π, π]
            for i, angle in enumerate(orientation):
                if not (-np.pi <= angle <= np.pi):
                    raise ValueError(f"方向角度[{i}]必须在-π到π之间, 当前值: {angle}")

            return list(orientation)

    def _validate_look_at(
        self, look_at: Optional[Union[list[float], tuple[float, ...]]]
    ) -> Optional[list[float]]:
        """观察点校验：世界坐标 ``[x, y, z]``（米）。

        参数:
        -------
        - look_at: list[float] | tuple[float, ...] | None
            待校验观察点；``None`` 表示不设置。

        返回:
        -------
        - list[float] | None
            规范化后的观察点列表，或 ``None``。
        """
        if look_at is None:
            return None
        else:
            if not isinstance(look_at, (list, tuple)):
                raise TypeError("朝向必须是列表或元组类型")

            if len(look_at) != 3:
                raise ValueError("朝向必须包含3个坐标值 (x, y, z)")

            for i, coord in enumerate(look_at):
                if not isinstance(coord, (int, float)):
                    raise TypeError(
                        f"朝向坐标[{i}]必须是数值类型，当前类型: {type(coord)}"
                    )

            return list(look_at)

    def _validate_type(
        self, transceiver_type: Union[str, List[str], Tuple[str, ...]]
    ) -> List[str]:
        """校验并规范化 ``transceiver_type``。

        接受单个字符串或字符串序列，保持原顺序；结果至少含 ``"tx"`` 或 ``"rx"`` 之一。

        参数:
        -------
        - transceiver_type: str | list[str] | tuple[str, ...]
            收发器类型配置。

        返回:
        -------
        - list[str]
            规范化后的类型列表（元素为 ``"tx"`` / ``"rx"``）。
        """
        if isinstance(transceiver_type, str):
            type_list = [transceiver_type]
        elif isinstance(transceiver_type, (list, tuple)):
            type_list = list(transceiver_type)
        else:
            raise TypeError(
                f"transceiver_type must be str or list[str]/tuple[str, ...], got {type(transceiver_type)}"
            )

        valid_types = {"tx", "rx"}
        for t in type_list:
            if not isinstance(t, str):
                raise TypeError(f"type list elements must be str, got {type(t)}")
            if t not in valid_types:
                raise ValueError(f"type must be one of {valid_types}, got '{t}'")

        if len(type_list) == 0:
            raise ValueError("type must contain at least one of 'tx' or 'rx'")

        return type_list
