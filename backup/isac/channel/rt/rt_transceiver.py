"""收发器模块，提供收发器配置和管理功能。"""

from typing import Any, Optional, Union, List, Tuple
import numpy as np
from sionna.rt import Transmitter, Receiver

# 默认收发机颜色
DEFAULT_TX_COLOR = (0, 0, 1)  # 蓝色
DEFAULT_RX_COLOR = (0.4, 0.8, 0.4)  # 绿色 0-1


class RTTransceiver:
    """收发器类

    管理发射器和接收器的配置，支持单独配置或同时配置。
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
        power_dbm: Optional[float] = None,
    ):
        """初始化收发器

        参数:
        -------
            - name (str): 收发器名称
            - position (list[float] | tuple[float, ...], optional): 位置坐标
            - orientation (list[float] | tuple[float, ...], optional): 方向
            - look_at (list[float] | tuple[float, ...], optional): 朝向
            - transceiver_type (str | list[str] | tuple[str, ...]): 类型，可以是 'tx'、'rx' 或两者组合
            - power_dbm (float | None): 发射功率 (dBm)，传入 Sionna ``Transmitter``；仅创建 TX 时有效；``None`` 用库默认
        """
        self._name = self._validate_name(name)
        self._position = self._validate_position(position)
        self._orientation = self._validate_orientation(orientation)
        self._look_at = self._validate_look_at(look_at)

        self.tx = None
        self.rx = None

        # 验证并规范化类型参数
        type_list = self._validate_type(transceiver_type)
        if "tx" in type_list:
            tx_kw: dict[str, Any] = {
                "name": name + "_tx",
                "position": position,
                "orientation": orientation,
                "look_at": look_at,
                "color": tx_color or DEFAULT_TX_COLOR,
            }
            if power_dbm is not None:
                tx_kw["power_dbm"] = power_dbm
            self.tx = Transmitter(**tx_kw)
        if "rx" in type_list:
            self.rx = Receiver(
                name=name + "_rx",
                position=position,
                orientation=orientation,
                look_at=look_at,
                color=rx_color or DEFAULT_RX_COLOR,
            )

    # ==================== 基本属性 ====================
    @property
    def name(self) -> str:
        """获取名称"""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """设置名称（带校验）"""
        self._name = self._validate_name(name)

    @property
    def position(self) -> list[float]:
        """获取位置"""
        return self._position.copy()

    @position.setter
    def position(self, position: list[float]) -> None:
        """设置位置（带校验）"""
        self._position = self._validate_position(position)

    @property
    def orientation(self) -> list[float]:
        """获取方向"""
        return self._orientation.copy()

    @orientation.setter
    def orientation(self, orientation: list[float]) -> None:
        """设置方向（带校验）"""
        self._orientation = self._validate_orientation(orientation)

    @property
    def look_at(self) -> list[float]:
        """获取朝向"""
        return self._look_at.copy()

    @look_at.setter
    def look_at(self, look_at: list[float]) -> None:
        """设置朝向（带校验）"""
        self._look_at = self._validate_look_at(look_at)

    # ==================== 验证函数 ====================
    def _validate_type(
        self, transceiver_type: Union[str, List[str], Tuple[str, ...]]
    ) -> List[str]:
        """类型校验和规范化

        参数:
        -------
            - transceiver_type: 类型，可以是 'tx'、'rx' 或两者组合

        返回:
        -------
            - list[str]: 规范化后的类型列表
        """
        # 统一转换成 list[str]
        if isinstance(transceiver_type, str):
            type_list = [transceiver_type]
        elif isinstance(transceiver_type, (list, tuple)):
            type_list = list(transceiver_type)
        else:
            raise TypeError(
                f"transceiver_type must be str or list[str]/tuple[str, ...], got {type(transceiver_type)}"
            )

        # 验证列表中的每个值
        valid_types = {"tx", "rx"}
        for t in type_list:
            if not isinstance(t, str):
                raise TypeError(f"type list elements must be str, got {type(t)}")
            if t not in valid_types:
                raise ValueError(f"type must be one of {valid_types}, got '{t}'")

        # 去重并保持顺序
        seen = set()
        result = []
        for t in type_list:
            if t not in seen:
                seen.add(t)
                result.append(t)

        if len(result) == 0:
            raise ValueError("type must contain at least one of 'tx' or 'rx'")

        return result

    def _validate_name(self, name: str) -> str:
        """名称校验"""
        if not isinstance(name, str):
            raise TypeError("名称必须是字符串类型")

        if len(name) == 0:
            raise ValueError("名称不能为空")

        return name

    def _validate_position(
        self, position: Optional[Union[list[float], tuple[float, ...]]]
    ) -> list[float]:
        """位置校验"""
        if not isinstance(position, (list, tuple)):
            raise TypeError("位置必须是列表或元组类型")

        if len(position) != 3:
            raise ValueError("位置必须包含3个坐标值 (x, y, z)")

        # 检查是否为数值类型
        for i, coord in enumerate(position):
            if not isinstance(coord, (int, float)):
                raise TypeError(f"位置坐标[{i}]必须是数值类型，当前类型: {type(coord)}")

        return list(position)

    def _validate_orientation(
        self, orientation: Optional[Union[list[float], tuple[float, ...]]]
    ) -> Optional[list[float]]:
        """方向校验"""
        if orientation is None:
            return None
        else:
            if not isinstance(orientation, (list, tuple)):
                raise TypeError("方向必须是列表或元组类型")

            if len(orientation) != 3:
                raise ValueError("方向必须包含3个角度值 (roll, pitch, yaw)")

            # 检查是否为数值类型
            for i, angle in enumerate(orientation):
                if not isinstance(angle, (int, float)):
                    raise TypeError(
                        f"方向角度[{i}]必须是数值类型，当前类型: {type(angle)}"
                    )

            # 检查角度范围（可选）
            for i, angle in enumerate(orientation):
                if not (-np.pi <= angle <= np.pi):
                    raise ValueError(f"方向角度[{i}]必须在-π到π之间, 当前值: {angle}")

            return list(orientation)

    def _validate_look_at(
        self, look_at: Optional[Union[list[float], tuple[float, ...]]]
    ) -> Optional[list[float]]:
        """朝向校验"""
        if look_at is None:
            return None
        else:
            if not isinstance(look_at, (list, tuple)):
                raise TypeError("朝向必须是列表或元组类型")

            if len(look_at) != 3:
                raise ValueError("朝向必须包含3个坐标值 (x, y, z)")

            # 检查是否为数值类型
            for i, coord in enumerate(look_at):
                if not isinstance(coord, (int, float)):
                    raise TypeError(
                        f"朝向坐标[{i}]必须是数值类型，当前类型: {type(coord)}"
                    )

            return list(look_at)
