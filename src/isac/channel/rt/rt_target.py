from typing import Any, Mapping
import numpy as np
import sionna.rt.scene
from sionna.rt import ITURadioMaterial
from sionna.rt import SceneObject

from ...data_structures.params.rt_scene import TrajectoryParams
from .trajectory import Trajectory


class RTTarget(SceneObject):
    """用于射线追踪仿真的目标：继承 Sionna `SceneObject` 并驱动轨迹运动。

    轨迹参数仅保存在本类，不写入 `SceneObject` 构造参数。待写场景属性缓存在
    ``_pending_attributes``；须在对象加入场景后调用 `set_attributes` 一次性下发。
    轨迹推进采用累计里程插值，到达轨迹终点后停止。
    """

    # SceneObject 支持的属性列表（这些属性必须在对象添加到场景后才能设置）
    _SCENE_OBJECT_ATTRIBUTES = [
        "position",
        "orientation",
        "look_at",
        "scaling",
        "velocity",
    ]

    def __init__(
        self,
        name: str,
        fname: str,
        radio_material: ITURadioMaterial,
        trajectory_params: TrajectoryParams | Mapping[str, Any],
    ):
        """
        初始化目标

        参数:
        ----------
            - name (str): 目标名称
            - fname: 目标文件名对象（从 sionna.rt.scene 获取的对象，如 sionna.rt.scene.low_poly_car）
            - radio_material (ITURadioMaterial): 目标材料
            - trajectory_points: 可选轨迹点列表（至少一个 3D 点）
            - trajectory_velocity: 可选轨迹速度（m/s，必须 > 0）
            - **kwargs: 其他参数，包括 position, orientation, look_at, scaling, velocity, acceleration
        """
        super().__init__(
            name=name,
            fname=self.resolve_fname(name, fname),
            radio_material=radio_material,
        )
        if isinstance(trajectory_params, TrajectoryParams):
            trajectory_points = trajectory_params.points
            trajectory_velocity = trajectory_params.velocity
        elif isinstance(trajectory_params, Mapping):
            trajectory_points = trajectory_params.get("points")
            trajectory_velocity = trajectory_params.get("velocity")
        else:
            raise TypeError(
                "trajectory_params 必须为 TrajectoryParams 或包含 points/velocity 的映射类型"
            )

        self.trajectory = Trajectory(
            points=np.array(trajectory_points),
            velocity=trajectory_velocity,
        )

    @staticmethod
    def resolve_fname(target_name: str, fname_str: str) -> Any:
        """根据字符串名称解析 sionna.rt.scene 中的目标对象。"""
        try:
            return getattr(sionna.rt.scene, fname_str)
        except AttributeError as e:
            raise ValueError(
                f"目标 '{target_name}' 的 fname '{fname_str}' 在 sionna.rt.scene 中不存在。"
            ) from e

    def update(self, **kwargs: Any) -> None:
        """设置目标属性（在对象添加到场景后调用）

        注意：此方法必须在 SceneObject 添加到场景后调用，否则会抛出 ValueError。
        """

        def _normalize_value(attr_name: str, value: Any) -> Any:
            if attr_name not in self._SCENE_OBJECT_ATTRIBUTES:
                return value
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            return arr.tolist()

        # 设置目标位置、朝向、速度等（只有当值不为 None 时才设置）
        # 注意：Sionna 的 SceneObject 属性不接受 None 值。
        for attr_name in self._SCENE_OBJECT_ATTRIBUTES:
            value = kwargs.get(attr_name)
            if value is not None:
                setattr(self, attr_name, _normalize_value(attr_name, value))
