from typing import Any
import numpy as np
import sionna.rt.scene
from sionna.rt import ITURadioMaterial
from sionna.rt import SceneObject


class RTTarget(SceneObject):
    """用于射线追踪仿真的目标：继承 Sionna `SceneObject`，位姿由外部 `update` 驱动。"""

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
    ):
        super().__init__(
            name=name,
            fname=self.resolve_fname(name, fname),
            radio_material=radio_material,
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
        """设置目标属性（在对象添加到场景后调用）。"""

        def _normalize_value(attr_name: str, value: Any) -> Any:
            if attr_name not in self._SCENE_OBJECT_ATTRIBUTES:
                return value
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            return arr.tolist()

        for attr_name in self._SCENE_OBJECT_ATTRIBUTES:
            value = kwargs.get(attr_name)
            if value is not None:
                setattr(self, attr_name, _normalize_value(attr_name, value))
