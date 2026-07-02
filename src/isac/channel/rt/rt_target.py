from typing import Any

import numpy as np
import sionna.rt.scene
from sionna.rt import ITURadioMaterial
from sionna.rt import SceneObject

from . import RT_SCENES_DIR


class RTTarget(SceneObject):
    """用于射线追踪仿真的目标：继承 Sionna `SceneObject`，位姿由外部 ``__call__`` 驱动。"""

    # 目标对象可实时跟新的属性
    _SCENE_OBJECT_ATTRIBUTES = [
        "position",  # 目标位置
        "orientation",  # 目标朝向
        "look_at",  # 目标观察点
        "velocity",  # 目标速度
    ]

    def __init__(
        self,
        name: str,  # 目标名称
        fname: str,  # mesh 逻辑名（如 cube）
        radio_material: ITURadioMaterial,  # 目标材质
        scaling: float = 1.0,  # 目标缩放比例
    ):
        """初始化目标对象"""
        super().__init__(
            name=name,
            fname=self._resolve_fname(name, fname),
            radio_material=radio_material,
        )
        self.scaling = scaling  # 只在目标初始化时设置，之后不可修改

    def _resolve_fname(self, target_name: str, fname_str: str) -> str:
        """解析目标 mesh 路径：``RT_SCENES_DIR/{name}.ply`` 或 Sionna 内置 mesh。

        参数:
        -------
        - target_name: str
            目标名称，仅用于错误信息。
        - fname_str: str
            mesh 逻辑名（如 ``cube``）。

        返回:
        -------
        - str
            绝对路径字符串（Mitsuba/Sionna 要求 ``fname`` 为 str，非 Path）。
        """
        # 检查 fname_str 是否为逻辑名
        if not fname_str or "/" in fname_str or "\\" in fname_str or "." in fname_str:
            raise ValueError(
                f"目标 '{target_name}' 的 fname '{fname_str}' 仅支持逻辑名（无路径、无扩展名）。"
            )

        # 检查本地 mesh 是否存在
        local_ply = RT_SCENES_DIR / f"{fname_str}.ply"
        if local_ply.is_file():
            return str(local_ply.resolve())

        # 检查 Sionna 内置 mesh 是否存在
        resolved = getattr(sionna.rt.scene, fname_str, None)
        if resolved is not None:
            return resolved
        else:
            raise ValueError(
                f"目标 '{target_name}' 的 fname '{fname_str}' 无法解析："
                f"未找到 {local_ply}，且 sionna.rt.scene 无此 mesh。"
            )

    def __call__(self, **kwargs: Any) -> None:
        """在对象已加入场景后更新位姿/速度（仅 ``_SCENE_OBJECT_ATTRIBUTES``）。

        参数:
        ----------
        - kwargs: Any
            更新属性，仅支持 ``_SCENE_OBJECT_ATTRIBUTES`` 中的属性。

        返回:
        -------
        - None
        """
        # 检查是否存在不支持的属性
        unknown = set(kwargs) - set(self._SCENE_OBJECT_ATTRIBUTES)
        if unknown:
            raise ValueError(
                f"不支持的目标属性 {sorted(unknown)!r}；"
                f"允许: {self._SCENE_OBJECT_ATTRIBUTES}"
            )

        # 更新目标属性
        for attr_name, value in kwargs.items():
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            setattr(self, attr_name, arr.tolist())
