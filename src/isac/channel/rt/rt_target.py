from pathlib import Path
from typing import Any

import numpy as np
import sionna.rt.scene
from sionna.rt import ITURadioMaterial
from sionna.rt import SceneObject

from . import RT_SCENES_DIR


class RTTarget(SceneObject):
    """用于射线追踪仿真的目标：继承 Sionna `SceneObject`，位姿由外部 `update` 驱动。"""

    # 目标对象可设置的属性（与 Sionna SceneObject 一致）
    _SCENE_OBJECT_ATTRIBUTES = [
        "position",
        "orientation",
        "look_at",
        "scaling",
        "velocity",
    ]

    def __init__(
        self,
        name: str,  # 目标名称
        fname: str,  # mesh 逻辑名、带扩展名文件名或绝对路径
        radio_material: ITURadioMaterial,
    ):
        """初始化目标对象"""
        super().__init__(
            name=name,
            fname=self.resolve_fname(name, fname),
            radio_material=radio_material,
        )

    @staticmethod
    def _resolve_local_mesh(fname_str: str) -> str | None:
        """在 ``RT_SCENES_DIR`` 主目录查找 mesh 文件，命中则返回绝对路径。"""
        path = Path(fname_str)
        stem = path.stem if path.suffix else fname_str
        for candidate in (
            RT_SCENES_DIR / f"{stem}.ply",
            RT_SCENES_DIR / f"{stem}.obj",
            RT_SCENES_DIR / fname_str,
        ):
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    @staticmethod
    def resolve_fname(target_name: str, fname_str: str) -> Any:
        """解析目标 mesh 路径。

        按顺序查找：
        1. ``fname_str`` 本身为已存在文件路径
        2. ``RT_SCENES_DIR/{name}.ply`` / ``.obj`` 或 ``RT_SCENES_DIR/{fname_str}``
        3. ``sionna.rt.scene`` 内置 mesh 名（如 ``low_poly_car``）
        """
        path = Path(fname_str)
        if path.is_file():
            return str(path.resolve())

        local = RTTarget._resolve_local_mesh(fname_str)
        if local is not None:
            return local

        try:
            return getattr(sionna.rt.scene, fname_str)
        except AttributeError as e:
            stem = path.stem if path.suffix else fname_str
            raise ValueError(
                f"目标 '{target_name}' 的 fname '{fname_str}' 无法解析："
                f"未在 {RT_SCENES_DIR} 找到 "
                f"'{stem}.ply' / '{stem}.obj' / '{fname_str}'，"
                f"且 sionna.rt.scene 中无同名属性。"
            ) from e

    def update(self, **kwargs: Any) -> None:
        """设置目标属性（在对象添加到场景后调用）。"""

        def _normalize_value(attr_name: str, value: Any) -> Any:
            if attr_name not in self._SCENE_OBJECT_ATTRIBUTES:
                return value
            else:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                return arr.tolist()

        for attr_name in self._SCENE_OBJECT_ATTRIBUTES:
            value = kwargs.get(attr_name)
            if value is not None:
                setattr(self, attr_name, _normalize_value(attr_name, value))
