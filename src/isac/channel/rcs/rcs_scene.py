"""RCS 点目标场景运行时状态（目标 + 自耦合等开关）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rcs_target import RCSTarget

if TYPE_CHECKING:
    from ...data_structures.params.channel_params.rcs_scene_params import RCSSceneParams


@dataclass
class RCSScene:
    """RCS 点目标场景；``RCSChannel`` 经 ``Callable[[], RCSScene]`` 延迟读取。"""

    target: RCSTarget
    self_coupling_db: float = -10.0
    rndm_phaseshift: bool = True
    self_coupling: bool = True

    @classmethod
    def from_params(cls, params: RCSSceneParams) -> RCSScene:
        return cls(
            target=RCSTarget.from_params(params.target),
            self_coupling_db=params.self_coupling_db,
            rndm_phaseshift=params.rndm_phaseshift,
            self_coupling=params.self_coupling,
        )
