"""
信道相关参数（SNR、射线追踪场景配置）
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .rt_scene_params import RtSceneParams


@dataclass
class ChannelParams:
    """信道配置：信噪比与可选射线追踪场景参数。"""

    snr_db: float = 10
    """接收端信噪比 (dB)：`no = E[|y_clean|^2] / 10^(snr_db/10)`。"""
    coderate: float = 1.0
    """码率，传入 ``ebnodb2no``；无 LDPC 时取 1（notebook 有 FEC 时填真实码率）。"""
    rt_scene: Optional[RtSceneParams] = None
    """射线追踪场景"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChannelParams":
        """从根级配置字典读取 ``snr_db``、``coderate`` 与 ``rt_scene``。"""
        rt_scene_cfg = config_dict.get("rt_scene")
        return cls(
            snr_db=float(config_dict.get("snr_db", 10)),
            coderate=float(config_dict.get("coderate", 1.0)),
            rt_scene=(
                RtSceneParams.from_dict(rt_scene_cfg)
                if isinstance(rt_scene_cfg, dict) and rt_scene_cfg
                else None
            ),
        )
