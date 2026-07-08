from .cfar import CFARDetector
from ...data_structures.types import MetricMode, SensMode

# 向后兼容别名
metric_mode = MetricMode

__all__ = [
    "CFARDetector",
    "MetricMode",
    "SensMode",
    "metric_mode",
]
