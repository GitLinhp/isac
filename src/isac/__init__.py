import os
from pathlib import Path

# 设置TensorFlow日志级别为ERROR，屏蔽INFO和WARNING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

from .utils import *
from .data_structures import *

__version__ = "0.1.0"
