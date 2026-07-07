"""pytest 全局 fixture：RT/Sionna 测试默认禁用 CUDA，避免无 GPU 环境失败。"""

from __future__ import annotations

import os


def pytest_configure(config: object) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
