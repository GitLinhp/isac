"""仓库 data/ 下录制目录解析。"""

from __future__ import annotations

from pathlib import Path


def repo_data_dir(*parts: str) -> str:
    """相对仓库根目录拼接路径并 mkdir(parents=True, exist_ok=True)。"""
    root = Path(__file__).resolve().parents[2]
    out = root.joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return str(out)
