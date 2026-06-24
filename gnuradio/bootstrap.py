"""统一 gnuradio 子目录的 sys.path 引导。

``isac`` 须通过 ``pip install -e .`` 安装；本模块仅引导 blocks/core/flowgraphs/tools。
"""
import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_GNURADIO_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _GNURADIO_ROOT.parent


def _resolve_gnuradio_root(caller_file: PathLike) -> Path:
    start = Path(caller_file).resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        if (directory / "bootstrap.py").is_file():
            return directory
    raise ImportError(f"无法从 {caller_file} 定位 gnuradio 根目录（缺少 bootstrap.py）")


def ensure_isac_importable() -> None:
    try:
        import isac  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "未找到 isac 包。请在仓库根目录执行: pip install -e ."
        ) from exc


def setup_gnuradio_paths() -> tuple[Path, Path]:
    """将 blocks/、core/、flowgraphs/、tools/ 加入 sys.path，返回 (gnuradio_root, repo_root)。"""
    for subdir in ("blocks", "core", "flowgraphs", "tools"):
        path = str(_GNURADIO_ROOT / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    gnuradio_root = str(_GNURADIO_ROOT)
    if gnuradio_root not in sys.path:
        sys.path.insert(0, gnuradio_root)
    return _GNURADIO_ROOT, _REPO_ROOT


def setup_gnuradio_paths_from(caller_file: PathLike) -> tuple[Path, Path]:
    """根据调用者文件定位 gnuradio 根目录并完成路径引导。"""
    root = _resolve_gnuradio_root(caller_file)
    gnuradio_str = str(root)
    if gnuradio_str not in sys.path:
        sys.path.insert(0, gnuradio_str)
    return setup_gnuradio_paths()
