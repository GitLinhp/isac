import tomli
from pathlib import Path

from .. import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"


def resolve_config_path(config_file: str | Path) -> Path:
    """解析 TOML 配置路径。

  查找顺序：
  1. 绝对路径且存在
  2. ``config/`` 下相对路径（如 ``simulation/sensing/sensing_monostatic.toml``）
  3. 仓库根相对路径（如 ``config/simulation/sensing/sensing_monostatic.toml``）
    """
    path = Path(config_file)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        candidates.append((CONFIG_DIR / path).resolve())
        candidates.append((PROJECT_ROOT / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"配置文件不存在: {config_file!r}（已尝试: {tried}）")


def load_config(config_file: str | Path) -> dict:
    config_path = resolve_config_path(config_file)
    with open(config_path, "rb") as f:
        return tomli.load(f)
