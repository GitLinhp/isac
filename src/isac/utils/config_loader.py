import tomli
from pathlib import Path

from .. import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(config_file: str | Path) -> dict:
    # 层级加载策略
    config_path = CONFIG_DIR / config_file

    # 加载toml配置文件
    if config_path.exists():
        with open(config_path, "rb") as f:
            config_config = tomli.load(f)
    else:
        config_config = {}

    return config_config
