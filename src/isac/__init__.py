"""ISAC 仿真与感知库。

子包地图
--------
- ``isac.system`` — 顶层编排（``System.transmit`` / ``receive``）
- ``isac.transmit_cache`` — 发射波形 ``.npy`` 磁盘缓存（``TransmitCache``）
- ``isac.data_structures`` — TOML 配置 dataclass 与 ``SystemComponents`` 工厂
- ``isac.channel`` — 信道仿真（RT / RCS / AWGN）
- ``isac.sensing`` — 感知 DSP（DD 谱、CFAR、MUSIC、MTI/MTD）
- ``isac.collection`` — 蒙特卡洛采集与 ``RTDataset``
- ``isac.models`` — 深度学习感知模型
- ``isac.utils`` — 横切工具（配置加载、类型转换、窗函数）

典型入口::

    from isac.system import System
    from isac.data_structures import SystemParams, SystemComponents
    from isac.sensing import MUSICEstimator, DelayDopplerSpectrum
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- 项目目录 ---
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "out"

# --- 采集数据 ---
DEFAULT_COLLECTION_OUT_DIR = DATA_DIR
DEFAULT_DATASET_H5 = (
    DATA_DIR
    / "empty_room_monostatic_30kHz"
    / "empty_room_monostatic_mc_sionna_dataset.h5"
)
DEFAULT_BISTATIC_DATASET_H5 = (
    DATA_DIR
    / "empty_room_bistatic_30kHz"
    / "empty_room_bistatic_mc_sionna_dataset.h5"
)

# --- 模型产物 ---
DEFAULT_SENSING_CNN_DIR = MODELS_DIR / "sensing_cnn"
DEFAULT_SENSING_CNN_MODEL = DEFAULT_SENSING_CNN_DIR / "monostatic" / "best_model.pth"
DEFAULT_BISTATIC_SENSING_CNN_MODEL = (
    DEFAULT_SENSING_CNN_DIR / "bistatic" / "best_model.pth"
)
# 向后兼容别名
DEFAULT_MONOSTATIC_CNN_DIR = DEFAULT_SENSING_CNN_DIR / "monostatic"
DEFAULT_MONOSTATIC_CNN_MODEL = DEFAULT_SENSING_CNN_MODEL

__version__ = "0.1.0"
