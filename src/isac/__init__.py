from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- 项目目录 ---
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "out"

# --- 采集数据 ---
DEFAULT_COLLECTION_OUT_DIR = DATA_DIR
DEFAULT_DATASET_H5 = DATA_DIR / "empty_room_mc_sionna_dataset.h5"

# --- 模型产物 ---
DEFAULT_MONOSTATIC_CNN_DIR = MODELS_DIR / "monostatic_cnn"
DEFAULT_MONOSTATIC_CNN_MODEL = DEFAULT_MONOSTATIC_CNN_DIR / "best_model.pth"

__version__ = "0.1.0"
