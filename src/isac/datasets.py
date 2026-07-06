"""兼容层：请改用 ``isac.collection``。"""
from isac import DEFAULT_COLLECTION_OUT_DIR
from .collection.dataset import *  # noqa: F403

__all__ = ["DEFAULT_COLLECTION_OUT_DIR"]
