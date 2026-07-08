"""采集子目录命名辅助函数单元测试。"""

from pathlib import Path

from isac.collection.h5_layout import (
    collection_dataset_dir,
    format_subcarrier_spacing_slug,
)


def test_format_subcarrier_spacing_slug_integer_khz() -> None:
    assert format_subcarrier_spacing_slug(30000) == "30kHz"
    assert format_subcarrier_spacing_slug(15000) == "15kHz"
    assert format_subcarrier_spacing_slug(60000) == "60kHz"


def test_format_subcarrier_spacing_slug_non_integer_khz() -> None:
    assert format_subcarrier_spacing_slug(30500) == "30p5kHz"


def test_collection_dataset_dir() -> None:
    out = collection_dataset_dir("empty_room", 30000, Path("data"))
    assert out == Path("data/empty_room_30kHz")
