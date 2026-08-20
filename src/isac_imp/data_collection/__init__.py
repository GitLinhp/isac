"""GNU Radio 数据采集工具（距离谱成对录制与离线加载）。"""

from typing import Any

from isac_imp.data_collection.cooperative_monostatic_dataset import (
    CooperativeMonostaticDataset,
    CooperativeMonostaticDatasetWriter,
    DEFAULT_COOPERATIVE_VLEN,
    build_cooperative_monostatic_h5,
    summarize_cooperative_monostatic_h5,
)
from isac_imp.data_collection.load_range_dataset import (
    PairedRangeDataset,
    load_session,
    summarize_session,
)
from isac_imp.data_collection.usrp_ofdm_single_bs_range_dataset import (
    DEFAULT_SINGLE_BS_RANGE_VLEN,
    SingleBsRangeDataset,
    SingleBsRangeDatasetWriter,
    build_usrp_ofdm_single_bs_range_h5,
    summarize_usrp_ofdm_single_bs_range_h5,
)

__all__ = [
    "CooperativeMonostaticDataset",
    "CooperativeMonostaticDatasetWriter",
    "DEFAULT_COOPERATIVE_VLEN",
    "DEFAULT_SINGLE_BS_RANGE_VLEN",
    "DevRangeProfileRecorder",
    "PairedRangeProfileRecorder",
    "RangeProfileSession",
    "PairedRangeDataset",
    "SingleBsRangeDataset",
    "SingleBsRangeDatasetWriter",
    "build_cooperative_monostatic_h5",
    "build_usrp_ofdm_single_bs_range_h5",
    "load_session",
    "summarize_cooperative_monostatic_h5",
    "summarize_session",
    "summarize_usrp_ofdm_single_bs_range_h5",
]

_GR_RECORDER_NAMES = frozenset(
    {
        "DevRangeProfileRecorder",
        "PairedRangeProfileRecorder",
        "RangeProfileSession",
    }
)


def __getattr__(name: str) -> Any:
    """惰性加载依赖 gnuradio 的录制器，避免离线训练导入失败。"""
    if name not in _GR_RECORDER_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from isac_imp.data_collection.range_profile_recorder import (
        DevRangeProfileRecorder,
        PairedRangeProfileRecorder,
        RangeProfileSession,
    )

    mapping = {
        "DevRangeProfileRecorder": DevRangeProfileRecorder,
        "PairedRangeProfileRecorder": PairedRangeProfileRecorder,
        "RangeProfileSession": RangeProfileSession,
    }
    for key, value in mapping.items():
        globals()[key] = value
    return mapping[name]
