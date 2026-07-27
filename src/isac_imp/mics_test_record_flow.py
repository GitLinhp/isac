"""mics_test / cooperative 流图录制辅助：GRC Snippet 在 main() 中调用 install_*_record_flow(tb)。"""

from __future__ import annotations

from pathlib import Path

from isac_imp.range_profile_record_limiter import bind_record_limit_handler
from isac_imp.record_target_metadata import append_cooperative_target_row


def install_mics_test_record_flow(tb) -> None:
    """注册录满 handler，由 RecordLimitBridge 在主线程调用 _apply_record_limit_stop。"""
    bind_record_limit_handler(tb)

    def _apply_record_limit_stop() -> None:
        tb.set_record_enable(False)

    tb._apply_record_limit_stop = _apply_record_limit_stop


def install_cooperative_record_flow(tb) -> None:
    """Cooperative dual-USRP：录满自动停录 + 开始录制时写目标位置 CSV。"""
    install_mics_test_record_flow(tb)

    original_set_record_enable = tb.set_record_enable

    def set_record_enable(record_enable: bool) -> None:
        was_enabled = tb.get_record_enable()
        original_set_record_enable(record_enable)
        if record_enable and not was_enabled:
            _append_target_metadata_on_record_start(tb)

    tb.set_record_enable = set_record_enable


def _append_target_metadata_on_record_start(tb) -> None:
    if not hasattr(tb, "record_file_path_dev0") or not hasattr(tb, "record_file_path_dev1"):
        return
    if not hasattr(tb, "target_pos_x_cm") or not hasattr(tb, "target_pos_y_cm"):
        return
    if not hasattr(tb, "record_output_dir_dev0"):
        return

    parent_dir = Path(tb.get_record_output_dir_dev0()).parent
    append_cooperative_target_row(
        parent_dir,
        target_x_cm=tb.get_target_pos_x_cm(),
        target_y_cm=tb.get_target_pos_y_cm(),
        dev0_file=tb.get_record_file_path_dev0(),
        dev1_file=tb.get_record_file_path_dev1(),
        record_max_frames=int(tb.get_record_max_frames()),
    )
