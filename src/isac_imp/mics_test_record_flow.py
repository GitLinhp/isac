"""mics_test 流图录制辅助：GRC Snippet 在 main() 中调用 install_mics_test_record_flow(tb)。"""

from __future__ import annotations

from isac_imp.range_profile_record_limiter import bind_record_limit_handler


def install_mics_test_record_flow(tb) -> None:
    """注册录满 handler，由 RecordLimitBridge 在主线程调用 _apply_record_limit_stop。"""
    bind_record_limit_handler(tb)

    def _apply_record_limit_stop() -> None:
        tb.set_record_enable(False)

    tb._apply_record_limit_stop = _apply_record_limit_stop
