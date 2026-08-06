"""GRC ``uhd_usrp_source`` 无法设置 ``issue_stream_cmd_on_start=False`` 时的补丁。"""

from __future__ import annotations

from typing import Any

_patched = False


def patch_usrp_source_factory() -> None:
    """在流图块实例化前调用，强制 ``uhd.usrp_source(..., False)``。"""
    global _patched
    if _patched:
        return

    import gnuradio.uhd as uhd

    orig = uhd.usrp_source

    def scheduled_usrp_source(
        device_addr,
        stream_args,
        issue_stream_cmd_on_start: bool = True,
    ):
        del issue_stream_cmd_on_start
        return orig(device_addr, stream_args, False)

    uhd.usrp_source = scheduled_usrp_source  # type: ignore[misc, assignment]
    _patched = True


def stop_continuous_rx_after_start(
    tb: Any,
    *,
    source_attr: str = "uhd_usrp_source_0",
) -> None:
    """兼容旧 Snippet：``patch_usrp_source_factory`` 已足够，此处为 no-op。"""
    del tb, source_attr
