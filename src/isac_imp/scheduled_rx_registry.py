"""Scheduled RX：在 GRC 生成代码中延迟绑定 ``uhd.usrp_source`` 句柄。"""

from __future__ import annotations

from typing import Any

_source: Any = None


def register_usrp_source(source: Any) -> None:
    global _source
    _source = source


def get_usrp_source() -> Any:
    return _source
