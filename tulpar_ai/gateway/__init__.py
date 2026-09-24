from __future__ import annotations

from ..config import get_settings
from ..store import Store
from .base import Gateway

_gw: Gateway | None = None


def build_gateway(store: Store) -> Gateway:
    if get_settings().is_real_mode:
        from .tulpar import TulparGateway

        return TulparGateway()
    from .demo import DemoGateway

    return DemoGateway(store)


def set_gateway(gw: Gateway | None) -> None:
    global _gw
    _gw = gw


def get_gateway() -> Gateway:
    if _gw is None:
        raise RuntimeError("gateway is not started")
    return _gw
