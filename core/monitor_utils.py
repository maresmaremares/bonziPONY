"""Multi-monitor helpers — thin cross-platform wrappers around platform_compat."""

from __future__ import annotations

import logging
from typing import NamedTuple

from core.platform_compat import (
    MonitorRect,
    get_monitor_rect_for_hwnd as _compat_for_hwnd,
    get_monitor_rect_for_point as _compat_for_point,
    get_monitor_screen_rect_for_hwnd as _compat_screen_for_hwnd,
    get_virtual_desktop_rect as _compat_virtual,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MonitorRect",
    "get_monitor_rect_for_point",
    "get_monitor_rect_for_hwnd",
    "get_monitor_screen_rect_for_hwnd",
    "get_virtual_desktop_rect",
]


def get_monitor_rect_for_point(x: int, y: int) -> MonitorRect:
    """Get the work-area rect of the monitor containing point (x, y)."""
    return _compat_for_point(int(x), int(y))


def get_monitor_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Get the work-area rect of the monitor the window is on."""
    return _compat_for_hwnd(int(hwnd))


def get_monitor_screen_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Get the full screen rect (not work area) of the monitor the window is on."""
    return _compat_screen_for_hwnd(int(hwnd))


def get_virtual_desktop_rect() -> MonitorRect:
    """Get the bounding box of all monitors (virtual desktop)."""
    return _compat_virtual()
