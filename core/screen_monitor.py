"""Local screen monitoring via win32gui — zero API cost, runs every few seconds."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    exe_name: Optional[str] = None   # e.g. "chrome.exe", "Minecraft.exe"
    is_fullscreen: bool = False       # taking up the whole monitor


@dataclass
class ScreenState:
    foreground: Optional[WindowInfo]
    foreground_duration_s: float
    open_windows: List[WindowInfo]
    recent_changes: List[str]
    timestamp: float
    is_media_fullscreen: bool = False  # user is watching video/media in fullscreen


# ── Media app detection ──────────────────────────────────────────────────

_MEDIA_EXES = {
    "vlc.exe", "mpv.exe", "mpc-hc64.exe", "mpc-hc.exe", "mpc-be64.exe",
    "potplayer.exe", "potplayer64.exe", "potplayermini64.exe",
    "wmplayer.exe", "smplayer.exe", "plex.exe", "plexmediaplayer.exe",
    "kodi.exe", "stremio.exe", "jellyfinmediaplayer.exe",
}

_MEDIA_TITLE_KEYWORDS = [
    "youtube", "netflix", "hulu", "disney+", "disneyplus", "crunchyroll",
    "prime video", "primevideo", "hbo max", "peacock", "paramount+",
    "plex", "jellyfin", "twitch", "funimation", "stremio",
    "vlc media player", "mpv",
]


def _is_media_app(exe_name: Optional[str], title: str) -> bool:
    """Check if a window is a media/video application."""
    if exe_name and exe_name.lower() in _MEDIA_EXES:
        return True
    title_lower = title.lower()
    return any(kw in title_lower for kw in _MEDIA_TITLE_KEYWORDS)


# ── Win32 helpers for getting process exe name ────────────────────────────

def _get_exe_name(hwnd: int) -> Optional[str]:
    """Get the executable name for a window handle using ctypes (no extra deps)."""
    try:
        # Get PID from hwnd
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return None

        # Open process with PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return None

        try:
            # QueryFullProcessImageNameW
            buf = ctypes.create_unicode_buffer(512)
            size = ctypes.c_ulong(512)
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            )
            if ok and buf.value:
                return os.path.basename(buf.value)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass
    return None


def _is_fullscreen(hwnd: int) -> bool:
    """Check if window covers the full monitor it's on."""
    try:
        import win32gui
        from core.monitor_utils import get_monitor_screen_rect_for_hwnd
        rect = win32gui.GetWindowRect(hwnd)
        mon = get_monitor_screen_rect_for_hwnd(hwnd)
        # Window covers the full monitor (or larger, for borderless)
        return (rect[0] <= mon.left and rect[1] <= mon.top
                and rect[2] >= mon.right and rect[3] >= mon.bottom)
    except Exception:
        return False


class ScreenMonitor:
    """Tracks open windows, foreground app, and changes using win32gui.

    Runs on a daemon thread, polling every ``poll_interval`` seconds.
    Call ``get_state()`` from any thread to get a snapshot.
    """

    def __init__(self, pet_hwnd: int = 0, poll_interval: float = 3.0) -> None:
        self._pet_hwnd = pet_hwnd
        self._excluded_hwnds: set[int] = {pet_hwnd} if pet_hwnd else set()
        self._poll_interval = poll_interval

        # Foreground tracking
        self._fg_hwnd: int = 0
        self._fg_since: float = 0.0

        # Window tracking
        self._known_windows: Dict[int, str] = {}  # hwnd → title

        # Exe name cache (PID lookups are slow, cache per hwnd)
        self._exe_cache: Dict[int, Optional[str]] = {}

        # Change log
        self._changes: List[str] = []
        self._start_time: float = time.monotonic()

        # Thread safety
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Current snapshot
        self._state = ScreenState(
            foreground=None,
            foreground_duration_s=0.0,
            open_windows=[],
            recent_changes=[],
            timestamp=time.monotonic(),
        )

    def exclude_hwnd(self, hwnd: int) -> None:
        """Add a window handle to the exclusion set (e.g. secondary pony windows)."""
        self._excluded_hwnds.add(hwnd)

    def include_hwnd(self, hwnd: int) -> None:
        """Remove a window handle from the exclusion set."""
        self._excluded_hwnds.discard(hwnd)

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="screen-monitor")
        self._thread.start()
        logger.info("ScreenMonitor started (poll_interval=%.1fs).", self._poll_interval)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("ScreenMonitor stopped.")

    def get_state(self) -> ScreenState:
        """Return a thread-safe snapshot of the current screen state."""
        with self._lock:
            # Update foreground duration live
            now = time.monotonic()
            fg_dur = (now - self._fg_since) if self._fg_hwnd else 0.0
            return ScreenState(
                foreground=self._state.foreground,
                foreground_duration_s=fg_dur,
                open_windows=list(self._state.open_windows),
                recent_changes=list(self._changes[-20:]),
                timestamp=now,
            )

    # ── Internal ────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: poll windows every interval."""
        while self._running:
            try:
                self._poll_once()
            except Exception as exc:
                logger.debug("ScreenMonitor poll error: %s", exc)
            time.sleep(self._poll_interval)

    def _get_cached_exe(self, hwnd: int) -> Optional[str]:
        """Get exe name with caching."""
        if hwnd not in self._exe_cache:
            self._exe_cache[hwnd] = _get_exe_name(hwnd)
        return self._exe_cache.get(hwnd)

    def _poll_once(self) -> None:
        try:
            import win32gui
        except ImportError:
            logger.debug("win32gui not available — screen monitor disabled.")
            self._running = False
            return

        now = time.monotonic()

        # ── Enumerate visible windows ─────────────────────────────────────
        current_windows: Dict[int, WindowInfo] = {}

        def _enum_callback(hwnd: int, _extra) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            if hwnd == self._pet_hwnd or hwnd in self._excluded_hwnds:
                return
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return
            try:
                class_name = win32gui.GetClassName(hwnd)
            except Exception:
                class_name = ""

            exe = self._get_cached_exe(hwnd)
            fullscreen = _is_fullscreen(hwnd) if hwnd == self._fg_hwnd else False

            current_windows[hwnd] = WindowInfo(
                hwnd=hwnd, title=title, class_name=class_name,
                exe_name=exe, is_fullscreen=fullscreen,
            )

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception as exc:
            logger.debug("EnumWindows failed: %s", exc)
            return

        # ── Detect foreground change ──────────────────────────────────────
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
        except Exception:
            fg_hwnd = 0

        fg_info = current_windows.get(fg_hwnd)
        # Update fullscreen status for new foreground
        if fg_info:
            fg_info.is_fullscreen = _is_fullscreen(fg_hwnd)

        with self._lock:
            # Foreground switch detection
            if fg_hwnd != self._fg_hwnd and fg_hwnd != 0:
                old_title = self._known_windows.get(self._fg_hwnd, "unknown")
                new_title = fg_info.title if fg_info else "unknown"
                new_exe = fg_info.exe_name if fg_info else None
                elapsed = self.__fmt_duration(now - self._fg_since) if self._fg_hwnd else "just now"
                exe_note = f" [{new_exe}]" if new_exe else ""
                self._add_change(f"Switched from \"{old_title}\" to \"{new_title}\"{exe_note} (was active {elapsed})")
                self._fg_hwnd = fg_hwnd
                self._fg_since = now

            # Detect new windows
            for hwnd, info in current_windows.items():
                if hwnd not in self._known_windows:
                    exe_note = f" [{info.exe_name}]" if info.exe_name else ""
                    self._add_change(f"Window opened: \"{info.title}\"{exe_note}")

            # Detect closed windows
            for hwnd, title in list(self._known_windows.items()):
                if hwnd not in current_windows:
                    self._add_change(f"Window closed: \"{title}\"")
                    # Clean up exe cache
                    self._exe_cache.pop(hwnd, None)

            # Update known windows
            self._known_windows = {h: info.title for h, info in current_windows.items()}

            # Update state
            fg_dur = (now - self._fg_since) if self._fg_hwnd else 0.0
            media_fs = (
                fg_info is not None
                and fg_info.is_fullscreen
                and _is_media_app(fg_info.exe_name, fg_info.title)
            )
            self._state = ScreenState(
                foreground=fg_info,
                foreground_duration_s=fg_dur,
                open_windows=list(current_windows.values()),
                recent_changes=list(self._changes[-20:]),
                timestamp=now,
                is_media_fullscreen=media_fs,
            )

    def _add_change(self, description: str) -> None:
        """Add a change event (must hold lock)."""
        elapsed = self.__fmt_duration(time.monotonic() - self._start_time)
        entry = f"[{elapsed} ago] {description}"
        self._changes.append(entry)
        if len(self._changes) > 50:
            self._changes = self._changes[-20:]
        logger.debug("Screen change: %s", description)

    @staticmethod
    def __fmt_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f} min"
        else:
            return f"{seconds / 3600:.1f}h"
