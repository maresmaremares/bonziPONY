"""Local screen monitoring — zero API cost, polls every few seconds.

Uses ``core.platform_compat`` so the same poll loop works on Windows
(via win32gui) and Linux (via wmctrl/xdotool/python-xlib).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from core import platform_compat
from core.platform_compat import WindowInfo as _CompatWindowInfo

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    exe_name: Optional[str] = None   # e.g. "chrome.exe", "firefox", "Minecraft.exe"
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

# Both Windows .exe names and Linux process names. Lowercased; ".exe" is
# stripped before comparison so "vlc" matches "vlc.exe" on Windows.
_MEDIA_EXES = {
    "vlc", "mpv", "mpc-hc64", "mpc-hc", "mpc-be64",
    "potplayer", "potplayer64", "potplayermini64",
    "wmplayer", "smplayer", "plex", "plexmediaplayer",
    "kodi", "stremio", "jellyfinmediaplayer",
    "totem", "celluloid", "parole", "rhythmbox",  # Linux media players
}

_MEDIA_TITLE_KEYWORDS = [
    "youtube", "netflix", "hulu", "disney+", "disneyplus", "crunchyroll",
    "prime video", "primevideo", "hbo max", "peacock", "paramount+",
    "plex", "jellyfin", "twitch", "funimation", "stremio",
    "vlc media player", "mpv",
]


def _is_media_app(exe_name: Optional[str], title: str) -> bool:
    """Check if a window is a media/video application."""
    if exe_name:
        normalized = exe_name.lower()
        if normalized.endswith(".exe"):
            normalized = normalized[:-4]
        if normalized in _MEDIA_EXES:
            return True
    title_lower = (title or "").lower()
    return any(kw in title_lower for kw in _MEDIA_TITLE_KEYWORDS)


def _is_window_fullscreen(hwnd: int) -> bool:
    """Check if window covers the full monitor it's on."""
    try:
        rect = platform_compat.window_get_rect(hwnd)
        if rect is None:
            return False
        mon = platform_compat.get_monitor_screen_rect_for_hwnd(hwnd)
        return (rect.left <= mon.left and rect.top <= mon.top
                and rect.right >= mon.right and rect.bottom >= mon.bottom)
    except Exception:
        return False


class ScreenMonitor:
    """Tracks open windows, foreground app, and changes.

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

    @staticmethod
    def _to_local(info: _CompatWindowInfo, fullscreen: bool) -> WindowInfo:
        return WindowInfo(
            hwnd=info.hwnd, title=info.title, class_name=info.class_name,
            exe_name=info.exe_name, is_fullscreen=fullscreen,
        )

    def _poll_once(self) -> None:
        compat_windows = platform_compat.enumerate_windows(skip_hwnds=self._excluded_hwnds)
        if not compat_windows and not platform_compat.IS_WINDOWS and not platform_compat.IS_LINUX:
            # No backend available — disable monitor permanently
            logger.debug("No window enumeration backend available; ScreenMonitor stopping.")
            self._running = False
            return

        now = time.monotonic()
        current_windows: Dict[int, WindowInfo] = {
            info.hwnd: self._to_local(info, fullscreen=False)
            for info in compat_windows
        }

        # ── Detect foreground change ──────────────────────────────────────
        fg_compat = platform_compat.get_active_window()
        fg_hwnd = fg_compat.hwnd if fg_compat else 0
        if fg_hwnd in self._excluded_hwnds:
            fg_hwnd = 0
            fg_compat = None

        fg_info = current_windows.get(fg_hwnd)
        if fg_info is None and fg_compat is not None:
            # Foreground window may not appear in enumerate_windows (e.g. has no
            # visible title) — synthesize an entry for it.
            fg_info = self._to_local(fg_compat, fullscreen=False)
            current_windows[fg_hwnd] = fg_info

        if fg_info:
            fg_info.is_fullscreen = _is_window_fullscreen(fg_hwnd)

        with self._lock:
            # Foreground switch detection
            if fg_hwnd != self._fg_hwnd and fg_hwnd != 0:
                old_title = self._known_windows.get(self._fg_hwnd, "unknown")
                new_title = fg_info.title if fg_info else "unknown"
                new_exe = fg_info.exe_name if fg_info else None
                elapsed = self.__fmt_duration(now - self._fg_since) if self._fg_hwnd else "just now"
                exe_note = f" [{new_exe}]" if new_exe else ""
                self._add_change(
                    f'Switched from "{old_title}" to "{new_title}"{exe_note} (was active {elapsed})'
                )
                self._fg_hwnd = fg_hwnd
                self._fg_since = now

            # Detect new windows
            for hwnd, info in current_windows.items():
                if hwnd not in self._known_windows:
                    exe_note = f" [{info.exe_name}]" if info.exe_name else ""
                    self._add_change(f'Window opened: "{info.title}"{exe_note}')

            # Detect closed windows
            for hwnd, title in list(self._known_windows.items()):
                if hwnd not in current_windows:
                    self._add_change(f'Window closed: "{title}"')

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
