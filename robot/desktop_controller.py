"""Desktop automation — executes named preset actions and parameterized commands."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config_loader import DesktopControlConfig
    from llm.response_parser import DesktopCommand

from robot.actions import RobotAction

logger = logging.getLogger(__name__)

# Named actions that operate on the foreground window
_WINDOW_ACTIONS = {
    RobotAction.CLOSE_WINDOW,
    RobotAction.MINIMIZE_WINDOW,
    RobotAction.MAXIMIZE_WINDOW,
    RobotAction.SNAP_WINDOW_LEFT,
    RobotAction.SNAP_WINDOW_RIGHT,
    RobotAction.SHAKE,
}

_VOLUME_ACTIONS = {
    RobotAction.VOLUME_UP,
    RobotAction.VOLUME_DOWN,
    RobotAction.VOLUME_MUTE,
}

# Default allowlist for OPEN command
_DEFAULT_ALLOWED_APPS = ["notepad", "calculator", "calc", "explorer", "chrome", "firefox", "mspaint"]

# Hotkeys that must never be sent
_BLOCKED_HOTKEYS = {
    "ctrl+alt+delete", "ctrl+alt+del",
    "alt+f4",          # could close our own console/window and kill the process
    "win+l",           # lock workstation
    "win+r",           # Run dialog — combined with PASTE = arbitrary command execution
    "win+e",           # File Explorer (path traversal risk with PASTE)
    "win+x",           # Power user menu
    "win+i",           # Settings
    "win+u",           # Accessibility settings
    "win+pause",       # System info
    "ctrl+shift+escape",  # Task manager
}

# Any hotkey starting with "win+" is blocked unless explicitly allowed here.
# This prevents prompt-injection attacks from using Windows key combos to
# reach shell/run dialogs for arbitrary command execution.
_ALLOWED_WIN_HOTKEYS = {
    "win+d",           # show desktop (used by ALT_TAB command)
    "win+down",        # minimize window
    "win+up",          # maximize window
    "win+left",        # snap left
    "win+right",       # snap right
    "win+m",           # minimize all
    "win+home",        # minimize all except foreground
}


class DesktopController:
    """Handles both named preset actions and parameterized [DESKTOP:...] commands."""

    def __init__(self, config: DesktopControlConfig, pet_hwnd: int = 0) -> None:
        import pyautogui
        pyautogui.FAILSAFE = True  # Mouse to corner (0,0) aborts

        self._config = config
        self._pet_hwnd = pet_hwnd
        self._pyautogui = pyautogui
        self._last_command_time = 0.0
        self._cooldown = 0.5  # seconds between desktop commands

        # Build sets from config
        self._allowed_apps = set(
            app.lower() for app in (config.allowed_apps or _DEFAULT_ALLOWED_APPS)
        )
        self._blocked_hotkeys = set(
            hk.lower().replace(":", "+") for hk in (config.blocked_hotkeys or [])
        ) | _BLOCKED_HOTKEYS

        self._blocked_url_patterns: list[str] = []  # standing rule patterns

        logger.info("DesktopController ready (pet_hwnd=%d).", pet_hwnd)

    def set_blocked_patterns(self, patterns: list[str]) -> None:
        """Update the URL blocklist from standing rules."""
        self._blocked_url_patterns = [p.lower() for p in patterns]

    def _get_monitor_rect(self, hwnd: int = 0):
        """Get work-area MonitorRect for the given window (or pet window if 0)."""
        try:
            from core.monitor_utils import get_monitor_rect_for_hwnd
            return get_monitor_rect_for_hwnd(hwnd or self._pet_hwnd)
        except Exception:
            w, h = self._pyautogui.size()
            from core.monitor_utils import MonitorRect
            return MonitorRect(0, 0, w, h, w, h)

    def _get_foreground_hwnd(self) -> int:
        """Return the HWND of the foreground window (Windows only)."""
        try:
            import win32gui
            return win32gui.GetForegroundWindow()
        except ImportError:
            logger.warning("win32gui not available — window actions disabled.")
            return 0

    def _is_pet_window(self, hwnd: int) -> bool:
        """Check if the given HWND is the pet window itself."""
        return self._pet_hwnd != 0 and hwnd == self._pet_hwnd

    # Cache of ancestor PIDs — computed once, never changes
    _ancestor_pids: set | None = None

    @staticmethod
    def _get_ancestor_pids() -> set:
        """Get PIDs of all ancestor processes (parent, grandparent, etc.)."""
        if DesktopController._ancestor_pids is not None:
            return DesktopController._ancestor_pids
        pids = {os.getpid()}
        try:
            import ctypes
            import ctypes.wintypes

            # Snapshot all processes to build parent chain
            TH32CS_SNAPPROCESS = 0x00000002

            class PROCESSENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", ctypes.wintypes.DWORD),
                    ("cntUsage", ctypes.wintypes.DWORD),
                    ("th32ProcessID", ctypes.wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", ctypes.wintypes.DWORD),
                    ("cntThreads", ctypes.wintypes.DWORD),
                    ("th32ParentProcessID", ctypes.wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260),
                ]

            kernel32 = ctypes.windll.kernel32
            snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
            if snap == -1:
                DesktopController._ancestor_pids = pids
                return pids

            # Build PID → parent PID map
            pid_parent = {}
            pe = PROCESSENTRY32()
            pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
            if kernel32.Process32First(snap, ctypes.byref(pe)):
                while True:
                    pid_parent[pe.th32ProcessID] = pe.th32ParentProcessID
                    if not kernel32.Process32Next(snap, ctypes.byref(pe)):
                        break
            kernel32.CloseHandle(snap)

            # Walk up the parent chain
            current = os.getpid()
            for _ in range(20):  # safety limit
                parent = pid_parent.get(current)
                if parent is None or parent == 0 or parent == current:
                    break
                pids.add(parent)
                current = parent

        except Exception as exc:
            logger.debug("Failed to get ancestor PIDs: %s", exc)

        DesktopController._ancestor_pids = pids
        return pids

    @staticmethod
    def _is_own_console(hwnd: int) -> bool:
        """Check if the window is the console or terminal hosting our process.

        Checks:
        1. GetConsoleWindow() — handles legacy conhost.exe
        2. Window's owning process is in our ancestor PID chain — handles
           Windows Terminal, cmd.exe, powershell.exe, VS Code terminal, etc.
        """
        try:
            import ctypes
            # Check 1: GetConsoleWindow (catches conhost.exe pseudo-console)
            console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if console_hwnd and hwnd == console_hwnd:
                return True

            # Check 2: window's process is one of our ancestors
            import ctypes.wintypes
            pid = ctypes.wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in DesktopController._get_ancestor_pids():
                return True
        except Exception:
            pass
        return False

    def _enforce_cooldown(self) -> None:
        """Wait if we're within cooldown period of last command."""
        elapsed = time.monotonic() - self._last_command_time
        if elapsed < self._cooldown:
            time.sleep(self._cooldown - elapsed)
        self._last_command_time = time.monotonic()

    # ── Named preset actions ────────────────────────────────────────────────

    def execute_action(self, action: RobotAction) -> None:
        """Execute a named desktop action (no parameters, operates on foreground window)."""
        if action in _WINDOW_ACTIONS:
            self._execute_window_action(action)
        elif action in _VOLUME_ACTIONS:
            self._execute_volume_action(action)
        else:
            logger.debug("DesktopController ignoring non-desktop action: %s", action)

    def _execute_window_action(self, action: RobotAction) -> None:
        try:
            import win32gui
            import win32con
        except ImportError:
            logger.warning("pywin32 not installed — window actions unavailable.")
            return

        self._enforce_cooldown()
        hwnd = self._get_foreground_hwnd()
        if hwnd == 0:
            logger.warning("No foreground window found.")
            return
        if self._is_pet_window(hwnd):
            logger.info("Skipping window action — foreground is pet window.")
            return
        if self._is_own_console(hwnd):
            logger.info("Skipping window action — foreground is our own console.")
            return

        try:
            if action == RobotAction.CLOSE_WINDOW:
                logger.info("Closing window HWND=%d", hwnd)
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

            elif action == RobotAction.MINIMIZE_WINDOW:
                logger.info("Minimizing window HWND=%d", hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)

            elif action == RobotAction.MAXIMIZE_WINDOW:
                logger.info("Maximizing window HWND=%d", hwnd)
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

            elif action == RobotAction.SNAP_WINDOW_LEFT:
                mon = self._get_monitor_rect(hwnd)
                logger.info("Snapping window left HWND=%d", hwnd)
                win32gui.MoveWindow(hwnd, mon.left, mon.top, mon.width // 2, mon.height, True)

            elif action == RobotAction.SNAP_WINDOW_RIGHT:
                mon = self._get_monitor_rect(hwnd)
                logger.info("Snapping window right HWND=%d", hwnd)
                win32gui.MoveWindow(hwnd, mon.left + mon.width // 2, mon.top, mon.width // 2, mon.height, True)

            elif action == RobotAction.SHAKE:
                logger.info("Shaking foreground window HWND=%d", hwnd)
                self.shake_window(hwnd)

        except Exception as exc:
            logger.warning("Window action %s failed: %s", action, exc)

    def _execute_volume_action(self, action: RobotAction) -> None:
        self._enforce_cooldown()
        try:
            if action == RobotAction.VOLUME_UP:
                logger.info("Volume up")
                self._pyautogui.press("volumeup")
            elif action == RobotAction.VOLUME_DOWN:
                logger.info("Volume down")
                self._pyautogui.press("volumedown")
            elif action == RobotAction.VOLUME_MUTE:
                logger.info("Volume mute")
                self._pyautogui.press("volumemute")
        except Exception as exc:
            logger.warning("Volume action %s failed: %s", action, exc)

    # ── Targeted window actions (by title) ─────────────────────────────────

    def _is_browser_hwnd(self, hwnd: int) -> bool:
        """Return True if *hwnd* belongs to a known browser process."""
        try:
            import win32process
            import ctypes
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
            if h:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                ctypes.windll.kernel32.QueryFullProcessImageNameW(
                    h, 0, buf, ctypes.byref(size))
                ctypes.windll.kernel32.CloseHandle(h)
                exe = buf.value.rsplit("\\", 1)[-1].lower()
                return exe in self._BROWSER_EXE_NAMES
        except Exception:
            pass
        return False

    def close_tab_by_title(self, title_substring: str) -> bool:
        """Focus the window matching *title_substring* and send Ctrl+W to close
        only the browser tab (not the whole browser).  Falls back to WM_CLOSE
        for non-browser windows.  Returns True if the window was found."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            logger.info("No window found matching %r to close tab.", title_substring)
            return False
        if self._is_pet_window(hwnd):
            logger.info("Skipping close_tab — matched window is pet window.")
            return False
        if self._is_own_console(hwnd):
            logger.info("Skipping close_tab — matched window is our console.")
            return False

        if self._is_browser_hwnd(hwnd):
            try:
                import win32gui
                import win32con
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                import time
                time.sleep(0.15)  # let the OS finish the focus switch
                self._pyautogui.hotkey("ctrl", "w")
                logger.info("Closed browser TAB matching %r (HWND=%d)", title_substring, hwnd)
                return True
            except Exception as exc:
                logger.warning("close_tab_by_title (Ctrl+W) failed: %s", exc)
                return False
        else:
            # Non-browser window — WM_CLOSE is fine
            return self.close_window_by_title(title_substring)

    def close_window_by_title(self, title_substring: str) -> bool:
        """Close the first window whose title contains the substring. Returns True if found."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            logger.info("No window found matching %r to close.", title_substring)
            return False
        if self._is_pet_window(hwnd):
            logger.info("Skipping close — matched window is pet window.")
            return False
        if self._is_own_console(hwnd):
            logger.info("Skipping close — matched window is our own console.")
            return False
        try:
            import win32gui
            import win32con
            logger.info("Closing window %r (HWND=%d)", title_substring, hwnd)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as exc:
            logger.warning("close_window_by_title failed: %s", exc)
            return False

    def minimize_window_by_title(self, title_substring: str) -> bool:
        """Minimize the first window whose title contains the substring. Returns True if found."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            logger.info("No window found matching %r to minimize.", title_substring)
            return False
        if self._is_pet_window(hwnd):
            logger.info("Skipping minimize — matched window is pet window.")
            return False
        if self._is_own_console(hwnd):
            logger.info("Skipping minimize — matched window is our own console.")
            return False
        try:
            import win32gui
            import win32con
            logger.info("Minimizing window %r (HWND=%d)", title_substring, hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as exc:
            logger.warning("minimize_window_by_title failed: %s", exc)
            return False

    def minimize_all_windows(self) -> int:
        """Minimize every visible window except the pet. Returns count minimized."""
        try:
            import win32gui
            import win32con
        except ImportError:
            return 0

        minimized = 0

        def _callback(hwnd, _extra):
            nonlocal minimized
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if self._is_pet_window(hwnd):
                return True
            if self._is_own_console(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            try:
                if not win32gui.IsIconic(hwnd):  # not already minimized
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                    minimized += 1
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass

        logger.info("Minimized %d windows.", minimized)
        return minimized

    def _is_prominent(self, hwnd: int) -> bool:
        """Check if a window is maximized or covers a significant portion of its monitor."""
        try:
            import win32gui
            if win32gui.IsZoomed(hwnd):
                return True
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            mon = self._get_monitor_rect(hwnd)
            return w > 0 and h > 0 and (w * h) >= (mon.width * mon.height) * 0.4
        except Exception:
            return False

    def _find_window_by_title(self, title_substring: str) -> int | None:
        """Find the first visible window whose title contains the substring (case-insensitive)."""
        try:
            import win32gui
        except ImportError:
            return None

        target = title_substring.lower()
        result = [None]

        def _callback(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if title and target in title.lower():
                result[0] = hwnd
                return False  # Stop enumeration
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass  # EnumWindows raises when callback returns False — that's our "found" signal

        return result[0]

    # ── Parameterized commands ──────────────────────────────────────────────

    def execute_command(self, cmd: DesktopCommand) -> None:
        """Execute a parameterized [DESKTOP:...] command."""
        self._enforce_cooldown()
        command = cmd.command.upper()

        try:
            if command == "CLICK":
                self._cmd_click(cmd.args)
            elif command == "TYPE":
                self._cmd_type(cmd.args)
            elif command == "PASTE":
                self._cmd_paste(cmd.args)
            elif command == "HOTKEY":
                self._cmd_hotkey(cmd.args)
            elif command == "OPEN":
                self._cmd_open(cmd.args)
            elif command == "BROWSE":
                self._cmd_browse(cmd.args)
            elif command == "SCROLL":
                self._cmd_scroll(cmd.args)
            elif command == "WRITE_NOTEPAD":
                self._cmd_write_notepad(cmd.args)
            elif command == "CLOSE":
                self._cmd_close(cmd.args)
            elif command == "CLOSE_TAB":
                self._cmd_close_tab()
            elif command == "SWITCH":
                self._cmd_switch(cmd.args)
            elif command == "DRAG":
                self._cmd_drag(cmd.args)
            elif command in ("CLOSE_WINDOW", "CLOSE_WIN"):
                self._cmd_close(cmd.args)
            elif command in ("MINIMIZE", "MINIMIZE_WINDOW"):
                hwnd = self._get_foreground_hwnd()
                if hwnd and not self._is_pet_window(hwnd):
                    try:
                        import win32gui, win32con
                        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                        logger.info("Minimized foreground window.")
                    except Exception as exc:
                        logger.warning("Minimize failed: %s", exc)
            else:
                logger.warning("Unknown desktop command: %s", command)
        except Exception as exc:
            logger.warning("Desktop command %s failed: %s", command, exc)

    def _cmd_click(self, args: list[str]) -> None:
        if not self._config.click_enabled:
            logger.info("Click disabled by config.")
            return
        if len(args) < 2:
            logger.warning("CLICK requires x and y args.")
            return

        x, y = int(args[0]), int(args[1])
        from core.monitor_utils import get_virtual_desktop_rect
        virt = get_virtual_desktop_rect()

        # Bounds check against entire virtual desktop (clicks can target any monitor)
        x = max(virt.left, min(x, virt.right - 1))
        y = max(virt.top, min(y, virt.bottom - 1))

        logger.info("Click at (%d, %d)", x, y)
        self._pyautogui.click(x, y)

    def _cmd_drag(self, args: list[str]) -> None:
        """Click-and-drag from (x1, y1) to (x2, y2).

        [DESKTOP:DRAG:x1:y1:x2:y2] or [DESKTOP:DRAG:x1:y1:x2:y2:duration]
        Used for dragging browser tabs, windows, etc.
        """
        if not self._config.click_enabled:
            logger.info("Drag disabled (click_enabled=False).")
            return
        if len(args) < 4:
            logger.warning("DRAG requires x1, y1, x2, y2 args.")
            return

        from core.monitor_utils import get_virtual_desktop_rect
        virt = get_virtual_desktop_rect()

        x1, y1 = int(args[0]), int(args[1])
        x2, y2 = int(args[2]), int(args[3])
        duration = float(args[4]) if len(args) > 4 else 1.0
        duration = max(0.2, min(duration, 10.0))

        # Bounds check
        x1 = max(virt.left, min(x1, virt.right - 1))
        y1 = max(virt.top, min(y1, virt.bottom - 1))
        x2 = max(virt.left, min(x2, virt.right - 1))
        y2 = max(virt.top, min(y2, virt.bottom - 1))

        logger.info("Drag from (%d,%d) to (%d,%d) over %.1fs", x1, y1, x2, y2, duration)
        self._pyautogui.moveTo(x1, y1, duration=0.2)
        import time as _time
        _time.sleep(0.1)
        self._pyautogui.mouseDown(x1, y1)
        _time.sleep(0.05)
        self._pyautogui.moveTo(x2, y2, duration=duration)
        self._pyautogui.mouseUp()

    def drag_to_position(self, from_x: int, from_y: int, to_x: int, to_y: int,
                         duration: float = 1.5) -> None:
        """High-level drag: click at (from_x, from_y) and drag to (to_x, to_y).

        Called by agent loop for tab-drag behavior.
        """
        self._cmd_drag([str(from_x), str(from_y), str(to_x), str(to_y), str(duration)])

    def _cmd_type(self, args: list[str]) -> None:
        if not self._config.type_enabled:
            logger.info("Type disabled by config.")
            return
        if not args:
            logger.warning("TYPE requires text arg.")
            return

        text = ":".join(args)  # Rejoin in case text contained colons
        # 2000-char limit for safety
        if len(text) > 2000:
            text = text[:2000]
            logger.warning("TYPE text truncated to 2000 chars.")

        logger.info("Typing: %r", text[:80])
        # Use clipboard paste for reliability (handles unicode, newlines, speed)
        self._paste_text(text)

    def _cmd_paste(self, args: list[str]) -> None:
        """Paste text into the focused app via clipboard. [DESKTOP:PASTE:text]"""
        if not self._config.type_enabled:
            logger.info("Type/paste disabled by config.")
            return
        if not args:
            logger.warning("PASTE requires text arg.")
            return

        text = ":".join(args)  # Rejoin in case text contained colons
        text = text.replace("\\n", "\n")  # Interpret \n as real newlines
        # 5000-char limit for safety
        if len(text) > 5000:
            text = text[:5000]
            logger.warning("PASTE text truncated to 5000 chars.")

        logger.info("PASTE: %d chars into foreground app", len(text))
        self._paste_text(text)

    def _paste_text(self, text: str) -> None:
        """Copy text to clipboard and Ctrl+V into the focused window."""
        try:
            import win32clipboard
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            time.sleep(0.05)
            self._pyautogui.hotkey("ctrl", "v")
        except ImportError:
            # Fallback: use pyperclip or pyautogui.write
            try:
                import pyperclip
                pyperclip.copy(text)
                self._pyautogui.hotkey("ctrl", "v")
            except ImportError:
                # Last resort: slow character-by-character
                self._pyautogui.write(text, interval=0.02)
        except Exception as exc:
            logger.warning("Paste failed: %s — falling back to write()", exc)
            self._pyautogui.write(text[:200], interval=0.02)

    def _cmd_hotkey(self, args: list[str]) -> None:
        if not args:
            logger.warning("HOTKEY requires key args.")
            return

        # Check blocklist
        combo = "+".join(a.lower() for a in args)
        if combo in self._blocked_hotkeys:
            logger.warning("Blocked hotkey: %s", combo)
            return

        # Block ALL win+* combos except the safe allow-list.
        # This prevents prompt-injection attacks from reaching Run dialog,
        # File Explorer, or other shell entry points.
        keys_lower = [a.lower() for a in args]
        if "win" in keys_lower or "winleft" in keys_lower or "winright" in keys_lower:
            if combo not in _ALLOWED_WIN_HOTKEYS:
                logger.warning("Blocked unsafe Windows hotkey: %s", combo)
                return

        logger.info("Hotkey: %s", "+".join(args))
        self._pyautogui.hotkey(*args)

    def _cmd_open(self, args: list[str]) -> None:
        if not args:
            logger.warning("OPEN requires app name arg.")
            return

        app_name = args[0].strip()
        app_lower = app_name.lower()

        # ── 1. Try the scanned app database (fuzzy match) ──
        if DesktopController._installed_apps:
            ok, matched = self.launch_app(app_name)
            if ok:
                logger.info("OPEN: launched '%s' via app scan.", matched)
                return

        # ── 2. Fallback: bare executable name (allowlist-gated) ──
        if app_lower in self._allowed_apps:
            logger.info("Opening app (Popen fallback): %s", app_lower)
            try:
                subprocess.Popen([app_lower])
                return
            except Exception as exc:
                logger.warning("Popen fallback failed for %s: %s", app_lower, exc)

        # ── 3. Last resort: Windows Start Menu search via shell ──
        try:
            os.startfile(app_name)
            logger.info("OPEN: os.startfile('%s') succeeded.", app_name)
            return
        except OSError:
            pass

        logger.warning("OPEN: could not find or launch '%s'.", app_name)

    # ── Site shortcuts and search URL construction ───────────────────────
    _SITE_SHORTCUTS: dict[str, str] = {
        "youtube": "youtube.com", "yt": "youtube.com",
        "google": "google.com",
        "reddit": "reddit.com",
        "twitter": "twitter.com", "x": "twitter.com",
        "twitch": "twitch.tv",
        "4chan": "4chan.org",
        "github": "github.com",
        "wikipedia": "wikipedia.org",
        "steam": "store.steampowered.com",
        "discord": "discord.com",
        "instagram": "instagram.com",
        "tiktok": "tiktok.com",
        "facebook": "facebook.com",
        "spotify": "open.spotify.com",
    }

    # Patterns that trigger site-specific search URLs
    _SEARCH_SITES: dict[str, str] = {
        "youtube": "https://www.youtube.com/results?search_query={q}",
        "yt": "https://www.youtube.com/results?search_query={q}",
        "google": "https://www.google.com/search?q={q}",
        "reddit": "https://www.reddit.com/search/?q={q}",
    }

    def _cmd_browse(self, args: list[str]) -> None:
        if not args:
            logger.warning("BROWSE requires a URL or site name.")
            return

        raw = ":".join(args).strip()  # Rejoin in case URL contained colons

        # Check against standing rule blocklist before doing anything
        if self._blocked_url_patterns:
            raw_lower = raw.lower()
            for pattern in self._blocked_url_patterns:
                if pattern in raw_lower:
                    logger.warning("Blocked BROWSE — matches standing rule: %s", pattern)
                    return

        import urllib.parse

        # 1. Full URL with scheme — use as-is
        if "://" in raw:
            url = raw
        # 2. Has a dot — treat as domain/path, just add https://
        elif "." in raw:
            url = f"https://{raw}"
        else:
            # 3. No dots, no scheme — check for "site searchterms" pattern
            parts = raw.split(None, 1)
            site_key = parts[0].lower() if parts else raw.lower()

            if len(parts) == 2 and site_key in self._SEARCH_SITES:
                # "youtube cat videos" → YouTube search
                query = urllib.parse.quote_plus(parts[1])
                url = self._SEARCH_SITES[site_key].format(q=query)
            elif site_key in self._SITE_SHORTCUTS:
                # Bare "youtube" or "reddit" → open the site
                url = f"https://www.{self._SITE_SHORTCUTS[site_key]}"
            else:
                # Unknown bare text → Google search
                query = urllib.parse.quote_plus(raw)
                url = f"https://www.google.com/search?q={query}"

        # Only allow http/https schemes — block javascript:, file:, data:, vbscript: etc.
        url_lower = url.lower().strip()
        if not url_lower.startswith(("http://", "https://")):
            logger.warning("Blocked non-http URL scheme: %s", url)
            return
        # Block data: URIs disguised with double-encoding or padding
        if "data:" in url_lower or "javascript:" in url_lower or "vbscript:" in url_lower:
            logger.warning("Blocked dangerous URL content: %s", url[:80])
            return

        logger.info("Opening URL: %s", url)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Failed to open URL %s: %s", url, exc)

    def _cmd_scroll(self, args: list[str]) -> None:
        if not args:
            logger.warning("SCROLL requires amount arg.")
            return

        amount = int(args[0])
        # Clamp scroll amount
        amount = max(-20, min(20, amount))

        logger.info("Scroll: %d", amount)
        self._pyautogui.scroll(amount)

    def _cmd_write_notepad(self, args: list[str]) -> None:
        """Open a new Notepad window and paste text content into it."""
        if not self._config.type_enabled:
            logger.info("Type/write disabled by config.")
            return
        if not args:
            logger.warning("WRITE_NOTEPAD requires content arg.")
            return

        text = ":".join(args)  # Rejoin in case content contained colons
        # Interpret \n as real newlines
        text = text.replace("\\n", "\n")
        # Safety cap
        if len(text) > 5000:
            text = text[:5000]
            logger.warning("WRITE_NOTEPAD text truncated to 5000 chars.")

        logger.info("WRITE_NOTEPAD: %d chars", len(text))

        # 1. Launch notepad
        try:
            proc = subprocess.Popen(["notepad.exe"])
        except Exception as exc:
            logger.warning("Failed to launch notepad: %s", exc)
            return

        # 2. Wait for the Notepad window to appear and get focus
        try:
            import win32gui
            import win32con

            notepad_hwnd = 0
            for _ in range(60):  # up to ~3 seconds
                time.sleep(0.05)
                fg = win32gui.GetForegroundWindow()
                try:
                    cls = win32gui.GetClassName(fg)
                except Exception:
                    cls = ""
                if cls == "Notepad" or "notepad" in cls.lower():
                    notepad_hwnd = fg
                    break

            if notepad_hwnd == 0:
                logger.warning("WRITE_NOTEPAD: Notepad window not found after launch.")
                return

            # Give it a moment to finish initializing
            time.sleep(0.2)

        except ImportError:
            # No win32gui — just wait and hope
            time.sleep(1.0)

        # 3. Paste via clipboard (handles newlines, unicode, and is fast)
        try:
            import win32clipboard

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()

            # Ctrl+V to paste
            self._pyautogui.hotkey("ctrl", "v")
            logger.info("WRITE_NOTEPAD: pasted %d chars into Notepad", len(text))

        except Exception as exc:
            logger.warning("WRITE_NOTEPAD clipboard paste failed: %s", exc)

    def _cmd_close(self, args: list[str]) -> None:
        """Close a window by title substring. [DESKTOP:CLOSE:title]"""
        if not args:
            logger.warning("CLOSE requires a window title arg.")
            return
        title = ":".join(args).strip()
        if not title:
            logger.warning("CLOSE: empty title.")
            return
        found = self.close_window_by_title(title)
        if not found:
            logger.info("CLOSE: no window matching %r found.", title)

    def _cmd_close_tab(self) -> None:
        """Close the current browser tab (Ctrl+W)."""
        hwnd = self._get_foreground_hwnd()
        if hwnd == 0:
            logger.warning("CLOSE_TAB: no foreground window.")
            return
        if self._is_pet_window(hwnd):
            logger.info("CLOSE_TAB: foreground is pet window, skipping.")
            return
        if self._is_own_console(hwnd):
            logger.info("CLOSE_TAB: foreground is our console, skipping.")
            return
        logger.info("CLOSE_TAB: sending Ctrl+W to HWND=%d", hwnd)
        self._pyautogui.hotkey("ctrl", "w")

    def _cmd_switch(self, args: list[str]) -> None:
        """Switch to a window by title substring. [DESKTOP:SWITCH:title]"""
        if not args:
            logger.warning("SWITCH requires a window title arg.")
            return
        title = ":".join(args).strip()
        if not title:
            logger.warning("SWITCH: empty title.")
            return
        hwnd = self._find_window_by_title(title)
        if hwnd is None:
            logger.info("SWITCH: no window matching %r found.", title)
            return
        if self._is_pet_window(hwnd):
            logger.info("SWITCH: matched window is pet window, skipping.")
            return
        try:
            import win32gui
            import win32con
            # Restore if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            logger.info("SWITCH: brought %r to foreground (HWND=%d)", title, hwnd)
        except Exception as exc:
            logger.warning("SWITCH failed: %s", exc)

    # ── Browser focus helpers (used by AFK mischief) ─────────────────────────

    _BROWSER_EXE_NAMES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe"}

    def focus_browser(self) -> bool:
        """Find the topmost browser window and bring it to the foreground.

        Returns True if a browser window was found and focused.
        """
        try:
            import win32gui
            import win32con
            import win32process
            import ctypes
        except ImportError:
            return False

        found = [None]

        def _callback(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
                if h:
                    buf = ctypes.create_unicode_buffer(260)
                    size = ctypes.c_ulong(260)
                    ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
                    ctypes.windll.kernel32.CloseHandle(h)
                    exe = buf.value.rsplit("\\", 1)[-1].lower()
                    if exe in self._BROWSER_EXE_NAMES:
                        found[0] = hwnd
                        return False
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass

        if found[0]:
            try:
                hwnd = found[0]
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                logger.debug("Focused browser window (HWND=%d)", hwnd)
                return True
            except Exception as exc:
                logger.debug("Failed to focus browser: %s", exc)
        return False

    def move_mouse_to_center(self) -> None:
        """Move the mouse cursor to the center of the primary monitor."""
        try:
            mon = self._get_monitor_rect()
            cx = mon.left + mon.width // 2
            cy = mon.top + mon.height // 2
            self._pyautogui.moveTo(cx, cy, duration=0.3)
        except Exception as exc:
            logger.debug("move_mouse_to_center failed: %s", exc)

    # ── Advanced actions (called by agent loop) ──────────────────────────────

    def pause_media(self) -> None:
        """Press the media play/pause key to pause/toggle media playback."""
        self._enforce_cooldown()
        try:
            logger.info("Pausing/toggling media playback")
            self._pyautogui.press("playpause")
        except Exception as exc:
            logger.warning("pause_media failed: %s", exc)

    @staticmethod
    def _is_fullscreen_window(hwnd: int) -> bool:
        """Check if a window is fullscreen on its monitor."""
        try:
            import win32gui
            from core.monitor_utils import get_monitor_screen_rect_for_hwnd
            rect = win32gui.GetWindowRect(hwnd)
            mon = get_monitor_screen_rect_for_hwnd(hwnd)
            return (rect[0] <= mon.left and rect[1] <= mon.top
                    and rect[2] >= mon.right and rect[3] >= mon.bottom)
        except Exception:
            return False

    def alt_tab(self) -> None:
        """Send Win+D to minimize all windows and show the desktop."""
        self._enforce_cooldown()
        try:
            import ctypes
            VK_LWIN = 0x5B
            VK_D = 0x44
            KEYEVENTF_KEYUP = 0x0002
            user32 = ctypes.windll.user32
            user32.keybd_event(VK_LWIN, 0, 0, 0)
            user32.keybd_event(VK_D, 0, 0, 0)
            user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            logger.info("Win+D sent (show desktop / minimize all)")
        except Exception as exc:
            logger.warning("alt_tab (Win+D) failed: %s", exc)

    def system_beep(self, frequency: int = 1000, duration_ms: int = 500) -> None:
        """Play an annoying system beep."""
        try:
            import winsound
            frequency = max(37, min(frequency, 32767))
            duration_ms = max(50, min(duration_ms, 3000))
            logger.info("System beep: %dHz for %dms", frequency, duration_ms)
            winsound.Beep(frequency, duration_ms)
        except Exception as exc:
            logger.warning("system_beep failed: %s", exc)

    def shake_window(self, hwnd: int = 0, duration: float = 5.0, intensity: int = 15) -> None:
        """Rapidly vibrate a window to get the user's attention — like an alarm clock."""
        try:
            import win32gui
        except ImportError:
            return

        if hwnd == 0:
            hwnd = self._get_foreground_hwnd()
        if hwnd == 0 or self._is_pet_window(hwnd) or self._is_own_console(hwnd):
            return
        if self._is_fullscreen_window(hwnd):
            logger.info("Skipping shake — window HWND=%d is fullscreen.", hwnd)
            return

        try:
            import win32gui
            rect = win32gui.GetWindowRect(hwnd)
            orig_x, orig_y = rect[0], rect[1]
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            import random as _rand
            end_time = time.monotonic() + duration
            while time.monotonic() < end_time:
                dx = _rand.randint(-intensity, intensity)
                dy = _rand.randint(-intensity, intensity)
                win32gui.MoveWindow(hwnd, orig_x + dx, orig_y + dy, width, height, True)
                time.sleep(0.03)

            # Restore original position
            win32gui.MoveWindow(hwnd, orig_x, orig_y, width, height, True)
            logger.info("Shook window HWND=%d for %.1fs", hwnd, duration)
        except Exception as exc:
            logger.warning("shake_window failed: %s", exc)

    def shake_all_windows(self, duration: float = 8.0, intensity: int = 12) -> None:
        """Shake visible windows — earthquake mode for high urgency."""
        try:
            import win32gui
        except ImportError:
            return

        hwnds_and_rects = []
        fg_hwnd = self._get_foreground_hwnd()

        def _callback(hwnd, _extra):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if self._is_pet_window(hwnd):
                return True
            if self._is_own_console(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or not title.strip():
                return True
            try:
                if self._is_fullscreen_window(hwnd):
                    return True  # skip fullscreen windows
                rect = win32gui.GetWindowRect(hwnd)
                # Always include the foreground window
                if hwnd == fg_hwnd:
                    hwnds_and_rects.append((hwnd, rect))
                elif win32gui.IsZoomed(hwnd):
                    hwnds_and_rects.append((hwnd, rect))
                else:
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    mon = self._get_monitor_rect(hwnd)
                    if w > 0 and h > 0 and (w * h) >= (mon.width * mon.height) * 0.15:
                        hwnds_and_rects.append((hwnd, rect))
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass

        if not hwnds_and_rects:
            return

        import random as _rand
        end_time = time.monotonic() + duration
        try:
            while time.monotonic() < end_time:
                dx = _rand.randint(-intensity, intensity)
                dy = _rand.randint(-intensity, intensity)
                for hwnd, rect in hwnds_and_rects:
                    try:
                        w = rect[2] - rect[0]
                        h = rect[3] - rect[1]
                        win32gui.MoveWindow(hwnd, rect[0] + dx, rect[1] + dy, w, h, True)
                    except Exception:
                        pass
                time.sleep(0.03)

            # Restore all
            for hwnd, rect in hwnds_and_rects:
                try:
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    win32gui.MoveWindow(hwnd, rect[0], rect[1], w, h, True)
                except Exception:
                    pass
            logger.info("Shook %d windows for %.1fs", len(hwnds_and_rects), duration)
        except Exception as exc:
            logger.warning("shake_all_windows failed: %s", exc)

    def mess_with_mouse(self, duration: float = 6.0, jitter: int = 80) -> None:
        """Jitter the mouse around chaotically — for high urgency nagging."""
        import random as _rand
        try:
            start_x, start_y = self._pyautogui.position()
            from core.monitor_utils import get_monitor_rect_for_point
            mon = get_monitor_rect_for_point(start_x, start_y)
            end_time = time.monotonic() + duration

            logger.info("Messing with mouse for %.1fs", duration)
            while time.monotonic() < end_time:
                dx = _rand.randint(-jitter, jitter)
                dy = _rand.randint(-jitter, jitter)
                new_x = max(mon.left + 5, min(mon.right - 5, start_x + dx))
                new_y = max(mon.top + 5, min(mon.bottom - 5, start_y + dy))
                self._pyautogui.moveTo(new_x, new_y, duration=0.05)
                time.sleep(0.05)

            # Return mouse to roughly where it was
            self._pyautogui.moveTo(start_x, start_y, duration=0.1)
        except Exception as exc:
            logger.warning("mess_with_mouse failed: %s", exc)

    def shake_window_by_title(self, title_substring: str, duration: float = 5.0, intensity: int = 15) -> bool:
        """Shake the first window matching the title. Returns True if found and prominent."""
        hwnd = self._find_window_by_title(title_substring)
        if hwnd is None:
            return False
        if self._is_pet_window(hwnd):
            return False
        if self._is_own_console(hwnd):
            return False
        self.shake_window(hwnd=hwnd, duration=duration, intensity=intensity)
        return True

    # ── App/game library ─────────────────────────────────────────────────

    _installed_apps: list = []  # cached list of (name, launch_path, source, app_id)

    def scan_installed_apps(self) -> list:
        """Scan Steam library, Desktop, and Start Menu for installed apps.
        Returns list of (name, launch_path, source, app_id) tuples.
        Thread-safe — stores result in _installed_apps."""
        apps = []

        # ── Steam games ──────────────────────────────────────────────
        try:
            apps.extend(self._scan_steam())
        except Exception as exc:
            logger.debug("Steam scan failed: %s", exc)

        # ── Desktop shortcuts ────────────────────────────────────────
        try:
            desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
            apps.extend(self._scan_shortcuts(desktop, "desktop"))
        except Exception as exc:
            logger.debug("Desktop shortcut scan failed: %s", exc)

        # ── Start Menu shortcuts ─────────────────────────────────────
        try:
            # User start menu
            user_start = os.path.join(
                os.environ.get("APPDATA", ""),
                "Microsoft", "Windows", "Start Menu", "Programs",
            )
            apps.extend(self._scan_shortcuts(user_start, "start_menu"))
            # All-users start menu
            all_start = os.path.join(
                os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
                "Microsoft", "Windows", "Start Menu", "Programs",
            )
            apps.extend(self._scan_shortcuts(all_start, "start_menu"))
        except Exception as exc:
            logger.debug("Start Menu scan failed: %s", exc)

        # Deduplicate by name (case-insensitive)
        seen = set()
        unique = []
        for name, path, source, app_id in apps:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique.append((name, path, source, app_id))

        DesktopController._installed_apps = unique
        logger.info("Scanned %d installed apps/games.", len(unique))
        print(f"[Apps] Found {len(unique)} installed apps/games.", flush=True)
        return unique

    @staticmethod
    def _scan_steam() -> list:
        """Parse Steam library for installed games."""
        import re as _re
        results = []
        steam_path = r"C:\Program Files (x86)\Steam"
        vdf_path = os.path.join(steam_path, "steamapps", "libraryfolders.vdf")
        if not os.path.exists(vdf_path):
            return results

        # Parse library folders from VDF
        lib_paths = [os.path.join(steam_path, "steamapps")]
        try:
            with open(vdf_path, encoding="utf-8") as f:
                content = f.read()
            for match in _re.finditer(r'"path"\s+"([^"]+)"', content):
                p = os.path.join(match.group(1), "steamapps")
                if os.path.isdir(p) and p not in lib_paths:
                    lib_paths.append(p)
        except Exception:
            pass

        # Parse ACF manifest files for installed games
        for lib in lib_paths:
            try:
                for fname in os.listdir(lib):
                    if not fname.startswith("appmanifest_") or not fname.endswith(".acf"):
                        continue
                    try:
                        with open(os.path.join(lib, fname), encoding="utf-8") as f:
                            acf = f.read()
                        name_m = _re.search(r'"name"\s+"([^"]+)"', acf)
                        appid_m = _re.search(r'"appid"\s+"(\d+)"', acf)
                        if name_m and appid_m:
                            name = name_m.group(1)
                            appid = appid_m.group(1)
                            # Skip Steamworks tools / redistributables
                            if any(kw in name.lower() for kw in (
                                "redistribut", "directx", "vcredist", "proton",
                                "steamworks", "steam linux", "compatibility",
                            )):
                                continue
                            results.append((name, f"steam://rungameid/{appid}", "steam", appid))
                    except Exception:
                        continue
            except Exception:
                continue
        return results

    @staticmethod
    def _scan_shortcuts(directory: str, source: str) -> list:
        """Scan a directory for .lnk shortcuts and resolve their targets."""
        results = []
        if not os.path.isdir(directory):
            return results

        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
        except ImportError:
            # Fallback: just list the shortcut names without resolving
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    if fname.lower().endswith(".lnk"):
                        name = fname[:-4]  # strip .lnk
                        if name.lower() not in ("uninstall", "readme", "help", "website"):
                            full = os.path.join(root, fname)
                            results.append((name, full, source, ""))
            return results

        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if not fname.lower().endswith(".lnk"):
                    continue
                name = fname[:-4]
                if name.lower() in ("uninstall", "readme", "help", "website"):
                    continue
                try:
                    full = os.path.join(root, fname)
                    shortcut = shell.CreateShortCut(full)
                    target = shortcut.TargetPath
                    if target:
                        results.append((name, target, source, ""))
                    else:
                        results.append((name, full, source, ""))
                except Exception:
                    results.append((name, os.path.join(root, fname), source, ""))
        return results

    # File extensions that os.startfile is allowed to open.
    # Blocks scripts (.bat, .cmd, .vbs, .ps1, .wsf, .js) that could execute
    # arbitrary code if a malicious shortcut is placed on the Desktop.
    _SAFE_LAUNCH_EXTS = frozenset({".exe", ".lnk", ".url", ".appref-ms"})

    def launch_app(self, name: str) -> tuple:
        """Launch an app by fuzzy name match. Returns (success, matched_name)."""
        if not DesktopController._installed_apps:
            return (False, name)

        name_lower = name.lower()

        # Try exact substring match first
        for app_name, path, source, app_id in DesktopController._installed_apps:
            if name_lower in app_name.lower() or app_name.lower() in name_lower:
                try:
                    if path.startswith("steam://"):
                        webbrowser.open(path)
                    else:
                        # Only allow safe file types — block scripts (.bat, .cmd, .vbs, .ps1)
                        ext = os.path.splitext(path)[1].lower()
                        if ext not in self._SAFE_LAUNCH_EXTS:
                            logger.warning("Blocked unsafe file type for launch_app: %s (%s)", path, ext)
                            return (False, app_name)
                        os.startfile(path)
                    logger.info("Launched app: %s (%s)", app_name, source)
                    return (True, app_name)
                except Exception as exc:
                    logger.warning("Failed to launch %s: %s", app_name, exc)
                    return (False, app_name)

        return (False, name)

    @staticmethod
    def get_installed_app_names() -> list:
        """Return list of installed app names (for injection into LLM prompts)."""
        return [name for name, _, _, _ in DesktopController._installed_apps]
