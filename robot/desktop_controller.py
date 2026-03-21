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

        logger.info("DesktopController ready (pet_hwnd=%d).", pet_hwnd)

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

    def _cmd_type(self, args: list[str]) -> None:
        if not self._config.type_enabled:
            logger.info("Type disabled by config.")
            return
        if not args:
            logger.warning("TYPE requires text arg.")
            return

        text = ":".join(args)  # Rejoin in case text contained colons
        # 200-char limit for safety
        if len(text) > 200:
            text = text[:200]
            logger.warning("TYPE text truncated to 200 chars.")

        logger.info("Typing: %r", text)
        self._pyautogui.write(text, interval=0.02)

    def _cmd_hotkey(self, args: list[str]) -> None:
        if not args:
            logger.warning("HOTKEY requires key args.")
            return

        # Check blocklist
        combo = "+".join(a.lower() for a in args)
        if combo in self._blocked_hotkeys:
            logger.warning("Blocked hotkey: %s", combo)
            return

        logger.info("Hotkey: %s", "+".join(args))
        self._pyautogui.hotkey(*args)

    def _cmd_open(self, args: list[str]) -> None:
        if not args:
            logger.warning("OPEN requires app name arg.")
            return

        app_name = args[0].lower().strip()
        if app_name not in self._allowed_apps:
            logger.warning("App not in allowlist: %s (allowed: %s)", app_name, self._allowed_apps)
            return

        logger.info("Opening app: %s", app_name)
        try:
            subprocess.Popen([app_name])
        except Exception as exc:
            logger.warning("Failed to open %s: %s", app_name, exc)

    def _cmd_browse(self, args: list[str]) -> None:
        if not args:
            logger.warning("BROWSE requires a URL or site name.")
            return

        raw = ":".join(args).strip()  # Rejoin in case URL contained colons

        # Figure out the URL
        if "://" in raw:
            url = raw
        elif "." in raw:
            url = f"https://{raw}"
        else:
            url = f"https://www.{raw}.com"

        # Only allow http/https schemes
        if not url.lower().startswith(("http://", "https://")):
            logger.warning("Blocked non-http URL scheme: %s", url)
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
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()

            # Ctrl+V to paste
            self._pyautogui.hotkey("ctrl", "v")
            logger.info("WRITE_NOTEPAD: pasted %d chars into Notepad", len(text))

        except Exception as exc:
            logger.warning("WRITE_NOTEPAD clipboard paste failed: %s", exc)

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
