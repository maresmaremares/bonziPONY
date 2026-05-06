"""
Cross-platform compatibility layer.

Centralizes every OS-specific operation so the rest of the codebase doesn't
need ``sys.platform`` branches or ``win32*`` imports. Each function exposes
a uniform interface and dispatches to a Windows or Linux backend internally.

Linux backend depends on (any subset of):
    wmctrl    — window list/close/focus/move
    xdotool   — active window, key send, fine-grained window ops
    xprintidle — idle ms (preferred over the python-xlib screensaver query)
    xdg-open  — file/URL launch
    xclip / xsel — clipboard (via the pyperclip package)
    tesseract — OCR (via the pytesseract package)
    screeninfo — monitor enumeration

Missing tools are detected at module load and the corresponding functions
log a warning and return a sentinel. Callers should treat None / 0 / False /
empty list as "feature unavailable".
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"


# ── Shared types ──────────────────────────────────────────────────────────

class WindowInfo(NamedTuple):
    hwnd: int
    title: str
    pid: int
    class_name: str
    exe_name: Optional[str]


class Rect(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class MonitorRect(NamedTuple):
    left: int
    top: int
    width: int
    height: int
    right: int
    bottom: int


@dataclass
class InstalledApp:
    name: str          # display name, e.g. "Firefox"
    exec_cmd: str      # command to invoke
    icon: Optional[str] = None
    categories: Optional[str] = None


# ── Tool availability (Linux) ──────────────────────────────────────────────

_LINUX_TOOLS: dict[str, Optional[str]] = {}
if IS_LINUX:
    for tool in ("wmctrl", "xdotool", "xprintidle", "xdg-open", "xprop"):
        _LINUX_TOOLS[tool] = shutil.which(tool)
        if not _LINUX_TOOLS[tool]:
            logger.debug("Linux tool not on PATH: %s (some features will degrade)", tool)


def _have(tool: str) -> bool:
    return bool(_LINUX_TOOLS.get(tool))


def _run(argv: Sequence[str], timeout: float = 3.0) -> Tuple[int, str]:
    """Run a subprocess, returning (rc, stdout). Logs and returns (-1, '') on failure."""
    try:
        result = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("subprocess failed (%s): %s", argv[0] if argv else "?", exc)
        return -1, ""


# ══════════════════════════════════════════════════════════════════════════
# File / URL opening
# ══════════════════════════════════════════════════════════════════════════

def open_path(path) -> bool:
    """Open a file, folder, or URL with the OS default handler. Returns True on success."""
    p = str(path)
    try:
        if IS_WINDOWS:
            os.startfile(p)  # type: ignore[attr-defined]
            return True
        if IS_LINUX:
            if not _have("xdg-open"):
                logger.warning("xdg-open not found; cannot open %r", p)
                return False
            subprocess.Popen([_LINUX_TOOLS["xdg-open"], p])
            return True
        if IS_MACOS:
            subprocess.Popen(["open", p])
            return True
    except Exception as exc:
        logger.warning("open_path(%r) failed: %s", p, exc)
    return False


def open_in_text_editor(path) -> bool:
    """Open a text file in a graphical text editor."""
    p = str(path)
    try:
        if IS_WINDOWS:
            subprocess.Popen(["notepad.exe", p])
            return True
        if IS_LINUX:
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
            candidates = []
            if editor:
                # $EDITOR may be terminal-only (vim/nano); only use if it's GUI-friendly
                base = os.path.basename(editor.split()[0])
                if base in ("gedit", "kate", "kwrite", "mousepad", "pluma", "code", "subl", "gnome-text-editor"):
                    candidates.append(editor.split() + [p])
            candidates.extend([
                ["gnome-text-editor", p],
                ["gedit", p],
                ["kate", p],
                ["mousepad", p],   # XFCE
                ["pluma", p],      # MATE
                ["xed", p],        # Cinnamon
            ])
            for cmd in candidates:
                if shutil.which(cmd[0]):
                    subprocess.Popen(cmd)
                    return True
            # Fall back to xdg-open (will pick the user's default for text/plain)
            return open_path(p)
        if IS_MACOS:
            subprocess.Popen(["open", "-t", p])
            return True
    except Exception as exc:
        logger.warning("open_in_text_editor(%r) failed: %s", p, exc)
    return False


# ══════════════════════════════════════════════════════════════════════════
# Idle time
# ══════════════════════════════════════════════════════════════════════════

if IS_WINDOWS:
    import ctypes

    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    try:
        ctypes.windll.user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
        ctypes.windll.user32.GetLastInputInfo.restype = ctypes.c_bool
        ctypes.windll.kernel32.GetTickCount.restype = ctypes.c_uint
    except Exception:
        pass


def get_idle_ms() -> int:
    """Milliseconds since the last user input (mouse/keyboard). 0 if unknown."""
    if IS_WINDOWS:
        try:
            lii = _LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                return 0
            now = ctypes.windll.kernel32.GetTickCount()
            return (now - lii.dwTime) & 0xFFFFFFFF
        except Exception:
            return 0
    if IS_LINUX:
        # xprintidle is simple and accurate; preferred when available
        if _have("xprintidle"):
            rc, out = _run([_LINUX_TOOLS["xprintidle"]], timeout=1.0)
            if rc == 0 and out.strip().isdigit():
                return int(out.strip())
        # Fallback: query the X11 screensaver extension via python-xlib
        try:
            from Xlib import display  # type: ignore
            from Xlib.ext import screensaver  # noqa: F401  — registers extension
            d = display.Display()
            root = d.screen().root
            info = root.screensaver_query_info()
            d.close()
            return int(getattr(info, "idle", 0))
        except Exception:
            return 0
    return 0


# ══════════════════════════════════════════════════════════════════════════
# Cursor
# ══════════════════════════════════════════════════════════════════════════

def set_cursor_pos(x: int, y: int) -> bool:
    """Move the mouse cursor to absolute screen coords. Returns True on success."""
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
            return True
        except Exception:
            return False
    # X11 / macOS — pyautogui works on both
    try:
        import pyautogui
        pyautogui.moveTo(int(x), int(y), _pause=False)
        return True
    except Exception:
        return False


def get_cursor_pos() -> Optional[Tuple[int, int]]:
    """Return the current mouse cursor position, or None if unavailable."""
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes
            pt = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        except Exception:
            return None
    try:
        import pyautogui
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)
    except Exception:
        return None


def lock_cursor_to_point(x: int, y: int, seconds: float, tick: float = 0.05) -> bool:
    """Keep the cursor pinned near a point for a short time.

    Windows uses ClipCursor for a true lock. Other platforms repeatedly move the
    pointer back to the point, which is weaker but preserves the enforcement
    behavior without Win32 APIs.
    """
    import time

    x = int(x)
    y = int(y)
    seconds = max(0.0, float(seconds))
    tick = max(0.01, float(tick))

    if IS_WINDOWS:
        try:
            import ctypes
            import ctypes.wintypes
            rect = ctypes.wintypes.RECT(x, y, x + 1, y + 1)
            end_time = time.monotonic() + seconds
            while time.monotonic() < end_time:
                ctypes.windll.user32.ClipCursor(ctypes.byref(rect))
                ctypes.windll.user32.SetCursorPos(x, y)
                time.sleep(tick)
            return True
        except Exception:
            return False
        finally:
            try:
                ctypes.windll.user32.ClipCursor(None)
            except Exception:
                pass

    end_time = time.monotonic() + seconds
    ok = False
    while time.monotonic() < end_time:
        ok = set_cursor_pos(x, y) or ok
        time.sleep(tick)
    return ok


# ══════════════════════════════════════════════════════════════════════════
# Clipboard
# ══════════════════════════════════════════════════════════════════════════

def clipboard_get() -> str:
    if IS_WINDOWS:
        try:
            import win32clipboard  # type: ignore
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                    return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT) or ""
            finally:
                win32clipboard.CloseClipboard()
            return ""
        except Exception:
            return ""
    try:
        import pyperclip  # type: ignore
        return pyperclip.paste() or ""
    except Exception as exc:
        logger.debug("clipboard_get via pyperclip failed: %s", exc)
        return ""


def clipboard_set(text: str) -> bool:
    if IS_WINDOWS:
        try:
            import win32clipboard  # type: ignore
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception:
            return False
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
        return True
    except Exception as exc:
        logger.debug("clipboard_set via pyperclip failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════
# Window management
# ══════════════════════════════════════════════════════════════════════════
#
# `hwnd` is treated as an opaque integer identifier:
#   - On Windows, it's an HWND.
#   - On Linux X11, it's the X window ID (matches PyQt5 winId() and
#     `xdotool getactivewindow` decimal output).
#   - wmctrl uses hex form `0x01234567`; we convert internally.

def _wmctrl_id(hwnd: int) -> str:
    return f"0x{hwnd:08x}"


def _proc_exe(pid: int) -> Optional[str]:
    """Return the basename of the executable for ``pid`` (Linux-only via /proc)."""
    if not pid:
        return None
    try:
        # /proc/<pid>/comm gives just the process name, truncated at 15 chars
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
        # /proc/<pid>/exe is a symlink to the actual binary (full name)
        try:
            link = os.readlink(f"/proc/{pid}/exe")
            return os.path.basename(link) or comm or None
        except OSError:
            return comm or None
    except Exception:
        return None


def get_active_window() -> Optional[WindowInfo]:
    """Return the foreground window, or None if there isn't one / lookup failed."""
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32process  # type: ignore
            import ctypes
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            title = win32gui.GetWindowText(hwnd) or ""
            try:
                cls = win32gui.GetClassName(hwnd)
            except Exception:
                cls = ""
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                pid = 0
            exe = _windows_exe_for_pid(pid) if pid else None
            return WindowInfo(hwnd=int(hwnd), title=title, pid=int(pid or 0),
                              class_name=cls, exe_name=exe)
        except ImportError:
            return None
        except Exception as exc:
            logger.debug("get_active_window (win32) failed: %s", exc)
            return None
    if IS_LINUX:
        if not _have("xdotool"):
            return None
        rc, out = _run([_LINUX_TOOLS["xdotool"], "getactivewindow"], timeout=1.0)
        if rc != 0 or not out.strip().isdigit():
            return None
        hwnd = int(out.strip())
        title = ""
        cls = ""
        pid = 0
        # Prefer wmctrl -lpx for one-shot lookup (title, pid, wm_class)
        if _have("wmctrl"):
            rc2, out2 = _run([_LINUX_TOOLS["wmctrl"], "-lpx"], timeout=1.0)
            if rc2 == 0:
                target_hex = _wmctrl_id(hwnd)
                for line in out2.splitlines():
                    parts = line.split(None, 4)
                    if len(parts) >= 5 and parts[0].lower() == target_hex:
                        # parts: id desktop pid wm_class hostname title
                        try:
                            pid = int(parts[2])
                        except ValueError:
                            pid = 0
                        cls = parts[3]  # e.g. "Navigator.firefox"
                        # title is everything after the host (parts[4] = "host title")
                        rest = parts[4].split(None, 1)
                        title = rest[1] if len(rest) > 1 else ""
                        break
        # Fallback for title via xdotool
        if not title:
            rc3, out3 = _run([_LINUX_TOOLS["xdotool"], "getwindowname", str(hwnd)], timeout=1.0)
            if rc3 == 0:
                title = out3.strip()
        exe = _proc_exe(pid)
        return WindowInfo(hwnd=hwnd, title=title, pid=pid, class_name=cls, exe_name=exe)
    return None


def _windows_exe_for_pid(pid: int) -> Optional[str]:
    """Get the basename of the .exe for a PID on Windows (via ctypes)."""
    if not IS_WINDOWS or not pid:
        return None
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return None
        try:
            buf = ctypes.create_unicode_buffer(512)
            size = ctypes.c_ulong(512)
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
            if ok and buf.value:
                return os.path.basename(buf.value)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        pass
    return None


def enumerate_windows(skip_hwnds: Optional[set[int]] = None) -> List[WindowInfo]:
    """List visible top-level windows."""
    skip = skip_hwnds or set()
    out: List[WindowInfo] = []
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32process  # type: ignore

            def _cb(hwnd, _extra):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                if hwnd in skip:
                    return
                title = win32gui.GetWindowText(hwnd) or ""
                if not title.strip():
                    return
                try:
                    cls = win32gui.GetClassName(hwnd)
                except Exception:
                    cls = ""
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                except Exception:
                    pid = 0
                exe = _windows_exe_for_pid(pid) if pid else None
                out.append(WindowInfo(hwnd=int(hwnd), title=title, pid=int(pid or 0),
                                      class_name=cls, exe_name=exe))

            win32gui.EnumWindows(_cb, None)
        except Exception as exc:
            logger.debug("enumerate_windows (win32) failed: %s", exc)
        return out
    if IS_LINUX:
        if not _have("wmctrl"):
            return out
        rc, raw = _run([_LINUX_TOOLS["wmctrl"], "-lpx"], timeout=1.5)
        if rc != 0:
            return out
        for line in raw.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            try:
                hwnd = int(parts[0], 16)
            except ValueError:
                continue
            if hwnd in skip:
                continue
            try:
                pid = int(parts[2])
            except ValueError:
                pid = 0
            cls = parts[3]
            rest = parts[4].split(None, 1)
            title = rest[1] if len(rest) > 1 else ""
            if not title.strip():
                continue
            out.append(WindowInfo(
                hwnd=hwnd, title=title, pid=pid,
                class_name=cls, exe_name=_proc_exe(pid),
            ))
        return out
    return out


def window_get_title(hwnd: int) -> str:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            return win32gui.GetWindowText(int(hwnd)) or ""
        except Exception:
            return ""
    if IS_LINUX and _have("xdotool"):
        rc, out = _run([_LINUX_TOOLS["xdotool"], "getwindowname", str(int(hwnd))], timeout=1.0)
        if rc == 0:
            return out.strip()
    return ""


def window_get_class(hwnd: int) -> str:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            return win32gui.GetClassName(int(hwnd)) or ""
        except Exception:
            return ""
    if IS_LINUX and _have("xdotool"):
        rc, out = _run([_LINUX_TOOLS["xdotool"], "getwindowclassname", str(int(hwnd))], timeout=1.0)
        if rc == 0:
            return out.strip()
    return ""


def window_get_pid(hwnd: int) -> int:
    if IS_WINDOWS:
        try:
            import win32process  # type: ignore
            _, pid = win32process.GetWindowThreadProcessId(int(hwnd))
            return int(pid or 0)
        except Exception:
            return 0
    if IS_LINUX and _have("xdotool"):
        rc, out = _run([_LINUX_TOOLS["xdotool"], "getwindowpid", str(int(hwnd))], timeout=1.0)
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
    return 0


def window_get_exe(hwnd: int) -> Optional[str]:
    if IS_WINDOWS:
        return _windows_exe_for_pid(window_get_pid(hwnd))
    if IS_LINUX:
        return _proc_exe(window_get_pid(hwnd))
    return None


def window_is_visible(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            return bool(win32gui.IsWindowVisible(int(hwnd)))
        except Exception:
            return False
    if IS_LINUX:
        # On X11, if it's in wmctrl's list and not hidden state → visible
        rc, out = _run([_LINUX_TOOLS["wmctrl"], "-l"], timeout=1.0) if _have("wmctrl") else (-1, "")
        target = _wmctrl_id(int(hwnd))
        if rc == 0:
            return any(line.lower().startswith(target) for line in out.splitlines())
        return False
    return False


def window_is_minimized(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            return bool(win32gui.IsIconic(int(hwnd)))
        except Exception:
            return False
    if IS_LINUX and _have("xprop"):
        rc, out = _run([_LINUX_TOOLS["xprop"], "-id", str(int(hwnd)), "_NET_WM_STATE"], timeout=1.0)
        return rc == 0 and "_NET_WM_STATE_HIDDEN" in out
    return False


def window_is_maximized(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            return bool(win32gui.IsZoomed(int(hwnd)))
        except Exception:
            return False
    if IS_LINUX and _have("xprop"):
        rc, out = _run([_LINUX_TOOLS["xprop"], "-id", str(int(hwnd)), "_NET_WM_STATE"], timeout=1.0)
        return rc == 0 and ("_NET_WM_STATE_MAXIMIZED_VERT" in out or "_NET_WM_STATE_MAXIMIZED_HORZ" in out)
    return False


def window_get_rect(hwnd: int) -> Optional[Rect]:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            l, t, r, b = win32gui.GetWindowRect(int(hwnd))
            return Rect(l, t, r, b)
        except Exception:
            return None
    if IS_LINUX and _have("xdotool"):
        rc, out = _run([_LINUX_TOOLS["xdotool"], "getwindowgeometry", "--shell", str(int(hwnd))], timeout=1.0)
        if rc != 0:
            return None
        env: dict[str, int] = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    env[k.strip()] = int(v.strip())
                except ValueError:
                    pass
        try:
            x, y, w, h = env["X"], env["Y"], env["WIDTH"], env["HEIGHT"]
            return Rect(x, y, x + w, y + h)
        except KeyError:
            return None
    return None


def window_close(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32con  # type: ignore
            win32gui.PostMessage(int(hwnd), win32con.WM_CLOSE, 0, 0)
            return True
        except Exception:
            return False
    if IS_LINUX and _have("wmctrl"):
        rc, _ = _run([_LINUX_TOOLS["wmctrl"], "-i", "-c", _wmctrl_id(int(hwnd))], timeout=1.5)
        return rc == 0
    return False


def window_minimize(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32con  # type: ignore
            win32gui.ShowWindow(int(hwnd), win32con.SW_MINIMIZE)
            return True
        except Exception:
            return False
    if IS_LINUX:
        if _have("xdotool"):
            rc, _ = _run([_LINUX_TOOLS["xdotool"], "windowminimize", str(int(hwnd))], timeout=1.0)
            if rc == 0:
                return True
        if _have("wmctrl"):
            rc, _ = _run([_LINUX_TOOLS["wmctrl"], "-i", "-r", _wmctrl_id(int(hwnd)),
                          "-b", "add,hidden"], timeout=1.0)
            return rc == 0
    return False


def window_maximize(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32con  # type: ignore
            win32gui.ShowWindow(int(hwnd), win32con.SW_MAXIMIZE)
            return True
        except Exception:
            return False
    if IS_LINUX and _have("wmctrl"):
        rc, _ = _run([_LINUX_TOOLS["wmctrl"], "-i", "-r", _wmctrl_id(int(hwnd)),
                      "-b", "add,maximized_vert,maximized_horz"], timeout=1.0)
        return rc == 0
    return False


def window_restore(hwnd: int) -> bool:
    """Un-minimize / un-maximize."""
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32con  # type: ignore
            win32gui.ShowWindow(int(hwnd), win32con.SW_RESTORE)
            return True
        except Exception:
            return False
    if IS_LINUX:
        if _have("wmctrl"):
            _run([_LINUX_TOOLS["wmctrl"], "-i", "-r", _wmctrl_id(int(hwnd)),
                  "-b", "remove,hidden,maximized_vert,maximized_horz"], timeout=1.0)
        if _have("xdotool"):
            _run([_LINUX_TOOLS["xdotool"], "windowactivate", str(int(hwnd))], timeout=1.0)
        return True
    return False


def window_focus(hwnd: int) -> bool:
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            import win32con  # type: ignore
            if win32gui.IsIconic(int(hwnd)):
                win32gui.ShowWindow(int(hwnd), win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(int(hwnd))
            return True
        except Exception:
            return False
    if IS_LINUX:
        if _have("wmctrl"):
            rc, _ = _run([_LINUX_TOOLS["wmctrl"], "-i", "-a", _wmctrl_id(int(hwnd))], timeout=1.0)
            if rc == 0:
                return True
        if _have("xdotool"):
            rc, _ = _run([_LINUX_TOOLS["xdotool"], "windowactivate", str(int(hwnd))], timeout=1.0)
            return rc == 0
    return False


def window_move(hwnd: int, x: int, y: int, w: int, h: int) -> bool:
    """Move and resize the window."""
    if IS_WINDOWS:
        try:
            import win32gui  # type: ignore
            win32gui.MoveWindow(int(hwnd), int(x), int(y), int(w), int(h), True)
            return True
        except Exception:
            return False
    if IS_LINUX and _have("wmctrl"):
        # Remove maximized state first so the move takes effect
        _run([_LINUX_TOOLS["wmctrl"], "-i", "-r", _wmctrl_id(int(hwnd)),
              "-b", "remove,maximized_vert,maximized_horz"], timeout=1.0)
        rc, _ = _run([_LINUX_TOOLS["wmctrl"], "-i", "-r", _wmctrl_id(int(hwnd)),
                      "-e", f"0,{int(x)},{int(y)},{int(w)},{int(h)}"], timeout=1.0)
        return rc == 0
    return False


# ══════════════════════════════════════════════════════════════════════════
# Monitors
# ══════════════════════════════════════════════════════════════════════════

if IS_WINDOWS:
    import ctypes.wintypes

    _MONITOR_DEFAULTTONEAREST = 2
    _SM_XVIRTUALSCREEN = 76
    _SM_YVIRTUALSCREEN = 77
    _SM_CXVIRTUALSCREEN = 78
    _SM_CYVIRTUALSCREEN = 79

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("rcMonitor", ctypes.wintypes.RECT),
            ("rcWork", ctypes.wintypes.RECT),
            ("dwFlags", ctypes.wintypes.DWORD),
        ]

    def _win_rect_to_mon(r) -> MonitorRect:
        return MonitorRect(
            left=r.left, top=r.top,
            width=r.right - r.left, height=r.bottom - r.top,
            right=r.right, bottom=r.bottom,
        )

    def _win_get_monitor_info(hmon) -> Optional[Tuple[MonitorRect, MonitorRect]]:
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            return _win_rect_to_mon(info.rcWork), _win_rect_to_mon(info.rcMonitor)
        return None


def _fallback_monitor() -> MonitorRect:
    """Last-resort: a single 1920x1080 desktop. Better than crashing."""
    try:
        import pyautogui
        w, h = pyautogui.size()
        return MonitorRect(0, 0, int(w), int(h), int(w), int(h))
    except Exception:
        return MonitorRect(0, 0, 1920, 1080, 1920, 1080)


def get_monitor_rect_for_point(x: int, y: int) -> MonitorRect:
    """Work-area rect of the monitor containing point (x, y)."""
    if IS_WINDOWS:
        try:
            import ctypes
            import ctypes.wintypes
            hmon = ctypes.windll.user32.MonitorFromPoint(
                ctypes.wintypes.POINT(int(x), int(y)), _MONITOR_DEFAULTTONEAREST,
            )
            res = _win_get_monitor_info(hmon)
            if res:
                return res[0]
            w = ctypes.windll.user32.GetSystemMetrics(0)
            h = ctypes.windll.user32.GetSystemMetrics(1)
            return MonitorRect(0, 0, w, h, w, h)
        except Exception:
            return _fallback_monitor()
    if IS_LINUX:
        try:
            from screeninfo import get_monitors  # type: ignore
            mons = list(get_monitors())
            for m in mons:
                if m.x <= x < m.x + m.width and m.y <= y < m.y + m.height:
                    return MonitorRect(m.x, m.y, m.width, m.height,
                                       m.x + m.width, m.y + m.height)
            if mons:
                m = mons[0]
                return MonitorRect(m.x, m.y, m.width, m.height,
                                   m.x + m.width, m.y + m.height)
        except Exception:
            pass
    return _fallback_monitor()


def get_monitor_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Work-area rect of the monitor the window lives on."""
    if IS_WINDOWS:
        try:
            import ctypes
            hmon = ctypes.windll.user32.MonitorFromWindow(int(hwnd), _MONITOR_DEFAULTTONEAREST)
            res = _win_get_monitor_info(hmon)
            if res:
                return res[0]
            w = ctypes.windll.user32.GetSystemMetrics(0)
            h = ctypes.windll.user32.GetSystemMetrics(1)
            return MonitorRect(0, 0, w, h, w, h)
        except Exception:
            return _fallback_monitor()
    if IS_LINUX:
        rect = window_get_rect(int(hwnd))
        if rect:
            cx = rect.left + rect.width // 2
            cy = rect.top + rect.height // 2
            return get_monitor_rect_for_point(cx, cy)
    return _fallback_monitor()


def get_monitor_screen_rect_for_hwnd(hwnd: int) -> MonitorRect:
    """Full screen rect (not work area) of the monitor the window lives on."""
    if IS_WINDOWS:
        try:
            import ctypes
            hmon = ctypes.windll.user32.MonitorFromWindow(int(hwnd), _MONITOR_DEFAULTTONEAREST)
            res = _win_get_monitor_info(hmon)
            if res:
                return res[1]
        except Exception:
            pass
    # Linux/macOS don't distinguish work-area vs full-screen here (X11 has no
    # "work area" concept that matches Windows precisely; screeninfo gives the
    # full monitor bounds, which is close enough).
    return get_monitor_rect_for_hwnd(int(hwnd))


def get_virtual_desktop_rect() -> MonitorRect:
    """Bounding box of all monitors."""
    if IS_WINDOWS:
        try:
            import ctypes
            left = ctypes.windll.user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
            top = ctypes.windll.user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
            width = ctypes.windll.user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
            height = ctypes.windll.user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
            return MonitorRect(left, top, width, height, left + width, top + height)
        except Exception:
            pass
    if IS_LINUX:
        try:
            from screeninfo import get_monitors  # type: ignore
            mons = list(get_monitors())
            if mons:
                left = min(m.x for m in mons)
                top = min(m.y for m in mons)
                right = max(m.x + m.width for m in mons)
                bottom = max(m.y + m.height for m in mons)
                return MonitorRect(left, top, right - left, bottom - top, right, bottom)
        except Exception:
            pass
    return _fallback_monitor()


# ══════════════════════════════════════════════════════════════════════════
# Process tree (for "is the foreground window our own terminal?")
# ══════════════════════════════════════════════════════════════════════════

def get_ancestor_pids() -> set[int]:
    """PIDs of the current process and all its ancestors."""
    pids: set[int] = {os.getpid()}
    try:
        import psutil  # type: ignore
        try:
            for p in psutil.Process(os.getpid()).parents():
                pids.add(p.pid)
        except Exception:
            pass
        return pids
    except ImportError:
        pass
    # Last-resort fallback that doesn't need psutil
    if IS_LINUX:
        cur = os.getpid()
        for _ in range(20):
            try:
                with open(f"/proc/{cur}/status") as f:
                    parent = 0
                    for line in f:
                        if line.startswith("PPid:"):
                            parent = int(line.split()[1])
                            break
            except Exception:
                break
            if parent == 0 or parent == cur:
                break
            pids.add(parent)
            cur = parent
    return pids


def is_own_console_window(hwnd: int) -> bool:
    """True if the window belongs to the terminal/console hosting our process."""
    pid = window_get_pid(int(hwnd))
    if not pid:
        return False
    return pid in get_ancestor_pids()


# ══════════════════════════════════════════════════════════════════════════
# Topmost (sprite stay-on-top reinforcement)
# ══════════════════════════════════════════════════════════════════════════

def ensure_window_topmost(hwnd: int) -> None:
    """Reinforce stay-on-top. On Windows uses HWND_TOPMOST; on Linux a no-op
    (PyQt5 ``Qt.WindowStaysOnTopHint`` is honored by EWMH-compliant WMs)."""
    if IS_WINDOWS:
        try:
            import ctypes
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            HWND_TOPMOST = -1
            ctypes.windll.user32.SetWindowPos(
                int(hwnd), HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass
    # Linux/macOS: no-op


# ══════════════════════════════════════════════════════════════════════════
# Tool / executable discovery
# ══════════════════════════════════════════════════════════════════════════

_git_exe_cache: Optional[str] = None


def find_git_executable() -> Optional[str]:
    """Locate ``git`` on PATH, plus common Windows install dirs as fallback."""
    global _git_exe_cache
    if _git_exe_cache:
        return _git_exe_cache

    found = shutil.which("git")
    if found:
        _git_exe_cache = found
        return found

    if IS_WINDOWS:
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "cmd", "git.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "cmd", "git.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd", "git.exe"),
            r"C:\Program Files\Git\cmd\git.exe",
            r"C:\Program Files (x86)\Git\cmd\git.exe",
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                _git_exe_cache = c
                return c
    return None


def find_app_executable(name: str) -> Optional[str]:
    """Locate an application binary by name (e.g. 'firefox', 'code')."""
    cmd = shutil.which(name)
    if cmd:
        return cmd
    if IS_LINUX:
        # Search .desktop files for a matching display name
        for app in enumerate_installed_apps():
            if app.name.lower() == name.lower():
                try:
                    first_token = shlex.split(app.exec_cmd)[0] if app.exec_cmd else ""
                except (ValueError, IndexError):
                    first_token = app.exec_cmd.split()[0] if app.exec_cmd else ""
                return shutil.which(first_token) or first_token or None
    return None


_installed_apps_cache: Optional[List[InstalledApp]] = None


def enumerate_installed_apps() -> List[InstalledApp]:
    """Return a list of installed GUI applications. Cached after first call."""
    global _installed_apps_cache
    if _installed_apps_cache is not None:
        return _installed_apps_cache

    apps: List[InstalledApp] = []
    if IS_LINUX:
        seen_names: set[str] = set()
        roots = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path.home() / ".local/share/applications",
            Path("/var/lib/flatpak/exports/share/applications"),
            Path.home() / ".local/share/flatpak/exports/share/applications",
            Path("/var/lib/snapd/desktop/applications"),
        ]
        for root in roots:
            if not root.is_dir():
                continue
            for desktop in root.glob("*.desktop"):
                try:
                    text = desktop.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                in_main_section = False
                name = ""
                exec_cmd = ""
                no_display = False
                hidden = False
                terminal = False
                icon = ""
                categories = ""
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        in_main_section = (line == "[Desktop Entry]")
                        continue
                    if not in_main_section or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip()
                    if k == "Name" and not name:
                        name = v
                    elif k == "Exec" and not exec_cmd:
                        # Strip %-codes (%U, %F, %f, %u)
                        try:
                            tokens = shlex.split(v)
                        except ValueError:
                            tokens = v.split()
                        exec_tokens = [t for t in tokens if not t.startswith("%")]
                        exec_cmd = shlex.join(exec_tokens)
                    elif k == "NoDisplay":
                        no_display = (v.lower() == "true")
                    elif k == "Hidden":
                        hidden = (v.lower() == "true")
                    elif k == "Terminal":
                        terminal = (v.lower() == "true")
                    elif k == "Icon":
                        icon = v
                    elif k == "Categories":
                        categories = v
                if not name or not exec_cmd or no_display or hidden or terminal:
                    continue
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                apps.append(InstalledApp(
                    name=name, exec_cmd=exec_cmd,
                    icon=icon or None, categories=categories or None,
                ))
    # Windows enumeration is left to the caller (existing Start-Menu .lnk scanner
    # in desktop_controller.py) — this module only adds a Linux backend.
    _installed_apps_cache = apps
    return apps


__all__ = [
    "IS_WINDOWS", "IS_LINUX", "IS_MACOS",
    "WindowInfo", "Rect", "MonitorRect", "InstalledApp",
    "open_path", "open_in_text_editor",
    "get_idle_ms",
    "set_cursor_pos", "get_cursor_pos", "lock_cursor_to_point",
    "clipboard_get", "clipboard_set",
    "get_active_window", "enumerate_windows",
    "window_get_title", "window_get_class", "window_get_pid", "window_get_exe",
    "window_is_visible", "window_is_minimized", "window_is_maximized",
    "window_get_rect", "window_close", "window_minimize", "window_maximize",
    "window_restore", "window_focus", "window_move",
    "get_monitor_rect_for_point", "get_monitor_rect_for_hwnd",
    "get_monitor_screen_rect_for_hwnd", "get_virtual_desktop_rect",
    "get_ancestor_pids", "is_own_console_window",
    "ensure_window_topmost",
    "find_git_executable", "find_app_executable", "enumerate_installed_apps",
]
