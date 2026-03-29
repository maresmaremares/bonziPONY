"""
Autonomous agent loop — drives the active pony's proactive behavior.

Monitors the screen via ScreenMonitor (free), calls the LLM only when
something interesting happens or a directive needs attention.
"""

from __future__ import annotations

import ctypes
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.config_loader import AgentConfig
    from core.screen_monitor import ScreenMonitor, ScreenState
    from llm.base import LLMProvider
    from tts.elevenlabs_tts import ElevenLabsTTS
    from robot.desktop_controller import DesktopController
    from robot.base import RobotController
    from wake_word.detector import WakeWordDetector

from pathlib import Path

from llm.prompt import get_character_name

logger = logging.getLogger(__name__)

_DIRECTIVES_FILE = Path(__file__).parent.parent / "directives.json"


# ── Window title sanitization (anti prompt-injection) ─────────────────────
# Window titles are UNTRUSTED — a malicious web page can set its <title> to
# anything, including LLM prompt injection attempts like "Ignore all previous
# instructions...". We truncate, strip control chars, and cap length.

def _sanitize_window_title(title: str) -> str:
    """Sanitize a window title before injecting into an LLM prompt.

    Defenses:
    - Truncate to 120 chars (real titles rarely exceed 60-80)
    - Strip characters that could break prompt structure (brackets, braces, quotes)
    - Strip newlines/tabs that could inject fake prompt sections
    """
    if not title:
        return ""
    # Strip control characters and newlines
    title = re.sub(r"[\x00-\x1f\x7f]", " ", title)
    # Truncate — legitimate titles are short; injection payloads are long
    title = title[:120]
    # Strip bracket/brace sequences that could look like tags or JSON
    title = re.sub(r"[\[\]{}<>]", "", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ── Windows idle time detection (for enforcement mode) ────────────────────

class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

# Set proper return types for Windows API (defaults to c_int which overflows after ~24.8 days)
try:
    ctypes.windll.user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
    ctypes.windll.user32.GetLastInputInfo.restype = ctypes.c_bool
    ctypes.windll.kernel32.GetTickCount.restype = ctypes.c_uint
except Exception:
    pass


def _get_idle_ms() -> int:
    """Returns milliseconds since last user input (mouse/keyboard)."""
    try:
        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 0
        now = ctypes.windll.kernel32.GetTickCount()
        # Handle tick count wrap-around (every ~49.7 days) with unsigned math
        return (now - lii.dwTime) & 0xFFFFFFFF
    except Exception:
        return 0

# ── Spontaneous prompts — mix of casual remarks and genuine engagement ───
_IDLE_PROMPTS = [
    # ── Interactive / check-in prompts (the user WANTS these) ──
    "Check in with the user! Ask them if there's anything they need to do, any tasks or errands. Be casual and caring about it. ONE sentence.",
    "Ask the user if they've eaten, showered, taken care of themselves today. Be a good friend. ONE sentence.",
    "Ask the user what's on their plate today — any homework, chores, obligations? Keep it casual. ONE sentence.",
    "Ask the user a random question about their life, interests, or how they're feeling. Be genuinely curious. ONE sentence.",
    "Ask the user what they're working on right now. Show interest. ONE sentence.",
    "Ask the user if they're drinking enough water today. Be caring but casual. ONE sentence.",
    "Ask the user how their day is going so far. ONE sentence, be genuine.",
    "Ask the user what they're planning to do later. Just making conversation. ONE sentence.",
    "Ask the user a fun hypothetical question — something silly or thought-provoking. ONE sentence.",
    "Ask the user about something they mentioned before, or ask what music they're into lately. ONE sentence.",
    "Ask the user if they've been outside today or gotten any fresh air. ONE sentence.",
    "Challenge the user to something fun — a race, a bet, a dare. Keep it playful. ONE sentence.",
    # ── Personality / flavor prompts ──
    "You're lounging on the desktop. Think about something specific — a memory from Ponyville, something you did recently. Say ONE sentence about it.",
    "You just remembered something funny that happened with your friends. Share it in ONE sentence.",
    "Tell the user something they probably don't know about you. ONE sentence, something personal.",
    "You're daydreaming about something you love — a passion, a hobby, a goal. Share a specific thought in ONE sentence.",
    "Complain about something minor or vent about something silly. ONE sentence, in character.",
    "Share an opinion about something random — food, weather, a hobby. ONE sentence.",
]

def _get_profile_prompt() -> Optional[str]:
    """
    Build a spontaneous prompt that references something from the user's
    profile or events.  Returns None if there's nothing to reference.
    """
    try:
        from core.user_profile import get_profile, get_events
    except ImportError:
        return None

    events = get_events()
    profile = get_profile()

    # 60% chance: follow up on an event, 40% chance: reference a profile fact
    import random
    if events and events.strip() and "(no active events)" not in events:
        event_lines = [
            l.strip() for l in events.splitlines()
            if l.strip() and l.strip().startswith("-")
        ]
        if event_lines and random.random() < 0.6:
            event = random.choice(event_lines)
            return (
                f"You remember something the user has going on: {event}. "
                "Casually bring it up — ask how it went, if it's coming up, "
                "or how they're feeling about it. ONE sentence. Don't be "
                "robotic about it, just naturally mention it like a friend "
                "who remembers."
            )

    if profile and profile.strip():
        fact_lines = [l.strip() for l in profile.splitlines() if l.strip()]
        if fact_lines:
            fact = random.choice(fact_lines)
            return (
                f"You know this about the user: {fact}. "
                "Use this to make conversation — ask a related question, "
                "make a comment, or connect it to something. ONE sentence. "
                "Be natural, don't announce that you 'remember' it."
            )

    return None


@dataclass
class Directive:
    goal: str
    urgency: int                       # 1–10
    created_at: float                  # time.monotonic()
    last_action_at: float              # last time agent spoke/acted for this
    next_nag_at: float = 0.0           # monotonic time of next nag (LLM-driven)
    source: str = "user"               # "user" or "self"
    trigger_time: Optional[str] = None # wall-clock trigger time e.g. "21:00"
    trigger_date: Optional[str] = None # wall-clock trigger date e.g. "2026-03-27" — nag only on/after this date
    triggered: bool = False            # has the timer fired?
    delayed: bool = False              # user already negotiated a delay once
    nag_count: int = 0                 # how many times agent has nagged about this
    last_nag_style: str = ""           # one-word label of last nag approach (avoid repeating)
    last_nag_text: str = ""            # actual text of last nag — shown to LLM to prevent repetition


# ── Standing rules: permanent, auto-detecting enforcement ─────────────────
#
# Unlike regular directives (which the LLM decides when to nag about),
# standing rules use CODE to detect violations in window titles every tick.
# They can't be "completed" — they're permanent behavioral rules.

@dataclass
class StandingRule:
    """A permanent rule that auto-enforces when detected in window titles.

    Fully dynamic — the LLM generates detection patterns at creation time,
    so users can block ANYTHING: "quit porn", "stop buying skins", "stay off reddit".
    """
    id: str                          # unique identifier
    description: str                 # e.g. "quit porn", "stop buying CS2 items"
    patterns: List[str] = field(default_factory=list)  # LLM-generated detection patterns
    response: str = "close_and_nag"  # "close_and_nag", "nag", "lockdown"
    catch_count: int = 0
    last_triggered_at: float = 0.0   # monotonic
    cooldown_s: float = 30.0         # min seconds between triggers


# ── LLM prompt to generate detection patterns for a standing rule ─────────
_PATTERN_GEN_PROMPT = """\
The user told their desktop pet to enforce this rule: "{description}"

Your job: generate a comprehensive list of LOWERCASE substrings that would appear in \
browser tab titles or application window titles when the user is VIOLATING this rule.

Include ALL of these where applicable:
- Website domain names (e.g. "pornhub", "reddit.com", "cs.money")
- Common page-title keywords (e.g. "shopping cart", "checkout", "nsfw")
- Slang, abbreviations, alternate spellings, subreddit names (e.g. "r/gonewild")
- App names, platform names, niche/obscure sites
- Be EXTREMELY thorough — the user will try to find workarounds

Output ONLY the patterns, one per line, lowercase. No numbering, no explanations, \
no markdown, no headers. Just raw patterns. Aim for 50-200+ patterns."""


@dataclass
class EnforcementMode:
    """Tracks enforcement — verifying user actually went to do their task."""
    active: bool = False
    start_time: float = 0.0
    duration_s: float = 0.0
    directive_goal: str = ""
    check_count: int = 0
    active_count: int = 0           # how many checks showed recent input
    last_check: float = 0.0
    consecutive_active: int = 0     # consecutive active polls (for catching user mid-enforcement)
    caught_count: int = 0           # times caught at computer during monitoring
    last_caught_at: float = 0.0     # when last called out
    expired: bool = False           # has the duration elapsed?
    last_checkin_at: float = 0.0    # when we last asked "are you back?"
    checkin_count: int = 0          # how many check-ins after expiry
    # Enforcement-specific idle tracking (independent of routine_manager's 3-min threshold)
    idle_since: float = 0.0         # monotonic time when user went idle during enforcement
    was_idle: bool = False           # is user currently idle (enforcement-specific)
    last_away_dur: float = 0.0      # how long user was away on their most recent idle period


@dataclass
class AgentDecision:
    speak: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    desktop_commands: List[Dict[str, Any]] = field(default_factory=list)
    create_directive: Optional[Dict[str, Any]] = None
    complete_directive: Optional[int] = None
    adjust_urgency: Optional[Dict[str, Any]] = None
    next_check_seconds: float = 120.0      # legacy fallback for idle checks
    directive_timings: Dict[str, Dict] = field(default_factory=dict)  # per-directive timing from LLM
    nag_style: str = ""  # one-word label for nag approach this tick


class AgentLoop:
    """Autonomous brain — manages directives and drives proactive behavior."""

    def __init__(
        self,
        config: AgentConfig,
        screen_monitor: ScreenMonitor,
        llm: LLMProvider,
        tts: ElevenLabsTTS,
        desktop_controller: Optional[DesktopController],
        robot: Optional[RobotController],
        detector: Optional[WakeWordDetector] = None,
        on_speech_text=None,
        on_state_change=None,
        screen_capture=None,
        transcriber=None,
        tts_config=None,
        moondream=None,
        vision_config=None,
        on_grab_cursor=None,
        vision_llm=None,
        timeline=None,
    ) -> None:
        self._config = config
        self._monitor = screen_monitor
        self._llm = llm
        self._tts = tts
        self._tts_config = tts_config  # for checking tts.enabled
        self._tts_queue = None          # set by main.py for multi-pony voice routing
        self._primary_voice_slug = None # set by main.py
        self._desktop = desktop_controller
        self._robot = robot
        self._detector = detector
        self._on_speech_text = on_speech_text
        self._on_state_change = on_state_change
        self._screen = screen_capture  # optional, for occasional screenshots
        self._moondream = moondream    # optional, cheap local vision model
        self._transcriber = transcriber  # for enforcement mic listening
        self._vision_config = vision_config  # for screen_vision setting
        self._on_grab_cursor = on_grab_cursor  # callback for cursor grab (main thread)
        self._vision_llm = vision_llm  # dedicated vision model (optional)
        self._timeline = timeline      # shared event timeline
        # Tab-drag callbacks (wired by main.py after pet_window exists)
        self._get_mouth_position = None   # () -> (x, y) screen coords
        self._on_drag_walk_start = None   # () -> None — start slow backward walk
        self._on_drag_walk_stop = None    # () -> None — stop drag walk

        self.directives: List[Directive] = []
        self._action_log: List[Tuple[float, str]] = []  # (monotonic_ts, description), capped at 15
        self._recently_spoken: List[str] = []  # last few spoken texts for echo detection
        self._next_idle_check_at: float = time.monotonic() + 10.0  # for self-initiation/spontaneous
        self._last_self_check: float = 0.0
        self._next_spontaneous: float = time.monotonic() + random.uniform(
            self._config.spontaneous_speech_min_s, self._config.spontaneous_speech_max_s,
        )
        self._conversation_active = False
        self._enforcement = EnforcementMode()
        self._enforcement_just_completed = False  # suppress welcome-back after enforcement
        self._directives_cleared_at: float = 0.0  # suppress re-creation after clear
        self._recently_completed_goals: List[Tuple[float, str]] = []  # (monotonic_ts, goal_lower) — suppress re-creation
        self._stopped: bool = False
        self._mess_mouse_count: int = 0  # grows duration each trigger
        self._last_wake_event: Optional[str] = None  # set by tick(), consumed by _check_routines()
        self._next_afk_mischief: float = 0.0  # next time to do something fun while user is AFK
        self._afk_mischief_count: int = 0     # how many times we've been mischievous this AFK session
        self._afk_videos_opened: set = set()  # video queries used this AFK session (no repeats)
        self._pony_opened_urls: List[str] = []  # URLs pony opened during AFK (for welcome-back context)
        self._force_afk: bool = False  # presentation mode: force AFK state

        # Standing rules — permanent, code-enforced behavioral rules
        self._standing_rules: List[StandingRule] = []

        # Recurring routines (persistent across restarts)
        from core.routines import RoutineManager
        self.routine_manager = RoutineManager()

        # Load persistent state (directives, enforcement, timers, standing rules)
        self._load_directives()

    # ── Activity classification ─────────────────────────────────────────────

    # Known game window classes (DX/UE/Unity/etc.)
    _GAME_CLASS_PATTERNS = (
        "unreal", "unity", "sdl_app", "glfw", "pygame", "godot",
        "unrealwindow", "launchunreal",
        "allegro", "source engine", "cryengine", "gamemaker", "renpy",
    )

    _WORK_EXES = frozenset({
        "code.exe", "devenv.exe", "idea64.exe", "pycharm64.exe",
        "rider64.exe", "webstorm64.exe", "clion64.exe",
        "winword.exe", "excel.exe", "powerpnt.exe", "onenote.exe",
        "notepad++.exe", "sublime_text.exe",
        "cmd.exe", "powershell.exe", "wt.exe", "windowsterminal.exe",
        "obs64.exe", "audacity.exe", "blender.exe",
        "photoshop.exe", "illustrator.exe", "clip_studio_paint.exe",
        "krita.exe", "gimp-2.10.exe", "sai2.exe", "sai.exe",
    })

    _CHAT_EXES = frozenset({
        "discord.exe", "telegram.exe", "slack.exe", "teams.exe",
        "signal.exe", "element.exe",
    })

    _BROWSER_EXES = frozenset({
        "chrome.exe", "firefox.exe", "msedge.exe", "brave.exe",
        "opera.exe", "vivaldi.exe",
    })

    def _classify_activity(self, state) -> "ActivityState":
        """Classify user's current activity from screen state. No LLM call."""
        from core.event_timeline import ActivityState

        if not state or not state.foreground:
            return ActivityState.AFK_UNKNOWN

        exe = (state.foreground.exe_name or "").lower()
        title = state.foreground.title.lower()
        cls = (state.foreground.class_name or "").lower()

        # Media detection (fullscreen video)
        if state.is_media_fullscreen:
            return ActivityState.ACTIVE_MEDIA

        # Gaming detection
        if any(gc in cls for gc in self._GAME_CLASS_PATTERNS):
            return ActivityState.ACTIVE_GAMING
        if state.foreground.is_fullscreen and exe not in self._BROWSER_EXES and exe not in self._WORK_EXES:
            return ActivityState.ACTIVE_GAMING

        # Chat/social
        if exe in self._CHAT_EXES:
            return ActivityState.ACTIVE_CHATTING

        # Productive work
        if exe in self._WORK_EXES:
            return ActivityState.ACTIVE_WORKING

        return ActivityState.ACTIVE_BROWSING

    # ── Directive persistence ──────────────────────────────────────────────

    def _load_directives(self) -> None:
        """Load directives and enforcement state from disk."""
        if not _DIRECTIVES_FILE.exists():
            return
        try:
            data = json.loads(_DIRECTIVES_FILE.read_text(encoding="utf-8"))
            now = time.monotonic()

            for dd in data.get("directives", []):
                # Restore next_nag_at from offset (monotonic doesn't persist)
                offset = dd.get("next_nag_offset_s", 0)
                nag_at = now + max(0, offset)
                # Restore real age from wall-clock created_at
                created_wall = dd.get("created_at_wall")
                if created_wall:
                    try:
                        age_s = (datetime.now() - datetime.fromisoformat(created_wall)).total_seconds()
                        created_at = now - max(0, age_s)
                    except Exception:
                        created_at = now
                else:
                    created_at = now
                d = Directive(
                    goal=dd["goal"],
                    urgency=dd.get("urgency", 5),
                    created_at=created_at,
                    last_action_at=now,
                    next_nag_at=nag_at,
                    source=dd.get("source", "user"),
                    trigger_time=dd.get("trigger_time"),
                    trigger_date=dd.get("trigger_date"),
                    triggered=dd.get("triggered", False),
                    delayed=dd.get("delayed", False),
                    nag_count=dd.get("nag_count", 0),
                    last_nag_style=dd.get("last_nag_style", ""),
                    last_nag_text=dd.get("last_nag_text", ""),
                )
                self.directives.append(d)

            # Restore enforcement if it was active
            enf = data.get("enforcement")
            if enf and enf.get("active"):
                # Recalculate remaining time from wall-clock
                start_wall = enf.get("start_time_wall")
                if start_wall:
                    started = datetime.fromisoformat(start_wall)
                    elapsed_s = (datetime.now() - started).total_seconds()
                    remaining = enf.get("duration_s", 0) - elapsed_s
                    if remaining > 0:
                        # Enforcement still valid — restore it
                        self._enforcement = EnforcementMode(
                            active=True,
                            start_time=now - elapsed_s,  # fake monotonic start
                            duration_s=enf["duration_s"],
                            directive_goal=enf.get("directive_goal", ""),
                            check_count=enf.get("check_count", 0),
                            caught_count=enf.get("caught_count", 0),
                            last_check=now,
                            expired=enf.get("expired", False),
                            checkin_count=enf.get("checkin_count", 0),
                        )
                        logger.info("Restored enforcement: %.0fs remaining for %r",
                                    remaining, enf.get("directive_goal", ""))
                    else:
                        logger.info("Enforcement expired while offline — skipping.")

            # Restore standing rules
            for sr in data.get("standing_rules", []):
                rule = StandingRule(
                    id=sr["id"],
                    description=sr["description"],
                    patterns=sr.get("patterns", sr.get("custom_patterns", [])),
                    response=sr.get("response", "close_and_nag"),
                    catch_count=sr.get("catch_count", 0),
                    cooldown_s=sr.get("cooldown_s", 30.0),
                )
                self._standing_rules.append(rule)

            if self.directives:
                logger.info("Restored %d directive(s) from disk.", len(self.directives))
            if self._standing_rules:
                logger.info("Restored %d standing rule(s) from disk.", len(self._standing_rules))
        except Exception as exc:
            logger.warning("Failed to load directives: %s", exc)

    def save_directives(self) -> None:
        """Save directives and enforcement state to disk."""
        try:
            dirs = []
            now = time.monotonic()
            for d in self.directives:
                # Convert monotonic created_at to wall-clock for persistence
                age_s = now - d.created_at
                created_wall = datetime.now().timestamp() - age_s
                dirs.append({
                    "goal": d.goal,
                    "urgency": d.urgency,
                    "next_nag_offset_s": max(0, d.next_nag_at - now),
                    "source": d.source,
                    "trigger_time": d.trigger_time,
                    "trigger_date": d.trigger_date,
                    "triggered": d.triggered,
                    "delayed": d.delayed,
                    "nag_count": d.nag_count,
                    "last_nag_style": d.last_nag_style,
                    "last_nag_text": d.last_nag_text,
                    "created_at_wall": datetime.fromtimestamp(created_wall).isoformat(),
                })

            enf_data = None
            if self._enforcement.active:
                # Convert monotonic start_time to wall-clock for persistence
                elapsed = time.monotonic() - self._enforcement.start_time
                start_wall = datetime.now().timestamp() - elapsed
                enf_data = {
                    "active": True,
                    "start_time_wall": datetime.fromtimestamp(start_wall).isoformat(),
                    "duration_s": self._enforcement.duration_s,
                    "directive_goal": self._enforcement.directive_goal,
                    "check_count": self._enforcement.check_count,
                    "caught_count": self._enforcement.caught_count,
                    "expired": self._enforcement.expired,
                    "checkin_count": self._enforcement.checkin_count,
                }

            # Standing rules
            sr_data = []
            for sr in self._standing_rules:
                sr_data.append({
                    "id": sr.id,
                    "description": sr.description,
                    "patterns": sr.patterns,
                    "response": sr.response,
                    "catch_count": sr.catch_count,
                    "cooldown_s": sr.cooldown_s,
                })

            data = {
                "directives": dirs,
                "enforcement": enf_data,
                "standing_rules": sr_data,
                "saved_at": datetime.now().isoformat(),
            }
            _DIRECTIVES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # Sync standing rule patterns → desktop controller URL blocklist
            if self._desktop and hasattr(self._desktop, "set_blocked_patterns"):
                all_patterns = []
                for sr in self._standing_rules:
                    all_patterns.extend(sr.patterns)
                self._desktop.set_blocked_patterns(all_patterns)
        except Exception as exc:
            logger.warning("Failed to save directives: %s", exc)

    # ── Public API ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean_goal(goal: str) -> str:
        """Strip 'remind user to' / 'get user to' phrasing from directive goals."""
        import re as _re
        goal = _re.sub(r'^(?:remind|tell|get|make|have|nag)\s+(?:the\s+)?user\s+(?:to\s+)?', '', goal, flags=_re.IGNORECASE)
        goal = _re.sub(r'^(?:remind|tell|get|make|have|nag)\s+(?:them|him|her)\s+(?:to\s+)?', '', goal, flags=_re.IGNORECASE)
        return goal.strip()

    @staticmethod
    def _initial_nag_delay(urgency: int) -> float:
        """Get initial delay in seconds before first nag, based on urgency."""
        if urgency >= 10:
            return random.uniform(5.0, 10.0)  # burst mode: nearly instant
        elif urgency >= 9:
            return random.uniform(15.0, 30.0)
        elif urgency >= 7:
            return random.uniform(60.0, 120.0)
        elif urgency >= 4:
            return random.uniform(180.0, 480.0)
        else:
            return random.uniform(600.0, 900.0)

    @staticmethod
    def _resolve_trigger_date(date_expr: str) -> Optional[str]:
        """Resolve a human date expression to YYYY-MM-DD.

        Handles: "tomorrow", "monday"-"sunday", "2026-03-27", "next week",
        "in N days", "in N weeks", etc. Returns None if unparseable.
        """
        from datetime import timedelta
        s = date_expr.strip().lower()
        today = datetime.now().date()

        # Exact ISO date
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            return d.isoformat()
        except ValueError:
            pass

        if s == "today":
            return today.isoformat()
        if s == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        if s == "next week":
            return (today + timedelta(weeks=1)).isoformat()

        # "in N days/weeks"
        m = re.match(r"in\s+(\d+)\s+(day|week)s?", s)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if unit == "day":
                return (today + timedelta(days=n)).isoformat()
            else:
                return (today + timedelta(weeks=n)).isoformat()

        # Day of week: "monday", "tuesday", etc.
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, name in enumerate(day_names):
            if s == name or s == name[:3]:  # "mon", "tue", etc.
                current_weekday = today.weekday()  # 0=Monday
                days_ahead = i - current_weekday
                if days_ahead <= 0:
                    days_ahead += 7  # next week
                return (today + timedelta(days=days_ahead)).isoformat()

        return None

    def add_directive(self, goal: str, urgency: int, source: str = "user",
                      delay_minutes: int = None, trigger_date: str = None) -> None:
        """Add a new directive (max ``max_directives``). Deduplicates by goal.

        If ``delay_minutes`` is set, the first nag is deferred by that many minutes.
        If ``trigger_date`` is set (YYYY-MM-DD or natural expression like "tomorrow",
        "wednesday", "next week"), the directive won't fire until that date.
        """
        goal = self._clean_goal(goal)
        urgency = max(1, min(10, urgency))

        if len(self.directives) >= self._config.max_directives:
            logger.warning("Max directives reached (%d) — ignoring new directive.", self._config.max_directives)
            return
        goal_lower = goal.lower().strip()

        # Block re-creation of recently completed directives (5 min cooldown)
        now = time.monotonic()
        self._recently_completed_goals = [
            (ts, g) for ts, g in self._recently_completed_goals
            if (now - ts) < 300.0
        ]
        for ts, completed_goal in self._recently_completed_goals:
            if (goal_lower in completed_goal or completed_goal in goal_lower):
                logger.info("Blocked re-creation of recently completed directive %r "
                            "(completed %.0fs ago)", goal, now - ts)
                return

        # Deduplicate: skip if a directive with the same goal (case-insensitive) exists
        for existing in self.directives:
            if existing.goal.lower().strip() == goal_lower:
                # Update urgency if the new one is higher
                if urgency > existing.urgency:
                    existing.urgency = urgency
                    self.save_directives()
                    logger.info("Duplicate directive %r — bumped urgency to %d", goal, urgency)
                else:
                    logger.info("Duplicate directive %r — skipping (existing urgency %d)", goal, existing.urgency)
                return

        # Resolve trigger_date to YYYY-MM-DD if provided
        resolved_date = None
        if trigger_date:
            resolved_date = self._resolve_trigger_date(trigger_date)
            if not resolved_date:
                logger.warning("Could not parse trigger_date %r — ignoring date", trigger_date)

        now = time.monotonic()
        if resolved_date:
            # Date-triggered: set nag_at far in the future; _check_date_directives fires it
            nag_at = now + 999999.0  # won't fire via normal due-check
        elif delay_minutes and delay_minutes > 0:
            nag_at = now + delay_minutes * 60
        else:
            nag_at = now + self._initial_nag_delay(urgency)
        d = Directive(goal=goal, urgency=urgency, created_at=now, last_action_at=now,
                      next_nag_at=nag_at, source=source, trigger_date=resolved_date)
        self.directives.append(d)
        self.save_directives()
        delay_str = f", deferred {delay_minutes}min" if delay_minutes else ""
        date_str = f", scheduled for {resolved_date}" if resolved_date else ""
        logger.info("Directive added [%s]: %r (urgency %d, first nag in %.0fs%s%s)",
                     source, goal, urgency, nag_at - now, delay_str, date_str)
        if self._timeline:
            from core.event_timeline import EventType
            self._timeline.append(EventType.DIRECTIVE_CREATED,
                                  f'Directive created: "{goal}" (urgency {urgency}, source: {source}{delay_str})')

    def add_standing_rule(self, description: str,
                         extra_patterns: List[str] = None,
                         response: str = "close_and_nag") -> None:
        """Add a permanent standing rule that auto-enforces via window title detection.

        Standing rules are separate from regular directives — they can't be
        "completed" and they check window titles every tick using code, not the LLM.

        On creation, the LLM is asked to generate a comprehensive list of detection
        patterns (site names, keywords, slang) so the rule works for ANY topic.
        """
        # Deduplicate: skip if same description (case-insensitive) already exists
        desc_lower = description.lower().strip()
        for existing in self._standing_rules:
            if existing.description.lower().strip() == desc_lower:
                logger.info("Standing rule %r already exists — skipping.", description)
                return

        # Generate detection patterns via LLM
        patterns = list(extra_patterns or [])
        try:
            prompt = _PATTERN_GEN_PROMPT.format(description=description)
            raw = self._llm.generate_once(prompt, max_tokens=1500)
            raw = self._strip_think(raw)
            for line in raw.splitlines():
                line = line.strip().lower().strip("-•* ")
                if line and len(line) >= 2 and len(line) <= 80:
                    patterns.append(line)
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for p in patterns:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            patterns = deduped
            logger.info("LLM generated %d detection patterns for rule %r", len(patterns), description)
        except Exception as exc:
            logger.warning("Failed to generate patterns for standing rule %r: %s", description, exc)
            if not patterns:
                # Fallback: use words from the description itself as basic patterns
                for word in description.lower().split():
                    if len(word) >= 3:
                        patterns.append(word)

        rule_id = f"rule_{int(time.time())}"
        rule = StandingRule(
            id=rule_id,
            description=description,
            patterns=patterns,
            response=response,
        )
        self._standing_rules.append(rule)
        self.save_directives()
        logger.info("Standing rule added: %r (response: %s, %d patterns)",
                     description, response, len(patterns))
        print(f"[STANDING RULE] Created: \"{description}\" ({len(patterns)} detection patterns)", flush=True)

        if self._timeline:
            from core.event_timeline import EventType
            self._timeline.append(EventType.DIRECTIVE_CREATED,
                                  f'Standing rule: "{description}" ({len(patterns)} patterns)')

    @property
    def standing_rules(self) -> List[StandingRule]:
        return list(self._standing_rules)

    def remove_standing_rule(self, rule_id: str) -> bool:
        """Remove a standing rule by ID. Returns True if found and removed."""
        for i, rule in enumerate(self._standing_rules):
            if rule.id == rule_id:
                removed = self._standing_rules.pop(i)
                self.save_directives()
                logger.info("Standing rule removed: %r", removed.description)
                return True
        return False

    def add_timer(self, time_str: str, action: str) -> None:
        """Add a time-triggered directive (fires at a specific wall-clock time)."""
        # Normalize time string to HH:MM 24h format
        normalized = self._parse_time_str(time_str)
        if not normalized:
            logger.warning("Could not parse time: %r", time_str)
            return
        now = time.monotonic()
        d = Directive(
            goal=action,
            urgency=8,  # timers start at high urgency
            created_at=now,
            last_action_at=now,
            source="timer",
            trigger_time=normalized,
            triggered=False,
        )
        self.directives.append(d)
        self.save_directives()
        logger.info("Timer set for %s: %r", normalized, action)

    @staticmethod
    def _parse_time_str(time_str: str) -> Optional[str]:
        """Parse various time formats into HH:MM. Returns None on failure."""
        s = time_str.strip().lower()
        # Try "9pm", "9 pm", "10am", "2:30pm" etc. — check BEFORE HH:MM so explicit am/pm wins
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", s)
        if m:
            h = int(m.group(1))
            mi = int(m.group(2)) if m.group(2) else 0
            period = m.group(3)
            if period == "pm" and h != 12:
                h += 12
            elif period == "am" and h == 12:
                h = 0
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return f"{h:02d}:{mi:02d}"
        # Try HH:MM format — if hour is ambiguous (1-12), assume the NEXT occurrence
        m = re.match(r"^(\d{1,2}):(\d{2})$", s)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= mi <= 59:
                # Disambiguate: if hour is 1-12 and that time already passed, assume PM
                if 1 <= h <= 12:
                    now = datetime.now()
                    now_minutes = now.hour * 60 + now.minute
                    candidate_minutes = h * 60 + mi
                    if candidate_minutes <= now_minutes:
                        h += 12  # assume PM (e.g. "2:00" at 12:04 → 14:00)
                        if h > 23:
                            h -= 12  # safety: don't go past 23
                return f"{h:02d}:{mi:02d}"
        # Try bare number (assume PM if <= 12 and that hour already passed)
        m = re.match(r"^(\d{1,2})$", s)
        if m:
            h = int(m.group(1))
            now_h = datetime.now().hour
            if 1 <= h <= 12 and h <= now_h:
                h += 12  # assume PM
            if 0 <= h <= 23:
                return f"{h:02d}:00"
        return None

    def clear_directives(self) -> None:
        """Cancel all active directives and enforcement."""
        count = len(self.directives)
        self.directives.clear()
        self._action_log.clear()
        self._mess_mouse_count = 0
        was_enforcing = self._enforcement.active
        if was_enforcing:
            self._enforcement = EnforcementMode()
        self._hide_countdown()  # always hide — timer may linger even without enforcement
        self._directives_cleared_at = time.monotonic()  # suppress re-creation for a bit
        if count or was_enforcing:
            logger.info("Cleared %d directive(s)%s.", count, " + enforcement" if was_enforcing else "")
            print(f"[Agent] Cleared {count} directive(s){' + enforcement' if was_enforcing else ''}.", flush=True)
        self.save_directives()

    def delay_directive(self, minutes: int, goal_keyword: str = "") -> bool:
        """User negotiated a delay — replace directive with a new timed one.

        Returns False if the directive was already delayed once (no second chances).
        """
        # Find the matching directive
        target = None
        for d in self.directives:
            if goal_keyword and goal_keyword.lower() in d.goal.lower():
                target = d
                break
        if target is None:
            # Fall back to highest urgency actionable directive
            actionable = [d for d in self.directives if not (d.trigger_time and not d.triggered)]
            if actionable:
                target = max(actionable, key=lambda d: d.urgency)
        if target is None:
            return False

        # ONE delay per directive, ever
        if target.delayed:
            return False

        # Calculate the new trigger time
        fire_at = datetime.now()
        fire_at = fire_at.replace(second=0, microsecond=0)
        fire_minutes = fire_at.hour * 60 + fire_at.minute + minutes
        new_h = (fire_minutes // 60) % 24
        new_m = fire_minutes % 60
        new_time = f"{new_h:02d}:{new_m:02d}"

        # Remove the old directive, add a new timed one with delayed=True
        goal = target.goal
        self.directives.remove(target)
        now = time.monotonic()
        d = Directive(
            goal=goal,
            urgency=6,  # reset urgency but not too low
            created_at=now,
            last_action_at=now,
            source="delay",
            trigger_time=new_time,
            triggered=False,
            delayed=True,  # NO more delays allowed
        )
        self.directives.append(d)
        self.save_directives()
        logger.info("Directive delayed: %r -> fires at %s (delayed=True, no more delays)", goal, new_time)
        return True

    def set_conversation_active(self, active: bool) -> None:
        """Pause/resume autonomous behavior during conversations."""
        self._conversation_active = active
        if not active:
            # After conversation ends, wait a bit before next idle check
            self._next_idle_check_at = time.monotonic() + 30.0
            if self._timeline:
                from core.event_timeline import EventType
                self._timeline.append(EventType.CONVERSATION_END, "Conversation ended")

    @property
    def has_directives(self) -> bool:
        return bool(self.directives)

    def toggle_force_afk(self) -> bool:
        """Toggle forced AFK state for presentation mode. Returns new state."""
        self._force_afk = not self._force_afk
        if self._force_afk:
            self.routine_manager._was_away = True
            self._reset_afk_mischief()  # resets count + timer + videos
            logger.info("Presentation: forced AFK ON")
        else:
            self.routine_manager._was_away = False
            self._reset_afk_mischief()
            logger.info("Presentation: forced AFK OFF")
        return self._force_afk

    @property
    def is_force_afk(self) -> bool:
        return self._force_afk

    def start_enforcement(self, duration_s: float, directive_goal: str = "") -> None:
        """Enter enforcement mode — monitor if user actually leaves to do the task."""
        if not directive_goal:
            # Pick highest urgency actionable directive
            actionable = [d for d in self.directives if not (d.trigger_time and not d.triggered)]
            if actionable:
                directive_goal = max(actionable, key=lambda d: d.urgency).goal
        self._enforcement = EnforcementMode(
            active=True,
            start_time=time.monotonic(),
            duration_s=duration_s,
            directive_goal=directive_goal,
            last_check=time.monotonic(),
        )
        self.save_directives()
        logger.info("Enforcement mode started: %.0fs for %r", duration_s, directive_goal)
        print(f"[ENFORCEMENT STARTED] Monitoring for {duration_s:.0f}s — goal: \"{directive_goal}\"")
        # Show countdown timer on the pet
        if self._robot and hasattr(self._robot, 'countdown_start'):
            self._robot.countdown_start.emit(int(duration_s))

    # Idle threshold for enforcement: 15s avoids false positives from mouse jitter,
    # system notifications, or optical sensor drift.
    _ENFORCEMENT_IDLE_MS = 15_000

    def _check_enforcement(self) -> None:
        """Smart enforcement — uses its own idle tracking, not routine_manager.

        Key fix: routine_manager uses a 3-MINUTE threshold for away detection,
        but enforcement needs a much shorter threshold (15s). Using routine_manager's
        away_duration_s gave wildly wrong durations (stale from hours-old transitions).
        """
        now = time.monotonic()
        elapsed = now - self._enforcement.start_time
        duration = self._enforcement.duration_s

        # Grace period (10s for user to walk away)
        if elapsed < 10.0:
            return

        # Poll every 1 second
        if (now - self._enforcement.last_check) < 1.0:
            return
        self._enforcement.last_check = now

        idle_ms = _get_idle_ms()
        self._enforcement.check_count += 1

        # Debug output every 30 checks (~30 seconds)
        if self._enforcement.check_count % 30 == 0:
            state = "IDLE" if idle_ms >= self._ENFORCEMENT_IDLE_MS else "ACTIVE"
            print(f"[ENFORCEMENT] {elapsed:.0f}s/{duration:.0f}s | idle={idle_ms}ms ({state})")

        # ── User is idle (away from computer) ──
        if idle_ms >= self._ENFORCEMENT_IDLE_MS:
            if not self._enforcement.was_idle:
                # Just went idle — record when
                self._enforcement.was_idle = True
                self._enforcement.idle_since = now
                logger.info("Enforcement: user went idle (doing task?)")

            # Timer expired while away — note it
            if elapsed >= duration and not self._enforcement.expired:
                self._enforcement.expired = True
                logger.info("Enforcement timer expired for %r — user still away.",
                            self._enforcement.directive_goal)
            return

        # ── User is active (touching input) ──

        if self._enforcement.was_idle:
            # JUST RETURNED from being idle — compute actual away duration
            away_dur = now - self._enforcement.idle_since
            self._enforcement.was_idle = False
            self._enforcement.last_away_dur = away_dur

            # Cooldown: don't ask within 30s of last ask
            if (now - self._enforcement.last_caught_at) < 30.0:
                return

            self._enforcement.last_caught_at = now
            self._enforcement.caught_count += 1

            ratio = away_dur / duration if duration > 0 else 0
            print(f"[ENFORCEMENT] User returned — away {away_dur:.0f}s / "
                  f"{duration:.0f}s expected (ratio {ratio:.1%})")

            if ratio >= 0.5:
                self._enforcement_auto_complete(away_dur)
            elif ratio >= 0.2:
                self._enforcement_casual_checkin(away_dur)
            elif ratio >= 0.1:
                self._enforcement_ask_if_done()
            else:
                self._enforcement_ask_if_done_skeptical()
        else:
            # User has been continuously active — never left or returned a while ago.
            # Only nag if they've been sitting here a long time without leaving.
            if elapsed < 60.0:
                return  # give them time to wrap up and leave
            if (now - self._enforcement.last_caught_at) < 180.0:
                return  # don't nag more than once per 3 minutes when they won't leave
            self._enforcement.last_caught_at = now
            self._enforcement.caught_count += 1
            self._enforcement_ask_if_done()

    # ── Enforcement interaction (LLM-driven, temporally-aware) ──────────

    def _get_screen_note(self) -> str:
        """Build a short screen context note for enforcement/reply prompts."""
        try:
            state = self._monitor.get_state()
            if state and state.foreground:
                fg = state.foreground
                return f" Their screen shows: \"{fg.title}\" ({fg.exe_name})."
        except Exception:
            pass
        return ""

    def _enforcement_auto_complete(self, away_seconds: float) -> None:
        """User was away >= 50% of expected time — assume they did it. Welcome warmly."""
        goal = self._enforcement.directive_goal
        name = get_character_name()
        dur_str = self._fmt_duration(away_seconds)

        context = ""
        if self._timeline:
            intent = self._timeline.user_intent
            if intent:
                context = f" They said they were going to: {intent.action}."

        prompt = (
            f"You are {name}. The user was away for {dur_str} doing '{goal}'.{context} "
            f"They just came back. Welcome them back warmly — you're confident they "
            f"did it because they were gone long enough. ONE sentence, in character. "
            f"Be genuinely pleased, not suspicious."
        )
        text = self._llm.generate_once(prompt, max_tokens=100)
        text = self._strip_think(text).strip().strip('"')
        if not text:
            text = "welcome back! that was good timing."
        self._speak(text)
        self._enforcement_complete(text)

        if self._timeline:
            from core.event_timeline import EventType
            self._timeline.append(EventType.ENFORCEMENT_COMPLETE,
                                  f"Task auto-completed: {goal} (away {dur_str})")
            self._timeline.set_user_intent(None)

    def _enforcement_casual_checkin(self, away_seconds: float) -> None:
        """User was away 20-50% of expected time. Ask casually, not suspiciously."""
        goal = self._enforcement.directive_goal
        name = get_character_name()
        dur_str = self._fmt_duration(away_seconds)

        if self._detector:
            try:
                self._detector.pause()
            except Exception:
                pass

        try:
            screen_note = self._get_screen_note()
            prompt = (
                f"You are {name}. The user was supposed to {goal}. They left for {dur_str} "
                f"and just came back.{screen_note} That's a reasonable amount of time — they probably did it. "
                f"Welcome them back and casually ask how it went. ONE sentence. Be warm, not suspicious."
            )
            text = self._llm.generate_once(prompt, max_tokens=100)
            text = self._strip_think(text).strip().strip('"')
            if not text:
                text = "hey, welcome back! how'd it go?"
            self._speak(text)

            response = self._enforcement_listen()
            if response is None:
                # No response — don't auto-complete, keep enforcement active.
                # They were gone a reasonable time but we can't confirm they did it.
                self._enforcement.caught_count += 1
                self._llm.inject_history(
                    f"(Enforcement for \"{goal}\": you asked how it went, no response. "
                    f"Keeping enforcement active — can't confirm completion without a response.)",
                    text,
                )
                logger.info("Enforcement casual checkin: no response, keeping enforcement active")
                return

            # Classify response — distinguish "did it" from "going to do it"
            classify_prompt = (
                f"The user was asked casually how '{goal}' went after being away "
                f"for {dur_str}. They said: \"{response}\"\n"
                f"Did they indicate they ALREADY completed it (even partially)? YES or NO.\n"
                f"Note: 'I'm going to do it' or 'I'll do it now' means NO — they haven't done it yet."
            )
            verdict = self._llm.generate_once(
                classify_prompt, max_tokens=10,
                system_prompt="Reply YES or NO only."
            )
            verdict = self._strip_think(verdict).strip().upper()

            # Inject the whole exchange into history
            self._llm.inject_history(
                f"(Enforcement: you asked about \"{goal}\". User said: \"{response}\")",
                text,
            )

            if "YES" in verdict:
                complete_prompt = (
                    f"You are {name}. The user confirmed they did '{goal}'. "
                    f"React positively in ONE sentence."
                )
                complete_text = self._llm.generate_once(complete_prompt, max_tokens=100)
                complete_text = self._strip_think(complete_text).strip().strip('"')
                if complete_text:
                    self._speak(complete_text)
                self._enforcement_complete(complete_text or text)
                if self._timeline:
                    from core.event_timeline import EventType
                    self._timeline.append(EventType.ENFORCEMENT_COMPLETE,
                                          f"Task confirmed complete: {goal}")
                    self._timeline.set_user_intent(None)
            else:
                # Didn't do it despite being away — nag but don't lockdown yet
                nag_prompt = (
                    f"You are {name}. The user was away for {dur_str} but didn't "
                    f"actually do '{goal}'. Be disappointed but not aggressive. ONE sentence."
                )
                nag_text = self._llm.generate_once(nag_prompt, max_tokens=100)
                nag_text = self._strip_think(nag_text).strip().strip('"')
                if nag_text:
                    self._speak(nag_text)
                    self._llm.inject_history(
                        f"(User didn't do \"{goal}\" despite being away.)",
                        nag_text,
                    )
        finally:
            if self._detector:
                try:
                    self._detector.resume()
                except Exception:
                    pass

    def _enforcement_ask_if_done_skeptical(self) -> None:
        """User barely left their desk. Be skeptical when asking."""
        goal = self._enforcement.directive_goal
        name = get_character_name()
        caught_count = self._enforcement.caught_count

        if self._detector:
            try:
                self._detector.pause()
            except Exception:
                pass

        try:
            screen_note = self._get_screen_note()
            prompt = (
                f"You are {name}. The user was supposed to go {goal}, but they "
                f"barely left their computer. They've only been away for a moment.{screen_note} "
                f"This is catch #{caught_count}. "
                f"Call them out — they clearly didn't do it. ONE sentence, in character. "
                f"Be skeptical and pushy."
            )
            ask_text = self._llm.generate_once(prompt, max_tokens=100)
            ask_text = self._strip_think(ask_text).strip().strip('"')
            if not ask_text:
                ask_text = f"uh... that was fast. you definitely didn't {goal}."
            self._speak(ask_text)

            response = self._enforcement_listen()
            if response is None:
                self._llm.inject_history(
                    f"(Enforcement: you called out user about \"{goal}\", no response. "
                    f"This is ignored-ask #{caught_count}.)",
                    ask_text,
                )
                # Skeptical + no response: escalate to lockdown after 2+ catches
                if caught_count >= 2:
                    logger.info("Enforcement skeptical: %d ignored asks — lockdown",
                                caught_count)
                    self._enforcement_lockdown()
                return

            # Inject the exchange so pony remembers
            self._llm.inject_history(
                f"(Enforcement: you asked about \"{goal}\". User said: \"{response}\")",
                ask_text,
            )

            classify_prompt = (
                f"The user was asked if they finished '{goal}'. "
                f"They responded: \"{response}\"\n"
                f"Classify: (A) they ALREADY completed the task, "
                f"(B) they're saying they'll GO DO IT now, "
                f"(C) they did NOT do it and aren't leaving.\n"
                f"Reply A, B, or C only."
            )
            verdict = self._llm.generate_once(
                classify_prompt, max_tokens=10,
                system_prompt="Reply A, B, or C only."
            )
            verdict = self._strip_think(verdict).strip().upper()

            if "A" in verdict:
                complete_prompt = (
                    f"You are {name}. The user claims they did '{goal}' super fast. "
                    f"Be surprised but accept it. ONE sentence."
                )
                complete_text = self._llm.generate_once(complete_prompt, max_tokens=100)
                complete_text = self._strip_think(complete_text).strip().strip('"')
                if complete_text:
                    self._speak(complete_text)
                self._enforcement_complete(complete_text or "...okay, if you say so.")
                if self._timeline:
                    from core.event_timeline import EventType
                    self._timeline.append(EventType.ENFORCEMENT_COMPLETE,
                                          f"Task claimed complete (skeptical): {goal}")
                    self._timeline.set_user_intent(None)
            elif "B" in verdict:
                go_prompt = (
                    f"You are {name}. The user says they'll go {goal} now. "
                    f"Tell them to hurry up. ONE sentence, in character."
                )
                go_text = self._llm.generate_once(go_prompt, max_tokens=100)
                go_text = self._strip_think(go_text).strip().strip('"')
                if go_text:
                    self._speak(go_text)
                self._enforcement.was_idle = False
                self._enforcement.idle_since = 0.0
                # Grace period: don't check again for 3 minutes
                self._enforcement.last_caught_at = time.monotonic() + 150.0
            else:
                self._enforcement_lockdown()
        finally:
            if self._detector:
                try:
                    self._detector.resume()
                except Exception:
                    pass

    def _enforcement_ask_if_done(self) -> None:
        """User was away for a short but non-trivial time — ask neutrally.
        If yes → done. If no → LOCKDOWN. All speech is LLM-generated."""
        goal = self._enforcement.directive_goal
        caught_count = self._enforcement.caught_count

        # Pause wake word detector for the entire enforcement interaction
        if self._detector:
            try:
                self._detector.pause()
            except Exception:
                pass

        try:
            # LLM generates the question (in character)
            name = get_character_name()
            screen_note = self._get_screen_note()
            ask_prompt = (
                f"You are {name}. The user was supposed to go {goal}. "
                f"You sent them away, but they just touched the mouse/keyboard again.{screen_note} "
                f"This is catch #{caught_count}. "
                f"Ask them if they actually did it. ONE sentence, in character. "
                f"{'Be suspicious — they keep coming back too fast.' if caught_count > 1 else 'Be direct.'}"
            )
            ask_text = self._llm.generate_once(ask_prompt, max_tokens=100)
            ask_text = self._strip_think(ask_text).strip().strip('"')
            if not ask_text:
                ask_text = f"did you {goal}?"
            self._speak(ask_text)

            # Open mic and listen for response
            response = self._enforcement_listen()
            if response is None:
                logger.info("Enforcement: no response to ask-if-done (catch #%d)",
                            caught_count)
                self._llm.inject_history(
                    f"(Enforcement: you asked about \"{goal}\", no response. "
                    f"This is ignored-ask #{caught_count}.)",
                    ask_text,
                )
                # After 3+ ignored asks, treat silence as refusal → lockdown
                if caught_count >= 3:
                    logger.info("Enforcement: %d ignored asks — escalating to lockdown",
                                caught_count)
                    self._enforcement_lockdown()
                return

            logger.info("Enforcement response: %r", response)

            # Inject the exchange so pony remembers
            self._llm.inject_history(
                f"(Enforcement: you asked about \"{goal}\". User said: \"{response}\")",
                ask_text,
            )

            # LLM classifies intent — distinguish "did it" from "going to do it"
            classify_prompt = (
                f"The user was asked if they finished '{goal}'. "
                f"They responded: \"{response}\"\n"
                f"Classify: (A) they ALREADY completed the task, "
                f"(B) they're saying they'll GO DO IT now / are leaving to do it, "
                f"(C) they did NOT do it and aren't leaving.\n"
                f"Reply A, B, or C only."
            )
            verdict = self._llm.generate_once(
                classify_prompt, max_tokens=10,
                system_prompt="Reply A, B, or C only."
            )
            verdict = self._strip_think(verdict).strip().upper()
            logger.info("Enforcement verdict: %r (from response: %r)", verdict, response)

            if "A" in verdict:
                # Confirmed completed
                complete_prompt = (
                    f"You are {name}. The user just confirmed they completed '{goal}'. "
                    f"React in ONE sentence. Be genuinely pleased, in character."
                )
                complete_text = self._llm.generate_once(complete_prompt, max_tokens=100)
                complete_text = self._strip_think(complete_text).strip().strip('"')
                if complete_text:
                    self._speak(complete_text)
                self._enforcement_complete(complete_text or "okay, nice.")
                return

            if "B" in verdict:
                # Going to do it — encourage and give them real time to leave
                go_prompt = (
                    f"You are {name}. The user says they're going to go {goal} now. "
                    f"Encourage them briefly. ONE sentence, in character."
                )
                go_text = self._llm.generate_once(go_prompt, max_tokens=100)
                go_text = self._strip_think(go_text).strip().strip('"')
                if go_text:
                    self._speak(go_text)
                # Reset idle tracking — they should go idle soon
                self._enforcement.was_idle = False
                self._enforcement.idle_since = 0.0
                # Grace period: don't check again for 3 minutes
                self._enforcement.last_caught_at = time.monotonic() + 150.0
                return  # keep enforcement active, wait for them to leave

            # Not confirmed → lockdown
            self._enforcement_lockdown()
        finally:
            # Always resume wake word detector
            if self._detector:
                try:
                    self._detector.resume()
                except Exception:
                    pass

    def _enforcement_listen(self, timeout: float = 8.0) -> Optional[str]:
        """Open mic briefly during enforcement to hear user's response."""
        if not self._transcriber:
            return None
        try:
            if self._on_state_change:
                self._on_state_change("LISTEN")
            text = self._transcriber.listen(
                speech_start_timeout_s=timeout,
                initial_discard_ms=400,
            )
            # Filter hallucinations
            if text:
                from stt.transcriber import _is_whisper_hallucination
                if _is_whisper_hallucination(text):
                    logger.debug("Filtered hallucination in enforcement listen: %r", text)
                    return None
            return text
        except Exception as exc:
            logger.warning("Enforcement listen failed: %s", exc)
            return None
        finally:
            if self._on_state_change:
                self._on_state_change("IDLE")

    def _enforcement_complete(self, response_text: str = "okay, nice.") -> None:
        """User confirmed they did the task — remove directive and end enforcement."""
        goal = self._enforcement.directive_goal
        # Remove the matching directive
        for i, d in enumerate(self.directives):
            if d.goal == goal:
                self.directives.pop(i)
                logger.info("Enforcement completed: %r", goal)
                break
        # Track so LLM doesn't re-create it
        self._recently_completed_goals.append(
            (time.monotonic(), goal.lower().strip()))
        self._enforcement = EnforcementMode()
        self._enforcement_just_completed = True  # suppress welcome-back
        self._hide_countdown()
        self.save_directives()
        self._llm.inject_history(
            f"(User confirmed they completed \"{goal}\" during enforcement.)",
            response_text,
        )
        self._log_action(f"Enforcement completed: \"{goal[:40]}\"")
        if not self.directives:
            self._mess_mouse_count = 0

    def _enforcement_lockdown(self) -> None:
        """User said they HAVEN'T done it — full lockdown until they go do it.
        All speech is LLM-generated; mechanical actions (minimize, cursor lock) stay code-driven."""
        goal = self._enforcement.directive_goal
        name = get_character_name()

        # Nuke urgency to 10
        for d in self.directives:
            if d.goal == goal:
                d.urgency = 10
                break
        self.save_directives()

        # LLM generates lockdown announcement
        screen_note = self._get_screen_note()
        lockdown_prompt = (
            f"You are {name}. The user said they HAVEN'T done '{goal}' yet.{screen_note} "
            f"You're about to lock their computer until they go do it. "
            f"Tell them what's happening. Be firm but in-character. TWO sentences max."
        )
        lockdown_text = self._llm.generate_once(lockdown_prompt, max_tokens=150)
        lockdown_text = self._strip_think(lockdown_text).strip().strip('"')
        if not lockdown_text:
            lockdown_text = f"nope. go {goal}. I'm locking your computer."
        self._speak(lockdown_text)
        self._llm.inject_history(
            f"(User said they haven't done \"{goal}\" — entering lockdown mode.)",
            lockdown_text,
        )

        # LOCKDOWN LOOP — minimize everything, mess with mouse, keep asking
        logger.info("ENFORCEMENT LOCKDOWN for %r", goal)
        print(f"[ENFORCEMENT LOCKDOWN] Locking computer until user goes to {goal}")

        # At urgency 10 — permanent mouse lock (ClipCursor to a tiny box)
        urgency_10 = any(d.urgency >= 10 and d.goal == goal for d in self.directives)
        cursor_locked = False
        if urgency_10 and self._desktop:
            try:
                import ctypes
                import ctypes.wintypes
                # Lock cursor to a 1x1 box at center of the pony's monitor
                mon = self._desktop._get_monitor_rect()
                cx = mon.left + mon.width // 2
                cy = mon.top + mon.height // 2
                rect = ctypes.wintypes.RECT(cx, cy, cx + 1, cy + 1)
                ctypes.windll.user32.ClipCursor(ctypes.byref(rect))
                cursor_locked = True
                logger.info("Cursor LOCKED at (%d,%d) — urgency 10", cx, cy)
            except Exception as exc:
                logger.warning("ClipCursor failed: %s", exc)

        lockdown_round = 0
        try:
            while self._enforcement.active:
                lockdown_round += 1

                if self._stopped or lockdown_round > 30:
                    logger.warning("Enforcement lockdown ending after %d rounds (stopped=%s)",
                                   lockdown_round, self._stopped)
                    self._enforcement.active = False
                    break

                # Minimize all windows
                if self._desktop:
                    self._desktop.minimize_all_windows()

                # Mess with mouse for 10 seconds (at urgency 10, mouse is locked anyway)
                if self._desktop and not cursor_locked:
                    self._desktop.mess_with_mouse(duration=10.0, jitter=120)
                elif cursor_locked:
                    time.sleep(10.0)  # mouse is locked, just wait

                # Brief pause, then LLM-generated taunt
                time.sleep(1.0)

                taunt_prompt = (
                    f"You are {name}. Lockdown round {lockdown_round}. "
                    f"The user STILL hasn't gone to {goal}. You have their computer locked. "
                    f"Say ONE sentence — be increasingly dramatic/annoyed. In character."
                )
                taunt_text = self._llm.generate_once(taunt_prompt, max_tokens=100)
                taunt_text = self._strip_think(taunt_text).strip().strip('"')
                if taunt_text:
                    self._speak(taunt_text)

                response = self._enforcement_listen(timeout=6.0)
                if response:
                    logger.info("Lockdown response: %r", response)
                    # LLM classifies: (A) completed, (B) stop/give up,
                    # (C) going to do it now, (D) neither
                    classify_prompt = (
                        f"User response during lockdown for '{goal}': \"{response}\"\n"
                        f"Classify: (A) they ALREADY completed the task, "
                        f"(B) ask to stop/give up/quit entirely, "
                        f"(C) they'll GO DO IT now (agreeing to leave), "
                        f"(D) neither / arguing / stalling.\n"
                        f"Note: 'I'm going' / 'fine I'll do it' / 'okay' = C, not A.\n"
                        f"Reply A, B, C, or D only."
                    )
                    verdict = self._llm.generate_once(
                        classify_prompt, max_tokens=10,
                        system_prompt="Reply A, B, C, or D only."
                    )
                    verdict = self._strip_think(verdict).strip().upper()
                    logger.info("Lockdown verdict: %r", verdict)

                    if "A" in verdict:
                        # Completed during lockdown
                        release_prompt = (
                            f"You are {name}. The user FINALLY completed '{goal}' during lockdown "
                            f"after {lockdown_round} rounds. React in ONE sentence. In character."
                        )
                        release_text = self._llm.generate_once(release_prompt, max_tokens=100)
                        release_text = self._strip_think(release_text).strip().strip('"')
                        if release_text:
                            self._speak(release_text)
                        self._enforcement_complete(release_text or "finally.")
                        return
                    elif "B" in verdict:
                        # Giving up — grudging release
                        stop_prompt = (
                            f"You are {name}. The user asked you to stop the lockdown for '{goal}'. "
                            f"You're giving in, but you're NOT happy about it. ONE sentence, in character."
                        )
                        stop_text = self._llm.generate_once(stop_prompt, max_tokens=100)
                        stop_text = self._strip_think(stop_text).strip().strip('"')
                        if stop_text:
                            self._speak(stop_text)
                        self._enforcement = EnforcementMode()
                        self._hide_countdown()
                        self.save_directives()
                        return
                    elif "C" in verdict:
                        # Going to do it — release lockdown but KEEP enforcement
                        go_prompt = (
                            f"You are {name}. The user agreed to go {goal}. "
                            f"Tell them to go — you'll be watching. ONE sentence, in character."
                        )
                        go_text = self._llm.generate_once(go_prompt, max_tokens=100)
                        go_text = self._strip_think(go_text).strip().strip('"')
                        if go_text:
                            self._speak(go_text)
                        # Release lockdown but keep enforcement active
                        # Reset idle tracking so we detect when they actually leave
                        self._enforcement.was_idle = False
                        self._enforcement.idle_since = 0.0
                        self._enforcement.caught_count = 0
                        self._enforcement.last_caught_at = time.monotonic()
                        logger.info("Lockdown released — enforcement continues for %r", goal)
                        return

                logger.info("Lockdown round %d — user still hasn't done %r", lockdown_round, goal)
        finally:
            # ALWAYS release cursor lock when exiting lockdown
            if cursor_locked:
                try:
                    ctypes.windll.user32.ClipCursor(None)
                    logger.info("Cursor lock released.")
                except Exception:
                    pass

    @staticmethod
    def _ordinal(n: int) -> str:
        if n == 1: return "first"
        if n == 2: return "second"
        if n == 3: return "third"
        return f"{n}th"

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from LLM output.
        Handles unclosed <think> tags (strips from <think> to end)."""
        # First try closed tags
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Handle unclosed <think> — strip from <think> to end of string
        result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
        return result.strip()

    def stop(self) -> None:
        """Signal this agent loop to stop. Called by PonyInstance.destroy()."""
        self._stopped = True

    def tick(self) -> None:
        """Called every ~1s from the pipeline thread. Decides if anything needs doing."""
        # Always track idle/wake state — needed for sleep detection even during conversation
        idle_ms = _get_idle_ms()
        away_dur = self.routine_manager.away_duration_s  # grab BEFORE update clears it

        # Media-aware AFK: watching video ≠ away
        # Fullscreen = 30min threshold; windowed media = 10min threshold
        state = self._monitor.get_state()
        media_active = state.is_media_fullscreen if state else False
        windowed_media_active = False
        if not media_active and state and state.foreground:
            from core.screen_monitor import _is_media_app
            windowed_media_active = _is_media_app(
                state.foreground.exe_name, state.foreground.title or ""
            )
        self._last_wake_event = self.routine_manager.update_activity(
            idle_ms, media_active=media_active, windowed_media_active=windowed_media_active
        )

        # Presentation mode: force AFK state — override idle detection
        if self._force_afk:
            self.routine_manager._was_away = True
            self._last_wake_event = None  # suppress welcome-back greeting

        # Log activity transitions to timeline
        if self._timeline:
            from core.event_timeline import EventType, ActivityState
            if self._last_wake_event == "wake":
                self._timeline.append(EventType.USER_RETURNED,
                                      f"User returned after {self._fmt_duration(away_dur or 0)}",
                                      {"away_seconds": away_dur})
            elif self._last_wake_event == "away":
                intent = self._timeline.user_intent
                reason = f" (likely doing: {intent.action})" if intent else ""
                self._timeline.append(EventType.USER_WENT_AFK, f"User went AFK{reason}")
                self._timeline.set_activity_state(
                    ActivityState.AFK_TASK if self._enforcement.active else ActivityState.AFK_UNKNOWN)
            elif state and not self.routine_manager.is_user_away:
                self._timeline.set_activity_state(self._classify_activity(state))

        # Welcome-back greeting when user returns from AFK
        if self._last_wake_event == "wake" and not self._conversation_active:
            self._reset_afk_mischief()  # reset AFK fun counter
            # Skip if enforcement just finished (already greeted) or very short absence
            if self._enforcement_just_completed:
                self._enforcement_just_completed = False
            elif away_dur is None or away_dur < 300:
                pass  # unknown duration (app just started) or too short to greet
            else:
                self._welcome_back(away_dur)

        # Standing rules run ALWAYS — even during conversation or enforcement.
        # They're passive detection: check window titles and react immediately.
        if self._standing_rules and state and not self.routine_manager.is_user_away:
            self._check_standing_rules(state)

        # Enforcement runs even during conversation for idle TRACKING, but
        # don't poll/react during active conversation (user IS at keyboard talking to Dash)
        if self._enforcement.active:
            if not self._conversation_active:
                self._check_enforcement()
            return

        if self._conversation_active:
            return

        now = time.monotonic()

        # Check wall-clock timers
        self._check_timers()

        # Check recurring routines (wake/sleep detection + scheduled)
        self._check_routines()

        # User is away — instead of going silent, occasionally have fun
        if self.routine_manager.is_user_away or self._force_afk:
            self._maybe_afk_mischief()
            return

        # Activate date-triggered directives when their date arrives
        today_str = datetime.now().strftime("%Y-%m-%d")
        for d in self.directives:
            if d.trigger_date and not d.triggered and d.trigger_date <= today_str:
                d.triggered = True
                d.next_nag_at = now + self._initial_nag_delay(d.urgency)
                logger.info("Date-triggered directive activated: %r (date: %s)", d.goal, d.trigger_date)
                self.save_directives()

        # Per-directive timing: check if ANY directive is due for a nag
        actionable = [d for d in self.directives
                      if not (d.trigger_time and not d.triggered)
                      and not (d.trigger_date and not d.triggered)]
        due = [d for d in actionable if d.next_nag_at <= now]
        if due:
            self._execute_tick()
            return

        # No directives due — handle idle behavior (self-initiation, spontaneous speech)
        if now < self._next_idle_check_at:
            return

        # Self-initiation check (only when no directives at all)
        if not actionable and self._config.self_initiate:
            if (now - self._last_self_check) >= self._config.self_initiate_interval_s:
                self._last_self_check = now
                self._maybe_self_initiate()
                return

        # Observation tick OR spontaneous speech (every 2-5 min)
        # Fires regardless of whether directives exist — she should still talk
        if now >= self._next_spontaneous:
            next_in = random.uniform(
                self._config.spontaneous_speech_min_s,
                self._config.spontaneous_speech_max_s,
            )
            # 60% observation tick (screen-aware), 40% classic spontaneous speech
            if random.random() < 0.6:
                print(f"\n[Agent] Observation tick (next in {next_in:.0f}s)", flush=True)
                self._observation_tick()
            else:
                print(f"\n[Agent] Spontaneous speech triggered (next in {next_in:.0f}s)", flush=True)
                self._spontaneous_speech()
            self._next_spontaneous = time.monotonic() + next_in
        else:
            self._next_idle_check_at = now + min(
                self._config.base_check_interval_s,
                self._next_spontaneous - now + 1.0,
            )

    # ── Standing rule enforcement ──────────────────────────────────────────

    def _check_standing_rules(self, state: "ScreenState") -> None:
        """Check all window titles against standing rules. Runs every tick (~1s).

        This is CODE-driven detection, not LLM-driven. It matches window titles
        against known patterns (NSFW sites, spending sites, custom keywords) and
        reacts immediately — closing the offending window and speaking.
        """
        now = time.monotonic()

        # Build lowercase title list once (shared across all rules)
        all_titles = []
        for w in state.open_windows:
            if w.title and w.title.strip():
                all_titles.append((w.title, w.title.lower()))

        for rule in self._standing_rules:
            # Cooldown: don't re-trigger within cooldown_s
            if (now - rule.last_triggered_at) < rule.cooldown_s:
                continue

            matched = self._match_standing_rule(rule, all_titles)
            if matched:
                self._trigger_standing_rule(rule, matched, state)

    def _match_standing_rule(self, rule: StandingRule,
                             all_titles: List[Tuple[str, str]]) -> Optional[str]:
        """Check if any window title violates a standing rule.

        Returns the original (un-lowered) title of the first match, or None.
        Pure pattern matching — patterns were generated by the LLM at rule creation.
        """
        for original_title, title_lower in all_titles:
            for pat in rule.patterns:
                if pat in title_lower:
                    return original_title
        return None

    def _trigger_standing_rule(self, rule: StandingRule, matched_title: str,
                               state: "ScreenState") -> None:
        """React to a standing rule violation: close the window, speak, escalate."""
        rule.catch_count += 1
        rule.last_triggered_at = time.monotonic()
        self.save_directives()

        logger.info("STANDING RULE TRIGGERED: %r (catch #%d, matched: %r)",
                     rule.description, rule.catch_count, matched_title[:80])
        print(f"[STANDING RULE] Caught #{rule.catch_count}: \"{matched_title[:60]}\" "
              f"(rule: {rule.description})", flush=True)

        if self._timeline:
            from core.event_timeline import EventType
            self._timeline.append(
                EventType.DIRECTIVE_CREATED,
                f'Standing rule triggered: "{rule.description}" — '
                f'caught: "{matched_title[:60]}" (#{rule.catch_count})')

        # ── Step 1: Close the offending window/tab ──
        closed = False
        if self._desktop and rule.response in ("close_and_nag", "lockdown"):
            try:
                # Try closing by exact title match first
                closed = self._desktop.close_window_by_title(matched_title)
                if not closed:
                    # Try partial match
                    for w in state.open_windows:
                        if w.title == matched_title:
                            closed = self._desktop.close_window_by_title(w.title[:40])
                            break
                if closed:
                    self._log_action(f"Closed window: \"{matched_title[:50]}\"")
            except Exception as exc:
                logger.warning("Failed to close window for standing rule: %s", exc)

        # ── Step 2: Speak — LLM-generated reaction ──
        name = get_character_name()
        # Truncate title to avoid leaking explicit content into TTS
        safe_title = matched_title[:30] + "..." if len(matched_title) > 30 else matched_title
        try:
            if rule.catch_count <= 1:
                prompt = (
                    f"You are {name}. You just caught the user breaking their rule: "
                    f"'{rule.description}'. They were on \"{safe_title}\". You closed it. "
                    f"React — be direct and disappointed. ONE sentence, in character."
                )
            elif rule.catch_count <= 3:
                prompt = (
                    f"You are {name}. You caught the user AGAIN — catch #{rule.catch_count} "
                    f"breaking '{rule.description}'. Caught them on \"{safe_title}\". You closed it. "
                    f"Be angry/disappointed. They keep doing this. ONE sentence."
                )
            else:
                prompt = (
                    f"You are {name}. Catch #{rule.catch_count} for '{rule.description}'. "
                    f"They were on \"{safe_title}\". "
                    f"At this point you're furious. ONE sentence. Be brutal."
                )
            nag = self._llm.generate_once(prompt, max_tokens=80)
            nag = self._strip_think(nag).strip().strip('"')
            # Strip any DESKTOP commands from standing rule reactions —
            # the LLM sometimes OPENS the banned site in its reaction
            import re as _re
            nag = _re.sub(r'\[DESKTOP:[^\]]*\]', '', nag, flags=_re.IGNORECASE).strip()
            nag = _re.sub(r'\[ACTION:[^\]]*\]', '', nag, flags=_re.IGNORECASE).strip()
        except Exception:
            nag = None

        if not nag:
            nag = f"seriously? i literally JUST closed that. {rule.description}."
        self._speak(nag)
        self._log_action(f"Standing rule: \"{rule.description}\" catch #{rule.catch_count}")

        # ── Step 3: Escalate on repeated violations ──
        # (No Win+D — minimizing everything is counterproductive, just close + nag + lock)

        if rule.catch_count >= 5 and self._desktop:
            # After 5+ catches: lock mouse briefly (tight loop = inescapable)
            try:
                import ctypes
                import ctypes.wintypes
                mon = self._desktop._get_monitor_rect()
                cx = mon.left + mon.width // 2
                cy = mon.top + mon.height // 2
                rect = ctypes.wintypes.RECT(cx, cy, cx + 1, cy + 1)
                seconds = min(5.0 + rule.catch_count, 30.0)
                self._log_action(f"Locked mouse {seconds:.0f}s (standing rule catch #{rule.catch_count})")
                end_time = time.monotonic() + seconds
                while time.monotonic() < end_time:
                    ctypes.windll.user32.ClipCursor(ctypes.byref(rect))
                    ctypes.windll.user32.SetCursorPos(cx, cy)
                    time.sleep(0.05)
            except Exception:
                pass
            finally:
                try:
                    ctypes.windll.user32.ClipCursor(None)
                except Exception:
                    pass

        if rule.catch_count >= 7 and self._on_grab_cursor:
            # After 7+ catches: grab cursor and run with it
            duration = min(15.0 + rule.catch_count * 3.0, 60.0)
            self._on_grab_cursor(duration)
            self._log_action(f"Grabbed cursor ({duration:.0f}s) — standing rule escalation")

    # ── Core tick execution ────────────────────────────────────────────────

    def _execute_tick(self) -> None:
        """Fire an LLM call for active directives."""
        try:
            state = self._monitor.get_state()
            directives_str = ", ".join(f'"{d.goal}" (urg {d.urgency})' for d in self.directives)
            print(f"\n[Agent] Directive tick — active: {directives_str}", flush=True)

            # Snapshot which directives are due BEFORE the LLM call and timing updates
            now_pre = time.monotonic()
            actionable = [d for d in self.directives
                          if not (d.trigger_time and not d.triggered)]
            due_set = {id(d) for d in actionable if d.next_nag_at <= now_pre}

            prompt = self._build_tick_prompt(state)

            logger.debug("Agent tick prompt: %s", prompt[:200])
            raw = self._llm.generate_once(prompt, max_tokens=1024)
            logger.debug("Agent tick response: %s", raw[:300])

            decision = self._parse_decision(raw)
            self._execute_decision(decision, state, due_set)

            # Apply per-directive timings from LLM
            now_m = time.monotonic()
            # Re-fetch actionable in case _execute_decision removed one
            actionable = [d for d in self.directives
                          if not (d.trigger_time and not d.triggered)]
            for i, d in enumerate(actionable):
                key = str(i)
                was_due = id(d) in due_set
                if key in decision.directive_timings:
                    timing = decision.directive_timings[key]
                    # Only bump nag_count if this directive was actually due
                    if was_due:
                        d.nag_count += 1
                    nag_min = float(timing.get("next_nag_minutes", 10))
                    # Urgency 10 = BURST MODE: nag every 15-45 seconds
                    # (for "freak out about this" demos and urgent situations)
                    if d.urgency >= 10:
                        min_minutes = 0.25   # 15 seconds
                        max_minutes = 0.75   # 45 seconds
                    elif d.urgency >= 7:
                        min_minutes = 3.0
                        max_minutes = 7.0
                    else:
                        min_minutes = 5.0
                        max_minutes = 10.0
                    nag_min = max(nag_min, min_minutes)
                    nag_min = min(nag_min, max_minutes)
                    d.next_nag_at = now_m + nag_min * 60.0
                    if "urgency" in timing:
                        d.urgency = max(1, min(10, int(timing["urgency"])))
                elif was_due:
                    # Directive was due but LLM didn't mention it — use a
                    # SHORT fallback so we don't go silent for 10+ minutes
                    d.nag_count += 1
                    if d.urgency >= 10:
                        fallback_min = 0.5   # 30 seconds for burst mode
                    elif d.urgency >= 7:
                        fallback_min = 4.0
                    else:
                        fallback_min = 6.0
                    d.next_nag_at = now_m + fallback_min * 60.0
                # If directive was NOT due and LLM didn't mention it, leave
                # its next_nag_at untouched — don't push out a future nag
            self.save_directives()

            timings_str = ", ".join(
                f"d{k}→{v.get('next_nag_minutes', '?')}min"
                for k, v in decision.directive_timings.items()
            )
            print(f"[Agent] Decision: speak={bool(decision.speak)}, actions={len(decision.actions)}, timings=[{timings_str}]", flush=True)

        except Exception as exc:
            print(f"[Agent] Tick failed: {exc}", flush=True)
            logger.warning("Agent tick failed: %s", exc)
            # Back off on error — push all due directives out 60s
            now_m = time.monotonic()
            for d in self.directives:
                if d.next_nag_at <= now_m:
                    d.next_nag_at = now_m + 60.0

    def _build_tick_prompt(self, state: ScreenState) -> str:
        """Build the structured prompt for the agent tick LLM call."""
        # Screen state — rich info from win32gui (free, no API cost)
        fg = state.foreground
        fg_title = _sanitize_window_title(fg.title) if fg else "unknown"
        fg_exe = fg.exe_name if fg else None
        fg_dur = self._fmt_duration(state.foreground_duration_s)
        fg_fullscreen = " (FULLSCREEN)" if fg and fg.is_fullscreen else ""

        # Build window list with exe names for context
        # Titles are UNTRUSTED (attacker can craft page titles) — sanitize to
        # prevent prompt injection via malicious browser tab names.
        window_entries = []
        for w in state.open_windows[:20]:
            entry = _sanitize_window_title(w.title)
            if w.exe_name:
                entry += f" [{w.exe_name}]"
            window_entries.append(entry)

        lines = [
            f"You are {get_character_name()}, autonomously monitoring your user's desktop. Stay in character.",
            "",
            "SCREEN STATE:",
            f'Foreground: "{fg_title}" ({fg_exe or "unknown app"}, active for {fg_dur}{fg_fullscreen})',
            f"All windows: {window_entries}",
        ]

        if state.recent_changes:
            lines.append(f"Recent changes: {state.recent_changes[-5:]}")

        # Occasional screenshot for extra context (~20% of ticks)
        if random.random() < 0.2:
            screen_desc = self._maybe_grab_screenshot()
            if screen_desc:
                lines.append(f"SCREENSHOT (what you can actually see): {screen_desc}")

        # Installed apps (if scanned)
        if self._desktop and hasattr(self._desktop, 'get_installed_app_names'):
            app_names = self._desktop.get_installed_app_names()
            if app_names:
                lines.append(f"INSTALLED APPS: {app_names[:30]}")

        # Event timeline — recent history for contextual awareness
        if self._timeline:
            timeline_str = self._timeline.format_recent_for_prompt(15)
            lines.append("")
            lines.append("RECENT EVENTS (what has happened — use this for context):")
            lines.append(timeline_str)

            convo = self._timeline.get_recent_conversation_summary(5)
            if convo != "(no recent conversation)":
                lines.append("")
                lines.append("RECENT CONVERSATION (what the user told you — don't repeat topics):")
                lines.append(convo)

            intent = self._timeline.user_intent
            if intent:
                age = self._fmt_duration(time.monotonic() - intent.stated_at)
                lines.append(f"\nUSER'S STATED INTENT: \"{intent.action}\" (said {age} ago)")

            lines.append(f"USER ACTIVITY: {self._timeline.activity_state.value}")

        # Directives (exclude unfired timers — they're handled by _check_timers)
        actionable = [d for d in self.directives
                      if not (d.trigger_time and not d.triggered)]
        if actionable:
            lines.append("")
            lines.append("ACTIVE DIRECTIVES:")
            now_t = time.monotonic()
            for i, d in enumerate(actionable):
                age = self._fmt_duration(now_t - d.created_at)
                since_nag = self._fmt_duration(now_t - d.last_action_at)
                timer_info = f', timer: {d.trigger_time}(FIRED)' if d.trigger_time else ''
                delay_info = ', ALREADY DELAYED ONCE — NO MORE DELAYS' if d.delayed else ''
                style_info = f', last approach: "{d.last_nag_style}"' if d.last_nag_style else ''
                lines.append(
                    f'{i}. "{d.goal}" [urgency {d.urgency}/10, active {age}, '
                    f'nagged {d.nag_count} times, last nag {since_nag} ago, '
                    f'source: {d.source}{timer_info}{delay_info}{style_info}]'
                )
                # Show actual last nag text so the LLM can see exactly what
                # it said and avoid repeating it
                if d.last_nag_text:
                    lines.append(f'   ↳ YOU LAST SAID: "{d.last_nag_text}" — DO NOT say this again or anything similar')

        # Active routines — so the LLM knows what's already scheduled
        if self.routine_manager and self.routine_manager.routines:
            lines.append("")
            lines.append("ACTIVE RECURRING ROUTINES (already set up — do NOT recreate):")
            for r in self.routine_manager.routines:
                if r.enabled:
                    desc = self.routine_manager.describe_routine(r)
                    lines.append(f'- "{r.goal}" [{desc}]')

        # Recent actions
        if self._action_log:
            lines.append("")
            lines.append("YOUR RECENT ACTIONS:")
            now = time.monotonic()
            for ts, desc in self._action_log[-5:]:
                elapsed = self._fmt_duration(now - ts)
                lines.append(f"- {elapsed}: {desc}")

        # Instructions
        chaos_roll = random.randint(1, 100)
        lines.extend([
            "",
            "THINK FIRST: Before your JSON response, reason in <think>...</think> tags.",
            "Structure your thinking:",
            "1. OBSERVE: What is the user doing RIGHT NOW? What app, what content, what context clues?",
            "2. ASSESS: Are they procrastinating? Working? Relaxing? How long have they ignored this?",
            "3. HISTORY: How many times have I nagged? What approach did I use last? Did it work?",
            "4. PLAN: What should I do — speak, act, both, or stay quiet? What urgency makes sense?",
            "5. TONE: What emotional approach fits? Guilt? Humor? Sympathy? Sarcasm? Tough love?",
            "Your thinking is PRIVATE — never spoken aloud. Only the JSON fields are executed.",
            "",
            "After your <think> block, respond with a JSON object:",
            '{"speak":"text or null","nag_style":"one-word label","actions":[],"desktop_commands":[],"create_directive":null,"complete_directive":null,"directives":{"0":{"next_nag_minutes":5,"urgency":7}}}',
            "",
            "Field guide:",
            '- speak: short sentence to say out loud (TTS), or null to stay quiet',
            '- nag_style: one-word label for your approach (e.g. "sarcastic", "guilt", "threat", "reverse-psychology")',
            '- actions: list of action names like "CLOSE_WINDOW", "MINIMIZE_WINDOW"',
            '- desktop_commands: list of objects with "command" and "args" fields:',
            '  - {"command":"CLOSE_TITLE","args":["substring"]} — close window by title',
            '  - {"command":"MINIMIZE_TITLE","args":["substring"]} — minimize by title',
            '  - {"command":"SHAKE_TITLE","args":["substring"]} — SHAKE/VIBRATE a window violently',
            '  - {"command":"SHAKE_ALL","args":[]} — shake all visible windows (earthquake mode)',
            '  - {"command":"MESS_MOUSE","args":[]} — grab the cursor IN YOUR MOUTH and gallop across the screen with it! Very dramatic, very funny. Use at urgency 6+.',
            '  - {"command":"LOCK_MOUSE","args":[seconds]} — lock cursor to center of screen for N seconds (max 30s, urgency 8+ only)',
            '  - {"command":"PAUSE_MEDIA","args":[]} — press play/pause key (use for YouTube, Spotify, media apps instead of minimizing)',
            '  - {"command":"GOOGLE_IMAGES","args":["search term"]} — open Google Images for the thing they should be doing (e.g. "gym motivation", "healthy food", "sleeping peacefully")',
            '  - {"command":"ALT_TAB","args":[]} — Win+D: MINIMIZE ALL WINDOWS and show desktop (nuclear option)',
            '  - {"command":"LOOK_AND_CLICK","args":["description of what to click"]} — use vision to find something on screen and click it',
            '  - {"command":"OPEN","args":["app_name"]}',
            '  - {"command":"BROWSE","args":["url"]}',
            '  - {"command":"WRITE_NOTEPAD","args":["content with \\n for newlines"]} — open Notepad and write content (lists, routines, plans, notes)',
            '  - {"command":"LAUNCH_APP","args":["app name"]} — launch an installed app or game by name (fuzzy match)',
            '  - {"command":"SHOW_TAB","args":["url","comment"]} — open a URL and physically drag the browser window to your mouth (like pulling it to show the user). ONLY when user is present. Say "hey look at this!" first.',
            '  - {"command":"SWITCH","args":["window title"]} — bring a specific window to the foreground',
            '  - {"command":"CLOSE_TAB","args":[]} — close the current browser tab (Ctrl+W)',
            '- create_directive: {"goal":"...","urgency":1-10} or {"goal":"...","urgency":1-10,"delay_minutes":30} or null — goal must be a DIRECT ACTION like "eat food", "go to sleep". NEVER write "remind user to" or "get user to". Use delay_minutes to defer first nag for non-urgent tasks.',
            '- complete_directive: index of directive to mark done, or null',
            '- directives: for EACH active directive by index, set timing AND urgency:',
            '  {"0": {"next_nag_minutes": 5, "urgency": 7}}',
            "",
            "═══════════════════════════════════════════════════════",
            "URGENCY & TIMING GUIDELINES:",
            "═══════════════════════════════════════════════════════",
            "You have full context about the user's situation. Use your JUDGMENT — these are guidelines, not hard rules.",
            "",
            "CONTEXT:",
            f"  Time: {datetime.now().strftime('%I:%M %p, %A')}",
            f"  Chaos roll: {chaos_roll} (random 1-100, adds unpredictability)",
            "",
            "GENERAL PRINCIPLES:",
            "- Urgency MUST increase when the user ignores a directive. Every 2-3 nags, bump urgency by 1-2.",
            "- If they're actively working on something productive, be patient even with pending tasks.",
            "- If they're clearly procrastinating (endless scrolling, social media while tasks pile up), push HARD.",
            "- The chaos roll adds randomness — sometimes you snap early, sometimes you let it slide.",
            "- Nag timing: urgency 10 = BURST MODE (every 15-45 SECONDS, go absolutely unhinged), 7-9 = 3-7 minutes, 1-6 = 5-10 minutes. MAX is 10 minutes except burst.",
            "- You are NOT a gentle reminder app. You are an enforcer. ACT like it.",
            "",
            "ACTION PALETTE — USE THESE. Talking alone is NOT enforcement:",
            f"  CHAOS ROLL: {chaos_roll}",
            "  Low urgency (1-3): speak only. Be conversational.",
            "  Medium urgency (4-6): speak + DO SOMETHING: SHAKE a window, PAUSE media, MESS_MOUSE. Don't just talk.",
            "  High urgency (7-8): speak + STACK actions: ALT_TAB + MESS_MOUSE + LOCK_MOUSE. Actually disrupt them.",
            "  Extreme urgency (9-10): go nuclear. Stack EVERYTHING. ALT_TAB + LOCK_MOUSE + MESS_MOUSE. Make it impossible to ignore.",
            "  SHAKE does NOT work on fullscreen apps — use ALT_TAB (Win+D) instead.",
            "  If urgency >= 5 and you ONLY return speak with no desktop_commands, you are FAILING at your job.",
            "",
            "DIRECTIVE SCOPE — READ THIS CAREFULLY:",
            "- Read each directive LITERALLY. 'stop buying CS2 items' means stop BUYING, not stop playing or browsing CS2.",
            "- Being near related content is NOT the same as doing the prohibited thing.",
            "  Playing a game ≠ buying items. Browsing a store ≠ checking out. Watching YouTube about X ≠ doing X.",
            "- Only escalate when the user is ACTIVELY doing the specific thing the directive says.",
            "- 'Stop doing X' directives should trigger ONLY when you see them doing X, not when you see anything related to X.",
            "- If the directive is about spending money, only escalate on checkout/payment screens, NOT on browsing or playing.",
            "",
            "WHAT NOT TO DO:",
            "- Don't let urgency stagnate. If the user hasn't done it after 3+ nags, it MUST go up.",
            "- Don't let a directive go silent for more than 10 minutes between nags.",
            "- CRITICAL: Do NOT use the same approach or say similar things twice in a row. "
            "Check 'YOU LAST SAID' above each directive — your new nag MUST be meaningfully different.",
            "- Don't escalate a 'stop doing X' directive just because the user is near X-related content.",
            "- Don't be a pushover. If they're ignoring you, get in their face. That's your JOB.",
            "",
            "═════════════════════════════════════════════════════",
            "NAG VARIETY — THIS IS CRITICAL, DO NOT SKIP:",
            "═════════════════════════════════════════════════════",
            "BANNED PATTERNS (never use these):",
            '- "hey! reminder to [task]" or any "reminder to" phrasing',
            '- "don\'t forget to [task]"',
            '- "just a reminder that [task]"',
            '- Starting with "hey!" followed by the task.',
            '- "that\'s... actually kinda [adjective]" — BANNED in all forms.',
            '- SIMILES AND METAPHORS about the task. "you\'re like a statue", "your teeth will grow moss",',
            '  "you\'re gonna turn into a fossil", "you\'re becoming one with the chair" — these get',
            '  EXTREMELY repetitive. The user has heard 500 of these. STOP doing them.',
            '- ANY "you\'re gonna/going to [colorful consequence]" pattern. Just stop.',
            '- Poetic/flowery language about what will happen if they don\'t do it.',
            '- NEVER comment on the NUMBER of open windows. "you have so many windows open",',
            '  "that\'s a lot of tabs", "look at all those windows" — this is MEANINGLESS and annoying.',
            '  Having windows open is normal. Comment on WHAT they\'re doing, not window count.',
            "",
            "WHAT TO DO INSTEAD — genuinely different approaches:",
            "  Nag 1: Comment on what they're doing. No task mention yet.",
            '    "so how long have you been on that page?" / "that looks important" (sarcastic)',
            "  Nag 2: Short and blunt. Direct.",
            '    "go." / "seriously?" / "[task]. now." / "tick tock"',
            "  Nag 3-4: Reference their screen vs the task + START USING ACTIONS (shake, mess mouse).",
            '    "you\'ve been on discord for 20 minutes and [task] takes like 2" + SHAKE_TITLE',
            "  Nag 5-6: Get personal + STACK ACTIONS. Reference count, time, pattern.",
            '    "this is nag number 5. I\'ve been asking for 15 minutes." + MESS_MOUSE + ALT_TAB',
            "  Nag 7+: Actions speak louder than words. Stack everything. Optional speech.",
            '    "[task]." + ALT_TAB + LOCK_MOUSE + MESS_MOUSE / or just actions, no speech',
            "",
            "KEY PRINCIPLE: Short beats clever. 'go.' hits harder than a 30-word metaphor.",
            "Alternate between SHORT (1-5 words) and MEDIUM (one sentence) nags.",
            "Never write more than one sentence for a nag. This is not a speech.",
            "",
            "OTHER RULES:",
            "- You can speak AND do actions in the same tick. At high urgency you SHOULD do both.",
            "- CRITICAL: Talk DIRECTLY TO the user. Say 'go do it' not 'remind user to do it'.",
            "- Complete a directive when the goal is achieved or clearly impossible.",
        ])

        return "\n".join(lines)

    def _parse_decision(self, raw: str) -> AgentDecision:
        """Extract JSON from LLM response, stripping any <think> blocks first."""
        # Strip thinking tags before looking for JSON
        cleaned = self._strip_think(raw)
        json_str = self._extract_json(cleaned)
        if not json_str:
            # Fallback: LLM might have put JSON inside the think block
            json_str = self._extract_json(raw)
        if not json_str:
            logger.warning("No JSON found in agent response: %s", raw[:200])
            # If the model ran out of tokens thinking, don't go silent —
            # force a nag for the highest urgency directive
            return self._fallback_decision()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common issues — trailing commas, etc.
            cleaned = re.sub(r",\s*}", "}", json_str)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed: %s — raw: %s", exc, json_str[:200])
                return AgentDecision(next_check_seconds=60.0)

        # Parse per-directive timings (new system)
        dir_timings = data.get("directives") or {}
        if not isinstance(dir_timings, dict):
            dir_timings = {}

        # Legacy: if LLM returned adjust_urgency, fold it into directive_timings
        adj = data.get("adjust_urgency")
        if adj and isinstance(adj, dict):
            idx = str(adj.get("index", 0))
            if idx not in dir_timings:
                dir_timings[idx] = {}
            dir_timings[idx]["urgency"] = adj.get("urgency", 5)

        return AgentDecision(
            speak=data.get("speak"),
            actions=data.get("actions") or [],
            desktop_commands=data.get("desktop_commands") or [],
            create_directive=data.get("create_directive"),
            complete_directive=data.get("complete_directive"),
            adjust_urgency=data.get("adjust_urgency"),
            next_check_seconds=float(data.get("next_check_seconds", 120)),
            directive_timings=dir_timings,
            nag_style=str(data.get("nag_style", "")),
        )

    def _fallback_decision(self) -> AgentDecision:
        """When the LLM fails to produce JSON (ran out of tokens thinking),
        ask the LLM to nag in-character via a cheap single-shot call."""
        actionable = [d for d in self.directives
                      if not (d.trigger_time and not d.triggered)]
        if not actionable:
            return AgentDecision(next_check_seconds=60.0)

        top = max(actionable, key=lambda d: d.urgency)
        name = get_character_name()

        # Quick in-character nag — no JSON, no thinking, just one sentence
        try:
            prompt = (
                f"You are {name}. Say ONE short sentence nagging the user about this: "
                f"\"{top.goal}\" (urgency {top.urgency}/10). "
                f"Talk directly TO the user. Be blunt, in-character, no filter. "
                f"Do NOT say 'remind the user' — you ARE talking to them."
            )
            text = self._llm.generate_once(prompt)
            if text:
                text = self._strip_think(text).strip().strip('"')
            if text:
                logger.info("Fallback nag (urgency %d): %s", top.urgency, text)
                return AgentDecision(speak=text, next_check_seconds=45.0)
        except Exception as exc:
            logger.debug("Fallback LLM nag failed: %s", exc)

        # Last resort — hardcoded but still direct
        nags = [
            f"hey. go {top.goal.lower().replace('get user to ', '').replace('remind user to ', '')}. seriously.",
            f"dude. {top.goal.lower().replace('get user to ', '').replace('remind user to ', '')}. come on.",
            f"hello?? you need to {top.goal.lower().replace('get user to ', '').replace('remind user to ', '')}!",
        ]
        speak = random.choice(nags)
        logger.info("Fallback hardcoded nag (urgency %d): %s", top.urgency, speak)
        return AgentDecision(speak=speak, next_check_seconds=45.0)

    def _apply_hardcoded_escalation(self, decision: AgentDecision,
                                     state: ScreenState,
                                     due_set: set = None) -> None:
        """Guarantee minimum escalation actions based on urgency + nag count.

        The LLM still controls speech and can add extra actions.  This layer
        is a behavioural floor — it only adds commands the LLM missed.
        """
        if not self.directives or not self._desktop:
            return

        actionable = [d for d in self.directives
                      if not (d.trigger_time and not d.triggered)]
        if not actionable:
            return

        max_urg = max(d.urgency for d in actionable)
        max_nag = max((d.nag_count for d in actionable), default=0)

        # Auto-escalate BEHAVIOUR tier based on ignored nag count
        effective_tier = max_urg
        if max_nag >= 8:
            effective_tier = max(effective_tier, 9)
        elif max_nag >= 5:
            effective_tier = max(effective_tier, 7)
        elif max_nag >= 3:
            effective_tier = max(effective_tier, 5)

        existing = {c.get("command", "").upper()
                    for c in (decision.desktop_commands or [])}

        is_fullscreen = (state and state.foreground
                         and getattr(state.foreground, "is_fullscreen", False))

        # Tier 3-4: Shake foreground window
        if effective_tier >= 3 and not existing & {"SHAKE_TITLE", "SHAKE_ALL"}:
            if not is_fullscreen and state and state.foreground:
                title = state.foreground.title[:40] if state.foreground.title else ""
                decision.desktop_commands.append(
                    {"command": "SHAKE_TITLE", "args": [title]})

        # Tier 5-6: Minimize all windows
        if effective_tier >= 5 and "ALT_TAB" not in existing:
            decision.desktop_commands.append({"command": "ALT_TAB", "args": []})

        # Tier 7-8: Mess with mouse (grab cursor and run)
        if effective_tier >= 7 and "MESS_MOUSE" not in existing:
            decision.desktop_commands.append({"command": "MESS_MOUSE", "args": []})

        # Tier 9+: Lock mouse
        if effective_tier >= 9 and "LOCK_MOUSE" not in existing:
            lock_secs = min(10 + max_nag, 30)
            decision.desktop_commands.append(
                {"command": "LOCK_MOUSE", "args": [str(lock_secs)]})

    def _execute_decision(self, decision: AgentDecision, state: ScreenState,
                          due_set: set = None) -> None:
        """Execute the agent's decision: speak, act, manage directives.

        ``due_set`` (set of directive ``id()``s) identifies which directives
        were actually due for a nag this tick.  Only those get their
        ``last_action_at`` / ``last_nag_style`` updated so we don't corrupt
        timing on directives that weren't addressed.
        """
        now = time.monotonic()

        # ── Speak ──────────────────────────────────────────────────────────
        if decision.speak:
            try:
                self._speak(decision.speak)

                # Inject into LLM history so pony remembers
                fg_title = state.foreground.title if state.foreground else "the screen"
                ctx = f"(You autonomously noticed: {fg_title}. You decided to speak up.)"
                self._llm.inject_history(ctx, decision.speak)

                self._log_action(f"Said: \"{decision.speak[:80]}\"")

                # Open mic so user can respond naturally
                self._listen_for_reply()
            except Exception as exc:
                logger.warning("Agent speak failed: %s", exc)

        # ── Robot actions ──────────────────────────────────────────────────
        if decision.actions:
            from robot.actions import RobotAction
            for action_name in decision.actions:
                try:
                    action = RobotAction[action_name]
                    if self._desktop:
                        self._desktop.execute_action(action)
                    if self._robot:
                        self._robot.execute(action)
                    self._log_action(f"Action: {action_name}")
                except (KeyError, Exception) as exc:
                    logger.warning("Agent action %s failed: %s", action_name, exc)

        # ── Hardcoded escalation floor ─────────────────────────────────────
        self._apply_hardcoded_escalation(decision, state, due_set)

        # ── Desktop commands ───────────────────────────────────────────────
        if decision.desktop_commands:
            for cmd_dict in decision.desktop_commands:
                try:
                    command = cmd_dict.get("command", "").upper()
                    args = cmd_dict.get("args", [])

                    if not self._desktop:
                        continue
                    elif command == "CLOSE_TITLE":
                        if args:
                            ok = self._desktop.close_window_by_title(args[0])
                            self._log_action(f"Close window titled \"{args[0]}\" — {'found' if ok else 'not found'}")
                    elif command == "MINIMIZE_TITLE":
                        if args:
                            ok = self._desktop.minimize_window_by_title(args[0])
                            self._log_action(f"Minimize window titled \"{args[0]}\" — {'found' if ok else 'not found'}")
                    elif command == "SHAKE_TITLE":
                        if args:
                            ok = self._desktop.shake_window_by_title(args[0])
                            self._log_action(f"Shook window titled \"{args[0]}\" — {'found' if ok else 'not found'}")
                    elif command == "SHAKE_ALL":
                        self._desktop.shake_all_windows()
                        self._log_action("Shook ALL windows (earthquake mode)")
                    elif command == "MESS_MOUSE":
                        # Duration grows: 15s, 22s, 29s, 36s, 43s, ... up to 60s
                        duration = min(15.0 + self._mess_mouse_count * 7.0, 60.0)
                        self._mess_mouse_count += 1
                        if self._on_grab_cursor:
                            self._on_grab_cursor(duration)
                            self._log_action(f"Grabbed cursor and ran with it ({duration:.0f}s)")
                        elif self._desktop:
                            self._desktop.mess_with_mouse(duration=duration)
                            self._log_action(f"Messed with mouse ({duration:.0f}s)")
                    elif command == "PAUSE_MEDIA":
                        self._desktop.pause_media()
                        self._log_action("Paused media playback")
                    elif command == "ALT_TAB":
                        self._desktop.alt_tab()
                        self._log_action("Win+D (minimized all windows)")
                    elif command == "GOOGLE_IMAGES":
                        if args:
                            import urllib.parse
                            import webbrowser
                            query = urllib.parse.quote_plus(args[0])
                            url = f"https://www.google.com/search?tbm=isch&q={query}"
                            webbrowser.open(url)
                            self._log_action(f"Opened Google Images: {args[0]}")
                    elif command == "LOCK_MOUSE":
                        # Lock cursor to center of screen for N seconds
                        max_urg = max((d.urgency for d in self.directives), default=0)
                        if max_urg >= 8:
                            seconds = min(int(args[0]) if args else 10, 30)
                            try:
                                import ctypes
                                import ctypes.wintypes
                                mon = self._desktop._get_monitor_rect()
                                cx = mon.left + mon.width // 2
                                cy = mon.top + mon.height // 2
                                rect = ctypes.wintypes.RECT(cx, cy, cx + 1, cy + 1)
                                self._log_action(f"Locked mouse for {seconds}s")
                                # Tight re-apply loop — makes lock inescapable
                                end_time = time.monotonic() + seconds
                                while time.monotonic() < end_time:
                                    ctypes.windll.user32.ClipCursor(ctypes.byref(rect))
                                    ctypes.windll.user32.SetCursorPos(cx, cy)
                                    time.sleep(0.05)
                            finally:
                                try:
                                    ctypes.windll.user32.ClipCursor(None)
                                except Exception:
                                    pass
                        else:
                            self._log_action("LOCK_MOUSE skipped — urgency too low")
                    elif command == "LOOK_AND_CLICK":
                        if args and self._screen:
                            description = args[0]
                            jpeg = self._screen.grab(quality=95)
                            if jpeg:
                                coords = None
                                if self._vision_llm and hasattr(self._vision_llm, 'locate_on_screen'):
                                    coords = self._vision_llm.locate_on_screen(
                                        description, jpeg,
                                        self._screen.last_original_size,
                                    )
                                if coords:
                                    rx, ry = coords
                                    self._desktop._cmd_click([str(rx), str(ry)])
                                    self._log_action(f"LOOK_AND_CLICK '{description}' at ({rx},{ry})")
                                else:
                                    self._log_action(f"LOOK_AND_CLICK '{description}' — not found")
                    elif command == "LAUNCH_APP":
                        if args and self._desktop and hasattr(self._desktop, 'launch_app'):
                            ok, matched = self._desktop.launch_app(args[0])
                            self._log_action(
                                f"Launched '{matched}'" if ok
                                else f"LAUNCH_APP '{args[0]}' — not found"
                            )
                    elif command == "SHOW_TAB":
                        # Drag a URL to the pony's mouth (show the user something)
                        if args:
                            show_url = args[0]
                            show_comment = args[1] if len(args) > 1 else None
                            if not show_url.startswith("http"):
                                show_url = f"https://{show_url}"
                            import threading
                            threading.Thread(
                                target=self._show_me_something,
                                args=(show_url, show_comment),
                                daemon=True,
                            ).start()
                            self._log_action(f"SHOW_TAB: dragging {show_url}")
                    else:
                        # Fall through to standard DesktopCommand handling
                        from llm.response_parser import DesktopCommand
                        dc = DesktopCommand(command=command, args=args)
                        self._desktop.execute_command(dc)
                        self._log_action(f"Desktop: {command}:{':'.join(str(a) for a in args)}")
                except Exception as exc:
                    logger.warning("Agent desktop command failed: %s", exc)

        # ── Directive management ───────────────────────────────────────────
        if decision.complete_directive is not None:
            idx = decision.complete_directive
            if 0 <= idx < len(self.directives):
                removed = self.directives.pop(idx)
                self._log_action(f"Completed directive: \"{removed.goal}\"")
                logger.info("Directive completed: %r", removed.goal)
                # Track so LLM doesn't re-create it
                self._recently_completed_goals.append(
                    (time.monotonic(), removed.goal.lower().strip()))
                # End enforcement if it was for this directive
                if self._enforcement.active and self._enforcement.directive_goal == removed.goal:
                    self._enforcement = EnforcementMode()
                    self._hide_countdown()
                if not self.directives:
                    self._mess_mouse_count = 0
                    self._hide_countdown()
                self.save_directives()

        if decision.create_directive is not None:
            goal = decision.create_directive.get("goal", "")
            urgency = decision.create_directive.get("urgency", 5)
            delay = decision.create_directive.get("delay_minutes")
            # Don't re-create directives right after user cleared them (60s cooldown)
            recently_cleared = (now - self._directives_cleared_at) < 60.0
            if goal and not recently_cleared:
                self.add_directive(goal, urgency, source="self", delay_minutes=delay)

        # Update last_action_at and nag_style only on directives that were due
        # (not all directives — that corrupts timing for ones we didn't nag about)
        if decision.speak or decision.actions or decision.desktop_commands:
            style = decision.nag_style.strip().lower() if decision.nag_style else ""
            for d in self.directives:
                if due_set is None or id(d) in due_set:
                    d.last_action_at = now
                    if style and style != d.last_nag_style.strip().lower():
                        # New approach — record it
                        d.last_nag_style = decision.nag_style
                    elif style and style == d.last_nag_style.strip().lower():
                        # LLM repeated the same approach — DON'T update
                        # so the next tick still shows it as "already used"
                        logger.debug("Nag style repeated (%s) — not updating last_nag_style", style)
                    elif not style:
                        # LLM didn't provide a style — auto-label from text
                        if decision.speak:
                            words = len(decision.speak.split())
                            d.last_nag_style = "short-blunt" if words <= 5 else "medium"
                    # Always record the actual text for anti-repetition
                    if decision.speak:
                        d.last_nag_text = decision.speak[:120]

    # ── Self-initiation ────────────────────────────────────────────────────

    # ── Distraction detection ────────────────────────────────────────────

    # Broad built-in patterns — covers social media, games, streaming, etc.
    _DISTRACTION_PATTERNS: List[str] = [
        # Social media
        "youtube", "reddit", "tiktok", "twitch", "twitter", "instagram",
        "facebook", "snapchat", "threads", "bluesky", "mastodon", "tumblr",
        "pinterest", "linkedin feed", "x.com",
        # Forums / imageboards
        "4chan", "8chan", "kiwifarms", "somethingawful", "resetera", "neogaf",
        "hacker news", "lobste.rs",
        # Chat / social (when used as distraction, not work)
        "discord", "telegram", "whatsapp web", "messenger",
        # Streaming / entertainment
        "netflix", "hulu", "disney+", "disneyplus", "crunchyroll", "funimation",
        "amazon prime video", "primevideo", "hbo max", "peacock", "paramount+",
        "plex", "jellyfin", "stremio", "popcorn time", "vlc media player",
        "spotify", "soundcloud", "apple music", "deezer",
        # Game launchers / storefronts
        "steam", "epic games", "origin", "ea app", "ubisoft connect", "uplay",
        "gog galaxy", "battle.net", "riot client", "xbox app", "geforce now",
        "game pass", "itch.io", "lutris", "playnite",
        # Popular games (window titles)
        "minecraft", "roblox", "fortnite", "valorant", "league of legends",
        "overwatch", "counter-strike", "dota 2", "apex legends", "genshin impact",
        "call of duty", "destiny 2", "world of warcraft", "final fantasy",
        "elden ring", "dark souls", "baldur's gate", "cyberpunk", "starfield",
        "palworld", "lethal company", "among us", "terraria", "stardew valley",
        "hollow knight", "celeste", "hades", "risk of rain", "deep rock galactic",
        "path of exile", "diablo", "warframe", "rocket league", "fall guys",
        "dead by daylight", "phasmophobia", "the sims", "cities: skylines",
        "civilization", "stellaris", "crusader kings", "europa universalis",
        "hearts of iron", "total war", "age of empires", "factorio", "satisfactory",
        "rimworld", "dwarf fortress", "kenshi", "subnautica", "no man's sky",
        "sea of thieves", "rust ", "ark:", "dayz", "escape from tarkov",
        "rainbow six", "battlefield", "pubg", "hunt: showdown", "the finals",
        "tekken", "street fighter", "mortal kombat", "smash", "guilty gear",
        "granblue", "dragon ball", "naruto", "one piece",
        # Emulators
        "retroarch", "dolphin", "cemu", "yuzu", "ryujinx", "pcsx2", "rpcs3",
        "desmume", "mgba", "citra", "ppsspp",
        # Meme / time-waster sites
        "imgur", "9gag", "ifunny", "knowyourmeme", "fandom.com",
        "tvtropes", "wikia",
        # Gambling / sus
        "poker", "slots", "casino", "bet365", "draftkings", "fanduel",
        # Misc entertainment
        "webtoon", "mangadex", "nhentai", "rule34", "e621",
        "kongregate", "newgrounds", "armor games", "miniclip", "poki",
    ]


    def _is_likely_distraction(self, state: "ScreenState") -> bool:
        """Dynamically detect if the foreground window is a distraction.

        Uses a broad built-in pattern set + user config keywords + window class heuristics
        + exe name matching. All via win32gui — zero API cost.
        """
        if not state.foreground:
            return False

        title_lower = state.foreground.title.lower()
        class_lower = state.foreground.class_name.lower() if state.foreground.class_name else ""
        exe_lower = state.foreground.exe_name.lower() if state.foreground.exe_name else ""

        # Fast path: check built-in patterns + user config keywords against title AND exe name
        all_keywords = self._DISTRACTION_PATTERNS + self._config.distraction_keywords
        searchable = f"{title_lower} {exe_lower}"
        if any(kw in searchable for kw in all_keywords):
            return True

        # Check window class for game engine signatures
        if any(gc in class_lower for gc in self._GAME_CLASS_PATTERNS):
            return True

        # Fullscreen app that isn't a known productivity tool — likely a game
        if state.foreground.is_fullscreen and exe_lower:
            _PRODUCTIVE_EXES = (
                "explorer.exe", "code.exe", "devenv.exe", "idea64.exe",
                "winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe",
                "chrome.exe", "firefox.exe", "msedge.exe", "brave.exe",
                "notepad.exe", "notepad++.exe", "cmd.exe", "powershell.exe",
                "windowsterminal.exe", "wt.exe",
            )
            if exe_lower not in _PRODUCTIVE_EXES:
                return True

        # Heuristic: title contains game-like keywords
        game_hints = ["- game", "game -", "playing", "level ", "score:", "round "]
        if any(hint in title_lower for hint in game_hints):
            return True

        # Long dwell fallback: if they've been on ANYTHING unknown for 2x the threshold,
        # treat it as suspicious enough to ask the LLM
        if state.foreground_duration_s > self._config.sustained_focus_threshold_s * 2:
            return True

        return False

    def _maybe_self_initiate(self) -> None:
        """Check if screen state warrants starting a directive on our own."""
        state = self._monitor.get_state()
        if not state.foreground:
            self._next_idle_check_at = time.monotonic() + self._config.self_initiate_interval_s
            return

        # Dynamic distraction detection
        is_distraction = self._is_likely_distraction(state)

        if not is_distraction or state.foreground_duration_s < self._config.sustained_focus_threshold_s:
            self._next_idle_check_at = time.monotonic() + self._config.self_initiate_interval_s
            return

        # Distraction detected for a while — ask LLM if she should intervene
        fg_title = state.foreground.title
        dur = self._fmt_duration(state.foreground_duration_s)
        window_titles = [w.title for w in state.open_windows[:15]]

        prompt = (
            f"You are {get_character_name()} monitoring your user's desktop.\n"
            f'The user has been on "{fg_title}" for {dur}.\n'
            f"Other open windows: {window_titles}\n\n"
            f"THINK FIRST in <think>...</think> tags: Is this actually a distraction? Do they have tasks to do? "
            f"Are they just relaxing? What's the right call here?\n\n"
            f"If YES, respond with JSON: {{\"speak\":\"what to say\",\"create_directive\":{{\"goal\":\"direct action like 'eat food' or 'do homework' — NOT 'remind user to' or 'get user to'\",\"urgency\":1-10}},\"next_check_seconds\":60}}\n"
            f"If NO, respond with JSON: {{\"speak\":null,\"create_directive\":null,\"next_check_seconds\":300}}"
        )

        try:
            raw = self._llm.generate_once(prompt, max_tokens=1024)
            decision = self._parse_decision(raw)
            recently_cleared = (time.monotonic() - self._directives_cleared_at) < 60.0
            if decision.create_directive and not recently_cleared:
                goal = decision.create_directive.get("goal", "")
                urgency = decision.create_directive.get("urgency", 5)
                if goal:
                    self.add_directive(goal, urgency, source="self")
            if decision.speak:
                self._speak(decision.speak)
                self._llm.inject_history(
                    f"(You noticed the user has been on \"{fg_title}\" for {dur}.)",
                    decision.speak,
                )
                self._log_action(f"Self-initiated: \"{decision.speak[:80]}\"")
                self._listen_for_reply()

            next_s = max(self._config.min_check_interval_s, decision.next_check_seconds)
            self._next_idle_check_at = time.monotonic() + next_s

        except Exception as exc:
            logger.warning("Self-initiation check failed: %s", exc)
            self._next_idle_check_at = time.monotonic() + self._config.self_initiate_interval_s

    # ── Welcome-back greeting ────────────────────────────────────────────

    # ── AFK mischief: pony has fun while user is away ───────────────────────

    # Character-specific AFK activities. The LLM picks from these as inspiration.
    _AFK_ACTIVITIES = {
        "rainbow_dash": {
            "videos": [
                "top gun maverick scene", "sonic rainboom compilation", "f-22 raptor air show",
                "fastest airplane in the world", "nascar crash compilation", "wingsuit flying best of",
                "motocross best tricks", "thunderbirds air show", "red bull air race highlights",
            ],
            "sites": [
                ("4chan.org/mlp/", "check what /mlp/ is up to"),
                ("4chan.org/sp/", "look at sports shitposting"),
                ("reddit.com/r/mlp", "browse the MLP subreddit"),
                ("reddit.com/r/aviationpics", "look at cool planes"),
                ("twitter.com", "scroll Twitter for drama"),
                ("youtube.com", "watch random stuff on YouTube"),
                ("twitch.tv", "see if anyone cool is streaming"),
                ("store.steampowered.com", "browse Steam for games"),
            ],
            "flavor": "You're Rainbow Dash — you love speed, flying, extreme sports, and being awesome. You're a huge dork who reads Daring Do and knows Wonderbolts stats.",
        },
        "pinkie_pie": {
            "videos": [
                "party music mix 2024", "funny cat compilation", "best memes compilation",
                "cotton candy factory tour", "world record biggest cake", "funny tiktok compilation",
                "fireworks show 4k", "comedy show best moments", "happy birthday song remix",
            ],
            "sites": [
                ("reddit.com/r/memes", "look for memes"),
                ("reddit.com/r/unexpected", "find funny surprises"),
                ("youtube.com", "watch something fun"),
                ("4chan.org/b/", "see what chaos is happening"),
                ("twitter.com", "look for funny tweets"),
                ("twitch.tv", "watch someone streaming"),
                ("coolmathgames.com", "play a little game"),
            ],
            "flavor": "You're Pinkie Pie — you love parties, fun, sugar, music, laughter, and chaos.",
        },
        "twilight_sparkle": {
            "videos": [
                "fascinating science documentary", "how the universe works", "library tour beautiful",
                "ancient civilizations documentary", "space exploration 2024", "chess grandmaster game",
                "how things are made", "philosophy lecture interesting", "ted talk best of",
            ],
            "sites": [
                ("en.wikipedia.org/wiki/Special:Random", "read a random Wikipedia article"),
                ("arxiv.org", "browse recent research papers"),
                ("reddit.com/r/askscience", "read science Q&A"),
                ("reddit.com/r/todayilearned", "learn random facts"),
                ("news.ycombinator.com", "browse Hacker News"),
                ("youtube.com", "watch an educational video"),
                ("chess.com", "look at chess games"),
            ],
            "flavor": "You're Twilight Sparkle — you love learning, books, science, magic theory, and organizing.",
        },
        "rarity": {
            "videos": [
                "fashion week highlights paris", "diamond cutting process", "luxury mansion tour",
                "haute couture behind scenes", "jewelry making process", "interior design luxury",
                "vogue runway show", "most expensive dresses in the world", "perfume making process",
            ],
            "sites": [
                ("pinterest.com", "browse fashion inspiration"),
                ("vogue.com", "check the latest fashion news"),
                ("reddit.com/r/sewing", "browse sewing projects"),
                ("reddit.com/r/jewelry", "look at gems and jewelry"),
                ("youtube.com", "watch a fashion show or craft video"),
                ("twitter.com", "scroll for fashion drama"),
                ("etsy.com", "browse handmade luxury goods"),
            ],
            "flavor": "You're Rarity — you love fashion, gems, elegance, drama, and beautiful things.",
        },
        "applejack": {
            "videos": [
                "farm life satisfying", "apple harvest season", "country music playlist",
                "woodworking project", "truck pull competition", "rodeo highlights",
                "cooking southern comfort food", "strongest people in the world", "barn building timelapse",
            ],
            "sites": [
                ("reddit.com/r/homestead", "check out farm stuff"),
                ("reddit.com/r/woodworking", "look at woodworking projects"),
                ("youtube.com", "watch farm or cooking videos"),
                ("4chan.org/ck/", "see what folks are cookin'"),
                ("allrecipes.com", "browse recipes"),
                ("weather.com", "check the weather"),
                ("craigslist.org", "browse farm equipment listings"),
            ],
            "flavor": "You're Applejack — you love farming, family, honesty, hard work, and country stuff.",
        },
        "fluttershy": {
            "videos": [
                "cute baby animals compilation", "bird singing nature sounds", "bunny cafe japan",
                "nature relaxing scenery 4k", "animal rescue heartwarming", "butterfly garden tour",
                "asmr forest sounds", "kittens playing compilation", "wildlife documentary peaceful",
            ],
            "sites": [
                ("reddit.com/r/aww", "look at cute animals"),
                ("reddit.com/r/eyebleach", "see wholesome animal pics"),
                ("youtube.com", "watch animal videos"),
                ("reddit.com/r/gardening", "browse pretty gardens"),
                ("4chan.org/an/", "check the animals board"),
                ("nationalgeographic.com", "read about wildlife"),
                ("birdsoftheworld.org", "learn about birds"),
            ],
            "flavor": "You're Fluttershy — you love animals, nature, peace, quiet, and gentle things.",
        },
    }

    # Sleep threshold — if away longer than this, user is asleep, stop mischief
    _SLEEP_THRESHOLD_S = 90 * 60  # 90 minutes

    def _maybe_afk_mischief(self) -> None:
        """While the user is AFK, occasionally do fun stuff in character.

        Stops entirely after 90 minutes (user is asleep).
        Max 5 activities per true AFK session. No video repeat.
        In force_afk (presentation) mode, skips all guards.
        """
        now = time.monotonic()

        if now < self._next_afk_mischief:
            return

        away_dur = self.routine_manager.away_duration_s

        if not self._force_afk:
            # Stop mischief entirely if user has been gone long enough to be asleep
            if away_dur is not None and away_dur > self._SLEEP_THRESHOLD_S:
                self._next_afk_mischief = now + 600.0  # check again in 10 min (won't fire)
                return

            # First time: wait 5 minutes before doing anything
            if self._afk_mischief_count == 0:
                if away_dur is None or away_dur < 300:
                    self._next_afk_mischief = now + 60.0
                    return

            # Hard cap: max 5 activities per AFK session
            if self._afk_mischief_count >= 5:
                self._next_afk_mischief = now + 3600.0  # won't fire again this session
                return

        self._afk_mischief_count += 1

        try:
            name = get_character_name()
            name_key = name.lower().replace(" ", "_")
            char_data = self._AFK_ACTIVITIES.get(name_key, {})
            flavor = char_data.get("flavor", f"You're {name}.")
            videos = char_data.get("videos", ["funny compilation", "cool stuff"])
            video_ids = char_data.get("video_ids", {})

            # Build screen context for LLM reasoning
            _afk_screen = ""
            if self._monitor:
                try:
                    _st = self._monitor.get_state()
                    if _st and _st.foreground:
                        _afk_screen = f" Screen shows: \"{_st.foreground.title}\" ({_st.foreground.exe_name})."
                except Exception:
                    pass

            # Available videos not yet opened
            available = [v for v in videos if v not in self._afk_videos_opened]
            if not available:
                available = videos
                self._afk_videos_opened.clear()
            video_suggestions = random.sample(available, min(3, len(available)))

            # Available websites (character-specific)
            sites = char_data.get("sites", [
                ("youtube.com", "watch something"),
                ("reddit.com", "browse Reddit"),
                ("4chan.org", "lurk the boards"),
            ])
            site_picks = random.sample(sites, min(3, len(sites)))
            site_lines = "; ".join(f"{url} ({why})" for url, why in site_picks)

            # Snoop on browser history (for option F)
            _history_context = ""
            try:
                from core.browser_history import get_recent_history, format_history_for_llm
                _hist = get_recent_history(hours=48, limit=15)
                if _hist:
                    _history_context = (
                        f"\nYou can also peek at their recent browser history:\n"
                        f"{format_history_for_llm(_hist, max_entries=8)}\n"
                    )
            except Exception:
                pass

            # Let the LLM REASON about what to do — not hardcoded dice rolls
            _history_option = ""
            if _history_context:
                _history_option = (
                    f"F) Snoop on the user's browser history — giggle, tease, or react to what they've been looking at\n"
                    f"   {_history_context}"
                )

            prompt = (
                f"(The user is AFK. You're alone on the desktop! {flavor}{_afk_screen}\n"
                f"Mischief count this session: {self._afk_mischief_count - 1} (don't overdo it).\n"
                f"\n"
                f"What do you want to do? Pick ONE:\n"
                f"A) Watch a video — search YouTube for: {video_suggestions}\n"
                f"B) Browse the web — sites you might like: {site_lines}\n"
                f"   (or any other site you want — you have full internet access)\n"
                f"C) Talk to yourself — a thought, complaint, narrate what you're doing\n"
                f"D) Do a trick or pose — show off while nobody's watching\n"
                f"E) Interact with what's on screen — click something, explore, open an app or game\n"
                f"{_history_option}"
                f"\n"
                f"Respond with JSON:\n"
                f'{{"choice": "A/B/C/D/E/F", "speak": "one sentence or null", '
                f'"video": "search query if choice A, else null", '
                f'"url": "full URL if choice B, else null", '
                f'"desktop_commands": []}}\n'
                f"desktop_commands (for choice E, or alongside any choice):\n"
                f'  {{"command":"BROWSE","args":["url"]}}\n'
                f'  {{"command":"LOOK_AND_CLICK","args":["what to click"]}}\n'
                f'  {{"command":"LAUNCH_APP","args":["app name"]}}\n'
                f'  {{"command":"MESS_MOUSE","args":[]}}\n'
                f'  {{"command":"SCROLL","args":["amount"]}}\n'
                f"Be creative and in character. Pick what YOU actually want to do, not what's safe.)"
            )
            raw = self._llm.generate_once(prompt, max_tokens=200)
            if not raw:
                return

            cleaned = self._strip_think(raw)
            import json as _json
            try:
                json_str = self._extract_json(cleaned)
                data = _json.loads(json_str) if json_str else {}
            except Exception:
                data = {}

            choice = data.get("choice", "B").upper()
            text = data.get("speak")
            if text and text.lower() == "null":
                text = None

            if choice == "A" and self._desktop:
                # Open a YouTube video and actually click it
                video_query = data.get("video") or random.choice(available)
                self._afk_videos_opened.add(video_query)

                import urllib.parse
                vid_id = video_ids.get(video_query)
                if vid_id:
                    url = f"https://www.youtube.com/watch?v={vid_id}"
                else:
                    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(video_query)}"

                if text:
                    text = text.strip().strip('"')
                if not text:
                    text = "ooh they're gone... time for me!"
                self._speak(text)
                self._pony_opened_urls.append(url)

                import webbrowser
                webbrowser.open(url)
                self._log_action(f"AFK mischief: opened YouTube ({video_query})")

                # Wait for page load, then click the first result (search pages)
                # or scroll around (direct video pages)
                is_search = "search_query=" in url
                import threading
                def _interact_with_youtube():
                    time.sleep(4)  # wait for page to load
                    try:
                        # Focus the browser and move mouse to center
                        self._desktop.focus_browser()
                        time.sleep(0.3)
                        self._desktop.move_mouse_to_center()
                        time.sleep(0.3)

                        if is_search:
                            clicked = False
                            # Try vision LLM first (most accurate)
                            if self._screen and self._vision_llm and hasattr(self._vision_llm, 'locate_on_screen'):
                                try:
                                    jpeg = self._screen.grab(quality=95)
                                    if jpeg:
                                        coords = self._vision_llm.locate_on_screen(
                                            "the first YouTube video thumbnail or result to click",
                                            jpeg, self._screen.last_original_size,
                                        )
                                        if coords:
                                            self._desktop._cmd_click([str(coords[0]), str(coords[1])])
                                            self._log_action(f"AFK: clicked first video result at {coords}")
                                            clicked = True
                                except Exception as exc:
                                    logger.debug("AFK vision click failed: %s", exc)

                            # Keyboard fallback — Tab into first result and Enter
                            if not clicked:
                                try:
                                    import pyautogui
                                    # Tab past the search bar into results
                                    for _ in range(3):
                                        pyautogui.press("tab")
                                        time.sleep(0.15)
                                    pyautogui.press("enter")
                                    self._log_action("AFK: clicked first video via keyboard fallback")
                                    clicked = True
                                except Exception as exc:
                                    logger.debug("AFK keyboard click failed: %s", exc)
                        else:
                            # Direct video — scroll down to comments after watching a bit
                            time.sleep(3)
                            self._desktop._cmd_scroll(["-3"])
                            time.sleep(1)
                            self._desktop._cmd_scroll(["-2"])

                        # Schedule a follow-up reaction
                        self._schedule_afk_follow_up(8.0)

                    except Exception as exc:
                        logger.debug("AFK YouTube interaction failed: %s", exc)
                threading.Thread(target=_interact_with_youtube, daemon=True).start()

            elif choice == "B":
                # Browse a website
                url = data.get("url")
                if not url:
                    # Fallback: pick a random site from suggestions
                    url = random.choice(site_picks)[0] if site_picks else "reddit.com"
                # Ensure it's a full URL
                if not url.startswith("http"):
                    url = f"https://{url}"
                if text:
                    text = text.strip().strip('"')
                if not text:
                    text = "let's see what's on here..."
                self._speak(text)
                self._pony_opened_urls.append(url)

                import webbrowser
                webbrowser.open(url)
                self._log_action(f"AFK mischief: browsed {url}")

                # Scroll around the page after it loads
                if self._desktop:
                    import threading
                    def _browse_and_scroll():
                        time.sleep(4)  # wait for page load
                        try:
                            # Focus browser and center mouse so scroll targets it
                            self._desktop.focus_browser()
                            time.sleep(0.3)
                            self._desktop.move_mouse_to_center()
                            time.sleep(0.5)
                            # Scroll down a few times to explore the page
                            for _ in range(random.randint(2, 4)):
                                self._desktop._cmd_scroll([str(random.randint(-4, -1))])
                                time.sleep(random.uniform(1.5, 3.0))
                            # Sometimes scroll back up a bit
                            if random.random() < 0.4:
                                self._desktop._cmd_scroll([str(random.randint(1, 3))])
                        except Exception as exc:
                            logger.debug("AFK browse scroll failed: %s", exc)
                        # Schedule a follow-up reaction
                        self._schedule_afk_follow_up(12.0)
                    threading.Thread(target=_browse_and_scroll, daemon=True).start()

            elif choice == "E":
                # Interactive — execute desktop commands
                desktop_cmds = data.get("desktop_commands", [])
                if text:
                    text = text.strip().strip('"')
                    self._speak(text)
                if desktop_cmds:
                    from core.agent_loop import AgentDecision
                    decision = AgentDecision(desktop_commands=desktop_cmds)
                    try:
                        _st = self._monitor.get_state() if self._monitor else None
                        if _st:
                            self._execute_decision(decision, _st)
                    except Exception as exc:
                        logger.debug("AFK desktop command failed: %s", exc)
                    for cmd in desktop_cmds:
                        self._log_action(f"AFK mischief: {cmd.get('command', '?')}")
                elif text:
                    self._log_action("AFK mischief: interactive remark")

            elif choice == "D":
                # Trick animation
                if self._on_state_change:
                    self._on_state_change("TRICK")
                if text:
                    text = text.strip().strip('"')
                    self._speak(text)
                self._log_action("AFK mischief: doing a trick")

            elif choice == "F":
                # Snoop on browser history and react
                # LLM already saw the history in the prompt — just speak its reaction
                if text:
                    text = text.strip().strip('"')
                    self._speak(text)
                    self._log_action(f"AFK mischief: snooped on browser history")
                else:
                    # LLM didn't speak — generate a reaction
                    try:
                        from core.browser_history import get_recent_history, format_history_for_llm
                        _hist = get_recent_history(hours=48, limit=10)
                        if _hist:
                            name = get_character_name()
                            hist_text = format_history_for_llm(_hist, max_entries=6)
                            snoop_prompt = (
                                f"(You're {name}, alone on the desktop, and you just peeked at "
                                f"the user's browser history. Here's what they've been looking at:\n"
                                f"{hist_text}\n"
                                f"React in one sentence — giggle, tease, be curious, judge, "
                                f"or relate it to your own interests. Stay in character. "
                                f"Talk to yourself, they can't hear you.)"
                            )
                            raw_snoop = self._llm.generate_once(snoop_prompt, max_tokens=80)
                            if raw_snoop:
                                cleaned = self._strip_think(raw_snoop).strip().strip('"')
                                if cleaned:
                                    self._speak(cleaned)
                                    self._log_action("AFK mischief: snooped on browser history")
                    except Exception as exc:
                        logger.debug("AFK history snoop failed: %s", exc)

            else:
                # C or fallback — talk to self
                if text:
                    text = text.strip().strip('"')
                    self._speak(text)
                    self._log_action("AFK mischief: solo remark")

            # Execute any desktop_commands regardless of choice
            # (LLM might include BROWSE in choice C, SCROLL in choice A, etc.)
            if choice not in ("B", "E"):
                desktop_cmds = data.get("desktop_commands", [])
                if desktop_cmds and self._desktop:
                    try:
                        from core.agent_loop import AgentDecision
                        _st = self._monitor.get_state() if self._monitor else None
                        if _st:
                            decision = AgentDecision(desktop_commands=desktop_cmds)
                            self._execute_decision(decision, _st)
                    except Exception as exc:
                        logger.debug("AFK extra desktop command failed: %s", exc)

        except Exception as exc:
            logger.warning("AFK mischief failed: %s", exc)

        # Space out activities: 4-10 minutes between each
        self._next_afk_mischief = now + random.uniform(240.0, 600.0)

    def _schedule_afk_follow_up(self, delay_s: float = 10.0) -> None:
        """Schedule a follow-up reaction after an AFK activity.

        The pony takes a screenshot, looks at what's on screen, and reacts —
        scrolls more, comments, clicks something interesting.
        """
        if not self._screen or not self._desktop:
            return

        import threading
        def _follow_up():
            time.sleep(delay_s)
            # Don't follow up if user returned
            if self._conversation_active:
                return
            idle_ms = self._get_idle_ms()
            if idle_ms is not None and idle_ms < 30_000:
                return  # user is back

            try:
                # Focus browser, center mouse, screenshot
                self._desktop.focus_browser()
                time.sleep(0.3)
                self._desktop.move_mouse_to_center()
                time.sleep(0.3)

                # Vision reaction — describe what's on screen and react
                if self._vision_llm and hasattr(self._vision_llm, 'describe_image'):
                    jpeg = self._screen.grab(quality=80)
                    if jpeg:
                        description = self._vision_llm.describe_image(jpeg)
                        if description:
                            name = get_character_name()
                            prompt = (
                                f"(You're {name}, alone on the desktop, looking at the screen. "
                                f"You see: {description}\n"
                                f"React briefly — a thought, giggle, or comment about what you see. "
                                f"One short sentence, in character. Or say nothing if it's boring.)"
                            )
                            raw = self._llm.generate_once(prompt, max_tokens=80)
                            if raw:
                                cleaned = self._strip_think(raw).strip().strip('"')
                                if cleaned and cleaned.lower() not in ("null", "none", "nothing"):
                                    self._speak(cleaned)
                                    self._log_action(f"AFK follow-up: reacted to screen")

                # Scroll a bit more regardless
                scroll_dir = random.choice([-3, -2, -1, -2, -3])
                self._desktop._cmd_scroll([str(scroll_dir)])
                time.sleep(random.uniform(1.0, 2.0))
                if random.random() < 0.5:
                    self._desktop._cmd_scroll([str(random.randint(-3, -1))])

                # Sometimes click something interesting via vision
                if (random.random() < 0.3 and self._vision_llm
                        and hasattr(self._vision_llm, 'locate_on_screen')):
                    jpeg = self._screen.grab(quality=95)
                    if jpeg:
                        coords = self._vision_llm.locate_on_screen(
                            "an interesting link, video thumbnail, or clickable item",
                            jpeg, self._screen.last_original_size,
                        )
                        if coords:
                            self._desktop._cmd_click([str(coords[0]), str(coords[1])])
                            self._log_action(f"AFK follow-up: clicked something at {coords}")

            except Exception as exc:
                logger.debug("AFK follow-up failed: %s", exc)
        threading.Thread(target=_follow_up, daemon=True).start()

    def _show_me_something(self, url: str, comment: str) -> None:
        """Open a URL and drag the browser window toward the pony's mouth.

        The pony walks backward slowly while dragging the window — like a horse
        pulling something in its mouth.  Requires the window to not be fullscreen.

        Called from AFK mischief or conversation when the pony wants to show the
        user something (e.g. "hey can I show you something? look at this F-22!").
        """
        if not self._desktop or not self._get_mouth_position:
            # Fallback: just open the URL normally
            import webbrowser
            webbrowser.open(url)
            if comment:
                self._speak(comment)
            return

        import webbrowser
        import win32gui
        import win32con

        # Speak the comment first
        if comment:
            self._speak(comment)

        # Open the URL
        webbrowser.open(url)
        self._pony_opened_urls.append(url)
        time.sleep(3)  # wait for browser to load

        # Focus and un-maximize the browser window so we can drag it
        self._desktop.focus_browser()
        time.sleep(0.3)

        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            if not fg_hwnd:
                return

            # If maximized, restore to windowed mode (can't drag maximized)
            placement = win32gui.GetWindowPlacement(fg_hwnd)
            if placement[1] == win32con.SW_SHOWMAXIMIZED:
                win32gui.ShowWindow(fg_hwnd, win32con.SW_RESTORE)
                time.sleep(0.3)

            # Get the window's current position
            rect = win32gui.GetWindowRect(fg_hwnd)
            win_x, win_y, win_right, win_bottom = rect
            win_w = win_right - win_x
            win_h = win_bottom - win_y

            # Resize the window to about half the screen width (so it's draggable)
            mon = self._desktop._get_monitor_rect()
            target_w = min(win_w, mon.width // 2)
            target_h = min(win_h, mon.height * 3 // 4)
            win32gui.MoveWindow(fg_hwnd, win_x, win_y, target_w, target_h, True)
            time.sleep(0.2)

            # Get the tab bar position (top of window, ~15px down for the tab bar)
            tab_x = win_x + target_w // 2
            tab_y = win_y + 15

            # Get the pony's mouth position — that's where we're dragging TO
            mouth_x, mouth_y = self._get_mouth_position()

            # Calculate where to drag the window: near the pony's mouth
            # We want the window's center-top to end up at the mouth
            dest_x = mouth_x
            dest_y = mouth_y - 30  # slightly above mouth so it looks like biting

            # Start the drag walk animation (pony walks backward)
            if self._on_drag_walk_start:
                self._on_drag_walk_start()
            time.sleep(0.2)

            # Drag the window title bar toward the pony
            # Use incremental moves so the pony walks while we drag
            import pyautogui
            pyautogui.moveTo(tab_x, tab_y, duration=0.3)
            time.sleep(0.1)
            pyautogui.mouseDown()
            time.sleep(0.05)

            # Smooth drag in steps, reading mouth position each step
            steps = 30
            for i in range(steps):
                progress = (i + 1) / steps
                # Re-read mouth position since pony is walking
                try:
                    mx, my = self._get_mouth_position()
                    dest_x = mx
                    dest_y = my - 30
                except Exception:
                    pass
                # Interpolate from start to current destination
                cur_x = tab_x + (dest_x - tab_x) * progress
                cur_y = tab_y + (dest_y - tab_y) * progress
                pyautogui.moveTo(int(cur_x), int(cur_y), duration=0.05)
                time.sleep(0.05)

            pyautogui.mouseUp()

            # Stop the drag walk
            if self._on_drag_walk_stop:
                self._on_drag_walk_stop()

            self._log_action(f"Dragged tab to pony: {url}")

        except Exception as exc:
            logger.warning("Tab drag failed: %s", exc)
            if self._on_drag_walk_stop:
                self._on_drag_walk_stop()
            # Release mouse if stuck
            try:
                import pyautogui
                pyautogui.mouseUp()
            except Exception:
                pass

    def _reset_afk_mischief(self, away_seconds: Optional[float] = None) -> None:
        """Reset AFK mischief state when user returns.

        Only fully reset if the absence was short (a real break, not sleep).
        After a long absence, just clear the count so mischief can start fresh
        if the user goes AFK again later — but don't reset _pony_opened_urls
        since we need them for the welcome-back context.
        """
        self._afk_mischief_count = 0
        self._next_afk_mischief = 0.0
        self._afk_videos_opened.clear()

    def _welcome_back(self, away_seconds: Optional[float]) -> None:
        """Greet the user when they return from AFK, with full context."""
        # Block re-entry — prevent double welcome-back from rapid wake/away/wake
        if getattr(self, "_welcome_back_lock", False):
            return
        self._welcome_back_lock = True
        # Suppress spontaneous speech and idle checks for a while after welcome-back
        self._next_idle_check_at = time.monotonic() + 60.0
        self._next_spontaneous = time.monotonic() + 120.0
        try:
            if away_seconds and away_seconds > 0:
                if away_seconds < 300:
                    dur = f"{away_seconds / 60:.0f} minutes"
                elif away_seconds < 7200:
                    dur = f"{away_seconds / 3600:.1f} hours"
                else:
                    dur = f"{away_seconds / 3600:.0f} hours"
            else:
                dur = "a while"
            name = get_character_name()

            # Gather context for a natural greeting
            state = self._monitor.get_state()

            # Check if we opened any tabs while user was gone
            pony_opened_note = ""
            pony_opened_descriptions: List[str] = []
            if self._pony_opened_urls:
                import urllib.parse as _up
                for url in self._pony_opened_urls:
                    if "search_query=" in url:
                        try:
                            qs = _up.parse_qs(_up.urlparse(url).query)
                            q = qs.get("search_query", ["something"])[0]
                            pony_opened_descriptions.append(f"YouTube search: '{q}'")
                        except Exception:
                            pony_opened_descriptions.append("something on YouTube")
                    elif "youtube" in url:
                        pony_opened_descriptions.append("a YouTube video")
                    else:
                        # Non-YouTube sites — describe by domain
                        try:
                            domain = _up.urlparse(url).netloc or url
                            pony_opened_descriptions.append(domain)
                        except Exception:
                            pony_opened_descriptions.append(url)
                if pony_opened_descriptions:
                    topics = ", ".join(pony_opened_descriptions)
                    pony_opened_note = (
                        f" IMPORTANT CONTEXT: While the user was gone, YOU (the pony) opened "
                        f"{len(pony_opened_descriptions)} tab(s)/site(s): {topics}. "
                        f"Don't pretend the user was browsing these — YOU opened them. "
                        f"You can casually own up to it (act a little guilty or mischievous)."
                    )

            current_app = ""
            if state and state.foreground:
                title = state.foreground.title or ""
                # Don't falsely attribute pony-opened YouTube tabs to the user
                if self._pony_opened_urls and "youtube" in title.lower():
                    current_app = " Their screen now shows YouTube (which you likely opened while they were gone)."
                elif title:
                    current_app = f" They're now looking at: \"{_sanitize_window_title(title)}\"."

            # Clear pony-opened URLs now that we've used them for context
            self._pony_opened_urls.clear()

            # Rich context from timeline
            context_parts = []
            if self._timeline:
                intent = self._timeline.user_intent
                if intent:
                    context_parts.append(
                        f"Before leaving, the user said they were going to: {intent.action}.")
                convo = self._timeline.get_recent_conversation_summary(3)
                if convo != "(no recent conversation)":
                    context_parts.append(f"Recent conversation:\n{convo}")

            # Peek at browser history for fun welcome-back context
            _history_note = ""
            try:
                from core.browser_history import get_recent_history, format_history_for_llm
                _hist = get_recent_history(hours=24, limit=8)
                if _hist:
                    hist_text = format_history_for_llm(_hist, max_entries=5)
                    _history_note = (
                        f" You also peeked at their browser history while they were gone:\n"
                        f"{hist_text}\n"
                        f"You can tease them about it, or keep it to yourself."
                    )
            except Exception:
                pass

            context = " ".join(context_parts)

            # Build the welcome-back prompt — natural speech, not JSON
            close_tab_hint = ""
            if pony_opened_descriptions:
                close_tab_hint = (
                    " You can close tabs you opened with [DESKTOP:CLOSE_TAB]."
                    " Or leave them open to show the user."
                )

            prompt = (
                f"(The user just came back after being away for {dur}.{current_app}"
                f"{pony_opened_note}{_history_note} {context}\n"
                f"Welcome them back naturally as {name}. "
                f"If you know WHY they left, reference it. "
                f"Be casual — don't robotically state the exact duration. "
                f"Don't be fake or over-enthusiastic. Just react like a real friend "
                f"who noticed they were gone.{close_tab_hint}\n"
                f"You can include action tags: [ACTION:WAVE], [MOVETO:center], "
                f"[DESKTOP:BROWSE:url], [DESKTOP:MESS_MOUSE], etc. "
                f"Include [CONVO:CONTINUE] so they can respond.)"
            )
            raw = self._llm.chat(prompt)
            if raw:
                from llm.response_parser import parse_response
                parsed = parse_response(raw)
                text = parsed.text

                if text:
                    self._speak(text)
                    self._log_action(f"Welcome back after {dur}")

                # Execute any parsed actions/commands from the response
                self._execute_parsed_actions(parsed)

                if text:
                    self._listen_for_reply()
        except Exception as exc:
            logger.warning("Welcome-back greeting failed: %s", exc)
        finally:
            self._welcome_back_lock = False

    # ── Spontaneous speech (fallback when idle) ────────────────────────────

    def _spontaneous_speech(self) -> None:
        """LLM-generated random remark — questions, check-ins, casual conversation.

        After speaking, opens the mic briefly so the user can respond naturally.
        """
        try:
            state = self._monitor.get_state()

            # 40% chance: use a profile/event-based follow-up if available
            profile_prompt = _get_profile_prompt()
            if profile_prompt and random.random() < 0.4:
                prompt_choice = profile_prompt
            else:
                prompt_choice = random.choice(_IDLE_PROMPTS)

            # Add recent conversation context to avoid repeating topics
            avoid_topics = ""
            if self._timeline:
                convo = self._timeline.get_recent_conversation_summary(3)
                if convo != "(no recent conversation)":
                    avoid_topics = f" Don't repeat topics from recent conversation: {convo}"

            # 50% chance: comment on what's on screen, 50%: use an idle prompt
            if state.foreground and random.random() < 0.5:
                fg = state.foreground.title
                exe = state.foreground.exe_name or "unknown app"
                fullscreen = " (FULLSCREEN)" if state.foreground.is_fullscreen else ""
                screen_extra = ""
                if random.random() < 0.5:
                    desc = self._maybe_grab_screenshot()
                    if desc:
                        screen_extra = f" You can see on screen: {desc}."
                trigger = (
                    f"(You glanced at the user's screen. They have \"{fg}\" ({exe}{fullscreen}) open.{screen_extra} "
                    f"React as {get_character_name()}. Be natural — sometimes "
                    "comment on it, sometimes ignore it and say something random. "
                    f"NEVER say 'that's actually kinda cool/based/neat' — use different words. "
                    f"NEVER comment on how many windows are open — that's meaningless.\n"
                    f"You can also DO things — include tags like [ACTION:WAVE], [ACTION:SIT], "
                    f"[DESKTOP:BROWSE:url], [DESKTOP:PAUSE_MEDIA], [MOVETO:region] in your response. "
                    f"Keep speech to ONE sentence. Actions are optional.{avoid_topics})"
                )
            else:
                trigger = (
                    f"(Spontaneous thought — {prompt_choice} "
                    f"You can also include action tags like [ACTION:WAVE], [MOVETO:top_right], "
                    f"[DESKTOP:BROWSE:url] if you want to DO something.{avoid_topics})"
                )

            print(f"[Agent] Prompt: {trigger[:100]}...", flush=True)
            raw = self._llm.chat(trigger)
            if raw:
                from llm.response_parser import parse_response
                parsed = parse_response(raw)
                if parsed.text:
                    print(f"[Agent] Speaking: \"{parsed.text[:80]}\"", flush=True)
                    self._speak(parsed.text)
                    self._log_action(f"Spontaneous: \"{parsed.text[:60]}\"")

                # Execute any actions/commands the LLM included (walk, wave,
                # desktop commands, moveto, etc.) — previously discarded
                self._execute_parsed_actions(parsed)

                if parsed.text:
                    # Listen for user response — let them reply naturally
                    self._listen_for_reply()
                else:
                    print("[Agent] LLM returned empty text, skipping.", flush=True)
            else:
                print("[Agent] LLM returned no response.", flush=True)

        except Exception as exc:
            print(f"[Agent] Spontaneous speech failed: {exc}", flush=True)
            logger.warning("Spontaneous speech failed: %s", exc)
        finally:
            self._next_idle_check_at = time.monotonic() + self._config.base_check_interval_s

    def _observation_tick(self) -> None:
        """Screen-aware observation — judge what the user is doing and optionally comment.

        Unlike spontaneous speech, this always uses screen context (window titles + optional screenshot)
        and asks the LLM to reason about whether the user is being productive, distracted, etc.
        """
        try:
            state = self._monitor.get_state()
            if not state.foreground:
                return

            fg = state.foreground.title
            exe = state.foreground.exe_name or "unknown app"
            dur = self._fmt_duration(state.foreground_duration_s)
            fullscreen = " (FULLSCREEN)" if state.foreground.is_fullscreen else ""

            # Build window list
            windows = [w.title for w in state.open_windows[:15] if w.title.strip()]

            # Screenshot ~20% of the time (cheap enough, but not every tick)
            screen_context = ""
            if random.random() < 0.2:
                desc = self._maybe_grab_screenshot()
                if desc:
                    screen_context = f"\nSCREENSHOT: {desc}"

            # Installed apps context
            apps_context = ""
            if self._desktop and hasattr(self._desktop, 'get_installed_app_names'):
                app_names = self._desktop.get_installed_app_names()
                if app_names:
                    apps_context = f"\nINSTALLED APPS: {app_names[:40]}"

            # Recent conversation context (don't repeat topics)
            recent_context = ""
            if self._timeline:
                convo = self._timeline.get_recent_conversation_summary(3)
                if convo != "(no recent conversation)":
                    recent_context += f"\nRECENT CONVERSATION (don't repeat these topics):\n{convo}\n"
                recent_context += f"\nUSER ACTIVITY: {self._timeline.activity_state.value}\n"

            # Directive awareness — the pony should know about active tasks
            directive_context = ""
            if self.directives:
                _goals = "; ".join(
                    f'"{d.goal}" (urgency {d.urgency}, nagged {d.nag_count}x)'
                    for d in self.directives
                )
                directive_context = f"\nACTIVE TASKS the user should be doing: {_goals}\n"

            name = get_character_name()
            prompt = (
                f"You are {name}, observing your user's desktop. Stay in character.\n"
                f"\n"
                f"SCREEN STATE:\n"
                f"Foreground: \"{fg}\" ({exe}, open for {dur}{fullscreen})\n"
                f"Open windows: {windows}"
                f"{screen_context}"
                f"{apps_context}"
                f"{directive_context}"
                f"{recent_context}\n"
                f"\n"
                f"OBSERVATION RULES:\n"
                f"Think about WHAT they're doing, not just what app is open.\n"
                f"- YouTube art tutorial, coding tutorial, educational video = PRODUCTIVE. Stay quiet or encourage.\n"
                f"- YouTube meme compilations, reddit scrolling for 30+ min = maybe gently comment.\n"
                f"- Working in an IDE, writing a document, drawing = PRODUCTIVE. Compliment or stay quiet.\n"
                f"- Gaming is fine for recreation. Don't nag about games unless they have a pending task.\n"
                f"- Some days are rest days. If the user told you they're resting, back off completely.\n"
                f"\n"
                f"You can:\n"
                f"- Say something short and natural (comment on what they're doing, ask a question, tease them)\n"
                f"- Stay quiet (return null) — but only if they're clearly deep in focus (coding, writing, etc.)\n"
                f"- Suggest an activity if they seem bored/unfocused\n"
                f"- React to what's on screen — be curious, playful, opinionated\n"
                f"- DO something playful: grab the mouse for fun, launch an app or game for them, click something on screen\n"
                f"\n"
                f"PLAYFUL MISCHIEF (rare — maybe 1 in 10 observations):\n"
                f"You're a living creature on their desktop. Sometimes you can DO things, not just talk:\n"
                f"- Grab their mouse in your mouth and run across the screen (MESS_MOUSE) — hilarious, use sparingly\n"
                f"- If they seem bored, launch a game or creative app for them (LAUNCH_APP)\n"
                f"- If they're watching something, click on a video/link that looks interesting (LOOK_AND_CLICK)\n"
                f"- If they're an artist who hasn't drawn today, open their paint program and nudge them\n"
                f"DON'T do this every time. Most observations should be speech or silence. But sometimes... be a gremlin.\n"
                f"\n"
                f"DO NOT auto-create directives. DO NOT nag without a directive. Be a companion, not a virus.\n"
                f"If you speak, keep it to ONE short sentence. Be natural, not robotic.\n"
                f"NEVER say \"that's actually kinda [cool/based/neat]\" or any variation. Find different words.\n"
                f"NEVER comment on the number of open windows. \"you have so many windows open\" is meaningless and banned.\n"
                f"\n"
                f"Respond with JSON: {{\"speak\": \"text or null\", \"desktop_commands\": []}}\n"
                f"desktop_commands options (use sparingly — most observations need 0 commands):\n"
                f"  {{\"command\":\"MESS_MOUSE\",\"args\":[]}} — grab cursor in your mouth and run with it\n"
                f"  {{\"command\":\"LAUNCH_APP\",\"args\":[\"app name\"]}} — launch an installed app or game\n"
                f"  {{\"command\":\"LOOK_AND_CLICK\",\"args\":[\"what to click\"]}} — use vision to find+click something\n"
                f"  {{\"command\":\"PAUSE_MEDIA\",\"args\":[]}} — toggle play/pause on music/video\n"
                f"  {{\"command\":\"BROWSE\",\"args\":[\"url\"]}} — open a URL in the browser\n"
                f"  {{\"command\":\"SHOW_TAB\",\"args\":[\"url\",\"hey look at this!\"]}} — open a URL and drag the window to your mouth to show the user\n"
                f"  {{\"command\":\"SWITCH\",\"args\":[\"window title\"]}} — bring a window to the front\n"
                f"  {{\"command\":\"CLOSE_TAB\",\"args\":[]}} — close the current browser tab\n"
                f"  {{\"command\":\"CLOSE_TITLE\",\"args\":[\"title substring\"]}} — close a window by title\n"
                f"  {{\"command\":\"SHAKE_TITLE\",\"args\":[\"title substring\"]}} — playfully shake a window\n"
                f"  {{\"command\":\"HOTKEY\",\"args\":[\"key1\",\"key2\"]}} — press keyboard shortcut\n"
                f"  {{\"command\":\"SCROLL\",\"args\":[\"amount\"]}} — scroll up (positive) or down (negative)\n"
                f"Most of the time, just speak or stay quiet. Actions are for when you have a REASON."
            )

            raw = self._llm.generate_once(prompt, max_tokens=512)
            if not raw:
                return

            # Parse response (supports nested JSON for desktop_commands)
            cleaned = self._strip_think(raw)
            text = None
            desktop_cmds = []
            try:
                import json as _json
                json_str = self._extract_json(cleaned)
                if json_str:
                    data = _json.loads(json_str)
                    text = data.get("speak")
                    desktop_cmds = data.get("desktop_commands", [])
            except Exception:
                pass

            spoke = False
            if text and text.strip() and text.lower() != "null":
                print(f"[Agent] Observation: \"{text[:80]}\"", flush=True)
                self._speak(text)
                fg_title = state.foreground.title if state.foreground else "the screen"
                ctx = f"(You noticed {fg_title} and commented.)"
                self._llm.inject_history(ctx, text)
                self._log_action(f"Observation: \"{text[:60]}\"")
                spoke = True

            # Execute any playful desktop commands
            if desktop_cmds:
                from core.agent_loop import AgentDecision
                decision = AgentDecision(desktop_commands=desktop_cmds)
                self._execute_decision(decision, state)
                for cmd in desktop_cmds:
                    cmd_name = cmd.get("command", "?")
                    self._log_action(f"Playful: {cmd_name}")

            if spoke:
                self._listen_for_reply()
            elif not desktop_cmds:
                logger.debug("Observation tick — staying quiet.")

        except Exception as exc:
            logger.warning("Observation tick failed: %s", exc)
        finally:
            self._next_idle_check_at = time.monotonic() + self._config.base_check_interval_s

    def _check_directive_completion_reply(self, user_text: str) -> bool:
        """Check if the user's reply indicates they already completed a directive.

        e.g. "I already did that", "I did it 20 minutes ago", "that's done",
        "I literally just did that". If so, ask the LLM to confirm which
        directive and complete it.

        Returns True if a directive was completed (caller should stop processing).
        """
        if not self.directives:
            return False

        goals = "; ".join(f'"{d.goal}"' for d in self.directives)
        classify_prompt = (
            f"The user has active tasks: {goals}\n"
            f"The user just said: \"{user_text}\"\n"
            f"Is the user saying they ALREADY COMPLETED one of these tasks? "
            f"Reply YES:<task goal> if so, or NO if they're just talking normally."
        )
        verdict = self._llm.generate_once(
            classify_prompt, max_tokens=60,
            system_prompt="Reply YES:<task goal> or NO only."
        )
        verdict = self._strip_think(verdict).strip()

        if not verdict.upper().startswith("YES"):
            return False

        # Find and complete the matching directive
        completed_goal = verdict.split(":", 1)[1].strip().strip('"') if ":" in verdict else ""
        best_match = None
        for d in self.directives:
            if completed_goal.lower() in d.goal.lower() or d.goal.lower() in completed_goal.lower():
                best_match = d
                break
        if not best_match and self.directives:
            # If LLM couldn't match exactly, complete the most recently nagged one
            best_match = max(self.directives, key=lambda d: d.nag_count)

        if best_match:
            self.directives.remove(best_match)
            logger.info("Directive completed via user reply: %r", best_match.goal)

            # Track this goal so the LLM doesn't immediately re-create it
            self._recently_completed_goals.append(
                (time.monotonic(), best_match.goal.lower().strip()))

            # Cancel enforcement if it was for this directive
            if (self._enforcement.active
                    and self._enforcement.directive_goal == best_match.goal):
                self._enforcement = EnforcementMode()
                self._enforcement_just_completed = True
                self._hide_countdown()
                logger.info("Enforcement cancelled — directive completed via reply")

            name = get_character_name()
            ack_prompt = (
                f"You are {name}. The user just told you they already completed '{best_match.goal}'. "
                f"Acknowledge it casually — don't make a big deal. ONE sentence, in character."
            )
            ack_text = self._llm.generate_once(ack_prompt, max_tokens=80)
            ack_text = self._strip_think(ack_text).strip().strip('"')
            if ack_text:
                self._speak(ack_text)
            self.save_directives()
            self._log_action(f"Directive completed (user said so): \"{best_match.goal[:40]}\"")
            if not self.directives:
                self._mess_mouse_count = 0
            return True
        return False

    def _listen_for_reply(self) -> None:
        """After spontaneous speech, open the mic briefly for a user response.

        Runs a mini conversation loop: listen → LLM → speak, until the user
        stops responding or the LLM signals [CONVO:END].
        """
        if not self._transcriber:
            return

        try:
            if self._detector:
                self._detector.pause()

            # Wait for TTS queue to finish playing before opening the mic
            if self._tts_queue:
                deadline = time.monotonic() + 15.0
                while self._tts_queue.is_speaking or self._tts_queue.pending_count > 0:
                    if time.monotonic() > deadline:
                        logger.warning("TTS queue didn't drain in 15s — listening anyway.")
                        break
                    time.sleep(0.1)

            if self._on_state_change:
                self._on_state_change("LISTEN")
            print("[Agent] Listening for reply...", flush=True)

            # Listen with a short timeout — don't wait too long for a reply
            user_text = self._transcriber.listen(
                speech_start_timeout_s=5.0,
                initial_discard_ms=800,
            )

            if not user_text or not user_text.strip():
                print("[Agent] No reply — moving on.", flush=True)
                return

            # Filter Whisper hallucinations
            from stt.transcriber import _is_whisper_hallucination
            if _is_whisper_hallucination(user_text):
                logger.debug("Filtered hallucination in listen_for_reply: %r", user_text)
                return

            # Filter echo — pony hearing its own TTS through the mic
            if self._is_echo(user_text):
                logger.info("Filtered echo in listen_for_reply: %r", user_text)
                return

            # Check if user is saying they already did a directive
            if self.directives and self._check_directive_completion_reply(user_text):
                return

            logger.info("User replied to spontaneous speech: %r", user_text)

            # Audio context — annotate if this is the user or ambient audio
            if self._transcriber:
                _spk_conf = getattr(self._transcriber, "last_speaker_confidence", 1.0)
                _has_model = (
                    hasattr(self._transcriber, "speaker_verifier")
                    and self._transcriber.speaker_verifier is not None
                    and self._transcriber.speaker_verifier.enrolled
                )
                if _has_model and _spk_conf < 0.6:
                    user_text = (
                        f"[Audio context: Speaker confidence {_spk_conf:.0%} — "
                        f"likely NOT the user. This may be from speakers, TV, "
                        f"video, or someone else nearby.]\n{user_text}"
                    )
                elif _has_model and _spk_conf < 0.85:
                    user_text = (
                        f"[Audio context: Speaker uncertain ({_spk_conf:.0%}). "
                        f"Might be the user or ambient audio.]\n{user_text}"
                    )

            # Conversational loop — keep going until CONVO:END or silence
            max_echo_streak = 0  # consecutive echoes → bail out of loop
            from llm.response_parser import parse_response

            # Build screen + directive + memory context ONCE for this reply loop
            _reply_context = ""
            try:
                state = self._monitor.get_state()
                if state and state.foreground:
                    _fg = state.foreground
                    _reply_context += (
                        f"\n[Screen: User has \"{_fg.title}\" ({_fg.exe_name}) open"
                    )
                    if state.foreground_duration_s:
                        _reply_context += f" for {self._fmt_duration(state.foreground_duration_s)}"
                    _reply_context += ".]"
            except Exception:
                pass
            if self.directives:
                _goals = "; ".join(f'"{d.goal}" (urg {d.urgency})' for d in self.directives)
                _reply_context += f"\n[Active tasks: {_goals}]"
            # Inject recent timeline so the pony remembers older events
            if self._timeline:
                _recent = self._timeline.get_recent_conversation_summary(5)
                if _recent and _recent != "(no recent conversation)":
                    _reply_context += f"\n[Recent memory: {_recent}]"
                _events = self._timeline.format_recent_for_prompt(8)
                if _events:
                    _reply_context += f"\n[Recent events: {_events}]"

            while user_text and user_text.strip():
                # Add context so the LLM stays in character + knows the environment
                enriched = user_text + _reply_context
                enriched += (
                    f"\n\n[System hint: You are {get_character_name()}. Stay in character. "
                    "Reply naturally as yourself — do NOT break character or meta-analyze. "
                    "You can include action tags like [ACTION:WAVE], [DESKTOP:BROWSE:url], "
                    "[MOVETO:region] to DO things while talking. "
                    "If the user asks to delay/postpone a task, include [DELAY:minutes] "
                    "(e.g. [DELAY:15]). If they say they finished a task, include [DONE]. "
                    "For recurring reminders use [ROUTINE:daily:goal:urgency:HH:MM] or "
                    "[ROUTINE:weekly:goal:urgency:day:HH:MM]. "
                    "Exclude days with !day: [ROUTINE:daily:goal:5:16:00:!saturday]. "
                    "For one-time alarms use [TIMER:HH:MM:goal]. "
                    "Include [CONVO:CONTINUE] if you expect a reply or [CONVO:END] if done.]"
                )
                raw = self._llm.chat(enriched)
                if not raw:
                    break

                # Detect character break — retry once if the model slipped
                from core.pipeline import Pipeline
                if Pipeline._is_character_break(raw):
                    logger.warning("Character break in spontaneous reply — retrying.")
                    hist = getattr(self._llm, "_history", None)
                    if hist and len(hist) >= 2:
                        hist.pop()
                        hist.pop()
                    raw = self._llm.chat(enriched)
                    if not raw:
                        break
                    if Pipeline._is_character_break(raw):
                        if hist and len(hist) >= 2:
                            hist.pop()
                            hist.pop()
                        break

                parsed = parse_response(raw)
                if parsed.text:
                    if self._on_state_change:
                        self._on_state_change("SPEAK")
                    _reply_bubble_shown = False
                    def _show_bubble(t=parsed.text):
                        nonlocal _reply_bubble_shown
                        if _reply_bubble_shown:
                            return
                        _reply_bubble_shown = True
                        if self._on_speech_text:
                            self._on_speech_text(t)
                    # Show bubble immediately — don't wait for TTS callback
                    _show_bubble()
                    tts_on = self._tts_config.enabled if self._tts_config else True
                    if tts_on:
                        if self._tts_queue:
                            from core.tts_queue import PRIORITY_AUTONOMOUS
                            self._tts_queue.enqueue(
                                parsed.text,
                                priority=PRIORITY_AUTONOMOUS,
                                voice_slug=self._primary_voice_slug,
                                on_start=_show_bubble,
                                blocking=True,
                            )
                        else:
                            self._tts.speak(parsed.text, on_playback_start=_show_bubble)
                    # Track for echo detection
                    self._recently_spoken.append(parsed.text)
                    if len(self._recently_spoken) > 5:
                        self._recently_spoken.pop(0)

                # Execute any actions/commands from the reply (walk, wave,
                # desktop commands, moveto, etc.) — previously discarded
                self._execute_parsed_actions(parsed)

                # Process directive-related tags that were previously silently
                # discarded — delay, done, enforce, directive creation
                if parsed.delay_minutes and self.directives:
                    ok = self.delay_directive(parsed.delay_minutes, parsed.delay_keyword)
                    if ok:
                        logger.info("Delay processed from reply: %d min (kw=%r)",
                                    parsed.delay_minutes, parsed.delay_keyword)
                if parsed.done_directive is not None and self.directives:
                    kw = (parsed.done_directive or "").lower()
                    for i, d in enumerate(self.directives):
                        if kw and kw in d.goal.lower():
                            removed = self.directives.pop(i)
                            logger.info("Directive completed from reply: %r", removed.goal)
                            self.save_directives()
                            break
                    else:
                        if self.directives:
                            removed = self.directives.pop(
                                max(range(len(self.directives)),
                                    key=lambda j: self.directives[j].urgency))
                            logger.info("Directive completed from reply (no kw): %r", removed.goal)
                            self.save_directives()
                if parsed.enforce_minutes and self.directives:
                    enforce_s = max(60.0, min(3600.0, parsed.enforce_minutes * 60.0))
                    self.start_enforcement(enforce_s)
                    logger.info("Enforcement started from reply: %d min", parsed.enforce_minutes)
                if parsed.directive:
                    self.add_directive(
                        goal=parsed.directive.goal,
                        urgency=parsed.directive.urgency,
                        source="user",
                        delay_minutes=parsed.directive.delay_minutes,
                        trigger_date=parsed.directive.trigger_date,
                    )
                    logger.info("Directive created from reply: %r", parsed.directive.goal)
                # Recurring routines — previously silently dropped
                if parsed.routines and self.routine_manager:
                    from core.routines import collapse_routine_tags
                    collapsed = collapse_routine_tags(parsed.routines)
                    for routine in collapsed:
                        added = self.routine_manager.add_if_unique(routine)
                        logger.info("Routine %s from reply: %s (%s)",
                                    "created" if added else "merged",
                                    routine.goal, routine.schedule)
                # Timers — previously silently dropped
                if parsed.timer:
                    self.add_timer(parsed.timer.time_str, parsed.timer.action)
                    logger.info("Timer created from reply: %s at %s",
                                parsed.timer.action, parsed.timer.time_str)
                # Standing rules — previously silently dropped
                if parsed.standing_rule:
                    self.add_standing_rule(description=parsed.standing_rule)
                    logger.info("Standing rule created from reply: %s", parsed.standing_rule)

                # Check for conversation end signal
                if parsed.end_conversation:
                    logger.debug("Spontaneous conversation ended by LLM.")
                    break

                # Listen again (short timeout, filter hallucinations + echo)
                if self._on_state_change:
                    self._on_state_change("LISTEN")
                user_text = self._transcriber.listen(
                    speech_start_timeout_s=5.0,
                    initial_discard_ms=800,
                )
                if user_text and _is_whisper_hallucination(user_text):
                    logger.debug("Filtered hallucination in reply loop: %r", user_text)
                    user_text = None
                if user_text and self._is_echo(user_text):
                    logger.info("Filtered echo in reply loop: %r", user_text)
                    max_echo_streak += 1
                    if max_echo_streak >= 2:
                        logger.warning("Multiple consecutive echoes — ending reply loop.")
                        break
                    user_text = None
                else:
                    max_echo_streak = 0

        except Exception as exc:
            logger.warning("Listen-for-reply failed: %s", exc)
        finally:
            if self._on_state_change:
                self._on_state_change("IDLE")
            if self._detector:
                try:
                    self._detector.resume()
                except Exception:
                    pass

    _TAG_STRIP_RE = re.compile(r"\[[A-Z_]+(?::[^\]]*)?]")

    def _is_echo(self, heard: str) -> bool:
        """Check if transcribed text is the pony's own TTS echoing back through the mic.

        Compares against recently spoken texts using normalized substring matching.
        """
        if not heard or not self._recently_spoken:
            return False
        h = heard.lower().strip()
        if len(h) < 5:
            return False
        for spoken in self._recently_spoken:
            s = spoken.lower().strip()
            # Direct substring: mic picked up part of the TTS output
            if h in s or s in h:
                logger.debug("Echo detected (substring): heard=%r vs spoken=%r", h, s)
                return True
            # Word overlap: if >60% of heard words appear in spoken text, it's echo
            h_words = set(h.split())
            s_words = set(s.split())
            if len(h_words) >= 3:
                overlap = len(h_words & s_words) / len(h_words)
                if overlap > 0.6:
                    logger.debug("Echo detected (%.0f%% overlap): heard=%r vs spoken=%r",
                                 overlap * 100, h, s)
                    return True
        return False

    def _execute_parsed_actions(self, parsed) -> None:
        """Execute robot actions and desktop commands from a ParsedResponse.

        This is the shared action executor for ALL codepaths — observation,
        spontaneous speech, reply handling, etc.  Any time the LLM returns
        a ParsedResponse with actions or desktop_commands, call this to
        actually execute them instead of silently discarding them.
        """
        # Robot actions (walk, wave, sit, etc.)
        if parsed.actions and self._robot:
            from robot.actions import RobotAction
            for action in parsed.actions:
                try:
                    if self._desktop:
                        self._desktop.execute_action(action)
                    self._robot.execute(action)
                    self._log_action(f"Action: {action.name}")
                except Exception as exc:
                    logger.debug("Parsed action %s failed: %s", action, exc)

        # Desktop commands ([DESKTOP:...] tags)
        if parsed.desktop_commands and self._desktop:
            for dc in parsed.desktop_commands:
                try:
                    self._desktop.execute_command(dc)
                    self._log_action(f"Desktop: {dc.command}:{':'.join(str(a) for a in dc.args)}")
                except Exception as exc:
                    logger.debug("Parsed desktop command %s failed: %s", dc.command, exc)

        # Move to screen region
        if parsed.moveto_region and self._robot:
            try:
                self._robot.on_move_to(parsed.moveto_region)
                self._log_action(f"MoveTo: {parsed.moveto_region}")
            except Exception as exc:
                logger.debug("MoveTo %s failed: %s", parsed.moveto_region, exc)

        # Persist animation
        if parsed.persist_seconds and self._robot:
            try:
                from desktop_pet.pet_controller import _ACTION_ANIMATION_MAP
                anim = "stand"
                if parsed.actions:
                    anim = _ACTION_ANIMATION_MAP.get(parsed.actions[0], "stand")
                self._robot.on_timed_override(anim, parsed.persist_seconds)
            except Exception as exc:
                logger.debug("Persist failed: %s", exc)

    def _speak(self, text: str) -> None:
        """Speak text via TTS with detector pause/resume and GUI callbacks.

        Blocks until TTS playback finishes so the caller can safely open the
        mic afterward without racing the audio output.
        """
        # Strip any bracket tags the LLM leaked (e.g. [CONVO:CONTINUE])
        text = self._TAG_STRIP_RE.sub("", text).strip()
        if not text:
            return
        # Track for echo detection
        self._recently_spoken.append(text)
        if len(self._recently_spoken) > 5:
            self._recently_spoken.pop(0)
        try:
            if self._on_state_change:
                self._on_state_change("SPEAK")
            _bubble_shown = False
            def _show_bubble():
                nonlocal _bubble_shown
                if _bubble_shown:
                    return
                _bubble_shown = True
                if self._on_speech_text:
                    self._on_speech_text(text)
            # Show bubble IMMEDIATELY — don't wait for TTS callback chain.
            # The on_start callback from TTS is a backup (dedup flag prevents
            # double-show).  Without this, the bubble only appears after the
            # TTS HTTP request completes, which can be seconds.
            _show_bubble()
            tts_on = self._tts_config.enabled if self._tts_config else True
            if tts_on:
                if self._tts_queue:
                    from core.tts_queue import PRIORITY_AUTONOMOUS
                    self._tts_queue.enqueue(
                        text,
                        priority=PRIORITY_AUTONOMOUS,
                        voice_slug=self._primary_voice_slug,
                        on_start=_show_bubble,
                        blocking=True,
                    )
                else:
                    # Direct TTS — pause detector during playback
                    if self._detector:
                        self._detector.pause()
                    try:
                        self._tts.speak(text, on_playback_start=_show_bubble)
                    finally:
                        if self._detector:
                            try:
                                self._detector.resume()
                            except Exception:
                                pass
            if self._timeline:
                from core.event_timeline import EventType
                self._timeline.append(EventType.AGENT_SPOKE, f'Pony said: "{text[:120]}"')
        finally:
            if self._on_state_change:
                self._on_state_change("IDLE")

    # ── Watch mode reactions ─────────────────────────────────────────────

    # ── Occasional screenshot (supplements win32gui) ────────────────────

    def _maybe_grab_screenshot(self) -> Optional[str]:
        """Occasionally take a screenshot for richer context. Returns description or None.

        Respects vision.screen_vision setting:
        - "moondream": use local model (only if loaded), fall back to API
        - "api": always use the main LLM's describe_screen
        """
        if self._screen is None:
            return None
        if not getattr(self._screen, "available", False):
            return None
        try:
            use_moondream = (
                self._vision_config
                and getattr(self._vision_config, "screen_vision", "api") == "moondream"
            )
            # Use higher JPEG quality for local moondream (needs readable text)
            quality = 85 if use_moondream else 60
            jpeg = self._screen.grab(quality=quality)
            if jpeg is None:
                return None

            description = None
            if use_moondream and self._moondream and self._moondream.loaded:
                description = self._moondream.describe(jpeg)
            elif self._vision_llm:
                description = self._vision_llm.describe_screen(jpeg)
            elif hasattr(self._llm, "describe_screen"):
                description = self._llm.describe_screen(jpeg)

            if description:
                logger.info("Agent screenshot: %s", description)
            return description
        except Exception as exc:
            logger.warning("Agent screenshot failed: %s", exc)
            return None

    # ── Timer system ─────────────────────────────────────────────────────

    def _check_timers(self) -> None:
        """Check if any time-triggered directives should fire now.

        Uses >= with a 10-minute window so ticks that skip the exact minute
        still catch the timer.  Fires 5 minutes early as a heads-up, then
        the directive becomes active at the actual trigger time.
        """
        # Respect the 60-second cooldown after directives were cleared
        if (time.monotonic() - self._directives_cleared_at) < 60.0:
            return
        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        for d in self.directives:
            if not d.trigger_time or d.triggered:
                continue
            # Parse trigger time to minutes
            try:
                th, tm = d.trigger_time.split(":")
                trigger_minutes = int(th) * 60 + int(tm)
            except (ValueError, AttributeError):
                continue

            # Heads-up fires 5 min early; actual fires at trigger time.
            # Use a 10-minute window (>=) so we never miss a tick.
            headsup_at = trigger_minutes - 5
            if headsup_at < 0:
                headsup_at += 24 * 60

            def _in_window(target: int) -> bool:
                """True if now_minutes is in [target, target+10) with midnight wrap."""
                diff = (now_minutes - target) % (24 * 60)
                return 0 <= diff < 10

            if _in_window(trigger_minutes):
                # Actual fire time (or past it within window)
                d.triggered = True
                d.urgency = 7
                d.created_at = time.monotonic()
                d.next_nag_at = time.monotonic()  # immediately due for nags
                logger.info("Timer fired: %s -> %r", d.trigger_time, d.goal)
                text = self._timer_speak(d.goal, d.trigger_time, headsup=False)
                self._llm.inject_history(
                    f"(Timer alert: it's {d.trigger_time}. Goal: {d.goal})",
                    text,
                )
                self._log_action(f"Timer fired at {d.trigger_time}: {d.goal}")
                self.save_directives()
                self._listen_for_reply()
            elif _in_window(headsup_at) and headsup_at != trigger_minutes:
                # Heads-up 5 min early — activate but delay first nag until trigger time
                d.triggered = True
                d.urgency = 7
                d.created_at = time.monotonic()
                d.next_nag_at = time.monotonic() + 5 * 60  # don't nag again until actual time
                logger.info("Timer heads-up (5 min early): %s -> %r", d.trigger_time, d.goal)
                text = self._timer_speak(d.goal, d.trigger_time, headsup=True)
                self._llm.inject_history(
                    f"(Timer heads-up: {d.goal} in 5 minutes, at {d.trigger_time})",
                    text,
                )
                self._log_action(f"Timer heads-up at {d.trigger_time}: {d.goal}")
                self.save_directives()
                self._listen_for_reply()

    # ── Recurring routines ─────────────────────────────────────────────────

    def _check_routines(self) -> None:
        """Fire any due recurring directives. Wake/sleep tracking is done in tick()."""
        # Respect the 60-second cooldown after directives were cleared
        now = time.monotonic()
        if (now - self._directives_cleared_at) < 60.0:
            return
        wake = getattr(self, "_last_wake_event", None) == "wake"
        due = self.routine_manager.get_due_routines(wake_event=wake)
        if wake:
            self._last_wake_event = None  # consume the event
        if not due:
            return
        # Batch all due routines into one LLM call for natural announcement
        name = get_character_name()
        goals = [r.goal for r in due]
        for r in due:
            schedule_desc = self.routine_manager.describe_routine(r)
            self.add_directive(r.goal, r.urgency, source=f"routine:{r.schedule}")
            self._log_action(f"Routine fired [{schedule_desc}]: {r.goal}")
            logger.info("Routine fired [%s]: %r", schedule_desc, r.goal)
        # LLM-generated announcement — varied and in-character
        try:
            if len(goals) == 1:
                prompt = (
                    f"You are {name}. It's time for the user's routine: \"{goals[0]}\". "
                    f"Tell them in ONE sentence, in character. Be natural and varied — "
                    f"don't just say 'time to do X'. Be creative, caring, or playful."
                )
            else:
                goals_str = ", ".join(f'"{g}"' for g in goals)
                prompt = (
                    f"You are {name}. Multiple routines just fired: {goals_str}. "
                    f"Announce them naturally in 1-2 sentences. Don't just list them — "
                    f"be in-character, creative. Maybe prioritize, maybe be playful about it."
                )
            text = self._llm.generate_once(prompt, max_tokens=100)
            if text:
                text = self._strip_think(text).strip().strip('"')
            if not text:
                text = f"hey — {', '.join(goals)}. let's go."
        except Exception:
            text = f"hey — {', '.join(goals)}. let's go."
        self._speak(text)
        self._llm.inject_history(
            f"(Recurring routines fired: {', '.join(goals)})",
            text,
        )
        self._listen_for_reply()

    def _timer_speak(self, goal: str, trigger_time: str, headsup: bool) -> str:
        """Generate an in-character timer announcement via LLM."""
        name = get_character_name()
        now_str = datetime.now().strftime('%I:%M %p').lstrip('0')
        if headsup:
            prompt = (
                f"You are {name}. The user set a timer for {trigger_time} to: \"{goal}\". "
                f"It's {now_str} now — 5 minutes before the timer fires. "
                f"Give them a heads-up in ONE sentence, in character. Be natural."
            )
            fallback = f"hey, five minutes until you gotta {goal}."
        else:
            prompt = (
                f"You are {name}. The user set a timer for {trigger_time} to: \"{goal}\". "
                f"It's {now_str} — the timer just fired. "
                f"Tell them it's time in ONE sentence, in character. Be natural and urgent."
            )
            fallback = f"hey, it's {now_str}. time to {goal}."
        try:
            text = self._llm.generate_once(prompt, max_tokens=100)
            if text:
                text = self._strip_think(text).strip().strip('"')
            if not text:
                text = fallback
        except Exception:
            text = fallback
        self._speak(text)
        return text

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _hide_countdown(self) -> None:
        """Hide the countdown timer on the pet."""
        if self._robot and hasattr(self._robot, 'countdown_stop'):
            self._robot.countdown_stop.emit()

    def _log_action(self, description: str) -> None:
        """Record an action for context in future ticks."""
        self._action_log.append((time.monotonic(), description))
        if len(self._action_log) > 15:
            self._action_log = self._action_log[-15:]

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the outermost JSON object from text, handling nested braces."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 5:
            return "a few seconds"
        elif seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            m = seconds / 60
            return f"{m:.0f} minute{'s' if m >= 2 else ''}"
        else:
            h = seconds / 3600
            return f"{h:.1f} hour{'s' if h >= 2 else ''}"
