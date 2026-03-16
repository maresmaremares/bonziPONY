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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
    triggered: bool = False            # has the timer fired?
    delayed: bool = False              # user already negotiated a delay once


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
    ) -> None:
        self._config = config
        self._monitor = screen_monitor
        self._llm = llm
        self._tts = tts
        self._tts_config = tts_config  # for checking tts.enabled
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

        self.directives: List[Directive] = []
        self._action_log: List[str] = []         # recent actions, capped at 15
        self._next_idle_check_at: float = time.monotonic() + 10.0  # for self-initiation/spontaneous
        self._last_self_check: float = 0.0
        self._next_spontaneous: float = time.monotonic() + random.uniform(
            self._config.spontaneous_speech_min_s, self._config.spontaneous_speech_max_s,
        )
        self._conversation_active = False
        self._enforcement = EnforcementMode()
        self._mess_mouse_count: int = 0  # grows duration each trigger
        self._last_wake_event: Optional[str] = None  # set by tick(), consumed by _check_routines()

        # Recurring routines (persistent across restarts)
        from core.routines import RoutineManager
        self.routine_manager = RoutineManager()

        # Load persistent state (directives, enforcement, timers)
        self._load_directives()

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
                d = Directive(
                    goal=dd["goal"],
                    urgency=dd.get("urgency", 5),
                    created_at=now,   # monotonic doesn't persist; reset to now
                    last_action_at=now,
                    next_nag_at=nag_at,
                    source=dd.get("source", "user"),
                    trigger_time=dd.get("trigger_time"),
                    triggered=dd.get("triggered", False),
                    delayed=dd.get("delayed", False),
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

            if self.directives:
                logger.info("Restored %d directive(s) from disk.", len(self.directives))
        except Exception as exc:
            logger.warning("Failed to load directives: %s", exc)

    def save_directives(self) -> None:
        """Save directives and enforcement state to disk."""
        try:
            dirs = []
            now = time.monotonic()
            for d in self.directives:
                dirs.append({
                    "goal": d.goal,
                    "urgency": d.urgency,
                    "next_nag_offset_s": max(0, d.next_nag_at - now),
                    "source": d.source,
                    "trigger_time": d.trigger_time,
                    "triggered": d.triggered,
                    "delayed": d.delayed,
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

            data = {
                "directives": dirs,
                "enforcement": enf_data,
                "saved_at": datetime.now().isoformat(),
            }
            _DIRECTIVES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
        if urgency >= 9:
            return random.uniform(15.0, 30.0)
        elif urgency >= 7:
            return random.uniform(60.0, 120.0)
        elif urgency >= 4:
            return random.uniform(180.0, 480.0)
        else:
            return random.uniform(600.0, 900.0)

    def add_directive(self, goal: str, urgency: int, source: str = "user") -> None:
        """Add a new directive (max ``max_directives``)."""
        if len(self.directives) >= self._config.max_directives:
            logger.warning("Max directives reached (%d) — ignoring new directive.", self._config.max_directives)
            return
        goal = self._clean_goal(goal)
        urgency = max(1, min(10, urgency))
        now = time.monotonic()
        nag_at = now + self._initial_nag_delay(urgency)
        d = Directive(goal=goal, urgency=urgency, created_at=now, last_action_at=now,
                      next_nag_at=nag_at, source=source)
        self.directives.append(d)
        self.save_directives()
        logger.info("Directive added [%s]: %r (urgency %d, first nag in %.0fs)",
                     source, goal, urgency, nag_at - now)

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
            self._hide_countdown()
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

    @property
    def has_directives(self) -> bool:
        return bool(self.directives)

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

    def _check_enforcement(self) -> None:
        """Enforcement mode — IMMEDIATELY detect any mouse/keyboard input, ask if done."""
        now = time.monotonic()
        elapsed = now - self._enforcement.start_time

        # Short grace period — user is walking away (10 seconds)
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
            print(f"[ENFORCEMENT] {elapsed:.0f}s/{self._enforcement.duration_s:.0f}s | idle={idle_ms}ms")

        # ANY input detected (idle < 3 seconds) — immediately ask
        if idle_ms < 3000:
            # Don't ask again if we asked recently (cooldown 30s)
            if (now - self._enforcement.last_caught_at) < 30.0:
                return
            self._enforcement.last_caught_at = now
            self._enforcement.caught_count += 1
            self._enforcement_ask_if_done()
            return

        # Timer expired and user is still away — periodic check-in
        if elapsed >= self._enforcement.duration_s and not self._enforcement.expired:
            self._enforcement.expired = True
            # Don't interrupt — they're away (idle). Just note it.
            logger.info("Enforcement timer expired for %r — user still away.", self._enforcement.directive_goal)

    # ── Enforcement response keywords ────────────────────────────────────

    _YES_KEYWORDS = ("yes", "yeah", "yep", "yup", "done", "did it", "finished",
                     "completed", "i did", "already did", "took care of it",
                     "all done", "i'm done", "it's done", "of course")
    _NO_KEYWORDS = ("no", "not yet", "nope", "haven't", "didn't", "not done",
                    "not really", "hold on", "give me a", "in a minute",
                    "in a sec", "almost", "working on it", "soon")

    def _enforcement_ask_if_done(self) -> None:
        """User touched mouse/keyboard during enforcement — ask if they completed the task.
        If yes → done. If no → LOCKDOWN."""
        goal = self._enforcement.directive_goal

        # Pause wake word detector for the entire enforcement interaction
        if self._detector:
            try:
                self._detector.pause()
            except Exception:
                pass

        try:
            # Ask if they did it
            self._speak(f"Hey, did you finish {goal}?")

            # Open mic and listen for response
            response = self._enforcement_listen()
            if response is None:
                # No response — ask again later
                logger.info("Enforcement: no response to 'did you finish it?'")
                return

            response_lower = response.lower()
            logger.info("Enforcement response: %r", response)

            # Check for affirmative
            if any(kw in response_lower for kw in self._YES_KEYWORDS):
                self._speak("nice, good job.")
                self._enforcement_complete()
                return

            # Check for negative or anything that's not clearly "yes"
            # If they said ANYTHING that isn't affirmative, it's lockdown time
            is_negative = any(kw in response_lower for kw in self._NO_KEYWORDS)
            if is_negative or not any(kw in response_lower for kw in self._YES_KEYWORDS):
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

    def _enforcement_complete(self) -> None:
        """User confirmed they did the task — remove directive and end enforcement."""
        goal = self._enforcement.directive_goal
        # Remove the matching directive
        for i, d in enumerate(self.directives):
            if d.goal == goal:
                self.directives.pop(i)
                logger.info("Enforcement completed: %r", goal)
                break
        self._enforcement = EnforcementMode()
        self._hide_countdown()
        self.save_directives()
        self._llm.inject_history(
            f"(User confirmed they completed \"{goal}\" during enforcement.)",
            "nice, good job.",
        )
        self._log_action(f"Enforcement completed: \"{goal[:40]}\"")
        if not self.directives:
            self._mess_mouse_count = 0

    def _enforcement_lockdown(self) -> None:
        """User said they HAVEN'T done it — full lockdown until they go do it."""
        goal = self._enforcement.directive_goal

        # Nuke urgency to 10
        for d in self.directives:
            if d.goal == goal:
                d.urgency = 10
                break
        self.save_directives()

        # Announce lockdown
        self._speak(
            f"not yet? well, I'm gonna lock your computer until you do. "
            f"if you want it back, you gotta go {goal}."
        )
        self._llm.inject_history(
            f"(User said they haven't done \"{goal}\" — entering lockdown mode.)",
            f"not yet? well, I'm gonna lock your computer until you do. "
            f"if you want it back, you gotta go {goal}.",
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

                # Minimize all windows
                if self._desktop:
                    self._desktop.minimize_all_windows()

                # Mess with mouse for 10 seconds (at urgency 10, mouse is locked anyway)
                if self._desktop and not cursor_locked:
                    self._desktop.mess_with_mouse(duration=10.0, jitter=120)
                elif cursor_locked:
                    time.sleep(10.0)  # mouse is locked, just wait

                # Brief pause, then ask again
                time.sleep(1.0)

                if lockdown_round % 2 == 0:
                    self._speak(f"have you done it yet? go {goal}!")
                else:
                    if cursor_locked:
                        self._speak(f"if you want control back, you gotta go {goal}!")
                    else:
                        self._speak("I'm not giving your computer back until you do it.")

                response = self._enforcement_listen(timeout=6.0)
                if response:
                    response_lower = response.lower()
                    if any(kw in response_lower for kw in self._YES_KEYWORDS):
                        self._speak("finally. okay, you're free.")
                        self._enforcement_complete()
                        return
                    _STOP = ("stop", "quit", "enough", "shut up", "knock it off", "cut it out")
                    if any(kw in response_lower for kw in _STOP):
                        self._speak("ugh, fine. but you still need to do it.")
                        self._enforcement = EnforcementMode()
                        self._hide_countdown()
                        self.save_directives()
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

    def tick(self) -> None:
        """Called every ~1s from the pipeline thread. Decides if anything needs doing."""
        # Always track idle/wake state — needed for sleep detection even during conversation
        idle_ms = _get_idle_ms()
        away_dur = self.routine_manager.away_duration_s  # grab BEFORE update clears it
        self._last_wake_event = self.routine_manager.update_activity(idle_ms)

        # Welcome-back greeting when user returns from AFK
        if self._last_wake_event == "wake" and not self._conversation_active:
            self._welcome_back(away_dur)

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

        # Don't do anything proactive while user is asleep/away
        if self.routine_manager.is_user_away:
            return

        # Per-directive timing: check if ANY directive is due for a nag
        actionable = [d for d in self.directives
                      if not (d.trigger_time and not d.triggered)]
        due = [d for d in actionable if d.next_nag_at <= now]
        if due:
            self._execute_tick()
            return

        # No directives due — handle idle behavior (self-initiation, spontaneous speech)
        if now < self._next_idle_check_at:
            return

        if not actionable:
            # No active directives — check self-initiation and spontaneous speech
            if self._config.self_initiate:
                if (now - self._last_self_check) >= self._config.self_initiate_interval_s:
                    self._last_self_check = now
                    self._maybe_self_initiate()
                    return

            # Spontaneous speech on its own dedicated timer (every 3-8 min)
            if now >= self._next_spontaneous:
                next_in = random.uniform(
                    self._config.spontaneous_speech_min_s,
                    self._config.spontaneous_speech_max_s,
                )
                print(f"\n[Agent] Spontaneous speech triggered (next in {next_in:.0f}s)", flush=True)
                self._spontaneous_speech()
                self._next_spontaneous = time.monotonic() + next_in
            else:
                self._next_idle_check_at = now + min(
                    self._config.base_check_interval_s,
                    self._next_spontaneous - now + 1.0,
                )

    # ── Core tick execution ────────────────────────────────────────────────

    def _execute_tick(self) -> None:
        """Fire an LLM call for active directives."""
        try:
            state = self._monitor.get_state()
            directives_str = ", ".join(f'"{d.goal}" (urg {d.urgency})' for d in self.directives)
            print(f"\n[Agent] Directive tick — active: {directives_str}", flush=True)
            prompt = self._build_tick_prompt(state)

            logger.debug("Agent tick prompt: %s", prompt[:200])
            raw = self._llm.generate_once(prompt, max_tokens=1024)
            logger.debug("Agent tick response: %s", raw[:300])

            decision = self._parse_decision(raw)
            self._execute_decision(decision, state)

            # Apply per-directive timings from LLM
            now_m = time.monotonic()
            actionable = [d for d in self.directives
                          if not (d.trigger_time and not d.triggered)]
            for i, d in enumerate(actionable):
                key = str(i)
                if key in decision.directive_timings:
                    timing = decision.directive_timings[key]
                    nag_min = float(timing.get("next_nag_minutes", 10))
                    d.next_nag_at = now_m + nag_min * 60.0
                    if "urgency" in timing:
                        d.urgency = max(1, min(10, int(timing["urgency"])))
                else:
                    # LLM didn't mention this directive — default 10 min
                    if d.next_nag_at <= now_m:
                        d.next_nag_at = now_m + 600.0
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
        fg_title = fg.title if fg else "unknown"
        fg_exe = fg.exe_name if fg else None
        fg_dur = self._fmt_duration(state.foreground_duration_s)
        fg_fullscreen = " (FULLSCREEN)" if fg and fg.is_fullscreen else ""

        # Build window list with exe names for context
        window_entries = []
        for w in state.open_windows[:20]:
            entry = w.title
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
                lines.append(
                    f'{i}. "{d.goal}" [urgency {d.urgency}/10, active {age}, '
                    f'last nag {since_nag} ago, source: {d.source}{timer_info}{delay_info}]'
                )

        # Recent actions
        if self._action_log:
            lines.append("")
            lines.append("YOUR RECENT ACTIONS:")
            for entry in self._action_log[-5:]:
                lines.append(f"- {entry}")

        # Instructions
        lines.extend([
            "",
            "THINK FIRST: Before your JSON response, reason through your decision in <think>...</think> tags.",
            "In your thinking, consider: What is the user doing? How long? Should I intervene or wait?",
            "What's the right tone? What actions make sense at this urgency level? Is this actually a distraction?",
            "Your thinking is PRIVATE — never spoken aloud. Only the JSON fields are executed.",
            "",
            "After your <think> block, respond with a JSON object:",
            '{"speak":"text or null","actions":[],"desktop_commands":[],"create_directive":null,"complete_directive":null,"directives":{"0":{"next_nag_minutes":15,"urgency":5}}}',
            "",
            "Field guide:",
            '- speak: short sentence to say out loud (TTS), or null to stay quiet',
            '- actions: list of action names like "CLOSE_WINDOW", "MINIMIZE_WINDOW"',
            '- desktop_commands: list of objects with "command" and "args" fields:',
            '  - {"command":"CLOSE_TITLE","args":["substring"]} — close window by title',
            '  - {"command":"MINIMIZE_TITLE","args":["substring"]} — minimize by title (only works on maximized/large windows)',
            '  - {"command":"SHAKE_TITLE","args":["substring"]} — SHAKE/VIBRATE a window violently (only targets maximized/large windows)',
            '  - {"command":"SHAKE_ALL","args":[]} — shake all MAXIMIZED windows (earthquake mode, high urgency only)',
            '  - {"command":"MESS_MOUSE","args":[]} — grab the cursor and run around with it (high urgency only)',
            '  - {"command":"PAUSE_MEDIA","args":[]} — press play/pause key (use for YouTube, Spotify, media apps instead of minimizing)',
            '  - {"command":"GOOGLE_IMAGES","args":["search term"]} — open Google Images for the thing they should be doing (e.g. "gym motivation", "healthy food", "sleeping peacefully")',
            '  - {"command":"ALT_TAB","args":[]} — Win+D: MINIMIZE ALL WINDOWS and show desktop (nuclear option)',
            '  - {"command":"OPEN","args":["app_name"]}',
            '  - {"command":"BROWSE","args":["url"]}',
            '  - {"command":"WRITE_NOTEPAD","args":["content with \\n for newlines"]} — open Notepad and write content (lists, routines, plans, notes)',
            '- create_directive: {"goal":"...","urgency":1-10} or null — goal must be a DIRECT ACTION like "eat food", "go to sleep", "do homework". NEVER write "remind user to" or "get user to" — just the action itself.',
            '- complete_directive: index of directive to mark done, or null',
            '- directives: for EACH active directive by index, decide timing and urgency:',
            '  {"0": {"next_nag_minutes": 15, "urgency": 5}, "1": {"next_nag_minutes": 45, "urgency": 3}}',
            '  Think about: What\'s the task? How urgent? How long since last nag? Is the user busy?',
            '  Would nagging NOW be effective? Low urgency = space it out (30-60 min). High urgency = nag often (1-5 min).',
            '  You can raise or lower urgency based on context. This replaces adjust_urgency.',
            "",
            "Rules:",
            "- If you have a directive, STAY FOCUSED on it. Escalate if user ignores you.",
            "- You can speak AND do actions in the same tick.",
            "- Escalation is GRADUATED. Follow this chaos roll system:",
            f"  CHAOS ROLL this tick: {random.randint(1, 100)}",
            "  Urgency 1-3: speak and nag only.",
            "  Urgency 4-5: nag harder. If chaos roll > 60, SHAKE a window (skip if fullscreen).",
            "  Urgency 6: GOOGLE_IMAGES the directive goal. SHAKE or ALT_TAB (minimizes ALL windows) if fullscreen.",
            "  Urgency 7: GOOGLE_IMAGES + ALT_TAB + PAUSE_MEDIA. If chaos roll > 40, MESS_MOUSE (grabs their cursor!).",
            "  Urgency 8: ALT_TAB + MESS_MOUSE. If chaos roll > 30, close/minimize windows.",
            "  Urgency 9-10: FULL NUCLEAR. ALT_TAB (minimize everything), MESS_MOUSE (grab cursor), close windows. Go wild.",
            "- IMPORTANT: SHAKE does NOT work on fullscreen apps/games. Use ALT_TAB (Win+D, minimizes ALL windows) instead.",
            "- The chaos roll adds unpredictability — sometimes you're chill, sometimes you snap early. Embrace it.",
            "- Be creative and in-character. You're a prankster with a mission.",
            "- CRITICAL: When speaking, talk DIRECTLY TO the user in second person. NEVER say 'remind the user' or 'get user to' — you ARE talking to them. Say 'go do it' not 'remind user to do it'.",
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

    def _execute_decision(self, decision: AgentDecision, state: ScreenState) -> None:
        """Execute the agent's decision: speak, act, manage directives."""
        now = time.monotonic()

        # ── Speak ──────────────────────────────────────────────────────────
        if decision.speak:
            try:
                self._speak(decision.speak)

                # Inject into LLM history so Dash remembers
                fg_title = state.foreground.title if state.foreground else "the screen"
                ctx = f"(You autonomously noticed: {fg_title}. You decided to speak up.)"
                self._llm.inject_history(ctx, decision.speak)

                self._log_action(f"Said: \"{decision.speak[:80]}\"")
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
                if not self.directives:
                    self._mess_mouse_count = 0
                self.save_directives()

        if decision.create_directive is not None:
            goal = decision.create_directive.get("goal", "")
            urgency = decision.create_directive.get("urgency", 5)
            if goal:
                self.add_directive(goal, urgency, source="self")

        # Update last_action_at on active directives if we did something
        if decision.speak or decision.actions or decision.desktop_commands:
            for d in self.directives:
                d.last_action_at = now

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

    # Window class names commonly used by game engines
    _GAME_CLASS_PATTERNS: List[str] = [
        "unrealwindow", "sdl_app", "unity", "godot", "pygame",
        "glfw", "allegro", "source engine", "cryengine",
        "gamemaker", "renpy",
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
            if decision.create_directive:
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

            next_s = max(self._config.min_check_interval_s, decision.next_check_seconds)
            self._next_idle_check_at = time.monotonic() + next_s

        except Exception as exc:
            logger.warning("Self-initiation check failed: %s", exc)
            self._next_idle_check_at = time.monotonic() + self._config.self_initiate_interval_s

    # ── Welcome-back greeting ────────────────────────────────────────────

    def _welcome_back(self, away_seconds: Optional[float]) -> None:
        """Greet the user when they return from AFK."""
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
            prompt = (
                f"(The user just came back after being away for {dur}. "
                f"Welcome them back in ONE short sentence as {name}. "
                f"Be casual — maybe comment on how long they were gone, "
                f"ask what they were up to, or just say hi. Keep it brief.)"
            )
            raw = self._llm.generate_once(prompt)
            if raw:
                from llm.response_parser import parse_response
                parsed = parse_response(raw)
                text = parsed.text or raw
                self._speak(text)
                self._llm.inject_history(
                    f"(User returned after being away for {dur}.)",
                    text,
                )
                self._log_action(f"Welcome back after {dur}")
        except Exception as exc:
            logger.warning("Welcome-back greeting failed: %s", exc)

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

            # 30% chance: comment on what's on screen, 70%: use an idle prompt
            if state.foreground and random.random() < 0.3:
                fg = state.foreground.title
                exe = state.foreground.exe_name or "unknown app"
                fullscreen = " (FULLSCREEN)" if state.foreground.is_fullscreen else ""
                screen_extra = ""
                if random.random() < 0.3:
                    desc = self._maybe_grab_screenshot()
                    if desc:
                        screen_extra = f" You can see on screen: {desc}."
                trigger = (
                    f"(You glanced at the user's screen. They have \"{fg}\" ({exe}{fullscreen}) open.{screen_extra} "
                    f"React in ONE short sentence as {get_character_name()}. Be natural — sometimes "
                    "comment on it, sometimes ignore it and say something random.)"
                )
            else:
                trigger = f"(Spontaneous thought — {prompt_choice})"

            print(f"[Agent] Prompt: {trigger[:100]}...", flush=True)
            raw = self._llm.chat(trigger)
            if raw:
                from llm.response_parser import parse_response
                parsed = parse_response(raw)
                if parsed.text:
                    print(f"[Agent] Speaking: \"{parsed.text[:80]}\"", flush=True)
                    self._speak(parsed.text)
                    self._log_action(f"Spontaneous: \"{parsed.text[:60]}\"")

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
            if self._on_state_change:
                self._on_state_change("LISTEN")

            # Listen with a short timeout — don't wait too long for a reply
            user_text = self._transcriber.listen(
                speech_start_timeout_s=5.0,
                initial_discard_ms=600,
            )

            if not user_text or not user_text.strip():
                logger.debug("No reply to spontaneous speech — moving on.")
                return

            # Filter Whisper hallucinations
            from stt.transcriber import _is_whisper_hallucination
            if _is_whisper_hallucination(user_text):
                logger.debug("Filtered hallucination in listen_for_reply: %r", user_text)
                return

            logger.info("User replied to spontaneous speech: %r", user_text)

            # Conversational loop — keep going until CONVO:END or silence
            from llm.response_parser import parse_response
            while user_text and user_text.strip():
                # Add context so the LLM stays in character (matches pipeline._run_turn)
                enriched = user_text
                enriched += (
                    f"\n\n[System hint: You are {get_character_name()}. Stay in character. "
                    "Reply naturally as yourself — do NOT break character or meta-analyze. "
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
                    def _show_bubble(t=parsed.text):
                        if self._on_speech_text:
                            self._on_speech_text(t)
                    tts_on = self._tts_config.enabled if self._tts_config else True
                    if tts_on:
                        self._tts.speak(parsed.text, on_playback_start=_show_bubble)
                    else:
                        _show_bubble()

                # Check for conversation end signal
                if parsed.end_conversation:
                    logger.debug("Spontaneous conversation ended by LLM.")
                    break

                # Listen again (short timeout, filter hallucinations)
                if self._on_state_change:
                    self._on_state_change("LISTEN")
                user_text = self._transcriber.listen(
                    speech_start_timeout_s=5.0,
                    initial_discard_ms=600,
                )
                if user_text and _is_whisper_hallucination(user_text):
                    logger.debug("Filtered hallucination in reply loop: %r", user_text)
                    user_text = None

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

    def _speak(self, text: str) -> None:
        """Speak text via TTS with detector pause/resume and GUI callbacks."""
        # Strip any bracket tags the LLM leaked (e.g. [CONVO:CONTINUE])
        text = self._TAG_STRIP_RE.sub("", text).strip()
        if not text:
            return
        try:
            if self._detector:
                self._detector.pause()
            if self._on_state_change:
                self._on_state_change("SPEAK")
            def _show_bubble():
                if self._on_speech_text:
                    self._on_speech_text(text)
            tts_on = self._tts_config.enabled if self._tts_config else True
            if tts_on:
                self._tts.speak(text, on_playback_start=_show_bubble)
            else:
                _show_bubble()
        finally:
            if self._on_state_change:
                self._on_state_change("IDLE")
            if self._detector:
                try:
                    self._detector.resume()
                except Exception:
                    pass

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
            jpeg = self._screen.grab()
            if jpeg is None:
                return None

            use_moondream = (
                self._vision_config
                and getattr(self._vision_config, "screen_vision", "api") == "moondream"
            )
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
                logger.info("Timer fired: %s -> %r", d.trigger_time, d.goal)
                self._speak(f"Hey! It's {now.strftime('%I:%M %p')}! {d.goal}")
                self._llm.inject_history(
                    f"(Timer alert: it's {d.trigger_time}. Goal: {d.goal})",
                    f"Hey! It's {now.strftime('%I:%M %p')}! Time to {d.goal}!",
                )
                self._log_action(f"Timer fired at {d.trigger_time}: {d.goal}")
                self.save_directives()
            elif _in_window(headsup_at) and headsup_at != trigger_minutes:
                # Heads-up — mark triggered so we don't repeat
                d.triggered = True
                d.urgency = 7
                d.created_at = time.monotonic()
                logger.info("Timer heads-up (5 min early): %s -> %r", d.trigger_time, d.goal)
                self._speak(f"Hey! Heads up — in five minutes you gotta {d.goal}!")
                self._llm.inject_history(
                    f"(Timer heads-up: {d.goal} in 5 minutes, at {d.trigger_time})",
                    f"Hey! Heads up — in five minutes you gotta {d.goal}!",
                )
                self._log_action(f"Timer heads-up at {d.trigger_time}: {d.goal}")
                self.save_directives()

    # ── Recurring routines ─────────────────────────────────────────────────

    def _check_routines(self) -> None:
        """Fire any due recurring directives. Wake/sleep tracking is done in tick()."""
        wake = getattr(self, "_last_wake_event", None) == "wake"
        due = self.routine_manager.get_due_routines(wake_event=wake)
        if wake:
            self._last_wake_event = None  # consume the event
        for r in due:
            schedule_desc = self.routine_manager.describe_routine(r)
            self.add_directive(r.goal, r.urgency, source=f"routine:{r.schedule}")
            self._speak(f"Hey! Routine reminder: {r.goal}")
            self._llm.inject_history(
                f"(Recurring routine fired [{schedule_desc}]: {r.goal})",
                f"Hey! Routine reminder — {r.goal}!",
            )
            self._log_action(f"Routine fired [{schedule_desc}]: {r.goal}")
            logger.info("Routine fired [%s]: %r", schedule_desc, r.goal)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _hide_countdown(self) -> None:
        """Hide the countdown timer on the pet."""
        if self._robot and hasattr(self._robot, 'countdown_stop'):
            self._robot.countdown_stop.emit()

    def _log_action(self, description: str) -> None:
        """Record an action for context in future ticks."""
        elapsed = self._fmt_duration(0)  # "just now"
        self._action_log.append(f"{elapsed}: {description}")
        if len(self._action_log) > 15:
            self._action_log = self._action_log[-15:]
        # Update timestamps to relative
        self._refresh_action_timestamps()

    def _refresh_action_timestamps(self) -> None:
        """Noop for now — timestamps are set at creation time."""
        pass

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
            return "just now"
        elif seconds < 60:
            return f"{seconds:.0f}s ago"
        elif seconds < 3600:
            return f"{seconds / 60:.0f} min ago"
        else:
            return f"{seconds / 3600:.1f}h ago"
