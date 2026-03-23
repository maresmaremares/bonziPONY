"""Shared event timeline — bridges Pipeline and AgentLoop context.

Both systems append events; the agent's tick prompt serializes recent
events so the LLM has full context about what happened (conversations,
AFK transitions, enforcement, directives).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional


# ── Event types ───────────────────────────────────────────────────────────────

class EventType(Enum):
    # Conversation events (written by Pipeline)
    USER_SAID = auto()
    PONY_SAID = auto()
    CONVERSATION_START = auto()
    CONVERSATION_END = auto()

    # Agent events (written by AgentLoop)
    AGENT_SPOKE = auto()
    DIRECTIVE_CREATED = auto()
    DIRECTIVE_COMPLETED = auto()
    ENFORCEMENT_START = auto()
    ENFORCEMENT_COMPLETE = auto()

    # Activity events (written by AgentLoop tick)
    USER_WENT_AFK = auto()
    USER_RETURNED = auto()


class ActivityState(Enum):
    ACTIVE_WORKING = "active_working"
    ACTIVE_BROWSING = "active_browsing"
    ACTIVE_MEDIA = "active_media"
    ACTIVE_GAMING = "active_gaming"
    ACTIVE_CHATTING = "active_chatting"
    AFK_TASK = "afk_task"
    AFK_UNKNOWN = "afk_unknown"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    type: EventType
    timestamp: float            # time.monotonic()
    wall_time: str              # "14:30:05"
    summary: str                # human-readable description
    metadata: Dict = field(default_factory=dict)


@dataclass
class UserIntent:
    """What the user said they're going to do."""
    action: str                 # "eat lunch", "take a shower"
    stated_at: float            # monotonic time
    expected_duration_s: Optional[float] = None


# ── Timeline ─────────────────────────────────────────────────────────────────

class EventTimeline:
    """Thread-safe shared event log.  Both Pipeline and AgentLoop read/write."""

    MAX_EVENTS = 100

    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []
        self._lock = threading.Lock()
        self._activity_state: ActivityState = ActivityState.ACTIVE_BROWSING
        self._user_intent: Optional[UserIntent] = None
        self._afk_reason: Optional[str] = None

    # ── Write ─────────────────────────────────────────────────────────────

    def append(self, event_type: EventType, summary: str,
               metadata: Optional[Dict] = None) -> None:
        evt = TimelineEvent(
            type=event_type,
            timestamp=time.monotonic(),
            wall_time=datetime.now().strftime("%H:%M:%S"),
            summary=summary,
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(evt)
            if len(self._events) > self.MAX_EVENTS:
                self._events = self._events[-self.MAX_EVENTS:]

    def set_activity_state(self, state: ActivityState) -> None:
        with self._lock:
            self._activity_state = state

    def set_user_intent(self, intent: Optional[UserIntent]) -> None:
        with self._lock:
            self._user_intent = intent

    def set_afk_context(self, reason: Optional[str]) -> None:
        with self._lock:
            self._afk_reason = reason

    # ── Read ──────────────────────────────────────────────────────────────

    @property
    def activity_state(self) -> ActivityState:
        with self._lock:
            return self._activity_state

    @property
    def user_intent(self) -> Optional[UserIntent]:
        with self._lock:
            return self._user_intent

    @property
    def afk_reason(self) -> Optional[str]:
        with self._lock:
            return self._afk_reason

    def recent(self, n: int = 20) -> List[TimelineEvent]:
        with self._lock:
            return list(self._events[-n:])

    # ── Prompt formatting ─────────────────────────────────────────────────

    @staticmethod
    def _age_str(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s ago"
        if seconds < 3600:
            return f"{seconds / 60:.0f}m ago"
        return f"{seconds / 3600:.1f}h ago"

    def format_recent_for_prompt(self, n: int = 15) -> str:
        """Compact timeline string for LLM prompts."""
        events = self.recent(n * 2)  # grab extra, then deduplicate
        if not events:
            return "(no recent events)"

        now = time.monotonic()
        lines: List[str] = []
        for evt in events:
            age = self._age_str(now - evt.timestamp)
            lines.append(f"[{evt.wall_time}, {age}] {evt.summary}")
            if len(lines) >= n:
                break

        return "\n".join(lines)

    def get_recent_conversation_summary(self, max_exchanges: int = 5) -> str:
        """Extract recent conversation events for agent context."""
        convo_types = {
            EventType.USER_SAID, EventType.PONY_SAID, EventType.AGENT_SPOKE,
        }
        events = self.recent(50)
        convo = [e for e in events if e.type in convo_types]
        if not convo:
            return "(no recent conversation)"

        now = time.monotonic()
        lines: List[str] = []
        for evt in convo[-(max_exchanges * 2):]:
            age = self._age_str(now - evt.timestamp)
            lines.append(f"[{age}] {evt.summary}")
        return "\n".join(lines)
