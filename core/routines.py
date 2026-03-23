"""
Recurring directives — persistent routines that fire on a schedule.

Schedule types:
  - "on_wake"    — fires when user returns after being idle/asleep
  - "on_sleep"   — fires ~N hours after wake-up (default 8), i.e. "nighttime"
  - "daily"      — fires once per day at a wall-clock time (HH:MM)
  - "weekly"     — fires once per week at a day + time
  - "interval"   — fires every N hours

Routines are saved to routines.json and survive restarts.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_ROUTINES_FILE = Path(__file__).parent.parent / "routines.json"
_WAKE_STATE_FILE = Path(__file__).parent.parent / "wake_state.json"

# How long the user must be idle before we consider them "asleep/away"
AWAY_THRESHOLD_MS = 3 * 60 * 1000   # 3 minutes
# If watching fullscreen media, allow much longer idle before "away"
MEDIA_AWAY_THRESHOLD_MS = 30 * 60 * 1000  # 30 minutes (probably fell asleep)


@dataclass
class Routine:
    id: str
    goal: str
    urgency: int               # 1-10
    schedule: str              # "on_wake", "on_sleep", "daily", "weekly", "interval"
    time: Optional[str] = None           # HH:MM for daily/weekly
    day: Optional[str] = None            # lowercase day name for weekly ("monday", etc.)
    interval_hours: Optional[float] = None  # for interval type
    sleep_offset_hours: float = 8.0      # hours after wake-up for on_sleep
    enabled: bool = True
    last_fired_date: Optional[str] = None  # ISO date (YYYY-MM-DD) — prevents double-fire per day
    last_fired_ts: Optional[str] = None    # ISO datetime — for interval tracking

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Routine":
        # Handle legacy or missing fields gracefully
        return Routine(
            id=d.get("id", str(uuid.uuid4())[:8]),
            goal=d.get("goal", ""),
            urgency=d.get("urgency", 5),
            schedule=d.get("schedule", "daily"),
            time=d.get("time"),
            day=d.get("day"),
            interval_hours=d.get("interval_hours"),
            sleep_offset_hours=d.get("sleep_offset_hours", 8.0),
            enabled=d.get("enabled", True),
            last_fired_date=d.get("last_fired_date"),
            last_fired_ts=d.get("last_fired_ts"),
        )


class RoutineManager:
    """Manages recurring directives with persistence and schedule evaluation."""

    def __init__(self) -> None:
        self.routines: List[Routine] = []
        self._wake_time: Optional[datetime] = None   # when the user last "woke up"
        self._was_away: bool = True                   # default; overridden by _load_wake_state
        self._away_since: Optional[datetime] = None   # when the user went away
        self._last_state_save: float = 0.0            # throttle disk writes
        self._load()
        self._load_wake_state()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _ROUTINES_FILE.exists():
            self.routines = []
            return
        try:
            data = json.loads(_ROUTINES_FILE.read_text(encoding="utf-8"))
            self.routines = [Routine.from_dict(r) for r in data]
            logger.info("Loaded %d routines from %s", len(self.routines), _ROUTINES_FILE)
        except Exception as exc:
            logger.warning("Failed to load routines: %s", exc)
            self.routines = []

    def save(self) -> None:
        try:
            _ROUTINES_FILE.write_text(
                json.dumps([r.to_dict() for r in self.routines], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save routines: %s", exc)

    # ── Wake state persistence ──────────────────────────────────────────

    def _load_wake_state(self) -> None:
        """Load wake_time and last_active from disk to survive restarts."""
        if not _WAKE_STATE_FILE.exists():
            return
        try:
            data = json.loads(_WAKE_STATE_FILE.read_text(encoding="utf-8"))
            if data.get("wake_time"):
                self._wake_time = datetime.fromisoformat(data["wake_time"])
                logger.info("Restored wake_time from disk: %s", self._wake_time.strftime("%H:%M"))
            if data.get("last_active"):
                last = datetime.fromisoformat(data["last_active"])
                elapsed_ms = (datetime.now() - last).total_seconds() * 1000
                if elapsed_ms < AWAY_THRESHOLD_MS:
                    # User was active recently — don't trigger false wake on restart
                    self._was_away = False
                    logger.info("User was active %.0fs ago — no false wake.", elapsed_ms / 1000)
                else:
                    self._was_away = True
                    logger.info("User was away %.0fs — next activity = wake.", elapsed_ms / 1000)
        except Exception as exc:
            logger.warning("Failed to load wake state: %s", exc)

    def _save_wake_state(self) -> None:
        """Save wake_time and last_active to disk."""
        try:
            data = {
                "wake_time": self._wake_time.isoformat() if self._wake_time else None,
                "last_active": datetime.now().isoformat(),
            }
            _WAKE_STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(self, routine: Routine) -> None:
        self.routines.append(routine)
        self.save()
        logger.info("Routine added: %s (%s)", routine.goal, routine.schedule)

    def remove(self, routine_id: str) -> bool:
        before = len(self.routines)
        self.routines = [r for r in self.routines if r.id != routine_id]
        if len(self.routines) < before:
            self.save()
            return True
        return False

    def toggle(self, routine_id: str) -> None:
        for r in self.routines:
            if r.id == routine_id:
                r.enabled = not r.enabled
                self.save()
                return

    # ── Wake/sleep detection ─────────────────────────────────────────────

    def update_activity(self, idle_ms: int, media_active: bool = False) -> Optional[str]:
        """Call every tick with current idle time.

        Returns "wake" if user just woke up, None otherwise.
        Updates internal wake/sleep tracking.

        If media_active is True (user watching fullscreen video), use a much
        longer idle threshold — they're present, just not touching input.
        """
        import time as _time

        threshold = MEDIA_AWAY_THRESHOLD_MS if media_active else AWAY_THRESHOLD_MS
        is_away = idle_ms > threshold

        if self._was_away and not is_away:
            # User just came back — they "woke up"
            self._wake_time = datetime.now()
            self._was_away = False
            self._save_wake_state()
            logger.info("User wake-up detected at %s", self._wake_time.strftime("%H:%M"))
            return "wake"

        if is_away and not self._was_away:
            self._was_away = True
            self._away_since = datetime.now()
            self._save_wake_state()
            return "away"

        # Throttle state saves to every ~30 seconds while user is active
        now = _time.monotonic()
        if not is_away and now - self._last_state_save > 30:
            self._last_state_save = now
            self._save_wake_state()

        return None

    @property
    def away_duration_s(self) -> Optional[float]:
        """How long the user was away (seconds).  Valid right after a wake event."""
        if self._away_since is None:
            return None
        return (datetime.now() - self._away_since).total_seconds()

    @property
    def wake_time(self) -> Optional[datetime]:
        return self._wake_time

    @property
    def hours_since_wake(self) -> Optional[float]:
        if self._wake_time is None:
            return None
        return (datetime.now() - self._wake_time).total_seconds() / 3600.0

    @property
    def is_user_away(self) -> bool:
        return self._was_away

    # ── Schedule evaluation ──────────────────────────────────────────────

    def get_due_routines(self, wake_event: bool = False) -> List[Routine]:
        """Return list of routines that should fire right now.

        Args:
            wake_event: True if the user just woke up this tick.
        """
        due: List[Routine] = []
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        now_hhmm = now.strftime("%H:%M")
        now_day = now.strftime("%A").lower()  # "monday", "tuesday", etc.

        for r in self.routines:
            if not r.enabled:
                continue

            fired = False

            if r.schedule == "on_wake":
                if wake_event and r.last_fired_date != today:
                    fired = True

            elif r.schedule == "on_sleep":
                # Fire N hours after wake-up (once per cycle)
                h = self.hours_since_wake
                if (h is not None and h >= r.sleep_offset_hours
                        and r.last_fired_date != today):
                    fired = True

            elif r.schedule == "daily":
                if r.time and r.time == now_hhmm and r.last_fired_date != today:
                    fired = True

            elif r.schedule == "weekly":
                if (r.day and r.day == now_day
                        and r.time and r.time == now_hhmm
                        and r.last_fired_date != today):
                    fired = True

            elif r.schedule == "interval":
                if r.interval_hours:
                    if r.last_fired_ts:
                        try:
                            last = datetime.fromisoformat(r.last_fired_ts)
                            elapsed_h = (now - last).total_seconds() / 3600.0
                            if elapsed_h >= r.interval_hours:
                                fired = True
                        except ValueError:
                            fired = True
                    else:
                        fired = True  # never fired before

            if fired:
                r.last_fired_date = today
                r.last_fired_ts = now.isoformat()
                due.append(r)

        if due:
            self.save()

        return due

    # ── Helpers ───────────────────────────────────────────────────────────

    def describe_routine(self, r: Routine) -> str:
        """Human-readable description of a routine's schedule."""
        if r.schedule == "on_wake":
            return "Every day when you wake up"
        elif r.schedule == "on_sleep":
            return f"~{r.sleep_offset_hours:.0f}h after waking up (bedtime)"
        elif r.schedule == "daily":
            return f"Daily at {r.time}"
        elif r.schedule == "weekly":
            return f"Every {r.day.title()} at {r.time}"
        elif r.schedule == "interval":
            return f"Every {r.interval_hours:.0f}h"
        return r.schedule
