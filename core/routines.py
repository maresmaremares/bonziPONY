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
AWAY_THRESHOLD_MS = 3 * 60 * 1000        # 3 minutes
# If watching fullscreen media, allow much longer idle before "away"
MEDIA_AWAY_THRESHOLD_MS = 30 * 60 * 1000  # 30 minutes
# If watching windowed media (YouTube in browser, VLC windowed, etc.)
WINDOWED_MEDIA_AWAY_THRESHOLD_MS = 10 * 60 * 1000  # 10 minutes


@dataclass
class Routine:
    id: str
    goal: str
    urgency: int               # 1-10
    schedule: str              # "on_wake", "on_sleep", "daily", "weekly", "interval"
    time: Optional[str] = None           # HH:MM for daily/weekly (base time)
    day: Optional[str] = None            # lowercase day name for weekly ("monday", etc.)
    interval_hours: Optional[float] = None  # for interval type
    sleep_offset_hours: float = 8.0      # hours after wake-up for on_sleep
    enabled: bool = True
    last_fired_date: Optional[str] = None  # ISO date (YYYY-MM-DD) — prevents double-fire per day
    last_fired_ts: Optional[str] = None    # ISO datetime — for interval tracking
    # ── Rich scheduling (per-day times + exclusions) ──
    day_times: Optional[dict] = None     # {"monday": "09:00", "friday": "15:00"} — overrides base time
    exclude_days: Optional[list] = None  # ["saturday", "sunday"] — skip these days

    def to_dict(self) -> dict:
        return asdict(self)

    def get_time_for_today(self) -> Optional[str]:
        """Return the scheduled time for today, considering day_times overrides and exclusions."""
        today = datetime.now().strftime("%A").lower()
        # Check exclusions first
        if self.exclude_days and today in self.exclude_days:
            return None
        # Check per-day override
        if self.day_times and today in self.day_times:
            return self.day_times[today]
        # Fall back to base time
        return self.time

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
            day_times=d.get("day_times"),
            exclude_days=d.get("exclude_days"),
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

    _SLEEP_GAP_S = 4 * 3600  # 4 hours — anything shorter is a restart, not sleep

    def _load_wake_state(self) -> None:
        """Load wake_time and last_active from disk to survive restarts.

        Distinguishes between *program restart during the day* (should NOT
        re-fire wake routines) and *first launch after sleep* (should fire).

        Rules:
        1. If wake_time is from today → already woke up, treat as restart.
        2. If last_active was < 4 hours ago → short gap, treat as restart.
        3. Otherwise → long gap, probably slept — trigger wake on next activity.
        """
        if not _WAKE_STATE_FILE.exists():
            return
        try:
            data = json.loads(_WAKE_STATE_FILE.read_text(encoding="utf-8"))
            now = datetime.now()

            saved_wake = None
            if data.get("wake_time"):
                saved_wake = datetime.fromisoformat(data["wake_time"])
                self._wake_time = saved_wake
                logger.info("Restored wake_time from disk: %s", saved_wake.strftime("%H:%M"))

            if data.get("last_active"):
                last = datetime.fromisoformat(data["last_active"])
                elapsed_s = (now - last).total_seconds()

                # Rule 1: already woke up today — just a restart
                if saved_wake and saved_wake.date() == now.date():
                    self._was_away = False
                    logger.info(
                        "Wake already recorded today at %s — restart, not wake.",
                        saved_wake.strftime("%H:%M"),
                    )
                # Rule 2: short gap — restart or brief absence
                elif elapsed_s < self._SLEEP_GAP_S:
                    self._was_away = False
                    logger.info("User was active %.0fs ago — restart, not wake.", elapsed_s)
                # Rule 3: long gap — probable sleep
                else:
                    self._was_away = True
                    self._wake_time = None  # clear stale wake so a fresh one fires
                    logger.info(
                        "User was away %.1fh — next activity = wake.", elapsed_s / 3600,
                    )
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

    def update_activity(self, idle_ms: int, media_active: bool = False,
                        windowed_media_active: bool = False) -> Optional[str]:
        """Call every tick with current idle time.

        Returns "wake" if user just woke up, None otherwise.
        Updates internal wake/sleep tracking.

        media_active: fullscreen video — 30 min threshold
        windowed_media_active: windowed YouTube/VLC/etc — 10 min threshold
        """
        import time as _time

        if media_active:
            threshold = MEDIA_AWAY_THRESHOLD_MS
        elif windowed_media_active:
            threshold = WINDOWED_MEDIA_AWAY_THRESHOLD_MS
        else:
            threshold = AWAY_THRESHOLD_MS
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
            if r.last_fired_date == today:
                continue  # already fired today

            fired = False

            if r.schedule == "on_wake":
                if wake_event:
                    fired = True

            elif r.schedule == "on_sleep":
                h = self.hours_since_wake
                if h is not None and h >= r.sleep_offset_hours:
                    fired = True

            elif r.schedule == "daily":
                # Check exclusions
                if r.exclude_days and now_day in r.exclude_days:
                    # Excluded day — but check if day_times has an override
                    if not (r.day_times and now_day in r.day_times):
                        continue

                # Determine today's target time
                target_time = r.get_time_for_today()
                if target_time and target_time <= now_hhmm:
                    fired = True

            elif r.schedule == "weekly":
                if r.day_times:
                    # Rich schedule: check if today is in day_times
                    if now_day in r.day_times:
                        target_time = r.day_times[now_day]
                        if target_time <= now_hhmm:
                            fired = True
                else:
                    # Simple weekly: single day + time
                    if r.day and r.day == now_day and r.time and r.time <= now_hhmm:
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

    def add_if_unique(self, routine: Routine) -> bool:
        """Add a routine only if no existing routine has the same goal.

        Returns True if added, False if a duplicate exists.
        """
        goal_lower = routine.goal.lower().strip()
        for existing in self.routines:
            if existing.goal.lower().strip() == goal_lower:
                # Duplicate — update urgency if higher, merge day_times
                if routine.urgency > existing.urgency:
                    existing.urgency = routine.urgency
                if routine.day_times:
                    if existing.day_times is None:
                        existing.day_times = {}
                    existing.day_times.update(routine.day_times)
                if routine.exclude_days:
                    if existing.exclude_days is None:
                        existing.exclude_days = []
                    for d in routine.exclude_days:
                        if d not in existing.exclude_days:
                            existing.exclude_days.append(d)
                self.save()
                logger.info("Routine merged into existing: %s", routine.goal)
                return False
        self.add(routine)
        return True

    def describe_routine(self, r: Routine) -> str:
        """Human-readable description of a routine's schedule."""
        if r.schedule == "on_wake":
            return "Every day when you wake up"
        elif r.schedule == "on_sleep":
            return f"~{r.sleep_offset_hours:.0f}h after waking up (bedtime)"
        elif r.schedule == "daily":
            desc = f"Daily at {r.time}"
            if r.day_times:
                overrides = ", ".join(f"{d.title()} at {t}" for d, t in r.day_times.items())
                desc += f" (except: {overrides})"
            if r.exclude_days:
                exc = ", ".join(d.title() for d in r.exclude_days)
                desc += f" (skip {exc})"
            return desc
        elif r.schedule == "weekly":
            if r.day_times:
                parts = [f"{d.title()} at {t}" for d, t in sorted(r.day_times.items())]
                return "Every " + ", ".join(parts)
            return f"Every {r.day.title() if r.day else '?'} at {r.time}"
        elif r.schedule == "interval":
            return f"Every {r.interval_hours:.0f}h" if r.interval_hours else "Recurring"
        return r.schedule


# ── Collapse multiple RoutineTags into rich Routines ────────────────────────

_ALL_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def collapse_routine_tags(tags) -> List[Routine]:
    """Collapse multiple RoutineTag objects with the same goal into rich Routines.

    When the LLM outputs multiple [ROUTINE:weekly:shower:5:tuesday:15:00] and
    [ROUTINE:weekly:shower:5:wednesday:16:00], this merges them into a single
    Routine with day_times = {"tuesday": "15:00", "wednesday": "16:00"}.

    When a daily tag and weekly overrides share the same goal, the daily base
    time is set and weekly tags become day_times overrides.
    """
    if not tags:
        return []

    # Group tags by normalized goal
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for tag in tags:
        key = tag.goal.lower().strip()
        groups[key].append(tag)

    result: List[Routine] = []
    for _, group in groups.items():
        # Separate daily vs weekly tags
        daily_tags = [t for t in group if t.schedule == "daily"]
        weekly_tags = [t for t in group if t.schedule == "weekly"]
        other_tags = [t for t in group if t.schedule not in ("daily", "weekly")]

        # Handle daily + weekly override combination
        if daily_tags:
            base = daily_tags[0]
            day_times = {}
            exclude_days = []

            # Collect exclusions from all daily tags
            for dt in daily_tags:
                if dt.exclude_days:
                    for d in dt.exclude_days:
                        if d not in exclude_days:
                            exclude_days.append(d)

            # Weekly tags become per-day overrides
            for wt in weekly_tags:
                if wt.day and wt.time:
                    day_times[wt.day] = wt.time
                    # If this day was excluded by the daily tag, remove from exclusions
                    # since it now has its own time
                    if wt.day in exclude_days:
                        exclude_days.remove(wt.day)

            routine = Routine(
                id=str(uuid.uuid4())[:8],
                goal=base.goal,
                urgency=max(t.urgency for t in group),
                schedule="daily",
                time=base.time,
                day_times=day_times if day_times else None,
                exclude_days=exclude_days if exclude_days else None,
            )
            result.append(routine)

        elif weekly_tags:
            # Multiple weekly tags, no daily base → merge into one Routine
            if len(weekly_tags) == 1:
                wt = weekly_tags[0]
                routine = Routine(
                    id=str(uuid.uuid4())[:8],
                    goal=wt.goal,
                    urgency=wt.urgency,
                    schedule="weekly",
                    day=wt.day,
                    time=wt.time,
                )
                result.append(routine)
            else:
                # Multiple weekly tags → merge into day_times
                day_times = {}
                best_urgency = 0
                for wt in weekly_tags:
                    if wt.day and wt.time:
                        day_times[wt.day] = wt.time
                    best_urgency = max(best_urgency, wt.urgency)

                routine = Routine(
                    id=str(uuid.uuid4())[:8],
                    goal=weekly_tags[0].goal,
                    urgency=best_urgency,
                    schedule="weekly",
                    day_times=day_times if day_times else None,
                    # Use first entry's time/day as fallback
                    day=weekly_tags[0].day,
                    time=weekly_tags[0].time,
                )
                result.append(routine)

        # Non-daily/weekly tags pass through as-is
        for tag in other_tags:
            routine = Routine(
                id=str(uuid.uuid4())[:8],
                goal=tag.goal,
                urgency=tag.urgency,
                schedule=tag.schedule,
                time=tag.time,
                day=tag.day,
                interval_hours=tag.hours if tag.schedule == "interval" else None,
                sleep_offset_hours=tag.hours if tag.schedule == "on_sleep" and tag.hours else 8.0,
            )
            result.append(routine)

    return result
