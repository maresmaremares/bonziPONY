"""
Parses LLM responses to extract [ACTION:XYZ], [DESKTOP:cmd:args],
[DIRECTIVE:goal:urgency], [TIMER:...], and [ROUTINE:...] tags and clean spoken text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from robot.actions import RobotAction

# Matches [ACTION:WALK_FORWARD], [action:sit], etc.
_ACTION_PATTERN = re.compile(r"\[ACTION:([A-Z_]+)\]", re.IGNORECASE)

# Matches [DESKTOP:CLICK:500:300], [DESKTOP:OPEN:notepad], etc.
_DESKTOP_PATTERN = re.compile(r"\[DESKTOP:([^\]]+)\]", re.IGNORECASE)

# Matches a truncated [DESKTOP:... tag at end of response (hit token limit before closing ])
_DESKTOP_TRUNCATED = re.compile(r"\[DESKTOP:([^\]]+)$", re.IGNORECASE | re.MULTILINE)

# Matches [DIRECTIVE:nag user to go to gym:7]
_DIRECTIVE_PATTERN = re.compile(r"\[DIRECTIVE:([^\]]+)\]", re.IGNORECASE)

# Matches [TIMER:21:00:close everything and tell user to sleep]
_TIMER_PATTERN = re.compile(r"\[TIMER:([^\]]+)\]", re.IGNORECASE)

# Matches [ROUTINE:on_wake:Brush teeth:5] or [ROUTINE:on_sleep:Brush teeth:5:8]
_ROUTINE_PATTERN = re.compile(r"\[ROUTINE:([^\]]+)\]", re.IGNORECASE)

# Matches [ENFORCE:15] — user is going to do the task, monitor for N minutes
_ENFORCE_PATTERN = re.compile(r"\[ENFORCE:(\d+)\]", re.IGNORECASE)

# Matches [DELAY:60] or [DELAY:30:keyword] — user negotiated a delay
_DELAY_PATTERN = re.compile(r"\[DELAY:(\d+)(?::([^\]]*))?\]", re.IGNORECASE)

# Matches [DONE] or [DONE:shower] — user completed a task
_DONE_PATTERN = re.compile(r"\[DONE(?::([^\]]*))?\]", re.IGNORECASE)

# Matches [CONVO:END] or [CONVO:CONTINUE] — conversation flow signal
_CONVO_PATTERN = re.compile(r"\[CONVO:\s*(END|CONTINUE)\s*\]", re.IGNORECASE)

# Matches [PERSIST:600] — keep current action for N seconds
_PERSIST_PATTERN = re.compile(r"\[PERSIST:\s*(\d+)\s*\]", re.IGNORECASE)

# Matches [MOVETO:top_left] — move pony to screen region
_MOVETO_PATTERN = re.compile(r"\[MOVETO:\s*([^\]]+?)\s*\]", re.IGNORECASE)

# Catch-all: strip any remaining [TAG:...] bracket expressions the LLM may produce
_LEFTOVER_TAG_PATTERN = re.compile(r"\[(?:MOVETO|PERSIST|ANIM|ACTION|CONVO|DESKTOP|DIRECTIVE|TIMER|ROUTINE|ENFORCE|DONE|DELAY)\s*:[^\]]*\]", re.IGNORECASE)

# Strip <think>...</think> blocks from reasoning models (DeepSeek, QwQ, etc.)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ── Speech sanitization patterns ──────────────────────────────────────────────
# These strip content that should never be spoken aloud through TTS.

# Code blocks: ```...``` or ```lang\n...\n```
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.DOTALL)
# Inline code: `code`
_INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
# Markdown headers: # Header, ## Header, etc.
_MD_HEADER_PATTERN = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
# Markdown bold/italic: **bold**, *italic*, __bold__, _italic_
_MD_EMPHASIS_PATTERN = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
# Markdown links: [text](url)
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Markdown images: ![alt](url)
_MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
# Markdown list markers: - item, * item, 1. item (at start of line)
_MD_LIST_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
# HTML tags: <div>, </span>, <br/>, etc.
_HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
# Raw URLs: http://... or https://...
_URL_PATTERN = re.compile(r"https?://\S+")
# Lines that look like code: indented 4+ spaces with code-like content
_INDENTED_CODE_PATTERN = re.compile(r"^[ \t]{4,}\S.*$", re.MULTILINE)
# Horizontal rules: --- or *** or ___
_MD_HR_PATTERN = re.compile(r"^[\s]*[-*_]{3,}[\s]*$", re.MULTILINE)


@dataclass
class DesktopCommand:
    command: str        # e.g. "CLICK"
    args: list[str]     # e.g. ["500", "300"]


@dataclass
class DirectiveTag:
    goal: str
    urgency: int
    delay_minutes: Optional[int] = None  # deferred directive — wait N min before first nag


@dataclass
class TimerTag:
    time_str: str       # e.g. "21:00" or "9pm"
    action: str         # e.g. "close everything and tell user to sleep"


@dataclass
class RoutineTag:
    schedule: str       # "on_wake", "on_sleep", "daily", "weekly", "interval"
    goal: str
    urgency: int
    time: Optional[str] = None       # HH:MM for daily/weekly
    day: Optional[str] = None        # lowercase day name for weekly ("monday", etc.)
    hours: Optional[float] = None    # hours for on_sleep / interval


@dataclass
class ParsedResponse:
    text: str                        # Speech text with tags stripped
    actions: List[RobotAction] = field(default_factory=list)
    desktop_commands: List[DesktopCommand] = field(default_factory=list)
    directive: Optional[DirectiveTag] = None
    timer: Optional[TimerTag] = None
    routines: List[RoutineTag] = field(default_factory=list)
    enforce_minutes: Optional[int] = None  # user is going to do the task, monitor for N min
    done_directive: Optional[str] = None   # user completed a task — keyword or empty string
    delay_minutes: Optional[int] = None    # user negotiated a delay — reschedule in N min
    delay_keyword: str = ""                # optional keyword to match directive for delay
    end_conversation: bool = False         # LLM signals conversation is over
    persist_seconds: Optional[int] = None  # keep action animation for N seconds
    moveto_region: Optional[str] = None    # move pony to screen region


def parse_response(raw: str) -> ParsedResponse:
    """Strip all tags from raw LLM text and extract structured data."""
    # Strip <think>...</think> blocks from reasoning models
    raw = _THINK_BLOCK_PATTERN.sub("", raw).strip()
    # Also strip unclosed <think> blocks (model hit token limit mid-thought)
    if "<think>" in raw.lower() and "</think>" not in raw.lower():
        idx = raw.lower().rfind("<think>")
        raw = raw[:idx].strip()

    actions: List[RobotAction] = []
    desktop_commands: List[DesktopCommand] = []
    directive: Optional[DirectiveTag] = None
    timer: Optional[TimerTag] = None
    routines: List[RoutineTag] = []

    for match in _ACTION_PATTERN.finditer(raw):
        tag = match.group(1).upper()
        try:
            actions.append(RobotAction[tag])
        except KeyError:
            pass  # Unknown action — ignore

    for match in _DESKTOP_PATTERN.finditer(raw):
        parts = match.group(1).split(":")
        if parts:
            desktop_commands.append(DesktopCommand(
                command=parts[0].strip(),
                args=[p.strip() for p in parts[1:]],
            ))

    # Handle truncated DESKTOP tag (response cut off by token limit before closing ])
    if "[DESKTOP:" in raw.upper():
        trunc_match = _DESKTOP_TRUNCATED.search(raw)
        if trunc_match:
            content = trunc_match.group(1).rstrip()
            parts = content.split(":")
            if parts:
                desktop_commands.append(DesktopCommand(
                    command=parts[0].strip(),
                    args=[p.strip() for p in parts[1:]],
                ))

    # Parse first DIRECTIVE tag (only one allowed per response)
    # Supports: [DIRECTIVE:goal:urgency] and [DIRECTIVE:goal:urgency:delay_minutes]
    dir_match = _DIRECTIVE_PATTERN.search(raw)
    if dir_match:
        content = dir_match.group(1)
        # Try goal:urgency:delay format first (rsplit from right, max 2 splits)
        parts3 = content.rsplit(":", 2)
        parsed_dir = False
        if len(parts3) == 3:
            try:
                urg = int(parts3[1].strip())
                delay = int(parts3[2].strip())
                directive = DirectiveTag(
                    goal=parts3[0].strip(),
                    urgency=max(1, min(10, urg)),
                    delay_minutes=delay if delay > 0 else None,
                )
                parsed_dir = True
            except ValueError:
                pass  # fall through to goal:urgency
        if not parsed_dir:
            parts2 = content.rsplit(":", 1)
            if len(parts2) == 2:
                try:
                    urg = int(parts2[1].strip())
                    directive = DirectiveTag(goal=parts2[0].strip(), urgency=max(1, min(10, urg)))
                except ValueError:
                    directive = DirectiveTag(goal=content.strip(), urgency=5)
            else:
                directive = DirectiveTag(goal=content.strip(), urgency=5)

    # Parse first TIMER tag
    timer_match = _TIMER_PATTERN.search(raw)
    if timer_match:
        # Format: [TIMER:HH:MM:action] or [TIMER:9pm:action]
        parts = timer_match.group(1).split(":", 2)  # split into at most 3 parts
        if len(parts) >= 2:
            # Check if first two parts form HH:MM
            try:
                int(parts[0])
                int(parts[1])
                # It's HH:MM:action format
                time_str = f"{parts[0]}:{parts[1]}"
                action = parts[2].strip() if len(parts) > 2 else "timer alert"
            except ValueError:
                # First part is like "9pm", rest is action
                time_str = parts[0].strip()
                action = ":".join(parts[1:]).strip() or "timer alert"
            timer = TimerTag(time_str=time_str, action=action)
        else:
            # Single-part: [TIMER:midnight] or [TIMER:9pm]
            time_str = parts[0].strip()
            if time_str:
                timer = TimerTag(time_str=time_str, action="timer alert")

    # Parse ALL ROUTINE tags (multiple allowed)
    for match in _ROUTINE_PATTERN.finditer(raw):
        parts = match.group(1).split(":")
        if len(parts) >= 3:
            schedule = parts[0].strip().lower()
            goal = parts[1].strip()
            try:
                urgency = int(parts[2].strip())
            except ValueError:
                urgency = 5
            urgency = max(1, min(10, urgency))

            rt = RoutineTag(schedule=schedule, goal=goal, urgency=urgency)

            # Parse optional extra fields
            if len(parts) >= 4:
                extra = parts[3].strip()
                if schedule == "daily":
                    # Could be HH:MM (possibly split across parts[3] and parts[4])
                    if len(parts) >= 5:
                        rt.time = f"{parts[3].strip()}:{parts[4].strip()}"
                    else:
                        rt.time = extra
                elif schedule == "weekly":
                    # [ROUTINE:weekly:goal:urgency:day:HH:MM]
                    rt.day = extra.lower()
                    if len(parts) >= 6:
                        rt.time = f"{parts[4].strip()}:{parts[5].strip()}"
                    elif len(parts) >= 5:
                        rt.time = parts[4].strip()
                elif schedule in ("on_sleep", "interval"):
                    try:
                        rt.hours = float(extra)
                    except ValueError:
                        pass

            routines.append(rt)

    # Parse [ENFORCE:minutes] tag
    enforce_minutes = None
    enforce_match = _ENFORCE_PATTERN.search(raw)
    if enforce_match:
        enforce_minutes = int(enforce_match.group(1))

    # Parse [DELAY:minutes] or [DELAY:minutes:keyword] tag
    delay_minutes = None
    delay_keyword = ""
    delay_match = _DELAY_PATTERN.search(raw)
    if delay_match:
        delay_minutes = int(delay_match.group(1))
        delay_keyword = (delay_match.group(2) or "").strip()

    # Parse [DONE] or [DONE:keyword] tag
    done_directive = None
    done_match = _DONE_PATTERN.search(raw)
    if done_match:
        done_directive = (done_match.group(1) or "").strip()

    # Parse [CONVO:END] / [CONVO:CONTINUE] tag
    end_conversation = False
    convo_match = _CONVO_PATTERN.search(raw)
    if convo_match:
        end_conversation = convo_match.group(1).upper() == "END"

    # Parse [PERSIST:seconds] tag
    persist_seconds = None
    persist_match = _PERSIST_PATTERN.search(raw)
    if persist_match:
        persist_seconds = int(persist_match.group(1))

    # Parse [MOVETO:region] tag
    moveto_region = None
    moveto_match = _MOVETO_PATTERN.search(raw)
    if moveto_match:
        moveto_region = moveto_match.group(1).strip().lower().replace(" ", "_")

    clean_text = _ACTION_PATTERN.sub("", raw)
    clean_text = _DESKTOP_PATTERN.sub("", clean_text)
    clean_text = _DESKTOP_TRUNCATED.sub("", clean_text)
    clean_text = _DIRECTIVE_PATTERN.sub("", clean_text)
    clean_text = _TIMER_PATTERN.sub("", clean_text)
    clean_text = _ROUTINE_PATTERN.sub("", clean_text)
    clean_text = _ENFORCE_PATTERN.sub("", clean_text)
    clean_text = _DELAY_PATTERN.sub("", clean_text)
    clean_text = _DONE_PATTERN.sub("", clean_text)
    clean_text = _CONVO_PATTERN.sub("", clean_text)
    clean_text = _PERSIST_PATTERN.sub("", clean_text)
    clean_text = _MOVETO_PATTERN.sub("", clean_text)
    clean_text = _LEFTOVER_TAG_PATTERN.sub("", clean_text).strip()
    # Sanitize for TTS — strip code, markdown, HTML, URLs
    clean_text = sanitize_for_speech(clean_text)
    return ParsedResponse(text=clean_text, actions=actions, desktop_commands=desktop_commands,
                          directive=directive, timer=timer, routines=routines,
                          enforce_minutes=enforce_minutes, done_directive=done_directive,
                          delay_minutes=delay_minutes, delay_keyword=delay_keyword,
                          end_conversation=end_conversation,
                          persist_seconds=persist_seconds, moveto_region=moveto_region)


def sanitize_for_speech(text: str) -> str:
    """Strip code, markdown, HTML, and URLs from text destined for TTS.

    This is the last line of defense — even if the LLM breaks character and
    outputs code/markdown, the user won't hear raw syntax through speakers.
    """
    if not text:
        return text

    # Strip code blocks first (```...```) — these are never speakable
    text = _CODE_BLOCK_PATTERN.sub("", text)
    # Strip inline code (`code`)
    text = _INLINE_CODE_PATTERN.sub("", text)
    # Strip markdown images (before links, since images contain links)
    text = _MD_IMAGE_PATTERN.sub("", text)
    # Strip markdown links but keep the link text: [text](url) → text
    text = _MD_LINK_PATTERN.sub(r"\1", text)
    # Strip raw URLs
    text = _URL_PATTERN.sub("", text)
    # Strip HTML tags
    text = _HTML_TAG_PATTERN.sub("", text)
    # Strip markdown headers (# Header → Header)
    text = _MD_HEADER_PATTERN.sub(lambda m: m.group(0).lstrip("# "), text)
    # Strip markdown emphasis markers but keep text: **bold** → bold
    text = _MD_EMPHASIS_PATTERN.sub(r"\2", text)
    # Strip markdown list markers: "- item" → "item"
    text = _MD_LIST_PATTERN.sub("", text)
    # Strip horizontal rules
    text = _MD_HR_PATTERN.sub("", text)
    # Strip indented code lines
    text = _INDENTED_CODE_PATTERN.sub("", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)

    return text.strip()
