"""
Persistent user profile — builds a living picture of the user over time.

Maintains two files in memory/:
  - user_profile.txt   — stable facts (name, age, location, job, interests, etc.)
  - user_events.txt    — time-sensitive events and follow-ups (job interview Friday, etc.)

After each conversation, the LLM extracts new information and the files
are updated.  Both files are injected into the system prompt so the
character remembers everything across sessions.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path(__file__).parent.parent / "memory"
_PROFILE_FILE = _MEMORY_DIR / "user_profile.txt"
_EVENTS_FILE = _MEMORY_DIR / "user_events.txt"

# ── Extraction prompts ──────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You just had a conversation with the user. Below is the conversation.

Your job: extract any NEW facts you learned about the user that are worth \
remembering long-term. Output two sections exactly as shown.

== PROFILE ==
Stable facts about the user: name, age, height, location, gender, job, \
school, hobbies, pets, relationship status, living situation, personality \
traits, preferences, favorite things, dislikes, health conditions, etc.

Only list NEW facts not already in the existing profile below. One fact per \
line, short and direct. If you learned nothing new, write "(nothing new)".

== EVENTS ==
Time-sensitive things to follow up on: upcoming job interviews, exams, \
deadlines, plans, dates, trips, goals they mentioned working toward, \
problems they're dealing with, things they said they'd do, etc.

One event per line. Include enough context to ask about it later.
Format: "- <event> (mentioned <today's date>)"
If there are no new events, write "(nothing new)".

Existing profile (don't repeat these):
---
{existing_profile}
---

Existing events (don't repeat these):
---
{existing_events}
---

Today's date: {today}

Conversation:
{conversation}

Now extract. Be concise. Only genuinely interesting or useful facts — skip \
trivial filler. Output the two sections and nothing else."""

_COMPACT_PROMPT = """\
Below is a user profile that has accumulated over many sessions. It may have \
duplicates, contradictions, outdated info, or trivial filler.

Rewrite it into a clean, compact profile. Rules:
- Keep ONLY important, useful facts (name, age, location, job/school, key \
interests, personality, living situation, health, relationships, preferences)
- Remove duplicates — if the same fact appears multiple times, keep it once
- If two facts contradict, keep the more recent/specific one
- Remove trivial filler ("user said hi", "user was tired today", etc.)
- Keep it as SHORT as possible while retaining all genuinely useful information
- One fact per line, short and direct, no bullets or prefixes
- If the profile is already clean, return it unchanged

Today's date: {today}

Current profile:
---
{profile}
---

Return ONLY the cleaned profile text, nothing else."""

_PRUNE_PROMPT = """\
Below is a list of time-sensitive events and follow-ups for a user. \
Today's date is {today}.

Remove any events that are clearly outdated or resolved (more than 2 weeks \
old with no indication they're ongoing). Keep anything that might still be \
relevant. Return the cleaned list, one item per line, prefixed with "- ".
If everything should stay, return the list unchanged.
If everything should be removed, return "(no active events)".

Events:
{events}"""


def _read_file(path: Path) -> str:
    """Read a file, returning empty string if it doesn't exist."""
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _write_file(path: Path, content: str) -> None:
    """Write content to file, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


# ── Public API ───────────────────────────────────────────────────────────────

def get_profile() -> str:
    """Return the current user profile text, or empty string."""
    return _read_file(_PROFILE_FILE)


def get_events() -> str:
    """Return current events/follow-ups text, or empty string."""
    return _read_file(_EVENTS_FILE)


def get_profile_for_prompt() -> Optional[str]:
    """
    Return a combined profile + events block for injection into the system
    prompt.  Returns None if both are empty.
    """
    profile = get_profile()
    events = get_events()

    if not profile and not events:
        return None

    parts = []
    if profile:
        parts.append(f"== What you know about the user ==\n{profile}")
    if events:
        parts.append(
            f"== Ongoing events & things to follow up on ==\n{events}\n"
            "(If any of these are relevant, you can casually bring them up. "
            "Don't force it — just naturally weave it in when it fits.)"
        )
    return "\n\n".join(parts)


def update_from_conversation(llm_provider, conversation_history: list[dict]) -> None:
    """
    Run a one-shot LLM call to extract new profile facts and events from
    the most recent conversation.  Appends to the persistent files.

    Called after a conversation ends (from pipeline or shutdown).
    """
    if not conversation_history:
        return

    # Build a readable conversation transcript
    lines = []
    for msg in conversation_history:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "system":
            continue
        # Strip system injections (screen context, hints) for cleaner extraction
        content = re.sub(r"\[Screen:.*?\]", "", content)
        content = re.sub(r"\[System hint:.*?\]", "", content, flags=re.DOTALL)
        content = re.sub(r"\[System:.*?\]", "", content)
        content = content.strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Pony"
        lines.append(f"{speaker}: {content}")

    if len(lines) < 2:
        # Too short to extract anything meaningful
        return

    conversation_text = "\n".join(lines[-40:])  # last 40 exchanges max

    existing_profile = get_profile() or "(empty — first session)"
    existing_events = get_events() or "(none yet)"
    today = datetime.now().strftime("%B %d, %Y")

    prompt = _EXTRACT_PROMPT.format(
        existing_profile=existing_profile,
        existing_events=existing_events,
        today=today,
        conversation=conversation_text,
    )

    try:
        raw = llm_provider.generate_once(
            prompt, max_tokens=1024,
            system_prompt="You are a helpful assistant that extracts structured information from conversations. Follow the instructions exactly. Do NOT role-play or respond in character.",
        )
        if not raw:
            return
        _parse_and_save(raw)
        logger.info("User profile updated from conversation.")
    except Exception as exc:
        logger.warning("Profile extraction failed: %s", exc)


def prune_events(llm_provider) -> None:
    """
    Remove stale events.  Call periodically (e.g. on startup).
    """
    events = get_events()
    if not events or events == "(no active events)":
        return

    today = datetime.now().strftime("%B %d, %Y")
    prompt = _PRUNE_PROMPT.format(today=today, events=events)

    _UTIL_SYSTEM = "You are a helpful assistant. Follow the instructions exactly. Do NOT role-play or respond in character."
    try:
        raw = llm_provider.generate_once(prompt, system_prompt=_UTIL_SYSTEM)
        if raw and raw.strip():
            _write_file(_EVENTS_FILE, raw.strip())
            logger.info("Events list pruned.")
    except Exception as exc:
        logger.warning("Event pruning failed: %s", exc)


def compact_profile(llm_provider) -> None:
    """
    Summarize and deduplicate the user profile.  Call on startup each session
    so the profile stays tight and doesn't grow unbounded.
    """
    profile = get_profile()
    if not profile:
        return

    # Only compact if the profile is getting long (>15 lines)
    line_count = len([l for l in profile.splitlines() if l.strip()])
    if line_count < 15:
        logger.debug("Profile is short (%d lines) — skipping compaction.", line_count)
        return

    today = datetime.now().strftime("%B %d, %Y")
    prompt = _COMPACT_PROMPT.format(today=today, profile=profile)

    _UTIL_SYSTEM = "You are a helpful assistant. Follow the instructions exactly. Do NOT role-play or respond in character."
    try:
        raw = llm_provider.generate_once(prompt, system_prompt=_UTIL_SYSTEM)
        if raw and raw.strip():
            compacted = raw.strip()
            new_count = len([l for l in compacted.splitlines() if l.strip()])
            _write_file(_PROFILE_FILE, compacted)
            logger.info("Profile compacted: %d lines → %d lines.", line_count, new_count)
    except Exception as exc:
        logger.warning("Profile compaction failed: %s", exc)


def _parse_and_save(raw: str) -> None:
    """Parse the LLM extraction output and append new facts/events."""
    # Split into profile and events sections
    profile_section = ""
    events_section = ""

    # Find == PROFILE == section
    profile_match = re.search(
        r"==\s*PROFILE\s*==\s*\n(.*?)(?=\n==\s*EVENTS\s*==|\Z)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if profile_match:
        profile_section = profile_match.group(1).strip()

    # Find == EVENTS == section
    events_match = re.search(
        r"==\s*EVENTS\s*==\s*\n(.*)",
        raw, re.DOTALL | re.IGNORECASE,
    )
    if events_match:
        events_section = events_match.group(1).strip()

    # Append new profile facts
    if profile_section and "(nothing new)" not in profile_section.lower():
        existing = get_profile()
        new_lines = [
            line.strip() for line in profile_section.splitlines()
            if line.strip() and not line.strip().startswith("==")
        ]
        if new_lines:
            combined = existing + "\n" + "\n".join(new_lines) if existing else "\n".join(new_lines)
            _write_file(_PROFILE_FILE, combined)
            logger.debug("Added %d profile facts.", len(new_lines))

    # Append new events
    if events_section and "(nothing new)" not in events_section.lower():
        existing = get_events()
        if existing == "(no active events)":
            existing = ""
        new_lines = [
            line.strip() for line in events_section.splitlines()
            if line.strip() and not line.strip().startswith("==")
        ]
        if new_lines:
            combined = existing + "\n" + "\n".join(new_lines) if existing else "\n".join(new_lines)
            _write_file(_EVENTS_FILE, combined)
            logger.debug("Added %d events.", len(new_lines))
