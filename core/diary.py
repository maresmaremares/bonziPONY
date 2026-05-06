"""Personal diary — in-character journal entries for the active pony."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DIARY_DIR = Path(__file__).parent.parent / "diary"


def _get_diary_file() -> Path:
    """Derive the diary filename from the active preset."""
    from llm.prompt import get_active_preset
    return _DIARY_DIR / f"{get_active_preset()}_diary.txt"


def write_entry(text: str, open_in_notepad: bool = False) -> Path:
    """Append a timestamped diary entry. Returns the file path."""
    _DIARY_DIR.mkdir(exist_ok=True)
    diary_file = _get_diary_file()
    timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    entry = f"\n{'~'*50}\n{timestamp}\n{'~'*50}\n{text}\n"

    with diary_file.open("a", encoding="utf-8") as f:
        f.write(entry)

    logger.info("Diary entry written (%d chars)", len(text))

    if open_in_notepad:
        try:
            from core.platform_compat import open_in_text_editor
            open_in_text_editor(diary_file)
        except Exception as exc:
            logger.warning("Failed to open diary in editor: %s", exc)

    return diary_file


def read_recent(n_entries: int = 5) -> str:
    """Read the last N diary entries."""
    diary_file = _get_diary_file()
    if not diary_file.exists():
        return "(No diary entries yet)"

    content = diary_file.read_text(encoding="utf-8")
    # Split on the separator lines
    separator = "~" * 50
    parts = content.split(separator)
    # Each entry is: separator + date + separator + content = 3 parts
    # Take the last n_entries worth
    if len(parts) > n_entries * 3:
        parts = parts[-(n_entries * 3):]
    return separator.join(parts).strip() or "(No diary entries yet)"


def get_diary_path() -> Path:
    """Return the diary file path for the active character."""
    return _get_diary_file()
