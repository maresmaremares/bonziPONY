"""
Read recent browser history from Chrome, Edge, and Firefox on Windows.

Browsers lock their SQLite databases while running, so we copy the file
to a temp location before reading.  Returns a deduplicated, filtered list
of recently visited pages — boring entries (new tabs, internal pages,
extensions) are stripped out.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Boring URL patterns to filter out ─────────────────────────────────────

_BORING_PREFIXES = (
    "chrome://", "chrome-extension://", "edge://", "about:",
    "chrome-native://", "devtools://", "chrome-search://",
    "chrome-distiller://", "chrome-untrusted://",
    "extension://", "moz-extension://",
    "file:///", "blob:", "data:",
)

_BORING_TITLES = {
    "", "new tab", "new tab - google chrome", "new tab - microsoft edge",
    "new tab — mozilla firefox", "start", "home", "speed dial",
    "most visited sites", "customize chrome",
}

_BORING_DOMAINS = {
    "newtab", "extensions", "settings", "flags", "downloads",
    "history", "bookmarks", "passwords", "apps",
}


def _is_boring(url: str, title: str) -> bool:
    """Return True if this history entry is uninteresting."""
    url_lower = url.lower()
    title_lower = title.lower().strip()

    if any(url_lower.startswith(p) for p in _BORING_PREFIXES):
        return True
    if title_lower in _BORING_TITLES:
        return True

    # Internal browser pages (chrome://settings, edge://flags, etc.)
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url_lower)
        if parsed.hostname in _BORING_DOMAINS:
            return True
        # Google search result pages (the search itself, not the result)
        if parsed.hostname and "google." in parsed.hostname and parsed.path == "/search":
            return True
    except Exception:
        pass

    return False


def _chromium_epoch_to_datetime(us: int) -> datetime:
    """Convert Chromium's microseconds-since-1601 to a Python datetime."""
    # Chromium epoch: Jan 1, 1601 UTC
    # Unix epoch offset: 11644473600 seconds
    try:
        unix_ts = (us / 1_000_000) - 11644473600
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return datetime.now(tz=timezone.utc)


def _read_chromium_history(db_path: Path, hours: int, limit: int) -> List[dict]:
    """Read history from a Chromium-based browser (Chrome, Edge, Brave, etc.)."""
    if not db_path.exists():
        return []

    # Copy to temp file (browser locks the DB)
    tmp = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(tmp_fd)
        tmp = tmp_path
        shutil.copy2(str(db_path), tmp)

        cutoff_us = int((time.time() + 11644473600) * 1_000_000) - (hours * 3600 * 1_000_000)

        conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT url, title, last_visit_time "
                "FROM urls "
                "WHERE last_visit_time > ? "
                "ORDER BY last_visit_time DESC "
                "LIMIT ?",
                (cutoff_us, limit * 3),  # over-fetch, then filter
            ).fetchall()
        finally:
            conn.close()

        results = []
        seen = set()
        for row in rows:
            url = row["url"]
            title = row["title"] or ""
            if _is_boring(url, title):
                continue
            # Deduplicate by domain+path (ignore query params)
            from urllib.parse import urlparse
            try:
                p = urlparse(url)
                key = f"{p.netloc}{p.path}".lower().rstrip("/")
            except Exception:
                key = url
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "url": url,
                "title": title,
                "visited_at": _chromium_epoch_to_datetime(row["last_visit_time"]),
            })
            if len(results) >= limit:
                break
        return results

    except Exception as exc:
        logger.debug("Failed to read Chromium history from %s: %s", db_path, exc)
        return []
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def _read_firefox_history(hours: int, limit: int) -> List[dict]:
    """Read history from Firefox (all profiles)."""
    profiles_root = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"
    if not profiles_root.exists():
        return []

    # Find all places.sqlite files
    db_paths = list(profiles_root.glob("*/places.sqlite"))
    if not db_paths:
        return []

    all_results = []
    for db_path in db_paths:
        tmp = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
            os.close(tmp_fd)
            tmp = tmp_path
            shutil.copy2(str(db_path), tmp)

            # Firefox uses microseconds since Unix epoch
            cutoff_us = int((time.time() - hours * 3600) * 1_000_000)

            conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT p.url, p.title, h.visit_date "
                    "FROM moz_places p "
                    "JOIN moz_historyvisits h ON h.place_id = p.id "
                    "WHERE h.visit_date > ? "
                    "ORDER BY h.visit_date DESC "
                    "LIMIT ?",
                    (cutoff_us, limit * 3),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                url = row["url"]
                title = row["title"] or ""
                if _is_boring(url, title):
                    continue
                try:
                    visited = datetime.fromtimestamp(
                        row["visit_date"] / 1_000_000, tz=timezone.utc
                    )
                except (ValueError, OSError, OverflowError):
                    visited = datetime.now(tz=timezone.utc)
                all_results.append({
                    "url": url,
                    "title": title,
                    "visited_at": visited,
                })
        except Exception as exc:
            logger.debug("Failed to read Firefox history from %s: %s", db_path, exc)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

    # Sort and deduplicate
    all_results.sort(key=lambda r: r["visited_at"], reverse=True)
    seen = set()
    deduped = []
    for r in all_results:
        from urllib.parse import urlparse
        try:
            p = urlparse(r["url"])
            key = f"{p.netloc}{p.path}".lower().rstrip("/")
        except Exception:
            key = r["url"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)
        if len(deduped) >= limit:
            break
    return deduped


# ── Public API ────────────────────────────────────────────────────────────

def get_recent_history(hours: int = 24, limit: int = 20) -> List[dict]:
    """Get recent browser history from all installed browsers.

    Returns a list of dicts with keys: url, title, visited_at (datetime).
    Sorted by most recent first.  Boring entries (new tabs, internal pages)
    are filtered out and duplicates are removed.
    """
    local = os.environ.get("LOCALAPPDATA", "")
    results = []

    # Chrome
    chrome_db = Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "History"
    results.extend(_read_chromium_history(chrome_db, hours, limit))

    # Edge
    edge_db = Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "History"
    results.extend(_read_chromium_history(edge_db, hours, limit))

    # Brave
    brave_db = Path(local) / "BraveSoftware" / "Brave-Browser" / "User Data" / "Default" / "History"
    results.extend(_read_chromium_history(brave_db, hours, limit))

    # Firefox
    results.extend(_read_firefox_history(hours, limit))

    # Global dedup + sort
    results.sort(key=lambda r: r["visited_at"], reverse=True)
    seen = set()
    final = []
    for r in results:
        from urllib.parse import urlparse
        try:
            p = urlparse(r["url"])
            key = f"{p.netloc}{p.path}".lower().rstrip("/")
        except Exception:
            key = r["url"]
        if key not in seen:
            seen.add(key)
            final.append(r)
        if len(final) >= limit:
            break
    return final


def format_history_for_llm(entries: List[dict], max_entries: int = 10) -> str:
    """Format history entries into a readable string for LLM context.

    Returns a compact summary like:
        - "How to make sourdough bread" (youtube.com, 2h ago)
        - "Reddit - r/programmerhumor" (reddit.com, 3h ago)
    """
    if not entries:
        return "(no interesting browser history found)"

    lines = []
    now = datetime.now(tz=timezone.utc)
    for entry in entries[:max_entries]:
        title = entry["title"].strip() or entry["url"]
        # Truncate long titles
        if len(title) > 80:
            title = title[:77] + "..."

        # Extract domain
        from urllib.parse import urlparse
        try:
            domain = urlparse(entry["url"]).netloc
            # Strip www.
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = "?"

        # Time ago
        delta = now - entry["visited_at"]
        if delta.total_seconds() < 3600:
            ago = f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            ago = f"{delta.total_seconds() / 3600:.0f}h ago"
        else:
            ago = f"{delta.days}d ago"

        lines.append(f'- "{title}" ({domain}, {ago})')

    return "\n".join(lines)
