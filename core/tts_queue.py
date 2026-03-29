"""
Serialised TTS playback queue for the multi-pony system.

Multiple ponies may want to speak simultaneously.  This queue ensures only
one audio stream plays at a time, with priority ordering so user-initiated
responses always cut ahead of spontaneous chatter.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Priority levels (lower = higher priority)
PRIORITY_USER_RESPONSE = 0       # direct reply to user speech
PRIORITY_INTER_PONY_REPLY = 1    # pony responding in a group convo the user started
PRIORITY_SPONTANEOUS_CHAT = 2    # inter-pony banter
PRIORITY_AUTONOMOUS = 3          # background autonomous speech


@dataclass(order=True)
class _TTSItem:
    """Internal queue item.  Ordered by (priority, sequence) so that
    equal-priority items are FIFO."""
    priority: int
    sequence: int
    # Everything below is excluded from comparison
    text: str = field(compare=False)
    voice_slug: Optional[str] = field(compare=False, default=None)
    on_start: Optional[Callable] = field(compare=False, default=None)
    on_done: Optional[Callable] = field(compare=False, default=None)
    skip_tts: bool = field(compare=False, default=False)


class TTSQueue:
    """Thread-safe, priority-ordered TTS playback queue.

    Parameters
    ----------
    tts_engine : object
        Any TTS engine with a ``speak(text, on_playback_start=...)`` method
        and optionally ``set_character(slug)``.
    pause_between : float
        Seconds to pause between consecutive utterances so conversations
        breathe (default 0.4 s).
    """

    def __init__(self, tts_engine: Any, pause_between: float = 0.4) -> None:
        self._tts = tts_engine
        self._pause = pause_between
        self._queue: queue.PriorityQueue[_TTSItem] = queue.PriorityQueue()
        self._seq = 0  # monotonic counter for FIFO within same priority
        self._seq_lock = threading.Lock()
        self._running = True
        self._current_item: Optional[_TTSItem] = None
        self._thread = threading.Thread(target=self._consumer, daemon=True, name="tts-queue")
        self._thread.start()

    # ── public API ──────────────────────────────────────────────────

    def enqueue(
        self,
        text: str,
        *,
        priority: int = PRIORITY_USER_RESPONSE,
        voice_slug: Optional[str] = None,
        on_start: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        skip_tts: bool = False,
        blocking: bool = False,
    ) -> None:
        """Add an utterance to the queue.

        Parameters
        ----------
        text : str
            What to say.
        priority : int
            One of the ``PRIORITY_*`` constants.
        voice_slug : str | None
            Character slug — if the TTS engine has ``set_character()``, this
            will be called before speaking.
        on_start : callable | None
            Invoked just before audio playback starts (e.g. show speech bubble).
        on_done : callable | None
            Invoked after playback finishes (e.g. hide speech bubble).
        skip_tts : bool
            If True, fire on_start/on_done callbacks but skip actual audio
            playback.  Used for characters without a TTS voice.
        blocking : bool
            If True, block until this item finishes playing.  Used by the
            pipeline for user-response speech so listening doesn't start
            while the pony is still talking.
        """
        if not text or not text.strip():
            return
        done_event = threading.Event() if blocking else None
        with self._seq_lock:
            seq = self._seq
            self._seq += 1

        # Wrap on_done to signal the blocking event
        original_on_done = on_done
        def _wrapped_on_done():
            if original_on_done:
                try:
                    original_on_done()
                except Exception:
                    pass
            if done_event:
                done_event.set()

        item = _TTSItem(
            priority=priority,
            sequence=seq,
            text=text,
            voice_slug=voice_slug,
            on_start=on_start,
            on_done=_wrapped_on_done if blocking else on_done,
            skip_tts=skip_tts,
        )
        self._queue.put(item)
        logger.debug("TTSQueue: enqueued prio=%d seq=%d voice=%s text=%r",
                      priority, seq, voice_slug, text[:60])

        if done_event:
            done_event.wait()  # block until consumer finishes this item

    def flush(self) -> None:
        """Drop all pending items (e.g. user interrupted)."""
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
                # Fire on_done so any blocking waiters unblock
                if item.on_done:
                    try:
                        item.on_done()
                    except Exception:
                        pass
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.info("TTSQueue: flushed %d pending items.", dropped)

    def interrupt(self) -> None:
        """Stop current playback and flush pending items.

        Called when the user presses PTT — immediately silences the pony
        so the user can speak.
        """
        self.flush()
        # Stop the audio that's currently playing
        if hasattr(self._tts, "stop"):
            try:
                self._tts.stop()
            except Exception:
                pass
        logger.debug("TTSQueue: interrupted.")

    def stop(self) -> None:
        """Shut down the consumer thread.  Drains remaining items first."""
        self.flush()  # drain pending items and fire their on_done callbacks
        self._running = False
        # Put a sentinel so the thread unblocks from .get()
        self._queue.put(_TTSItem(priority=999, sequence=999_999_999, text=""))
        self._thread.join(timeout=5.0)
        logger.info("TTSQueue: stopped.")

    @property
    def is_speaking(self) -> bool:
        return self._current_item is not None

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    # ── consumer thread ─────────────────────────────────────────────

    def _consumer(self) -> None:
        """Background thread that pulls items and plays them sequentially."""
        last_spoke = 0.0

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Sentinel / shutdown check
            if not self._running or not item.text.strip():
                continue

            # Set current_item BEFORE breathing pause so is_speaking=True
            # from the moment the item is dequeued.  Without this, callers
            # that poll is_speaking/pending_count (e.g. _listen_for_reply)
            # see both as False during the gap and skip the wait.
            self._current_item = item

            # Breathing pause between utterances
            elapsed = time.monotonic() - last_spoke
            if last_spoke > 0 and elapsed < self._pause:
                time.sleep(self._pause - elapsed)

            if item.skip_tts:
                # No voice available — just show bubble, pause, move on
                if item.on_start:
                    try:
                        item.on_start()
                    except Exception:
                        pass
                # Pause so the bubble is visible for a reasonable time
                time.sleep(max(1.0, len(item.text) * 0.05))
            else:
                try:
                    if item.voice_slug and hasattr(self._tts, "set_character"):
                        try:
                            self._tts.set_character(item.voice_slug)
                        except Exception as exc:
                            logger.warning("TTSQueue: set_character(%r) failed: %s",
                                           item.voice_slug, exc)
                    _on_start_fired = False
                    def _guarded_on_start():
                        nonlocal _on_start_fired
                        if not _on_start_fired:
                            _on_start_fired = True
                            if item.on_start:
                                item.on_start()
                    self._tts.speak(item.text, on_playback_start=_guarded_on_start)
                except Exception as exc:
                    logger.error("TTSQueue: speak() failed: %s", exc)
                    # Still fire on_start so speech bubble shows (if not already fired)
                    if item.on_start and not _on_start_fired:
                        try:
                            item.on_start()
                        except Exception:
                            pass

            # Done callback
            if item.on_done:
                try:
                    item.on_done()
                except Exception as exc:
                    logger.warning("TTSQueue: on_done callback failed: %s", exc)

            last_spoke = time.monotonic()
            self._current_item = None
