"""
Wake word detector using local Whisper for keyword spotting.

Uses the ``speech_recognition`` library's energy-based VAD to detect when
someone starts talking, records a short clip, transcribes it locally with
OpenAI Whisper, then checks if the transcription contains any of the
configured wake phrases.

Fully offline — no API calls, no network required.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Sensible defaults per character — override via config wake_word.phrases
DEFAULT_PHRASES: Dict[str, List[str]] = {
    "rainbow_dash":     ["hey dash", "hey dashie", "rainbow dash", "dash"],
    "twilight_sparkle":  ["hey twilight", "twilight sparkle", "twilight", "hey twi"],
    "pinkie_pie":        ["hey pinkie", "pinkie pie", "pinkie", "hey pinkie pie",
                          "hey pinky", "pinky pie", "pinky", "hey pinky pie",
                          "pink e", "pink key", "pinky pi"],
    "rarity":            ["hey rarity", "rarity"],
    "applejack":         ["hey applejack", "applejack", "hey aj"],
    "fluttershy":        ["hey fluttershy", "fluttershy", "hey flutter"],
}

# Common Whisper mistranscriptions → canonical form
_WHISPER_NORMALIZATIONS = {
    "pinky": "pinkie",
    "pink e": "pinkie",
    "pink key": "pinkie",
    "pinky pi": "pinkie pie",
    "twighlight": "twilight",
    "twilite": "twilight",
    "apple jack": "applejack",
    "flutter shy": "fluttershy",
    "rain bow": "rainbow",
    "rainbow": "rainbow",
}


def _normalize_whisper_text(text: str) -> str:
    """Normalize common Whisper mistranscriptions for better wake word matching."""
    result = text.lower()
    for wrong, right in _WHISPER_NORMALIZATIONS.items():
        result = result.replace(wrong, right)
    return result


def _auto_generate_phrases(slug: str) -> List[str]:
    """Generate wake phrases for characters without hand-tuned defaults."""
    from core.character_registry import get_display_name
    display_name = get_display_name(slug)

    # Strip parenthetical content and "PP " prefix
    import re
    clean = re.sub(r"\s*\(.*?\)", "", display_name).strip()
    if clean.upper().startswith("PP "):
        clean = clean[3:].strip()

    words = clean.lower().split()
    if not words:
        return ["hey pony"]

    phrases = []
    last_word = words[-1]
    full = " ".join(words)

    if len(words) > 1:
        phrases.append(f"hey {last_word}")
        phrases.append(full)
        phrases.append(last_word)
    else:
        phrases.append(f"hey {full}")
        phrases.append(full)

    phrases.append("hey pony")
    return phrases


def get_phrases_for(preset: str, config_phrases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Return the wake phrases for a preset, preferring config overrides."""
    if config_phrases and preset in config_phrases:
        return config_phrases[preset]
    if preset in DEFAULT_PHRASES:
        return DEFAULT_PHRASES[preset]
    return _auto_generate_phrases(preset)


class WakeWordDetector:
    """Thread-safe wake word detector via local Whisper keyword matching.

    Uses OpenAI Whisper for fully offline transcription — no network needed.
    """

    def __init__(
        self,
        wake_phrases: List[str],
        input_device_index: int = -1,
        language: str = "en",
        whisper_model: str = "tiny",
    ) -> None:
        self._phrases: List[str] = [p.lower().strip() for p in wake_phrases]
        self._input_device_index = input_device_index
        self._language = language
        self._whisper_model_name = whisper_model

        self._event_queue: queue.Queue[int] = queue.Queue()
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Audio buffer — saved when a wake phrase fires (for voice verification)
        self._wake_audio: Optional[bytes] = None

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def wake_phrases(self) -> List[str]:
        return list(self._phrases)

    def set_wake_phrases(self, phrases: List[str]) -> None:
        """Hot-swap the active wake phrases (e.g. on character switch)."""
        self._phrases = [p.lower().strip() for p in phrases]
        logger.info("Wake phrases updated: %s", self._phrases)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background detection thread."""
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WakeWordThread",
        )
        self._thread.start()
        logger.info(
            "Wake word detector started (Whisper %s). Listening for: %s",
            self._whisper_model_name,
            ", ".join(self._phrases),
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._paused.set()  # unblock any wait inside the thread
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Wake word detector stopped.")

    def pause(self) -> None:
        """Pause detection (e.g. while TTS / STT is using the mic)."""
        self._paused.set()
        logger.debug("Wake word detector paused.")

    def resume(self) -> None:
        """Resume detection after pipeline finishes."""
        self._paused.clear()
        logger.debug("Wake word detector resumed.")

    # ── Blocking query ────────────────────────────────────────────────────

    def wait_for_wake_word(self, timeout: Optional[float] = None) -> Optional[int]:
        """Block until a wake phrase is detected.  Returns phrase index or None."""
        try:
            return self._event_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_wake_audio(self) -> Optional[np.ndarray]:
        """Return the audio around the last detection as float32.  One-shot."""
        raw = self._wake_audio
        if raw is None:
            return None
        self._wake_audio = None
        try:
            audio_i16 = np.frombuffer(raw, dtype=np.int16)
            return audio_i16.astype(np.float32) / 32768.0
        except Exception:
            return None

    # ── Background thread ─────────────────────────────────────────────────

    def _run(self) -> None:
        import speech_recognition as sr
        import whisper

        # Load Whisper model once (tiny is fast enough for short wake phrases)
        logger.info("Loading Whisper model '%s' for wake word detection...", self._whisper_model_name)
        model = whisper.load_model(self._whisper_model_name)
        logger.info("Whisper model loaded.")

        recognizer = sr.Recognizer()
        # Short pause threshold — wake phrases are brief
        recognizer.pause_threshold = 0.8
        recognizer.non_speaking_duration = 0.3
        # Very sensitive — false triggers don't matter (Whisper filters them)
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 50
        recognizer.dynamic_energy_adjustment_damping = 0.05
        recognizer.dynamic_energy_ratio = 1.2

        mic_kwargs: dict = {"sample_rate": SAMPLE_RATE}
        if self._input_device_index >= 0:
            mic_kwargs["device_index"] = self._input_device_index

        while not self._stop.is_set():
            # ── Paused: spin-wait until resumed ────────────────────────
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            try:
                from stt.mic_lock import safe_microphone
                with safe_microphone(**mic_kwargs) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.15)

                    # Inner loop: listen while not paused/stopped
                    while not self._stop.is_set() and not self._paused.is_set():
                        try:
                            audio = recognizer.listen(
                                source,
                                timeout=3.0,           # wait at most 3s for speech to START
                                phrase_time_limit=5.0,  # record at most 5s once speech starts
                            )
                        except sr.WaitTimeoutError:
                            continue

                        # Save raw PCM for potential voice verification
                        raw_pcm = audio.get_raw_data(
                            convert_rate=SAMPLE_RATE, convert_width=2,
                        )

                        # Convert to float32 numpy array for Whisper
                        audio_i16 = np.frombuffer(raw_pcm, dtype=np.int16)
                        audio_f32 = audio_i16.astype(np.float32) / 32768.0

                        # Transcribe locally with Whisper
                        try:
                            result = model.transcribe(
                                audio_f32,
                                language=self._language,
                                fp16=False,
                                no_speech_threshold=0.3,
                            )
                            text = result.get("text", "").strip()
                        except Exception as exc:
                            logger.warning("Whisper transcription failed: %s", exc)
                            continue

                        if not text:
                            continue

                        text_lower = text.lower()
                        text_normalized = _normalize_whisper_text(text_lower)
                        logger.debug("Wake detector heard: %r (normalized: %r)", text_lower, text_normalized)

                        # Check for any matching wake phrase (try both raw and normalized)
                        for idx, phrase in enumerate(self._phrases):
                            if phrase in text_lower or phrase in text_normalized:
                                logger.info(
                                    "Wake phrase detected: '%s' in '%s'",
                                    phrase, text_lower,
                                )
                                self._wake_audio = raw_pcm
                                self._event_queue.put(idx)
                                # Wait for main thread to pause us before
                                # re-opening the mic (avoids contention)
                                self._paused.wait(timeout=3.0)
                                break
                        else:
                            continue
                        # Matched — break inner loop to re-check pause flag
                        break

            except (OSError, AttributeError) as exc:
                # AttributeError: speech_recognition's Microphone.__exit__
                # calls self.stream.close() even when the stream was never
                # opened (e.g. wrong device index, mic busy/missing).
                if not self._stop.is_set():
                    logger.warning("Microphone error in wake detector: %s", exc)
                    time.sleep(2.0)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.error("Wake detector error: %s", exc)
                    time.sleep(1.0)
