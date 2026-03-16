"""
Speech-to-text transcriber.

Records from microphone using the SpeechRecognition library's energy-based
endpoint detection, then transcribes locally with OpenAI Whisper.

The library automatically calibrates to ambient noise, detects when speech
starts (energy above threshold), and stops recording after a configurable
pause in speech — much better at knowing when you're done talking than
raw VAD frame counting.

Flow:
  1. Open mic stream (16 kHz, mono)
  2. Calibrate to ambient noise level
  3. Wait for speech energy above threshold
  4. Record until pause_threshold seconds of silence
  5. Transcribe locally with Whisper
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1

# Common Whisper hallucinations when processing silence/noise
_WHISPER_HALLUCINATIONS = {
    "", "thank you.", "thanks for watching.", "thank you for watching.",
    "bye.", "goodbye.", "you", "the", "i", "a", "so", "okay",
    "thanks.", "thank you!", "bye!", "hmm.", "hm.", "ah.",
    "subtitles by the amara.org community",
    "subscribe", "like and subscribe",
    "please subscribe", "thanks for watching",
    "thank you for listening", "thank you so much for watching",
    "see you next time", "see you in the next video",
    "music", "applause", "laughter",
}


def _is_whisper_hallucination(text: str) -> bool:
    """Check if transcribed text is a known Whisper hallucination."""
    if not text:
        return True
    clean = text.strip().lower()
    # Exact match against known hallucinations
    if clean in _WHISPER_HALLUCINATIONS:
        return True
    # Single character or just punctuation
    if len(clean) <= 2:
        return True
    # Music/sound markers: ♪, [Music], (music), etc.
    if clean.startswith(("♪", "[", "(")) and len(clean) < 20:
        return True
    # Repeated single word/character: "you you you you"
    words = clean.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


class _ListenInterrupted(Exception):
    """Raised by the stream wrapper when listening is interrupted by user click."""
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames


class _InterruptableStream:
    """Wraps a PyAudio stream so we can interrupt recording via a threading.Event."""

    def __init__(self, stream, stop_event: threading.Event) -> None:
        self._stream = stream
        self._stop_event = stop_event
        self._frames: list[bytes] = []

    def read(self, size, **kwargs):
        if self._stop_event.is_set():
            raise _ListenInterrupted(list(self._frames))
        data = self._stream.read(size, **kwargs)
        self._frames.append(data)
        return data

    def close(self):
        return self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class Transcriber:
    """Mic → Energy-based endpoint detection → Voice filter → Whisper STT."""

    def __init__(
        self,
        model_name: str = "base",
        language: str = "en",
        vad_aggressiveness: int = 2,
        silence_duration_ms: int = 800,
        input_device_index: int = -1,
    ) -> None:
        self.language = language
        self.model_name = model_name
        self.silence_duration_s = silence_duration_ms / 1000.0
        self.input_device_index = input_device_index if input_device_index >= 0 else None

        self._recognizer = None  # lazy-loaded
        self._whisper_model = None  # lazy-loaded
        self._stop_event = threading.Event()

        # Called right after recording finishes, before Whisper runs
        # Lets the GUI transition away from LISTEN state immediately
        self.on_recording_done = None

        # Speaker verification — lazy-loaded, only if profile exists
        self._voice_filter = None
        self._voice_filter_checked = False

    def interrupt_listening(self) -> None:
        """Interrupt active listening — process whatever audio was captured so far."""
        self._stop_event.set()

    def _get_recognizer(self):
        if self._recognizer is None:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            # pause_threshold: seconds of non-speech before recording stops
            # 1.3s = responds quickly after you stop talking but still handles
            # natural mid-sentence pauses (breathing, thinking)
            self._recognizer.pause_threshold = 1.3
            # non_speaking_duration: how much silence BEFORE speech to include
            # Helps capture the start of words that begin softly
            self._recognizer.non_speaking_duration = 0.5
            # Let the library auto-adjust energy threshold based on ambient noise
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.energy_threshold = 150
            self._recognizer.dynamic_energy_adjustment_damping = 0.10
            self._recognizer.dynamic_energy_ratio = 1.4
            logger.info(
                "Recognizer initialized (pause_threshold=%.1fs)",
                self._recognizer.pause_threshold,
            )
        return self._recognizer

    def _get_whisper_model(self):
        if self._whisper_model is None:
            import whisper
            logger.info("Loading Whisper model '%s' for transcription...", self.model_name)
            self._whisper_model = whisper.load_model(self.model_name)
            logger.info("Whisper model '%s' loaded.", self.model_name)
        return self._whisper_model

    def _get_voice_filter(self):
        """Lazy-load voice filter. Returns None if resemblyzer not installed."""
        if not self._voice_filter_checked:
            self._voice_filter_checked = True
            try:
                from stt.voice_filter import VoiceFilter
                self._voice_filter = VoiceFilter()
                if self._voice_filter.enrolled:
                    logger.info("Voice filter active — only user's voice will be transcribed.")
                else:
                    logger.info("Voice filter available but no profile enrolled. Run voice enrollment to enable.")
            except ImportError:
                logger.info("resemblyzer not installed — voice filter disabled.")
            except Exception as exc:
                logger.warning("Voice filter init failed: %s", exc)
        return self._voice_filter

    @property
    def voice_filter(self):
        """Access the voice filter (for enrollment from GUI)."""
        return self._get_voice_filter()

    def listen(self, speech_start_timeout_s: float = 0.0, initial_discard_ms: int = 0) -> Optional[str]:
        """
        Record until silence, then return transcription via local Whisper.
        Returns None if nothing was captured or transcription is empty.

        speech_start_timeout_s: if > 0, give up waiting for speech to BEGIN
            after this many seconds (used for conversation follow-up windows).
        initial_discard_ms: discard this many ms of mic input before listening.
            Use after TTS playback to flush echo/bleed from the input buffer.
        """
        import speech_recognition as sr
        from stt.mic_lock import safe_microphone

        self._stop_event.clear()
        recognizer = self._get_recognizer()

        mic_kwargs = {"sample_rate": SAMPLE_RATE}
        if self.input_device_index is not None:
            mic_kwargs["device_index"] = self.input_device_index

        try:
            with safe_microphone(**mic_kwargs) as source:
                # Calibrate to ambient noise — also drains any TTS echo from the buffer
                calibrate_s = max(0.3, initial_discard_ms / 1000.0)
                recognizer.adjust_for_ambient_noise(source, duration=calibrate_s)

                logger.debug("Listening for speech…")

                # Wrap the stream so a user click can interrupt recording
                wrapper = _InterruptableStream(source.stream, self._stop_event)
                source.stream = wrapper

                timeout = speech_start_timeout_s if speech_start_timeout_s > 0 else None
                try:
                    audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
                except _ListenInterrupted as exc:
                    logger.info("Listening interrupted — processing %d captured chunks.", len(exc.frames))
                    if not exc.frames:
                        return None
                    frame_data = b"".join(exc.frames)
                    # Need at least ~0.5s of audio (8000 samples at 16kHz) for Whisper
                    if len(frame_data) < SAMPLE_RATE:
                        logger.info("Interrupted audio too short (%d bytes) — skipping.", len(frame_data))
                        return None
                    audio = sr.AudioData(frame_data, SAMPLE_RATE, 2)
                except sr.WaitTimeoutError:
                    logger.debug("Speech start timeout — no speech detected.")
                    return None

            # Recording done — notify GUI so mic icon goes away before Whisper runs
            if self.on_recording_done:
                try:
                    self.on_recording_done()
                except Exception:
                    pass

            # Get raw audio for voice filter check and Whisper transcription
            audio_data = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
            audio_f32 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Speaker verification — skip transcription if audio isn't the user
            vf = self._get_voice_filter()
            if vf and vf.enrolled and not vf.is_user(audio_f32):
                logger.info("Voice filter: rejected audio (not the enrolled user).")
                return None

            # Transcribe locally with Whisper
            logger.debug("Transcribing %d samples via Whisper (%s)…", len(audio_f32), self.model_name)
            try:
                model = self._get_whisper_model()
                result = model.transcribe(
                    audio_f32,
                    language=self.language,
                    fp16=False,
                )
                text = result.get("text", "").strip()
                if text:
                    if _is_whisper_hallucination(text):
                        logger.debug("Whisper hallucination filtered: %r", text)
                        return None
                    logger.debug("Transcription: %r", text)
                    return text
                else:
                    logger.debug("Whisper returned empty transcription.")
                    return None
            except Exception as exc:
                logger.error("Whisper transcription failed: %s", exc)
                return None

        except OSError as exc:
            logger.error("Microphone error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Listening failed: %s", exc)
            return None
