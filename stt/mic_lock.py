"""
Global lock for PyAudio initialization/termination.

PortAudio (the C library under PyAudio) has global state — calling
Pa_Initialize and Pa_Terminate from multiple threads simultaneously
causes heap corruption and access violations.

Both the wake word detector and the transcriber create sr.Microphone
instances (which create/destroy PyAudio). This lock serializes those
calls while allowing actual audio capture to happen concurrently.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

def _ensure_pyaudio_importable() -> None:
    """Ensure `import pyaudio` works (SpeechRecognition expects it).

    PyAudioWPatch ships prebuilt wheels on Windows but installs as
    `pyaudiowpatch`, so we alias it to the expected module name.
    """
    import sys
    try:
        import pyaudio  # noqa: F401
        return
    except Exception:
        pass

    try:
        import pyaudiowpatch as _pa
        sys.modules.setdefault("pyaudio", _pa)
    except Exception:
        # Neither is available; callers will fail gracefully when opening mic.
        return


_ensure_pyaudio_importable()

import speech_recognition as sr

_mic_lock = threading.Lock()


@contextmanager
def safe_microphone(**kwargs):
    """Context manager that wraps sr.Microphone with thread-safe init/exit.

    Acquires a global lock during PyAudio creation (Microphone.__init__ +
    __enter__) and destruction (__exit__ + PyAudio.terminate), but releases
    it during actual listening so both detector and transcriber aren't
    blocked from capturing audio simultaneously when properly sequenced.
    """
    with _mic_lock:
        mic = sr.Microphone(**kwargs)
        source = mic.__enter__()
    try:
        yield source
    finally:
        with _mic_lock:
            try:
                mic.__exit__(None, None, None)
            except AttributeError:
                # sr.Microphone.__exit__ calls self.stream.close() even when
                # the stream was never opened (e.g. device missing/busy).
                # Clean up PyAudio directly if possible.
                if hasattr(mic, "audio") and mic.audio is not None:
                    try:
                        mic.audio.terminate()
                    except Exception:
                        pass
                    mic.stream = None
