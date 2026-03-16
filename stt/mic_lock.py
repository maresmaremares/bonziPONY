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
            mic.__exit__(None, None, None)
