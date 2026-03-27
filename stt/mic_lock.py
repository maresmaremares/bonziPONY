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

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# PortAudio error codes that indicate the device/backend is unavailable
# (not a hard programming error — worth retrying with a different device)
_PYAUDIO_DEVICE_ERRORS = (-9999, -9996, -9998, -9986, -9985)

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


def _open_microphone_with_fallback(**kwargs):
    """Try to open an sr.Microphone, falling back on PortAudio device errors.

    Fallback order:
      1. Exact kwargs (user's configured device + sample_rate)
      2. Same device, no explicit sample_rate (use device default; get_raw_data resamples)
      3. Enumerate all input devices, try each without explicit sample_rate
    """
    def _try(**kw):
        m = sr.Microphone(**kw)
        s = m.__enter__()
        return m, s

    # ── Attempt 1: exactly as requested ──────────────────────────
    try:
        return _try(**kwargs)
    except OSError as first_err:
        if first_err.errno not in _PYAUDIO_DEVICE_ERRORS:
            raise
        logger.warning("Mic open failed (%s) — trying fallbacks", first_err)

    # ── Attempt 2: drop sample_rate, keep device (if specified) ──
    kw_no_rate = {k: v for k, v in kwargs.items() if k != "sample_rate"}
    if kw_no_rate != kwargs:
        try:
            m, s = _try(**kw_no_rate)
            logger.info("Mic: opened without explicit sample_rate (device default)")
            return m, s
        except OSError:
            pass

    # ── Attempt 3: enumerate all input devices ────────────────────
    try:
        import pyaudio
        pa_tmp = pyaudio.PyAudio()
        device_count = pa_tmp.get_device_count()
        device_names = {
            i: pa_tmp.get_device_info_by_index(i).get("name", "?")
            for i in range(device_count)
            if pa_tmp.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
        }
        pa_tmp.terminate()
    except Exception:
        device_names = {}

    for idx, name in device_names.items():
        try:
            m, s = _try(**{**kw_no_rate, "device_index": idx})
            logger.info("Mic: using fallback device %d (%s)", idx, name)
            return m, s
        except Exception:
            continue

    # All attempts failed
    raise OSError(-9999, "No working audio input device found. "
                  "Run: python scripts/list_audio_devices.py "
                  "and set input_device_index in config.yaml")


@contextmanager
def safe_microphone(**kwargs):
    """Context manager that wraps sr.Microphone with thread-safe init/exit.

    Acquires a global lock during PyAudio creation (Microphone.__init__ +
    __enter__) and destruction (__exit__ + PyAudio.terminate), but releases
    it during actual listening so both detector and transcriber aren't
    blocked from capturing audio simultaneously when properly sequenced.

    Automatically falls back through available devices if the default/requested
    device fails with a PortAudio error (e.g. -9999 in VMs or with WASAPI).
    """
    with _mic_lock:
        mic, source = _open_microphone_with_fallback(**kwargs)
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
