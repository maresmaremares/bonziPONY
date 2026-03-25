"""
ElevenLabs TTS → PCM → sounddevice playback.

Uses output_format="pcm_22050" to receive raw PCM directly,
bypassing any MP3 decode dependency.
"""

from __future__ import annotations

import logging
import struct

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 22050
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # 16-bit


class ElevenLabsTTS:
    """Streams ElevenLabs TTS audio and plays it via sounddevice."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2",
        output_format: str = "pcm_22050",
        output_device_index: int = -1,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.output_format = output_format
        self.output_device_index = output_device_index if output_device_index >= 0 else None

        # Parse sample rate from format string, e.g. "pcm_22050" → 22050
        try:
            self._sample_rate = int(output_format.split("_")[1])
        except (IndexError, ValueError):
            self._sample_rate = PCM_SAMPLE_RATE
            logger.warning(
                "Could not parse sample rate from output_format '%s'. "
                "Defaulting to %d Hz.",
                output_format,
                PCM_SAMPLE_RATE,
            )

    def speak(self, text: str, on_playback_start=None) -> None:
        """Convert text to speech and play it. Blocks until playback finishes.

        Args:
            on_playback_start: Optional callback invoked right before audio playback begins.
        """
        if not text.strip():
            return

        logger.debug("TTS: %r", text)

        try:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=self.api_key)

            # SDK v1.x: text_to_speech.convert() returns an iterator of bytes chunks
            audio_chunks = client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model,
                output_format=self.output_format,
            )

            pcm_bytes = b"".join(audio_chunks)

        except Exception as exc:
            logger.error("ElevenLabs TTS request failed: %s", exc)
            # Still show the speech bubble even if TTS fails
            if on_playback_start:
                try:
                    on_playback_start()
                except Exception:
                    pass
            return

        if not pcm_bytes:
            logger.warning("ElevenLabs returned empty audio.")
            if on_playback_start:
                try:
                    on_playback_start()
                except Exception:
                    pass
            return

        if on_playback_start:
            try:
                on_playback_start()
            except Exception as exc:
                logger.warning("on_playback_start callback failed: %s", exc)
        self._play_pcm(pcm_bytes)

    def _play_pcm(self, pcm_bytes: bytes) -> None:
        """Convert raw 16-bit PCM bytes to float32 and play via sounddevice."""
        num_samples = len(pcm_bytes) // PCM_SAMPLE_WIDTH
        samples_int16 = struct.unpack(f"{num_samples}h", pcm_bytes[:num_samples * PCM_SAMPLE_WIDTH])
        audio_f32 = np.array(samples_int16, dtype=np.float32) / 32768.0

        play_kwargs = {"samplerate": self._sample_rate}
        if self.output_device_index is not None:
            play_kwargs["device"] = self.output_device_index

        # Stop any stuck/leftover audio before starting new playback
        try:
            sd.stop()
        except Exception:
            pass

        sd.play(audio_f32, **play_kwargs)
        sd.wait()
        logger.debug("TTS playback complete.")

    @staticmethod
    def stop() -> None:
        """Immediately stop any playing audio (e.g. user pressed PTT)."""
        try:
            sd.stop()
        except Exception:
            pass
