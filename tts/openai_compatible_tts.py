"""
OpenAI-compatible TTS — sends text to a local or remote server that
implements the OpenAI /v1/audio/speech endpoint.

Works with ponyvoicetool, AllTalk, OpenedAI-Speech, and any other
server that speaks the same format.
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2  # 16-bit

# ── ponyvoicetool voice mapping ──────────────────────────────────────────────
# Maps character slugs to the voice name ponyvoicetool expects.
# Variant characters (filly, etc.) are resolved to their base voice automatically.
_PVT_VOICE_MAP: dict[str, str] = {
    "pinkie_pie": "Pinkie Pie",
    "applejack": "Applejack",
    "rainbow_dash": "Rainbow Dash",
    "rarity": "Rarity",
    "twilight_sparkle": "Twilight Sparkle",
    "princess_twilight_sparkle": "Twilight Sparkle",
    "fluttershy": "Fluttershy",
    "spike": "Spike",
    "trixie": "Trixie",
    "starlight_glimmer": "Starlight Glimmer",
    "princess_celestia": "Princess Celestia",
    "princess_luna": "Princess Luna",
    "princess_cadance": "Princess Cadance",
    "nightmare_moon": "Nightmare Moon",
    "discord": "Discord",
    "apple_bloom": "Apple Bloom",
    "scootaloo": "Scootaloo",
    "sweetie_belle": "Sweetie Belle",
    "octavia": "Octavia",
    "derpy_hooves": "Derpy Hooves",
    "queen_chrysalis": "Queen Chrysalis",
    "cozy_glow": "Cozy Glow",
    "tirek": "Tirek",
    "flim": "Flim",
    "flam": "Flam",
}


def _resolve_pvt_voice(slug: str) -> Optional[str]:
    """Resolve a character slug to a ponyvoicetool voice name.

    Handles variant characters (e.g. "applejack_filly" → "Applejack")
    by trying progressively shorter slug prefixes.
    Returns None if no matching voice is found.
    """
    if slug in _PVT_VOICE_MAP:
        return _PVT_VOICE_MAP[slug]
    # Try progressively shorter prefixes for variants
    parts = slug.split("_")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:i])
        if prefix in _PVT_VOICE_MAP:
            return _PVT_VOICE_MAP[prefix]
    return None


def get_pvt_voice_for(slug: str) -> Optional[str]:
    """Public accessor — returns the PVT voice name for a slug, or None."""
    return _resolve_pvt_voice(slug)


def has_pvt_voice(slug: str) -> bool:
    """Return True if this character has a ponyvoicetool voice."""
    return _resolve_pvt_voice(slug) is not None


class OpenAICompatibleTTS:
    """Sends TTS requests to an OpenAI-compatible /v1/audio/speech endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8069/v1",
        model: str = "ponyvoicetool",
        voice: str = "default",
        response_format: str = "pcm",
        sample_rate: int = 24000,
        output_device_index: int = -1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self._sample_rate = sample_rate
        self.output_device_index = output_device_index if output_device_index >= 0 else None

    def set_character(self, slug: str) -> None:
        """Switch to the correct ponyvoicetool voice for a character.

        If the character has a PVT voice, use it. Otherwise keep current voice.
        """
        voice = _resolve_pvt_voice(slug)
        if voice:
            self.voice = voice
            logger.info("PVT voice set to: %s (character: %s)", voice, slug)
        else:
            logger.debug("No PVT voice for %s — keeping current voice: %s", slug, self.voice)

    def speak(self, text: str, on_playback_start=None) -> None:
        """Convert text to speech via the OpenAI-compatible endpoint. Blocks until done.

        Args:
            on_playback_start: Optional callback invoked right before audio playback begins.
        """
        if not text.strip():
            return

        logger.debug("TTS (OpenAI-compat): %r", text)

        try:
            import requests

            url = f"{self.base_url}/audio/speech"
            payload = {
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": self.response_format,
            }

            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            pcm_bytes = resp.content

        except Exception as exc:
            logger.error("OpenAI-compatible TTS request failed: %s", exc)
            # Still show the speech bubble even if TTS fails
            if on_playback_start:
                try:
                    on_playback_start()
                except Exception:
                    pass
            return

        if not pcm_bytes:
            logger.warning("TTS server returned empty audio.")
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
        """Play audio bytes — auto-detects WAV vs raw PCM."""
        import io

        # Detect WAV format (starts with "RIFF" header)
        if pcm_bytes[:4] == b"RIFF":
            import soundfile as sf
            data, sample_rate = sf.read(io.BytesIO(pcm_bytes), dtype="float32")
            audio_f32 = data
        else:
            # Raw 16-bit PCM
            num_samples = len(pcm_bytes) // PCM_SAMPLE_WIDTH
            samples_int16 = struct.unpack(f"{num_samples}h", pcm_bytes[:num_samples * PCM_SAMPLE_WIDTH])
            audio_f32 = np.array(samples_int16, dtype=np.float32) / 32768.0
            sample_rate = self._sample_rate

        play_kwargs: dict = {"samplerate": sample_rate}
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
