"""
Speaker identification using MFCC embeddings — no extra dependencies.

Extracts mel-frequency cepstral coefficients from audio, averages them into
a fixed-size speaker embedding, and compares via cosine similarity.  This is
lightweight and fast (~5 ms per clip) but accurate enough to distinguish
"user talking into mic" from "video/TV audio through speakers."

Usage:
    verifier = SpeakerVerifier()
    verifier.enroll([clip1_f32, clip2_f32, clip3_f32])   # numpy float32 arrays @ 16 kHz
    confidence = verifier.verify(new_clip_f32)            # 0.0–1.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# MFCC extraction parameters
_SAMPLE_RATE = 16000
_N_MFCC = 20          # number of cepstral coefficients
_N_FFT = 512          # FFT window size (~32 ms at 16 kHz)
_HOP_LENGTH = 160     # hop size (~10 ms at 16 kHz)
_N_MELS = 40          # mel filterbank size
_FMIN = 80.0          # minimum frequency for mel filterbank
_FMAX = 7600.0        # maximum frequency for mel filterbank

# Verification threshold — above this = "likely the user"
DEFAULT_THRESHOLD = 0.82

# Where enrollment data lives
_PROFILE_DIR = Path(__file__).resolve().parent.parent / "voice_profile"


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """Build a mel-scale filterbank matrix (n_mels × (n_fft//2 + 1))."""
    def _hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_lo = _hz_to_mel(fmin)
    mel_hi = _hz_to_mel(min(fmax, sr / 2.0))
    mels = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if lo == mid:
            mid = lo + 1
        if mid == hi:
            hi = mid + 1
        for k in range(lo, mid):
            fb[i, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            fb[i, k] = (hi - k) / (hi - mid)
    return fb


def _extract_mfcc(audio: np.ndarray, sr: int = _SAMPLE_RATE) -> np.ndarray:
    """Extract MFCCs from a float32 audio array.  Returns (n_mfcc, T) matrix."""
    from scipy.fftpack import dct

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    n_frames = 1 + (len(audio) - _N_FFT) // _HOP_LENGTH
    if n_frames < 1:
        # Audio too short — pad to minimum length
        audio = np.pad(audio, (0, _N_FFT - len(audio)))
        n_frames = 1

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, _N_FFT),
        strides=(audio.strides[0] * _HOP_LENGTH, audio.strides[0]),
    ).copy()

    # Hamming window
    window = np.hamming(_N_FFT).astype(np.float32)
    frames *= window

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, n=_N_FFT)) ** 2
    mag = np.maximum(mag, 1e-10)

    # Mel filterbank
    fb = _mel_filterbank(sr, _N_FFT, _N_MELS, _FMIN, _FMAX)
    mel_spec = mag @ fb.T
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT → MFCCs
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :_N_MFCC]
    return mfcc.T  # (n_mfcc, T)


def _embedding_from_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Collapse a (n_mfcc, T) matrix into a fixed-size embedding.

    Uses mean + standard deviation across time → 2 * n_mfcc dimensions.
    This captures both average vocal quality and variability.
    """
    if mfcc.shape[1] == 0:
        return np.zeros(2 * _N_MFCC, dtype=np.float32)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.  Returns 0.0–1.0."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    sim = dot / (na * nb)
    # Clamp to [0, 1] — negative similarity = very different speaker
    return float(max(0.0, min(1.0, sim)))


class SpeakerVerifier:
    """MFCC-based speaker verification.

    Enroll the user's voice with a few audio clips, then verify new clips
    against the enrolled embedding.  Returns a confidence score (0.0–1.0).

    The enrolled embedding is saved to ``voice_profile/embedding.npy`` and
    persists across restarts.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 profile_dir: Path | None = None) -> None:
        self._threshold = threshold
        self._profile_dir = profile_dir or _PROFILE_DIR
        self._embedding: Optional[np.ndarray] = None
        self._enrolled = False
        self._load()

    @property
    def enrolled(self) -> bool:
        """True if a voice profile has been enrolled."""
        return self._enrolled

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    # ── Enrollment ──────────────────────────────────────────────────────

    def enroll(self, clips: List[np.ndarray], sr: int = _SAMPLE_RATE) -> float:
        """Enroll the user's voice from multiple audio clips.

        Parameters
        ----------
        clips : list of numpy float32 arrays
            3–5 clips of the user speaking (each at least 1 second).
        sr : int
            Sample rate of the clips (default 16000).

        Returns
        -------
        float
            Average self-similarity across enrollment clips (quality metric).
            Values above 0.9 mean good enrollment quality.
        """
        if not clips:
            raise ValueError("Need at least one audio clip for enrollment.")

        embeddings = []
        for i, clip in enumerate(clips):
            if len(clip) < sr * 0.5:  # at least 0.5 seconds
                logger.warning("Enrollment clip %d too short (%.1fs), skipping.",
                               i, len(clip) / sr)
                continue
            mfcc = _extract_mfcc(clip, sr)
            emb = _embedding_from_mfcc(mfcc)
            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid clips for enrollment (all too short).")

        # Centroid embedding = average of all clip embeddings
        self._embedding = np.mean(embeddings, axis=0).astype(np.float32)
        self._enrolled = True
        self._save()

        # Quality metric: average similarity of each clip to the centroid
        sims = [_cosine_similarity(e, self._embedding) for e in embeddings]
        quality = float(np.mean(sims))
        logger.info("Voice enrolled: %d clips, quality=%.3f, embedding_dim=%d",
                     len(embeddings), quality, len(self._embedding))
        return quality

    def clear(self) -> None:
        """Delete the enrolled voice profile."""
        self._embedding = None
        self._enrolled = False
        emb_path = self._profile_dir / "embedding.npy"
        meta_path = self._profile_dir / "metadata.json"
        for p in (emb_path, meta_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("Voice profile cleared.")

    # ── Verification ────────────────────────────────────────────────────

    def verify(self, audio: np.ndarray, sr: int = _SAMPLE_RATE) -> float:
        """Verify whether an audio clip matches the enrolled user.

        Parameters
        ----------
        audio : numpy float32 array
            Audio clip to verify (at 16 kHz).

        Returns
        -------
        float
            Confidence score 0.0–1.0.  Above ``self.threshold`` = likely the user.
            Returns 1.0 if no voice is enrolled (assume everything is the user).
        """
        if not self._enrolled or self._embedding is None:
            return 1.0  # no profile → assume it's the user

        if len(audio) < sr * 0.3:  # less than 0.3 seconds
            return 0.5  # too short to tell

        mfcc = _extract_mfcc(audio, sr)
        emb = _embedding_from_mfcc(mfcc)
        sim = _cosine_similarity(emb, self._embedding)
        logger.debug("Speaker verification: similarity=%.3f (threshold=%.2f)",
                     sim, self._threshold)
        return sim

    def is_user(self, audio: np.ndarray, sr: int = _SAMPLE_RATE) -> bool:
        """Convenience: True if confidence >= threshold."""
        return self.verify(audio, sr) >= self._threshold

    def verify_segments(self, audio: np.ndarray, sr: int = _SAMPLE_RATE,
                        segment_s: float = 1.5,
                        hop_s: float = 0.5) -> List[dict]:
        """Verify speaker across overlapping segments of the audio.

        Returns a list of dicts with start_s, end_s, confidence for each
        segment.  Useful for understanding which parts of a recording
        are the user vs ambient audio.
        """
        if not self._enrolled or self._embedding is None:
            return [{"start_s": 0.0, "end_s": len(audio) / sr, "confidence": 1.0}]

        seg_samples = int(segment_s * sr)
        hop_samples = int(hop_s * sr)
        results = []

        pos = 0
        while pos + seg_samples <= len(audio):
            segment = audio[pos:pos + seg_samples]
            conf = self.verify(segment, sr)
            results.append({
                "start_s": pos / sr,
                "end_s": (pos + seg_samples) / sr,
                "confidence": conf,
            })
            pos += hop_samples

        # Handle tail (if audio doesn't divide evenly)
        if pos < len(audio) and len(audio) - pos > sr * 0.3:
            segment = audio[pos:]
            conf = self.verify(segment, sr)
            results.append({
                "start_s": pos / sr,
                "end_s": len(audio) / sr,
                "confidence": conf,
            })

        return results

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save enrolled embedding to disk."""
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        if self._embedding is not None:
            np.save(self._profile_dir / "embedding.npy", self._embedding)
            meta = {
                "threshold": self._threshold,
                "embedding_dim": len(self._embedding),
                "n_mfcc": _N_MFCC,
            }
            (self._profile_dir / "metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            logger.debug("Voice profile saved to %s", self._profile_dir)

    def _load(self) -> None:
        """Load enrolled embedding from disk if it exists."""
        emb_path = self._profile_dir / "embedding.npy"
        if emb_path.exists():
            try:
                self._embedding = np.load(emb_path)
                self._enrolled = True
                logger.info("Voice profile loaded (%d-dim embedding).",
                            len(self._embedding))
            except Exception as exc:
                logger.warning("Failed to load voice profile: %s", exc)
                self._embedding = None
                self._enrolled = False


# ── CLI enrollment entrypoint ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import speech_recognition as sr
    from stt.mic_lock import safe_microphone

    logging.basicConfig(level=logging.INFO)
    verifier = SpeakerVerifier()

    print("=== Voice Enrollment ===")
    print("You'll record 3 clips of your voice (3 seconds each).")
    print("Speak naturally — say anything.\n")

    clips = []
    for i in range(3):
        input(f"Press Enter to record clip {i+1}/3...")
        recognizer = sr.Recognizer()
        with safe_microphone(sample_rate=_SAMPLE_RATE) as source:
            print("  Recording... (speak for 3 seconds)")
            audio = recognizer.record(source, duration=3)
        raw = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
        clips.append(raw)
        print(f"  Clip {i+1} captured ({len(raw)/16000:.1f}s)")

    quality = verifier.enroll(clips)
    print(f"\nEnrollment complete! Quality: {quality:.3f}")
    if quality > 0.9:
        print("Excellent enrollment quality.")
    elif quality > 0.8:
        print("Good enrollment quality.")
    else:
        print("Low quality — try re-enrolling in a quieter environment.")
