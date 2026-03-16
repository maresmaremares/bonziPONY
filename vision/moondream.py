"""Moondream — lightweight local vision model for cheap screen descriptions."""

from __future__ import annotations

import io
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe what's on this computer screen concisely in 2-3 sentences. "
    "Focus on which applications or windows are open, what content is displayed, "
    "and any notable text or activity. Ignore any small animated sprite overlay."
)


class MoondreamDescriber:
    """Moondream2 vision model for local screen descriptions.

    The model is loaded ONLY via ``start_background_load()`` — never lazily
    during a pipeline call.  This prevents the 1–2 GB download / load from
    blocking conversations or crashing the app.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if use_gpu else "cpu"
        self._available = True
        self._loading = False
        self._lock = threading.Lock()

    # ── Loading ──────────────────────────────────────────────────────────

    def start_background_load(self) -> None:
        """Kick off model loading in a daemon thread.  Safe to call multiple times."""
        with self._lock:
            if self._model is not None or self._loading or not self._available:
                return
            self._loading = True
        t = threading.Thread(target=self._load, daemon=True, name="moondream-loader")
        t.start()

    def _load(self) -> bool:
        """Load model.  Called from background thread only."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            free_gb = mem.available / (1024 ** 3)
            if free_gb < 2.0:
                logger.warning(
                    "Moondream skipped — only %.1f GB RAM free (need ≥2 GB).", free_gb
                )
                self._available = False
                self._loading = False
                return False
        except ImportError:
            pass  # psutil not installed, skip the check

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "vikhyatk/moondream2"
            logger.info("Loading Moondream2 (%s)...", self._device)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True,
            ).to(self._device).eval()
            with self._lock:
                self._tokenizer = tokenizer
                self._model = model
                self._loading = False
            logger.info("Moondream2 ready on %s.", self._device)
            return True
        except Exception as exc:
            logger.warning("Moondream2 failed to load: %s", exc)
            with self._lock:
                self._available = False
                self._loading = False
            return False

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def loaded(self) -> bool:
        """True only if model is already in memory."""
        return self._model is not None

    # ── Inference ────────────────────────────────────────────────────────

    def describe(self, jpeg_bytes: bytes) -> Optional[str]:
        """Describe a screenshot.  Returns None if model isn't loaded yet."""
        if not self._available or self._model is None:
            return None
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            enc_image = self._model.encode_image(img)
            result = self._model.answer_question(enc_image, _PROMPT, self._tokenizer)
            return result.strip() if result else None
        except Exception as exc:
            logger.warning("Moondream describe failed: %s", exc)
            return None
