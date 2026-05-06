"""
CLIP + OCR continuous screen understanding — lets Dash watch along with the user.

Runs on a daemon thread, captures the screen every few seconds, uses CLIP for
scene understanding and OCR for subtitle extraction. Zero API cost.
"""

from __future__ import annotations

import collections
import difflib
import logging
import sys
import threading
import time
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _resolve_ocr_engine(name: str) -> str:
    """Resolve an OCR engine name. 'auto' picks the best for the current platform.

    'winocr' is Windows-only and silently falls through to 'pytesseract' on Linux/macOS
    (logged once so the user knows their config setting was overridden)."""
    requested = (name or "auto").lower().strip()
    if requested == "auto":
        return "winocr" if sys.platform == "win32" else "pytesseract"
    if requested == "winocr" and sys.platform != "win32":
        logger.info("OCR engine 'winocr' is Windows-only; using 'pytesseract' instead.")
        return "pytesseract"
    return requested

# Scene classification prompts for CLIP zero-shot
_SCENE_PROMPTS = [
    "an action scene with fighting or explosions",
    "a dialogue scene with people talking",
    "a romantic or intimate scene",
    "a comedy scene, something funny",
    "a dramatic or emotional scene, someone crying",
    "a chase or pursuit scene",
    "a horror or scary scene",
    "a musical performance or concert",
    "a landscape or scenic establishing shot",
    "a title screen or opening credits",
    "a news broadcast or interview",
    "an advertisement or commercial break",
    "a sports event or competition",
    "a video game being played",
    "an animated cartoon or anime",
    "a nature or wildlife documentary",
    "a cooking or food scene",
    "a quiet calm peaceful scene",
]


class WatchMode:
    """Continuous screen understanding via local CLIP + OCR."""

    def __init__(
        self,
        screen_capture,
        capture_interval: float = 2.5,
        scene_change_threshold: float = 0.85,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        ocr_engine: str = "auto",
        subtitle_region_pct: float = 0.20,
        use_gpu: bool = False,
    ) -> None:
        self._screen = screen_capture
        self._capture_interval = capture_interval
        self._scene_threshold = scene_change_threshold
        self._clip_model_name = clip_model_name
        self._ocr_engine = _resolve_ocr_engine(ocr_engine)
        self._subtitle_pct = subtitle_region_pct
        self._use_gpu = use_gpu
        logger.info("WatchMode OCR engine: %s", self._ocr_engine)

        self.enabled = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # CLIP (lazy loaded)
        self._clip_model = None
        self._clip_processor = None
        self._text_embeddings = None  # cached scene prompt embeddings
        self._device = "cpu"

        # State
        self._prev_embedding = None
        self._current_scene: str = "unknown"
        self._current_confidence: float = 0.0
        self._scene_change_count: int = 0
        self._last_scene_change: float = 0.0
        self._subtitle_buffer: Deque[Tuple[float, str]] = collections.deque(maxlen=20)
        self._event_log: Deque[Tuple[float, str]] = collections.deque(maxlen=30)
        self._interesting_event: Optional[str] = None

    def start(self) -> None:
        """Load models and start the capture thread."""
        if self._running:
            return
        if self._screen is None:
            logger.warning("WatchMode: no screen capture available.")
            return

        logger.info("WatchMode starting — loading CLIP model...")
        try:
            self._load_clip()
        except Exception as exc:
            logger.error("WatchMode: failed to load CLIP: %s", exc)
            return

        self.enabled = True
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="watch-mode")
        self._thread.start()
        logger.info("WatchMode started (interval=%.1fs, model=%s).", self._capture_interval, self._clip_model_name)

    def stop(self) -> None:
        """Stop the capture thread."""
        self._running = False
        self.enabled = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("WatchMode stopped.")

    # ── Public API ────────────────────────────────────────────────────────

    def get_context(self) -> Optional[str]:
        """Return a compact text summary of what's happening on screen, or None if disabled."""
        if not self.enabled:
            return None

        with self._lock:
            lines = ["[WATCH MODE — watching with user]"]
            lines.append(f"Scene: {self._current_scene} (confidence: {self._current_confidence:.2f})")

            # Recent scene changes
            now = time.monotonic()
            recent_changes = sum(1 for t, _ in self._event_log if (now - t) < 120)
            lines.append(f"Scene changes in last 2 min: {recent_changes}")

            # Recent subtitles
            if self._subtitle_buffer:
                lines.append("Recent subtitles:")
                for _, text in list(self._subtitle_buffer)[-5:]:
                    lines.append(f'  - "{text}"')

            # Recent events
            recent_events = [(t, e) for t, e in self._event_log if (now - t) < 60]
            if recent_events:
                lines.append("Recent events:")
                for t, event in recent_events[-5:]:
                    ago = int(now - t)
                    lines.append(f"  - [{ago}s ago] {event}")

            return "\n".join(lines)

    def get_interesting_event(self) -> Optional[str]:
        """Pop and return the last interesting event (one-shot). Returns None if nothing."""
        with self._lock:
            event = self._interesting_event
            self._interesting_event = None
            return event

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(features):
        """Extract raw tensor from CLIP output (handles both old and new transformers)."""
        import torch
        if isinstance(features, torch.Tensor):
            return features
        # Newer transformers may return BaseModelOutputWithPooling or similar
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if hasattr(features, "last_hidden_state"):
            return features.last_hidden_state[:, 0]
        # Fallback — try indexing
        return features[0]

    def _load_clip(self) -> None:
        """Load CLIP model and pre-compute text embeddings for scene prompts."""
        import torch
        from transformers import CLIPProcessor, CLIPModel

        self._device = "cuda" if (self._use_gpu and torch.cuda.is_available()) else "cpu"
        logger.info("WatchMode: loading CLIP on %s", self._device)

        self._clip_model = CLIPModel.from_pretrained(self._clip_model_name)
        self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)
        self._clip_model.eval()
        self._clip_model.to(self._device)

        # Pre-compute text embeddings for all scene prompts
        inputs = self._clip_processor(text=_SCENE_PROMPTS, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self._to_tensor(self._clip_model.get_text_features(**inputs))
            self._text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

        logger.info("WatchMode: CLIP loaded, %d scene prompts cached.", len(_SCENE_PROMPTS))

    def _capture_loop(self) -> None:
        """Daemon thread: capture and analyze screen at interval."""
        while self._running:
            try:
                self._process_frame()
            except Exception as exc:
                logger.debug("WatchMode frame error: %s", exc)
            time.sleep(self._capture_interval)

    def _process_frame(self) -> None:
        """Capture one frame, run CLIP + OCR."""
        import torch

        # Grab screen as PIL image
        img = self._screen.grab_pil()
        if img is None:
            return

        # CLIP: embed the frame
        inputs = self._clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self._to_tensor(self._clip_model.get_image_features(**inputs))
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)

        # Scene change detection
        scene_changed = False
        if self._prev_embedding is not None:
            similarity = (embedding @ self._prev_embedding.T).item()
            if similarity < self._scene_threshold:
                scene_changed = True
                big_change = similarity < 0.70
        else:
            scene_changed = True
            big_change = False

        # Scene classification
        scores = (embedding @ self._text_embeddings.T).squeeze(0)
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        new_scene = _SCENE_PROMPTS[best_idx]

        # OCR on subtitle region
        subtitle = self._run_ocr(img)

        # Update state
        with self._lock:
            old_scene = self._current_scene
            self._current_scene = new_scene
            self._current_confidence = best_score
            self._prev_embedding = embedding

            if scene_changed:
                self._scene_change_count += 1
                self._last_scene_change = time.monotonic()
                if old_scene != "unknown":
                    event_text = f"Scene changed: {old_scene.split(',')[0]} -> {new_scene.split(',')[0]}"
                    self._event_log.append((time.monotonic(), event_text))
                    # Big scene changes are interesting
                    if big_change:
                        self._interesting_event = event_text

            if subtitle:
                # Deduplicate: only add if sufficiently different from last subtitle
                if self._subtitle_buffer:
                    last_sub = self._subtitle_buffer[-1][1]
                    ratio = difflib.SequenceMatcher(None, last_sub.lower(), subtitle.lower()).ratio()
                    if ratio < 0.8:
                        self._subtitle_buffer.append((time.monotonic(), subtitle))
                        self._event_log.append((time.monotonic(), f'Subtitle: "{subtitle}"'))
                else:
                    self._subtitle_buffer.append((time.monotonic(), subtitle))
                    self._event_log.append((time.monotonic(), f'Subtitle: "{subtitle}"'))

    def _run_ocr(self, img) -> Optional[str]:
        """Run OCR on the bottom subtitle region of the frame."""
        try:
            # Crop bottom region
            w, h = img.size
            top = int(h * (1.0 - self._subtitle_pct))
            cropped = img.crop((0, top, w, h))

            if self._ocr_engine == "winocr":
                return self._ocr_winocr(cropped)
            else:
                return self._ocr_tesseract(cropped)
        except Exception as exc:
            logger.debug("WatchMode OCR failed: %s", exc)
            return None

    def _ocr_winocr(self, img) -> Optional[str]:
        """OCR via Windows built-in OCR (winocr package). Windows-only."""
        try:
            import asyncio
            from winocr import recognize_pil  # type: ignore
        except ImportError:
            logger.warning("winocr not available — OCR disabled for this frame.")
            self._ocr_engine = "pytesseract"  # don't keep retrying
            return None
        result = asyncio.run(recognize_pil(img, lang="en"))
        text = result.text.strip() if result and result.text else ""
        return text if len(text) > 3 else None  # ignore tiny noise

    def _ocr_tesseract(self, img) -> Optional[str]:
        """OCR via pytesseract (needs the `tesseract` binary on PATH)."""
        try:
            import pytesseract  # type: ignore
        except ImportError:
            logger.warning("pytesseract not installed — OCR disabled for this frame.")
            return None
        try:
            text = pytesseract.image_to_string(img, config="--psm 6").strip()
        except pytesseract.TesseractNotFoundError:
            logger.warning("tesseract binary not found on PATH; install 'tesseract-ocr' to enable OCR.")
            return None
        except Exception as exc:
            logger.debug("pytesseract failed: %s", exc)
            return None
        return text if len(text) > 3 else None
