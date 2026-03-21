"""Dedicated vision LLM provider with API key cycling.

Uses a separate (cheaper/faster) model for describe_screen/describe_image
calls, cycling through multiple API keys to spread rate limits.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import date
from typing import List, Optional

logger = logging.getLogger(__name__)


class VisionProvider:
    """Lightweight vision-only provider with round-robin key cycling."""

    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        max_requests_per_key_per_day: int = 100,
        temperature: float = 0.3,
    ) -> None:
        from openai import OpenAI

        if not api_keys:
            raise ValueError("VisionProvider requires at least one API key.")

        self._api_keys = api_keys
        self._model = model
        self._base_url = base_url
        self._max_per_key = max_requests_per_key_per_day
        self._temperature = temperature

        # One client per key
        self._clients: List[OpenAI] = []
        for key in api_keys:
            self._clients.append(OpenAI(api_key=key, base_url=base_url))

        # Daily request counters: {key_index: count}
        self._daily_counts: List[int] = [0] * len(api_keys)
        self._count_date: date = date.today()
        self._current_index: int = 0

        logger.info(
            "VisionProvider: %d key(s), model=%s, max %d req/key/day",
            len(api_keys), model, max_requests_per_key_per_day,
        )

    def _reset_if_new_day(self) -> None:
        """Reset counters at midnight."""
        today = date.today()
        if today != self._count_date:
            self._daily_counts = [0] * len(self._api_keys)
            self._count_date = today
            self._current_index = 0
            logger.info("VisionProvider: daily counters reset.")

    def _next_client(self) -> Optional["OpenAI"]:
        """Get the next available client, cycling through keys.
        Returns None if all keys are exhausted for today."""
        self._reset_if_new_day()
        n = len(self._clients)
        for _ in range(n):
            idx = self._current_index % n
            if self._daily_counts[idx] < self._max_per_key:
                self._daily_counts[idx] += 1
                self._current_index = (idx + 1) % n
                remaining = sum(
                    self._max_per_key - c for c in self._daily_counts
                )
                if self._daily_counts[idx] % 20 == 0:
                    logger.info(
                        "VisionProvider: key %d used %d/%d today (%d total remaining)",
                        idx, self._daily_counts[idx], self._max_per_key, remaining,
                    )
                return self._clients[idx]
            self._current_index = (idx + 1) % n
        logger.warning("VisionProvider: all keys exhausted for today!")
        return None

    def describe_screen(self, jpeg_bytes: bytes) -> Optional[str]:
        """Describe what's on a computer screen."""
        client = self._next_client()
        if client is None:
            return None

        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a screen reader providing detailed descriptions of a computer screen "
                            "for someone who cannot see it. Your output is consumed by another AI, not a human."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this screenshot in detail. Include:\n"
                                    "1. APPLICATIONS: Which programs/windows are open, which is focused\n"
                                    "2. TEXT/OCR: Read and transcribe any visible text — titles, tabs, chat messages, "
                                    "code, articles, captions, notifications, URLs. Quote key text verbatim.\n"
                                    "3. MEDIA: If a video/stream/game is playing, describe what's happening in it\n"
                                    "4. ACTIVITY: What the user appears to be doing (browsing, coding, chatting, gaming, etc.)\n"
                                    "Ignore the small animated pony sprite — that's a desktop pet overlay, not relevant.\n"
                                    "Be thorough. The more detail you provide, the better."
                                ),
                            },
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=self._temperature,
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] vision describe_screen() took %.2fs", elapsed)
            return response.choices[0].message.content or None
        except Exception as exc:
            logger.warning("VisionProvider describe_screen failed: %s", exc)
            return None

    def describe_image(self, jpeg_bytes: bytes) -> Optional[str]:
        """Describe what's in an image (webcam)."""
        client = self._next_client()
        if client is None:
            return None

        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise visual observer. Describe what you see in the image concisely in 1-3 sentences. Focus on people, objects, environment, and notable details.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": "What do you see?"},
                        ],
                    },
                ],
                max_tokens=150,
                temperature=self._temperature,
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] vision describe_image() took %.2fs", elapsed)
            return response.choices[0].message.content or None
        except Exception as exc:
            logger.warning("VisionProvider describe_image failed: %s", exc)
            return None

    def locate_on_screen(
        self,
        description: str,
        jpeg_bytes: bytes,
        original_size: tuple[int, int],
    ) -> Optional[tuple[int, int]]:
        """Find something on screen and return its real pixel coordinates.

        Args:
            description: What to look for (e.g. "the blue button", "the blueberry image")
            jpeg_bytes: Screenshot JPEG (may be scaled down to max_width)
            original_size: (width, height) of the original unscaled screenshot

        Returns:
            (x, y) in real screen coordinates, or None if not found.
        """
        client = self._next_client()
        if client is None:
            return None

        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        try:
            # Determine the scaled image dimensions for coordinate mapping
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(jpeg_bytes))
            img_w, img_h = img.size

            t0 = time.time()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise visual locator. Given a screenshot and a description, "
                            "return the approximate CENTER pixel coordinates of the described element "
                            f"within this {img_w}x{img_h} image. "
                            "Return ONLY a JSON object, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    f'Find: "{description}"\n'
                                    f'Return JSON: {{"x": <pixel_x>, "y": <pixel_y>}}\n'
                                    f'If not found, return: {{"x": null, "y": null}}'
                                ),
                            },
                        ],
                    },
                ],
                max_tokens=50,
                temperature=0.1,
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] vision locate_on_screen() took %.2fs", elapsed)

            raw = response.choices[0].message.content or ""
            # Extract JSON from response
            import json, re
            json_match = re.search(r'\{[^}]+\}', raw)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                return None

            # Scale from image coordinates to real screen coordinates
            orig_w, orig_h = original_size
            scale_x = orig_w / img_w
            scale_y = orig_h / img_h
            real_x = int(x * scale_x)
            real_y = int(y * scale_y)
            logger.info("locate_on_screen(%r): image(%d,%d) -> real(%d,%d)", description, x, y, real_x, real_y)
            return (real_x, real_y)

        except Exception as exc:
            logger.warning("VisionProvider locate_on_screen failed: %s", exc)
            return None
