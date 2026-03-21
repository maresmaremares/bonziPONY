"""OpenAI (and OpenAI-compatible) LLM provider."""

from __future__ import annotations

import base64
import logging
import time
from typing import List, Optional

import re

from llm.base import LLMProvider
from llm.prompt import get_system_prompt

_MAX_RETRIES = 5
_RETRY_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """GPT-4o (or any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.85,
        max_tokens: int = 600,
        max_history_turns: int = 10,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._history: List[dict] = []

    def _call_with_retry(self, **kwargs):
        """Call chat.completions.create with retries on errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is not None and status >= 400
                if not retryable:
                    retryable = isinstance(exc, (ConnectionError, TimeoutError, OSError))
                if retryable and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning("API error (attempt %d/%d), retrying in %.0fs: %s",
                                   attempt + 1, _MAX_RETRIES, wait, exc)
                    time.sleep(wait)
                    continue
                raise

    def chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        messages = [{"role": "system", "content": get_system_prompt()}]
        # Character prefill: inject an assistant greeting so the model sees
        # itself already in-character.  Skip for Claude models (no prefill support).
        if len(self._history) == 1:
            model_lower = self.model.lower()
            is_claude = any(k in model_lower for k in ("claude", "opus", "sonnet", "haiku"))
            if not is_claude:
                from llm.prompt import get_character_name
                name = get_character_name()
                messages.append({"role": "assistant", "content": f"(I am {name}. I stay in character at all times.)"})
        messages.extend(self._history)

        t0 = time.time()
        response = self._call_with_retry(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = time.time() - t0
        logger.info("[TIMING] chat() API call took %.2fs", elapsed)
        print(f"[LLM] chat() took {elapsed:.2f}s", flush=True)

        # If the response was truncated (hit token limit), retry with a higher limit.
        # WRITE_NOTEPAD gets unlimited (16k) so she can write as much as she wants.
        finish = getattr(response.choices[0], "finish_reason", None) if response.choices else None
        if finish == "length":
            partial = response.choices[0].message.content or ""
            if "WRITE_NOTEPAD" in partial:
                retry_tokens = 16384
                logger.info("WRITE_NOTEPAD truncated — retrying with %d tokens.", retry_tokens)
            else:
                retry_tokens = max(self.max_tokens * 4, 4096)
                logger.info("Response truncated — retrying with %d tokens.", retry_tokens)
            t0 = time.time()
            response = self._call_with_retry(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=retry_tokens,
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] chat() retry API call took %.2fs", elapsed)
            print(f"[LLM] chat() retry took {elapsed:.2f}s", flush=True)

        assistant_text = response.choices[0].message.content or ""
        assistant_text = self._strip_think(assistant_text)
        self._history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def has_history(self) -> bool:
        return bool(self._history)

    def reset_history(self) -> None:
        self._history.clear()
        logger.debug("OpenAI history cleared.")

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from reasoning models."""
        text = _THINK_RE.sub("", text)
        # Handle unclosed <think> (model hit token limit mid-thought)
        lower = text.lower()
        if "<think>" in lower and "</think>" not in lower:
            idx = lower.rfind("<think>")
            text = text[:idx]
        return text.strip()

    def generate_once(self, prompt: str, max_tokens: int | None = None,
                      system_prompt: str | None = None) -> str:
        """One-shot call — does not touch self._history."""
        if system_prompt is None:
            from llm.prompt import get_system_prompt
            system_prompt = get_system_prompt()
        t0 = time.time()
        response = self._call_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        elapsed = time.time() - t0
        logger.info("[TIMING] generate_once() took %.2fs", elapsed)
        print(f"[LLM] generate_once() took {elapsed:.2f}s", flush=True)
        return self._strip_think(response.choices[0].message.content or "")

    def describe_image(self, jpeg_bytes: bytes) -> Optional[str]:
        """One-shot vision call — returns a plain description of the image."""
        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        try:
            t0 = time.time()
            response = self._call_with_retry(
                model=self.model,
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
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] describe_image() took %.2fs", elapsed)
            return response.choices[0].message.content or None
        except Exception as exc:
            logger.warning("Vision call failed (model may not support images): %s", exc)
            return None

    def describe_screen(self, jpeg_bytes: bytes) -> Optional[str]:
        """One-shot vision call — describe what's on a computer screen."""
        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        try:
            t0 = time.time()
            response = self._call_with_retry(
                model=self.model,
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
                max_tokens=600,
            )
            elapsed = time.time() - t0
            logger.info("[TIMING] describe_screen() took %.2fs", elapsed)
            return response.choices[0].message.content or None
        except Exception as exc:
            logger.warning("Screen vision call failed (model may not support images): %s", exc)
            return None

    def inject_history(self, user_message: str, assistant_message: str) -> None:
        """Inject a fake exchange into history so Dash remembers autonomous actions."""
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_message})
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep only the most recent max_history_turns pairs."""
        max_messages = self.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
