"""Anthropic Claude LLM provider."""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import List, Optional

from llm.base import LLMProvider
from llm.prompt import get_system_prompt

_MAX_RETRIES = 3
_RETRY_BACKOFF = (1.0, 3.0, 5.0)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.85,
        max_tokens: int = 600,
        max_history_turns: int = 10,
        base_url: Optional[str] = None,
    ) -> None:
        from anthropic import Anthropic

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_turns = max_history_turns

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Anthropic(**client_kwargs)
        self._history: List[dict] = []

    def _call_with_retry(self, **kwargs):
        """Call messages.create with retries on 5xx / connection errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is not None and status >= 500
                if not retryable:
                    # Also retry on connection / timeout errors
                    retryable = isinstance(exc, (ConnectionError, TimeoutError))
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

        response = self._call_with_retry(
            model=self.model,
            system=get_system_prompt(),
            messages=self._history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # If the response was truncated (hit token limit), retry with a higher limit.
        # WRITE_NOTEPAD gets unlimited (16k) so she can write as much as she wants.
        if getattr(response, "stop_reason", None) == "max_tokens":
            partial = response.content[0].text if response.content else ""
            if "WRITE_NOTEPAD" in partial:
                retry_tokens = 16384
                logger.info("WRITE_NOTEPAD truncated — retrying with %d tokens.", retry_tokens)
            else:
                retry_tokens = max(self.max_tokens * 4, 4096)
                logger.info("Response truncated — retrying with %d tokens.", retry_tokens)
            response = self._call_with_retry(
                model=self.model,
                system=get_system_prompt(),
                messages=self._history,
                temperature=self.temperature,
                max_tokens=retry_tokens,
            )

        assistant_text = response.content[0].text if response.content else ""
        assistant_text = self._strip_think(assistant_text)
        self._history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def has_history(self) -> bool:
        return bool(self._history)

    def reset_history(self) -> None:
        self._history.clear()
        logger.debug("Anthropic history cleared.")

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from reasoning models."""
        text = _THINK_RE.sub("", text)
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
        response = self._call_with_retry(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        return self._strip_think(response.content[0].text if response.content else "")

    def describe_image(self, jpeg_bytes: bytes) -> Optional[str]:
        """One-shot vision call — returns a plain description of the image."""
        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        response = self._call_with_retry(
            model=self.model,
            system="You are a precise visual observer. Describe what you see in the image concisely in 1-3 sentences. Focus on people, objects, environment, and notable details.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": "What do you see?"},
                    ],
                }
            ],
            max_tokens=150,
        )
        return response.content[0].text.strip() if response.content else None

    def describe_screen(self, jpeg_bytes: bytes) -> Optional[str]:
        """One-shot vision call — describe what's on a computer screen."""
        b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")
        response = self._call_with_retry(
            model=self.model,
            system=(
                "You are observing a computer screen. Describe what you see concisely in 2-3 sentences. "
                "Focus on: which applications/windows are open, what content is displayed, any notable "
                "text or activity. Ignore the small animated pony sprite — that's you."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": "What's on the screen?"},
                    ],
                }
            ],
            max_tokens=200,
        )
        return response.content[0].text.strip() if response.content else None

    def inject_history(self, user_message: str, assistant_message: str) -> None:
        """Inject a fake exchange into history so Dash remembers autonomous actions."""
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_message})
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = self.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
