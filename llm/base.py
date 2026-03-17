"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Base class for all LLM backends."""

    @abstractmethod
    def chat(self, user_message: str) -> str:
        """Send a user message and return the assistant's response."""

    @abstractmethod
    def reset_history(self) -> None:
        """Clear conversation history."""

    @abstractmethod
    def generate_once(self, prompt: str, max_tokens: int | None = None,
                      system_prompt: str | None = None) -> str:
        """One-shot generation that does NOT affect conversation history.

        If system_prompt is provided, it overrides the default character
        system prompt.  Use this for utility tasks (summarization, profile
        extraction) that should NOT be in-character.
        """

    def has_history(self) -> bool:
        """Return True if there is any conversation history to summarize."""
        return False

    def describe_image(self, jpeg_bytes: bytes) -> Optional[str]:
        """
        One-shot vision call: describe what's in the image.
        Returns a plain-text description, or None if unsupported.
        Override in providers that support vision.
        """
        return None

    def describe_screen(self, jpeg_bytes: bytes) -> Optional[str]:
        """
        One-shot vision call: describe what's on a computer screen.
        Returns a plain-text description, or None if unsupported.
        Override in providers that support vision.
        """
        return None

    def inject_history(self, user_message: str, assistant_message: str) -> None:
        """Inject a user/assistant exchange into history without an API call.

        Used by the agent loop so Dash remembers autonomous actions.
        Override in providers that maintain conversation history.
        """
        pass
