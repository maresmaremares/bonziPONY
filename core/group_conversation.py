"""
GroupConversation — inter-pony + user conversation coordinator.

Manages turn-taking between ponies, maintains a shared conversation log,
and stops when all ponies PASS or max depth is reached.
"""

from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.pony_instance import PonyInstance
    from core.pony_manager import PonyManager

logger = logging.getLogger(__name__)

# Regex to strip pipeline tags the LLM may include — these should never be spoken aloud
_TAG_RE = re.compile(
    r'\[(?:CONVO|DIRECTIVE|TIMER|ROUTINE|DONE|ENFORCE|DELAY|MOVETO|PERSIST)\s*(?::[^\]]*?)?\]',
    re.IGNORECASE,
)

_TURN_PROMPT_TEMPLATE = (
    "(Group chat on the desktop. Here's what just happened:\n"
    "{log_block}\n\n"
    "It's your turn. You can respond naturally, or say [PASS] if you have nothing to add.\n"
    "Keep it short — 1-2 sentences, like real banter between friends.\n"
    "IMPORTANT: Do NOT make up things you can't actually see or know right now. "
    "Only reference real things: your conversation history, things the user told you, "
    "or what you've actually observed. Do NOT describe imaginary weather, clouds, scenery, etc.\n"
    "Be yourself — not a caricature. Avoid leaning into your most stereotypical trait "
    "in every single line. Real people don't do that.\n"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)

_PIGGYBACK_PROMPT_TEMPLATE = (
    "({speaker} just responded to the user.\n"
    "[User]: \"{user_text}\"\n"
    "[{speaker}]: \"{response_text}\"\n\n"
    "You overheard this. If you want to jump in with a quick comment, go for it.\n"
    "Otherwise say [PASS]. Keep it short — one sentence max.\n"
    "Be yourself — not a walking stereotype. Say what YOU would actually say, "
    "not what a parody of you would say.\n"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)

_SPONTANEOUS_PROMPT_TEMPLATE = (
    "(You're hanging out on the desktop with {companions}. "
    "Say something casual to start a conversation — bring up something from "
    "a previous chat, ask what they've been up to, share a thought about "
    "something real that happened, or ask about the user. "
    "Keep it short — 1-2 sentences max.\n"
    "IMPORTANT: Do NOT make up things you can't see. You don't have screen access "
    "right now, so don't describe weather, clouds, scenery, or anything visual. "
    "Stick to things you actually know.\n"
    "Be a real character, not a caricature. Don't lean into your most obvious trait "
    "every single time — real ponies have range.\n"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)


class GroupConversation:
    """Coordinates a multi-pony conversation with turn-taking."""

    def __init__(
        self,
        manager: "PonyManager",
        max_depth: int = 6,
    ) -> None:
        self._manager = manager
        self._log: list[tuple[str, str]] = []  # (speaker_name, text)
        self._depth = 0
        self._max_depth = max_depth

    def start(self, initiator: "PonyInstance", trigger: str = "spontaneous") -> None:
        """Kick off a conversation.  The initiator speaks first, then others
        get a chance to respond in turn."""
        from core.tts_queue import PRIORITY_SPONTANEOUS_CHAT

        # Generate initiator's opening line
        companions = [p.display_name for p in self._manager.ponies if p is not initiator]
        if not companions:
            return

        prompt = _SPONTANEOUS_PROMPT_TEMPLATE.format(
            companions=", ".join(companions),
        )

        try:
            opening = initiator.llm.generate_once(prompt, max_tokens=150)
        except Exception as exc:
            logger.error("Group conversation start failed: %s", exc)
            return

        opening = self._clean_reply(opening)
        if not opening:
            return

        self._log.append((initiator.display_name, opening))
        self._depth += 1

        # Queue the speech
        self._speak(initiator, opening, PRIORITY_SPONTANEOUS_CHAT)

        # Offer turns to other ponies
        self._offer_rounds(exclude=initiator)

    def piggyback(
        self,
        pony: "PonyInstance",
        original_speaker: str,
        user_text: str,
        response_text: str,
    ) -> None:
        """Give a pony a chance to jump in after another pony responded to the user."""
        from core.tts_queue import PRIORITY_INTER_PONY_REPLY

        if getattr(pony, "_destroyed", False):
            return

        prompt = _PIGGYBACK_PROMPT_TEMPLATE.format(
            speaker=original_speaker,
            user_text=user_text,
            response_text=response_text,
        )

        try:
            reply = pony.llm.generate_once(prompt, max_tokens=100)
        except Exception as exc:
            logger.debug("Piggyback LLM call failed: %s", exc)
            return

        reply = self._clean_reply(reply)
        if not reply:
            return

        self._speak(pony, reply, PRIORITY_INTER_PONY_REPLY)

    def inject_user(self, text: str) -> None:
        """User jumps into an active conversation."""
        self._log.append(("[User]", text))

    def _offer_rounds(self, exclude: Optional["PonyInstance"] = None) -> None:
        """Offer turns to all ponies (except *exclude*) in a round-robin
        until everyone PASSes or we hit max depth."""
        from core.tts_queue import PRIORITY_SPONTANEOUS_CHAT

        while self._depth < self._max_depth:
            anyone_spoke = False
            # Randomise turn order each round
            others = [p for p in self._manager.ponies if p is not exclude]
            random.shuffle(others)

            for pony in others:
                if self._depth >= self._max_depth:
                    break
                if getattr(pony, "_destroyed", False):
                    continue
                reply = self._offer_turn(pony)
                if reply:
                    self._log.append((pony.display_name, reply))
                    self._depth += 1
                    anyone_spoke = True
                    self._speak(pony, reply, PRIORITY_SPONTANEOUS_CHAT)
                    # After someone speaks, reset exclude so the original
                    # initiator can respond next round
                    exclude = pony

            if not anyone_spoke:
                break  # everyone passed

    def _offer_turn(self, pony: "PonyInstance") -> Optional[str]:
        """Ask a pony if she wants to respond.  Returns her reply or None."""
        log_block = "\n".join(f"[{name}]: \"{text}\"" for name, text in self._log[-6:])
        prompt = _TURN_PROMPT_TEMPLATE.format(log_block=log_block)

        try:
            reply = pony.llm.generate_once(prompt, max_tokens=150)
        except Exception as exc:
            logger.debug("Turn offer LLM call failed for %s: %s", pony.display_name, exc)
            return None

        return self._clean_reply(reply)

    def _speak(self, pony: "PonyInstance", text: str, priority: int) -> None:
        """Enqueue speech for a pony.  No-ops if the pony was destroyed."""
        if getattr(pony, "_destroyed", False):
            return

        def _show_bubble():
            # Emit pet_controller signal to show bubble — this properly marshals
            # to the Qt main thread via BlockingQueuedConnection
            if getattr(pony, "_destroyed", False):
                return
            try:
                pony.pet_controller.speech_text.emit(text)
            except Exception as exc:
                logger.debug("GroupConversation bubble emit failed: %s", exc)

        self._manager.tts_queue.enqueue(
            text,
            priority=priority,
            voice_slug=pony.slug,
            on_start=_show_bubble,
            skip_tts=not getattr(pony, "has_voice", True),
        )

    @staticmethod
    def _clean_reply(text: str) -> Optional[str]:
        """Strip [PASS], pipeline tags, and surrounding quotes from LLM output.

        Returns None if the model said [PASS] or if nothing remains after cleaning.
        """
        if not text:
            return None
        cleaned = text.strip()

        # Check for PASS
        if "[PASS]" in cleaned.upper() or cleaned.upper() == "PASS":
            return None

        # Strip pipeline tags like [CONVO:CONTINUE], [DIRECTIVE:...], etc.
        cleaned = _TAG_RE.sub('', cleaned).strip()

        # Remove surrounding quotes/parens the model sometimes adds
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()

        # Strip any remaining asterisk-wrapped actions like *giggles*
        # that some models add — keep only if there's real text too
        only_actions = re.sub(r'\*[^*]+\*', '', cleaned).strip()
        if not only_actions:
            # Only had action text like "*giggles*" — keep it
            pass

        return cleaned if cleaned else None
