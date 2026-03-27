"""
GroupConversation — inter-pony + user conversation coordinator.

Manages turn-taking between ponies, maintains a shared conversation log,
and stops when all ponies PASS or max depth is reached.
"""

from __future__ import annotations

import logging
import random
import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from core.pony_instance import PonyInstance
    from core.pony_manager import PonyManager

logger = logging.getLogger(__name__)

# Regex to strip pipeline tags the LLM may include — these should never be spoken aloud
_TAG_RE = re.compile(
    r'\[(?:CONVO|DIRECTIVE|TIMER|ROUTINE|DONE|ENFORCE|DELAY|MOVETO|PERSIST|RULE)\s*(?::[^\]]*?)?\]',
    re.IGNORECASE,
)

# ── Conversation topics for variety ─────────────────────────────────────
# The initiator picks a random topic SEED so conversations don't all start
# with "hey what are you working on?"  Topics are character-agnostic nudges.
_TOPIC_SEEDS = [
    "something funny that happened recently or a joke you want to tell",
    "a random opinion or hot take about something you feel strongly about",
    "a question for one of your friends here — something you've been curious about",
    "a memory or story from Equestria — something nostalgic",
    "something annoying or frustrating that's been on your mind",
    "a random fun fact or piece of trivia you think is interesting",
    "a compliment or roast of one of the other ponies present",
    "what you'd be doing RIGHT NOW if you could do anything",
    "something about the user — an observation, question, or thought about them",
    "a debate topic — ask the group something where ponies might disagree",
    "gossip — talk about one of the ponies who might or might not be here",
    "your honest review of something — a food, a show, an activity",
    "a challenge or dare for one of the other ponies",
    "something weird or random that just popped into your head",
    "a 'would you rather' or hypothetical question for the group",
    "complain about something — vent a little, you're among friends",
]

_TURN_PROMPT_TEMPLATE = (
    "(Group chat on the desktop. {screen_info}\n"
    "Here's what just happened:\n"
    "{log_block}\n\n"
    "It's your turn. You can respond naturally, or say [PASS] if you have nothing to add.\n"
    "Keep it short — 1-2 sentences, like real banter between friends.\n"
    "RULES:\n"
    "- Do NOT make up things you can't see. The screen info above is ALL you know about "
    "the user's screen. If it's not listed there, you DON'T know about it.\n"
    "- Do NOT repeat what another pony already said. If someone already made an observation, "
    "don't say the same thing differently.\n"
    "- Do NOT parrot system errors, connection messages, or technical info from window titles. "
    "You are NOT tech support.\n"
    "- ADVANCE the conversation — react to what was said, add a NEW thought, or disagree. "
    "Don't just agree and restate.\n"
    "- Be yourself — not a caricature. Don't lean into your most stereotypical trait every time.\n"
    "{recent_topics_warning}"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)

_PIGGYBACK_PROMPT_TEMPLATE = (
    "({speaker} just responded to the user.\n"
    "[User]: \"{user_text}\"\n"
    "[{speaker}]: \"{response_text}\"\n\n"
    "You overheard this. If you want to jump in with a quick comment, go for it.\n"
    "Otherwise say [PASS]. Keep it short — one sentence max.\n"
    "Don't just agree — add something new, a different angle, or a joke.\n"
    "Be yourself — not a walking stereotype.\n"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)

_SPONTANEOUS_PROMPT_TEMPLATE = (
    "(You're hanging out on the desktop with {companions}. "
    "{screen_info}\n"
    "{recent_topics_warning}"
    "Start a conversation — pick any topic you genuinely feel like talking about right now. "
    "Be specific and in character. Think of something the REAL you would bring up.\n"
    "EXPLICITLY BANNED openers:\n"
    "- 'would you rather...' — do NOT use this\n"
    "- 'what are you working on?' — boring, don't use it\n"
    "- Generic small talk like 'how are you?' or 'what's up?'\n"
    "RULES:\n"
    "- ONLY reference things from the screen info above, your conversation history, "
    "or things the user has told you. Do NOT invent things you can't see.\n"
    "- Do NOT comment on technical details of window titles (errors, status messages, etc.) "
    "— you're a pony, not IT support.\n"
    "Keep it short — 1-2 sentences max. Be casual and natural.\n"
    "Be a real character with range — don't default to your most obvious personality trait.\n"
    "Do NOT include any tags like [CONVO:...] — just speak naturally.)"
)


class GroupConversation:
    """Coordinates a multi-pony conversation with turn-taking."""

    # Class-level recent topic tracking — shared across all conversations
    # to prevent the same topics coming up repeatedly.
    _recent_topics: List[str] = []
    _MAX_RECENT_TOPICS = 15

    def __init__(
        self,
        manager: "PonyManager",
        max_depth: int = 6,
    ) -> None:
        self._manager = manager
        self._log: list[tuple[str, str]] = []  # (speaker_name, text)
        self._depth = 0
        self._max_depth = max_depth
        self._screen_info = ""  # shared screen context for ALL turns
        self.interrupted: bool = False  # set True from outside to stop mid-conversation

    @classmethod
    def _record_topic(cls, opening_line: str) -> None:
        """Record a conversation opening so we can warn against repetition."""
        # Keep a short summary (first 80 chars)
        summary = opening_line[:80].strip()
        cls._recent_topics.append(summary)
        if len(cls._recent_topics) > cls._MAX_RECENT_TOPICS:
            cls._recent_topics.pop(0)

    @classmethod
    def _get_recent_topics_warning(cls) -> str:
        """Build a warning string listing recent topics to avoid."""
        if not cls._recent_topics:
            return ""
        recent = cls._recent_topics[-8:]  # last 8 conversation openers
        lines = ", ".join(f'"{t}"' for t in recent)
        return (
            f"RECENT conversations already covered these topics (DO NOT repeat them, "
            f"find something NEW): {lines}\n"
        )

    def start(self, initiator: "PonyInstance", trigger: str = "spontaneous",
              screen_context: str = "") -> None:
        """Kick off a conversation.  The initiator speaks first, then others
        get a chance to respond in turn."""
        from core.tts_queue import PRIORITY_SPONTANEOUS_CHAT

        # Generate initiator's opening line
        companions = [p.display_name for p in self._manager.ponies if p is not initiator]
        if not companions:
            return

        # Store screen context for ALL turns in this conversation
        self._screen_info = screen_context or "No screen info available."

        recent_warning = self._get_recent_topics_warning()

        prompt = _SPONTANEOUS_PROMPT_TEMPLATE.format(
            companions=", ".join(companions),
            screen_info=self._screen_info,
            recent_topics_warning=recent_warning,
        )

        try:
            opening = initiator.llm.generate_once(prompt, max_tokens=150)
        except Exception as exc:
            logger.error("Group conversation start failed: %s", exc)
            return

        opening = self._clean_reply(opening)
        if not opening:
            return

        # Track this topic for future diversity
        self._record_topic(opening)

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
            if self.interrupted:
                logger.debug("Group conversation interrupted by PTT")
                break
            anyone_spoke = False
            # Randomise turn order each round
            others = [p for p in self._manager.ponies if p is not exclude]
            random.shuffle(others)

            for pony in others:
                if self.interrupted:
                    break
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

        recent_warning = self._get_recent_topics_warning()

        prompt = _TURN_PROMPT_TEMPLATE.format(
            log_block=log_block,
            screen_info=self._screen_info,
            recent_topics_warning=recent_warning,
        )

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

        # Strip <think>...</think> blocks from reasoning models
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

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
