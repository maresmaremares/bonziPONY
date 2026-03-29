"""
State machine pipeline: IDLE → ACKNOWLEDGE → LISTEN → THINK → SPEAK

After Dash speaks, stays in conversation mode for `conversation.timeout_s`
seconds so the user can reply without repeating the wake word.

Also exposes `speak_spontaneously()` and `summarize_session()`.
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from core.config_loader import AppConfig
    from wake_word.detector import WakeWordDetector
    from acknowledgement.player import AcknowledgementPlayer
    from stt.transcriber import Transcriber
    from llm.base import LLMProvider
    from llm.response_parser import ParsedResponse
    from tts.elevenlabs_tts import ElevenLabsTTS
    from robot.base import RobotController
    from vision.camera import Camera
    from vision.screen import ScreenCapture
    from robot.desktop_controller import DesktopController
    from core.agent_loop import AgentLoop
    from core.screen_monitor import ScreenMonitor

from llm.prompt import get_character_name

logger = logging.getLogger(__name__)

_PROFILE_COOLDOWN_S = 4 * 3600  # 4 hours between profile extractions


def _unrelated_prompts() -> list[str]:
    """Return spontaneous prompts using the active character's name."""
    name = get_character_name()
    return [
        f"You're just hanging out and have a random thought. Say ONE short thing as {name}.",
        f"You're bored. Say ONE thing. {name} style.",
        "Say ONE thing about whatever is on your mind. One sentence.",
        f"Make a quick offhand comment about anything. One sentence, {name}.",
    ]


class PipelineState(Enum):
    IDLE = auto()
    ACKNOWLEDGE = auto()
    LISTEN = auto()
    THINK = auto()
    SPEAK = auto()
    ERROR = auto()


class Pipeline:
    """Wires all stages together."""

    def __init__(
        self,
        config: AppConfig,
        detector: WakeWordDetector,
        ack_player: AcknowledgementPlayer,
        transcriber: Transcriber,
        llm_provider: LLMProvider,
        tts: ElevenLabsTTS,
        robot: RobotController | None = None,
        camera: Camera | None = None,
        screen: ScreenCapture | None = None,
        desktop_controller: DesktopController | None = None,
        agent_loop: AgentLoop | None = None,
        screen_monitor: ScreenMonitor | None = None,
        moondream=None,
        vision_llm=None,
        timeline=None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.ack_player = ack_player
        self.transcriber = transcriber
        self.llm = llm_provider
        self.tts = tts
        self.tts_queue = None          # set by main.py for multi-pony voice routing
        self.primary_voice_slug = None  # set by main.py
        self.pony_manager = None       # set by main.py for piggyback
        self.robot = robot
        self.camera = camera
        self.screen = screen
        self.desktop_controller = desktop_controller
        self.agent_loop = agent_loop
        self.screen_monitor = screen_monitor
        self.moondream = moondream
        self.vision_llm = vision_llm  # dedicated vision model (optional)
        self._timeline = timeline     # shared event timeline
        self.state = PipelineState.IDLE

        self._recent_topics: List[str] = []
        self._visual_memory: List[str] = []
        self._recently_spoken: List[str] = []  # echo detection
        self._last_end_conversation: bool = False  # LLM signaled conversation over
        self._last_profile_extraction: float = 0.0  # monotonic timestamp
        self._active_responder: Any = None  # non-primary pony currently responding (Fix 10/11)

        # Optional GUI callbacks
        self._on_state_change = None
        self._on_speech_text = None
        self._on_heard_text = None
        self._on_conversation_start = None
        self._on_conversation_end = None

    def set_callbacks(
        self,
        on_state_change=None,
        on_speech_text=None,
        on_heard_text=None,
        on_conversation_start=None,
        on_conversation_end=None,
    ) -> None:
        """Set optional callbacks for GUI integration."""
        self._on_state_change = on_state_change
        self._on_speech_text = on_speech_text
        self._on_heard_text = on_heard_text
        self._on_conversation_start = on_conversation_start
        self._on_conversation_end = on_conversation_end

    @property
    def active_speech_bubble(self):
        """Return the speech bubble for the currently-responding pony (Fix 11).

        Returns the non-primary pony's speech_bubble when routing is active,
        otherwise None (caller should fall back to primary pony's bubble).
        """
        if self._active_responder and hasattr(self._active_responder, "speech_bubble"):
            return self._active_responder.speech_bubble
        return None

    def _is_echo(self, heard: str) -> bool:
        """Check if transcribed text is the pony's own TTS output echoing through the mic."""
        if not heard or not self._recently_spoken:
            return False
        h = heard.lower().strip()
        if len(h) < 5:
            return False
        for spoken in self._recently_spoken:
            s = spoken.lower().strip()
            if h in s or s in h:
                logger.debug("Echo detected (substring): heard=%r", h)
                return True
            h_words = set(h.split())
            s_words = set(s.split())
            if len(h_words) >= 3:
                overlap = len(h_words & s_words) / len(h_words)
                if overlap > 0.6:
                    logger.debug("Echo detected (%.0f%% overlap): heard=%r", overlap * 100, h)
                    return True
        return False

    def _speak_with_queue(
        self, text: str, show_bubble_cb, priority: int = 0,
        responder=None, blocking: bool | None = None,
    ) -> None:
        """Speak through TTSQueue if available, else direct TTS.

        This ensures voice switching works correctly in multi-pony mode.
        When *blocking* is None (default), blocks for user-response priority.
        Pass ``blocking=True`` explicitly to block for any priority level.
        When *responder* is a non-primary PonyInstance, uses their voice slug.
        """
        voice_slug = self.primary_voice_slug
        if responder and hasattr(responder, "slug"):
            voice_slug = responder.slug

        if blocking is None:
            from core.tts_queue import PRIORITY_USER_RESPONSE
            blocking = (priority == PRIORITY_USER_RESPONSE)

        if self.tts_queue:
            self.tts_queue.enqueue(
                text,
                priority=priority,
                voice_slug=voice_slug,
                on_start=show_bubble_cb,
                blocking=blocking,
            )
        elif self.config.tts.enabled:
            self.tts.speak(text, on_playback_start=show_bubble_cb)
        else:
            show_bubble_cb()

    # ── Public entry points ────────────────────────────────────────────────────

    def run_conversation(self) -> None:
        """
        Handle one wake-word-triggered interaction, then stay in conversation
        mode for `conversation.timeout_s` seconds so the user can keep talking
        without repeating the wake word.
        """
        if self._on_conversation_start:
            try:
                self._on_conversation_start()
            except Exception:
                pass
        if self._timeline:
            from core.event_timeline import EventType
            self._timeline.append(EventType.CONVERSATION_START, "Conversation started")

        if self.agent_loop:
            self.agent_loop.set_conversation_active(True)

        try:
            self._last_end_conversation = False
            spoke = self._run_turn(play_ack=True)
            if not spoke and not self._last_end_conversation:
                return

            # If LLM signaled end on first turn (e.g. user said "goodnight")
            if self._last_end_conversation:
                logger.info("LLM signaled end of conversation after first turn.")
                print("[Conversation ended naturally]")
                return

            cfg = self.config.conversation
            # Minimum 15s conversation window — lower values make multi-turn
            # conversations nearly impossible (user can't respond in time)
            convo_timeout = max(cfg.timeout_s, 15.0)
            deadline = time.monotonic() + convo_timeout
            just_spoke = True  # TTS just played; first follow-up listen needs echo drain

            print("\n[Conversation mode — just keep talking, no wake word needed]")

            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                wait = min(cfg.listen_timeout_s, remaining)
                if wait <= 0:
                    break

                print(f"\n[Listening... {remaining:.0f}s remaining]", flush=True)
                self._transition(PipelineState.LISTEN)

                discard_ms = 800 if just_spoke else 0
                user_text = self.transcriber.listen(speech_start_timeout_s=wait, initial_discard_ms=discard_ms)
                just_spoke = False

                if not user_text or not user_text.strip():
                    logger.debug("No follow-up speech — ending conversation.")
                    break

                # Filter Whisper hallucinations (ambient noise transcribed as garbage)
                from stt.transcriber import _is_whisper_hallucination
                if _is_whisper_hallucination(user_text):
                    logger.debug("Filtered hallucination in follow-up: %r", user_text)
                    continue

                # Filter echo — pony hearing its own TTS through the mic
                if self._is_echo(user_text):
                    logger.info("Filtered echo in conversation: %r", user_text)
                    continue

                spoke = self._run_turn(play_ack=False, user_text=user_text)
                if self._last_end_conversation:
                    logger.info("LLM signaled end of conversation.")
                    print("[Conversation ended naturally]")
                    break
                if spoke:
                    deadline = time.monotonic() + convo_timeout
                    just_spoke = True

            print("[Conversation ended — say the wake word to start again]")
            self._extract_user_profile()
        finally:
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)

    def run_conversation_with_text(self, text: str) -> None:
        """Like run_conversation but with pre-supplied text for the first turn.

        Used by push-to-talk: the audio is already transcribed, so we skip
        the first listen but still enter the follow-up conversation loop.
        """
        if self._on_conversation_start:
            try:
                self._on_conversation_start()
            except Exception:
                pass

        if self.agent_loop:
            self.agent_loop.set_conversation_active(True)

        try:
            self._last_end_conversation = False
            if self._on_heard_text:
                try:
                    self._on_heard_text(text)
                except Exception:
                    pass
            spoke = self._run_turn(play_ack=False, user_text=text)
            if not spoke and not self._last_end_conversation:
                return

            if self._last_end_conversation:
                return

            # Enter follow-up conversation loop (same as run_conversation)
            cfg = self.config.conversation
            convo_timeout = max(cfg.timeout_s, 15.0)
            deadline = time.monotonic() + convo_timeout
            just_spoke = True

            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                wait = min(cfg.listen_timeout_s, remaining)
                if wait <= 0:
                    break

                self._transition(PipelineState.LISTEN)
                discard_ms = 800 if just_spoke else 0
                user_text = self.transcriber.listen(speech_start_timeout_s=wait, initial_discard_ms=discard_ms)
                just_spoke = False

                if not user_text or not user_text.strip():
                    break

                from stt.transcriber import _is_whisper_hallucination
                if _is_whisper_hallucination(user_text):
                    continue

                if self._is_echo(user_text):
                    logger.info("Filtered echo in PTT conversation: %r", user_text)
                    continue

                spoke = self._run_turn(play_ack=False, user_text=user_text)
                if self._last_end_conversation:
                    break
                if spoke:
                    deadline = time.monotonic() + convo_timeout
                    just_spoke = True

            self._extract_user_profile()
        finally:
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)

    def run_text_conversation(self, text: str) -> None:
        """Handle a typed message — skips STT, goes straight to LLM."""
        if self._on_conversation_start:
            try:
                self._on_conversation_start()
            except Exception:
                pass

        if self.agent_loop:
            self.agent_loop.set_conversation_active(True)

        self._last_end_conversation = False
        spoke = self._run_turn(play_ack=False, user_text=text)

        if self.agent_loop:
            self.agent_loop.set_conversation_active(False)
        if self._on_conversation_end:
            try:
                self._on_conversation_end()
            except Exception:
                pass
        self._transition(PipelineState.IDLE)

    def speak_spontaneously(self) -> None:
        """
        Generate an unprompted remark that goes INTO history so Dash remembers it.
        70% chance: related to recent topics. 30%: random.
        """
        try:
            use_recent = self._recent_topics and random.random() < 0.70

            if use_recent:
                recent = ", ".join(self._recent_topics[-3:])
                trigger = (
                    f"(You spontaneously think of something related to what you've been talking about: {recent}. "
                    "Say ONE short thing — continue the thread naturally or make an offhand comment about it. "
                    "One sentence, no setup.)"
                )
            else:
                trigger = f"(Spontaneous thought — {random.choice(_unrelated_prompts())})"

            # Inject screen context so the pony knows what's happening
            screen_note = self._inject_screen_state("")
            if screen_note.strip():
                trigger += f"\n{screen_note.strip()}"

            logger.debug("Spontaneous trigger: %r", trigger)

            # Use chat() so the exchange lands in history — Dash will remember it
            raw = self.llm.chat(trigger)
            if not raw:
                return

            from llm.response_parser import parse_response
            parsed = parse_response(raw)
            logger.info("Spontaneous: %r", parsed.text)

            if parsed.text:
                self._transition(PipelineState.SPEAK)
                _bubble_shown = False
                def _show_bubble():
                    nonlocal _bubble_shown
                    if _bubble_shown:
                        return
                    _bubble_shown = True
                    if self._on_speech_text:
                        try:
                            self._on_speech_text(parsed.text)
                        except Exception:
                            pass
                # Show bubble immediately — don't wait for TTS callback chain
                _show_bubble()
                from core.tts_queue import PRIORITY_AUTONOMOUS
                self._speak_with_queue(parsed.text, _show_bubble, priority=PRIORITY_AUTONOMOUS, blocking=True)

            # Execute actions/commands from the response (previously discarded)
            if parsed.actions:
                for action in parsed.actions:
                    try:
                        if self.desktop_controller:
                            self.desktop_controller.execute_action(action)
                        if self.robot:
                            self.robot.execute(action)
                    except Exception as exc:
                        logger.debug("Spontaneous action %s failed: %s", action, exc)
            if parsed.desktop_commands and self.desktop_controller:
                for dc in parsed.desktop_commands:
                    try:
                        self.desktop_controller.execute_command(dc)
                    except Exception as exc:
                        logger.debug("Spontaneous desktop cmd %s failed: %s", dc.command, exc)
            if parsed.moveto_region and self.robot:
                try:
                    self.robot.on_move_to(parsed.moveto_region)
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Spontaneous speech failed: %s", exc)
        finally:
            self._transition(PipelineState.IDLE)

    def summarize_session(self) -> None:
        """
        Generate a brief summary of this session and save it to memory/sessions.txt.
        Called on shutdown. Skipped if no conversation happened.
        """
        if not self.llm.has_history():
            return

        try:
            # Build conversation transcript from history so the LLM can see
            # what was actually said (generate_once only sends system prompt)
            history = list(getattr(self.llm, "_history", []))
            if len(history) < 2:
                return

            transcript_lines = []
            for msg in history:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if role == "system":
                    continue
                speaker = "User" if role == "user" else "Pony"
                transcript_lines.append(f"{speaker}: {content}")

            if not transcript_lines:
                return

            transcript = "\n".join(transcript_lines[-30:])
            prompt = (
                "Summarize this conversation in 3-5 bullet points. "
                "Be as brief as possible — key topics, anything important said, "
                "notable moments. Plain text, no formatting.\n\n"
                f"Conversation:\n{transcript}"
            )
            summary = self.llm.generate_once(
                prompt, max_tokens=512,
                system_prompt="You are a helpful assistant that summarizes conversations. Be concise and factual. Do NOT role-play or respond in character.",
            )
            if summary and summary.strip():
                from core.memory import save_summary
                save_summary(summary)
                logger.info("Session summary saved.")
        except Exception as exc:
            logger.warning("Failed to save session summary: %s", exc)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _run_turn(self, play_ack: bool = True, user_text: str | None = None) -> bool:
        """
        Execute one listen→think→speak turn.
        Returns True if Dash successfully spoke, False otherwise.
        """
        _orig_llm = self.llm  # save for finally-block restore (Fix 10)
        try:
            if play_ack:
                # Smart ack: check if user is already mid-sentence before interrupting
                self._transition(PipelineState.LISTEN)
                quick_text = self.transcriber.listen(speech_start_timeout_s=1.2)
                if quick_text and quick_text.strip():
                    # User was already talking — skip ack, use what they said
                    logger.info("User spoke immediately after wake word — skipping ack.")
                    user_text = quick_text
                else:
                    # User just said the wake word and stopped — play ack, then full listen
                    self._transition(PipelineState.ACKNOWLEDGE)
                    self.ack_player.play()

            if user_text is None:
                self._transition(PipelineState.LISTEN)
                print("\n[Listening...]", flush=True)
                user_text = self.transcriber.listen()
                if not user_text or not user_text.strip():
                    logger.info("No speech detected.")
                    print("[No speech detected]")
                    return False

            logger.info("User said: %r", user_text)
            if self._timeline:
                from core.event_timeline import EventType
                self._timeline.append(EventType.USER_SAID,
                                      f'User said: "{user_text[:150]}"')
            if self._on_heard_text:
                try:
                    self._on_heard_text(user_text)
                except Exception:
                    pass
            original_user_text = user_text  # save before injections for heuristic check
            self._remember_topic(user_text)

            # Fix 6: Inject user speech into active group conversation + interrupt it
            # Fix 10: Route speech to the named pony, swap LLM for this turn
            if self.pony_manager:
                if self.pony_manager._active_convo is not None:
                    try:
                        self.pony_manager._active_convo.inject_user(original_user_text)
                        self.pony_manager._active_convo.interrupted = True
                    except Exception:
                        pass
                if len(self.pony_manager.ponies) > 1:
                    try:
                        target = self.pony_manager.route_user_speech(original_user_text)
                        if target and not target.is_primary and hasattr(target, "llm"):
                            self.llm = target.llm
                            self._active_responder = target
                            # Inject recent conversation context so the addressed
                            # pony knows what was discussed (their LLM history is
                            # likely empty — without this they hallucinate)
                            primary = self.pony_manager.primary
                            if primary and hasattr(primary, "llm"):
                                _ph = getattr(primary.llm, "_history", [])
                                if _ph:
                                    _ctx = []
                                    for _m in _ph[-6:]:
                                        _role = _m.get("role", "")
                                        _c = _m.get("content", "")
                                        if _role == "user":
                                            _c = _c.split("[System hint:")[0].split("[IMPORTANT:")[0].strip()
                                            if _c:
                                                _ctx.append(f"[User]: {_c[:200]}")
                                        elif _role == "assistant":
                                            _c = re.sub(r'\[(?:CONVO|DIRECTIVE|TIMER|DONE|ENFORCE|DELAY|MOVETO|PERSIST|RULE)\s*(?::[^\]]*?)?\]', '', _c).strip()
                                            if _c:
                                                _ctx.append(f"[{primary.display_name}]: {_c[:200]}")
                                    if _ctx:
                                        user_text = (
                                            f"[Context — recent conversation before you were addressed:]\n"
                                            + "\n".join(_ctx[-4:])
                                            + f"\n\n{user_text}"
                                        )
                        else:
                            self._active_responder = None
                    except Exception:
                        self._active_responder = None

            # Stop keyword detection — clear all directives if user says stop
            if self.agent_loop and self.agent_loop.has_directives:
                _STOP_KW = ("stop", "knock it off", "enough", "quit it", "shut up", "leave me alone", "chill", "cut it out")
                if any(kw in user_text.lower() for kw in _STOP_KW):
                    self.agent_loop.clear_directives()
                    user_text = f"[System: User told you to stop. All your active directives have been cleared.]\n\n{user_text}"

            # Audio context — tell LLM whether this is the user or ambient audio
            user_text = self._build_audio_context(user_text)

            # ALWAYS inject win32gui screen context (free)
            user_text = self._inject_screen_state(user_text)

            user_text = self._maybe_inject_vision(user_text)

            # Nudge LLM to pick up on tasks/needs, compliance, and conversation flow
            if self.agent_loop:
                hint = (
                    f"\n\n[System hint: Today is {datetime.now().strftime('%A, %Y-%m-%d')}. "
                    "Pick the RIGHT tag for the situation:\n"
                    "- RIGHT NOW → [DIRECTIVE:goal:urgency]\n"
                    "- LATER TODAY → [DIRECTIVE:goal:urgency:delay_minutes] (number = minutes to defer)\n"
                    "- SPECIFIC FUTURE DATE → [DIRECTIVE:goal:urgency:tomorrow] or [DIRECTIVE:goal:urgency:wednesday] "
                    "or [DIRECTIVE:goal:urgency:next week] or [DIRECTIVE:goal:urgency:2026-03-30]\n"
                    "- SPECIFIC TIME → [TIMER:HH:MM:goal]\n"
                    "- RECURRING (daily/weekly/etc.) → [ROUTINE:...]\n"
                    "  FORMATS:\n"
                    "  [ROUTINE:daily:goal:urgency:HH:MM] — every day at time\n"
                    "  [ROUTINE:daily:goal:urgency:HH:MM:!saturday] — every day EXCEPT saturday\n"
                    "  [ROUTINE:daily:goal:urgency:HH:MM:!saturday,!sunday] — exclude multiple days\n"
                    "  [ROUTINE:weekly:goal:urgency:day:HH:MM] — one specific day per week\n"
                    "  COMBINING:\n"
                    "  'every day at 4 except friday at 3' = [ROUTINE:daily:goal:5:16:00:!friday] AND [ROUTINE:weekly:goal:5:friday:15:00]\n"
                    "  'tuesday at 3, wednesday at 4' = [ROUTINE:weekly:goal:5:tuesday:15:00] AND [ROUTINE:weekly:goal:5:wednesday:16:00]\n"
                    "  'every day except saturday' = [ROUTINE:daily:goal:5:16:00:!saturday]\n"
                    "  'every weekday' = [ROUTINE:daily:goal:5:09:00:!saturday,!sunday]\n"
                    "  USE MULTIPLE [ROUTINE:...] TAGS in one response for complex schedules.\n"
                    "- PERMANENTLY BLOCKED (quit porn, stop buying skins, stay off reddit) → [RULE:description]\n"
                    "- Just MENTIONING (plans, stories) → NO TAG, just talk about it.\n"
                    "Goal must be a DIRECT ACTION like 'eat food', 'shower' — NEVER 'remind user to' or 'get user to'."
                )
                if self.agent_loop.has_directives:
                    directives_list = "; ".join(
                        f'"{d.goal}" (urgency {d.urgency}{", ALREADY DELAYED" if d.delayed else ""})'
                        for d in self.agent_loop.directives
                    )
                    hint += (
                        f" ACTIVE DIRECTIVES: {directives_list}."
                        " If the user says they COMPLETED a task from this list, use [DONE] or"
                        " [DONE:keyword] to mark it done."
                        " If the user says they're LEAVING to go do a task (away from computer),"
                        " use [ENFORCE:minutes] with an estimated duration. Do NOT ask how long —"
                        " just estimate (shower=15, eating=20, gym=60, errands=45, brb=10)."
                    )
                hint += (
                    " If the user asks you to DELAY a directive (e.g. 'give me an hour', 'I'll do it later'),"
                    " you may grant ONE delay per directive using [DELAY:minutes] or [DELAY:minutes:keyword]."
                    " Example: 'okay, you have one hour' [DELAY:60]. But if the directive was ALREADY delayed"
                    " once, REFUSE. Say 'no, you already got your extension. go do it NOW.' Do NOT give a"
                    " second delay under any circumstances — the user had their chance."
                )
                hint += "]"
                user_text += hint

            # ALWAYS inject conversation flow hint (even without agent loop)
            # so the LLM outputs [CONVO:CONTINUE/END] tags reliably
            user_text += (
                "\n\n[IMPORTANT: End your reply with [CONVO:CONTINUE] or [CONVO:END]. "
                "Default to [CONVO:CONTINUE] — conversations should keep going unless the user "
                "is explicitly saying goodbye, leaving, or going AFK. A short reply like "
                "'ok cool' or 'thanks' from the user is NOT a reason to end — they might "
                "have more to say. Only use [CONVO:END] for clear goodbyes and sign-offs.]"
            )

            self._transition(PipelineState.THINK)
            raw_response = self.llm.chat(user_text)
            logger.info("LLM response: %r", raw_response)

            # Detect character break — model meta-analyzing the prompt instead of role-playing
            if self._is_character_break(raw_response):
                # Try stripping the meta preamble first — the in-character part might be fine
                stripped = self._strip_meta_preamble(raw_response)
                if stripped != raw_response and not self._is_character_break(stripped):
                    logger.warning("Character break detected — stripped meta preamble.")
                    raw_response = stripped
                    # Update history with the cleaned version
                    hist = getattr(self.llm, "_history", None)
                    if hist and hist[-1].get("role") == "assistant":
                        hist[-1]["content"] = stripped
                else:
                    logger.warning("Character break detected — retrying once.")
                    # Remove the broken exchange from history and retry
                    hist = getattr(self.llm, "_history", None)
                    if hist is not None and len(hist) >= 2:
                        hist.pop()  # broken assistant response
                        hist.pop()  # our user message (chat() will re-add it)
                    # Prepend a strong reminder to stay in character
                    name = get_character_name()
                    retry_text = (
                        f"[System: STAY IN CHARACTER. You are {name}. Do NOT analyze, "
                        f"explain, or output code/markdown. Just respond naturally as {name} "
                        f"in spoken words.]\n\n{user_text}"
                    )
                    raw_response = self.llm.chat(retry_text)
                    logger.info("LLM retry response: %r", raw_response)
                    if self._is_character_break(raw_response):
                        logger.warning("Character break on retry — using fallback.")
                        # Remove the second broken exchange too
                        if hist is not None and len(hist) >= 2:
                            hist.pop()
                            hist.pop()
                        raw_response = "hmm, uh, sorry I kinda spaced out for a second there. what were you saying? [CONVO:CONTINUE]"

            from llm.response_parser import parse_response
            parsed: ParsedResponse = parse_response(raw_response)

            if parsed.actions:
                from robot.actions import RobotAction as _RA
                _DESKTOP_ACTIONS = {
                    _RA.CLOSE_WINDOW, _RA.MINIMIZE_WINDOW, _RA.MAXIMIZE_WINDOW,
                    _RA.SNAP_WINDOW_LEFT, _RA.SNAP_WINDOW_RIGHT,
                    _RA.VOLUME_UP, _RA.VOLUME_DOWN, _RA.VOLUME_MUTE,
                    _RA.SHAKE,
                }
                for action in parsed.actions:
                    try:
                        if action in _DESKTOP_ACTIONS and self.desktop_controller:
                            self.desktop_controller.execute_action(action)
                        if self.robot:
                            self.robot.execute(action)
                    except Exception as exc:
                        logger.warning("Action %s failed: %s", action, exc)

            if parsed.desktop_commands and self.desktop_controller:
                for cmd in parsed.desktop_commands:
                    try:
                        self.desktop_controller.execute_command(cmd)
                    except Exception as exc:
                        logger.warning("Desktop command %s failed: %s", cmd.command, exc)

            # Create directive from conversation if LLM used [DIRECTIVE:...] tag
            # Skip if a TIMER tag is also present — the timer handles it, no duplicate nagging
            if parsed.directive and self.agent_loop and not parsed.timer:
                self.agent_loop.add_directive(
                    goal=parsed.directive.goal,
                    urgency=parsed.directive.urgency,
                    source="user",
                    delay_minutes=parsed.directive.delay_minutes,
                    trigger_date=parsed.directive.trigger_date,
                )

            # Create timer from conversation if LLM used [TIMER:...] tag
            if parsed.timer and self.agent_loop:
                self.agent_loop.add_timer(
                    time_str=parsed.timer.time_str,
                    action=parsed.timer.action,
                )

            # Create standing rule from conversation if LLM used [RULE:...] tag
            if parsed.standing_rule and self.agent_loop:
                self.agent_loop.add_standing_rule(description=parsed.standing_rule)

            # Create recurring routines from conversation if LLM used [ROUTINE:...] tags
            # Uses collapse_routine_tags to merge multiple tags with the same goal
            # (e.g. two weekly tags → one routine with day_times)
            if parsed.routines and self.agent_loop:
                from core.routines import collapse_routine_tags
                collapsed = collapse_routine_tags(parsed.routines)
                for routine in collapsed:
                    added = self.agent_loop.routine_manager.add_if_unique(routine)
                    logger.info("Routine %s from conversation: %s (%s)",
                                "created" if added else "merged",
                                routine.goal, routine.schedule)

            # Mark directive as completed if LLM used [DONE] or [DONE:keyword]
            if parsed.done_directive is not None and self.agent_loop and self.agent_loop.has_directives:
                try:
                    keyword = parsed.done_directive.lower()
                    removed = False
                    removed_d = None
                    if keyword:
                        # Match by keyword against directive goals
                        for i, d in enumerate(self.agent_loop.directives):
                            if keyword in d.goal.lower():
                                removed_d = self.agent_loop.directives.pop(i)
                                logger.info("Directive completed via [DONE:%s]: %r", keyword, removed_d.goal)
                                removed = True
                                break
                    if not removed:
                        # No keyword or no match — remove highest urgency directive
                        if self.agent_loop.directives:
                            best = max(range(len(self.agent_loop.directives)),
                                       key=lambda i: self.agent_loop.directives[i].urgency)
                            removed_d = self.agent_loop.directives.pop(best)
                            logger.info("Directive completed via [DONE]: %r", removed_d.goal)
                    # End enforcement if the completed directive was the one being enforced
                    if removed_d and self.agent_loop._enforcement.active:
                        if removed_d.goal == self.agent_loop._enforcement.directive_goal:
                            self.agent_loop._enforcement.active = False
                            logger.info("Enforcement ended: directive completed via conversation.")
                    if not self.agent_loop.directives:
                        self.agent_loop._mess_mouse_count = 0
                    self.agent_loop.save_directives()
                except (IndexError, RuntimeError) as exc:
                    logger.debug("Directive modification race: %s", exc)

            # MOVETO — move pony to a screen region
            if parsed.moveto_region and self.robot:
                try:
                    self.robot.on_move_to(parsed.moveto_region)
                except Exception as exc:
                    logger.warning("MOVETO failed: %s", exc)

            # PERSIST — keep the action animation for N seconds
            if parsed.persist_seconds and self.robot:
                from desktop_pet.pet_controller import _ACTION_ANIMATION_MAP
                try:
                    anim_name = "stand"
                    if parsed.actions:
                        anim_name = _ACTION_ANIMATION_MAP.get(parsed.actions[0], "stand")
                    self.robot.on_timed_override(anim_name, parsed.persist_seconds)
                except Exception as exc:
                    logger.warning("PERSIST failed: %s", exc)

            # Conversation flow — LLM signals whether to keep listening or end
            # Check if ANY convo tag was present (END or CONTINUE)
            _convo_tag_present = bool(re.search(r"\[CONVO:\s*(?:END|CONTINUE)\s*\]", raw_response, re.IGNORECASE))
            self._last_end_conversation = parsed.end_conversation
            if not self._last_end_conversation and not _convo_tag_present:
                # LLM forgot the tag entirely — use heuristic fallback
                self._last_end_conversation = self._heuristic_convo_end(
                    original_user_text, parsed.text
                )
            logger.debug("Conversation flow: end=%s (tag_present=%s, parsed_end=%s)",
                         self._last_end_conversation, _convo_tag_present, parsed.end_conversation)

            # Delay negotiation — user convinced pony to delay a directive
            if parsed.delay_minutes and self.agent_loop and self.agent_loop.has_directives:
                ok = self.agent_loop.delay_directive(parsed.delay_minutes, parsed.delay_keyword)
                if ok:
                    logger.info("Directive delayed by %d minutes (keyword=%r)", parsed.delay_minutes, parsed.delay_keyword)
                else:
                    logger.info("Delay rejected — directive already delayed or not found")

            # Enforcement mode — LLM detected user is going to do the task
            if parsed.enforce_minutes and self.agent_loop and self.agent_loop.has_directives:
                enforce_s = max(60.0, min(3600.0, parsed.enforce_minutes * 60.0))
                self.agent_loop.start_enforcement(enforce_s)
                logger.info("Enforcement started via LLM tag: %d min", parsed.enforce_minutes)
                if self._timeline:
                    from core.event_timeline import EventType, UserIntent
                    import time as _time
                    goal = self.agent_loop._enforcement.directive_goal
                    self._timeline.set_user_intent(UserIntent(
                        action=goal, stated_at=_time.monotonic(),
                        expected_duration_s=enforce_s))
                    self._timeline.set_afk_context(f"going to: {goal}")
                    self._timeline.append(EventType.ENFORCEMENT_START,
                                          f"Enforcement started: {goal} ({parsed.enforce_minutes}min)")

            self._transition(PipelineState.SPEAK)
            if parsed.text:
                _bubble_shown = False
                _target = self._active_responder  # may be None (primary)

                def _show_bubble():
                    nonlocal _bubble_shown
                    if _bubble_shown:
                        return
                    _bubble_shown = True
                    # Route bubble to the correct pony's pet_controller
                    if _target and hasattr(_target, "pet_controller"):
                        try:
                            _target.pet_controller.speech_text.emit(parsed.text)
                        except Exception:
                            pass
                    elif self._on_speech_text:
                        try:
                            self._on_speech_text(parsed.text)
                        except Exception:
                            pass

                # Show bubble IMMEDIATELY — don't wait for TTS HTTP request.
                # The on_start callback from TTS is a backup (dedup flag prevents
                # double-show).  Without this, bubble only appears after TTS starts.
                _show_bubble()
                self._speak_with_queue(parsed.text, _show_bubble, responder=_target)
                # Fallback in case neither immediate nor TTS callback fired
                _show_bubble()
                # Track for echo detection
                self._recently_spoken.append(parsed.text)
                if len(self._recently_spoken) > 5:
                    self._recently_spoken.pop(0)
                if self._timeline:
                    from core.event_timeline import EventType
                    self._timeline.append(EventType.PONY_SAID,
                                          f'Pony replied: "{parsed.text[:150]}"')
                # Offer piggyback to other ponies after user response
                if self.pony_manager and len(self.pony_manager.ponies) > 1:
                    try:
                        _responder = _target or self.pony_manager.primary
                        self.pony_manager.offer_piggyback(
                            _responder,
                            original_user_text or "",
                            parsed.text,
                        )
                    except Exception as exc:
                        logger.debug("Piggyback offer failed: %s", exc)
                return True
            return False

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._transition(PipelineState.ERROR)
            logger.exception("Pipeline turn error: %s", exc)
            return False
        finally:
            # Always restore primary LLM and clear active responder (Fix 10/11)
            self.llm = _orig_llm
            self._active_responder = None

    # Phrases that indicate the model broke character and is meta-analyzing
    _CHARACTER_BREAK_PHRASES = (
        "system prompt", "character configuration", "character card",
        "character prompt", "desktop companion",
        "claude on claude", "i'm claude", "i am claude", "as claude",
        "i'm an ai", "i am an ai", "as an ai", "language model",
        "i'm chatgpt", "i am chatgpt", "as chatgpt",
        "desktop companion prompt", "bonzipony conversation", "bonzipony",
        "sharing this with me", "sharing your prompt",
        "looking at this document", "analyze this prompt",
        "let me understand what's happening",
        "let me break down", "let me analyze", "let me examine",
        "roleplay", "role-play", "stay in character", "in-character",
        "the user is asking me to", "the user is asking",
        "i'm an assistant", "i am an assistant",
        "how can i help you today",
        "i'd be happy to help", "i'd be happy to assist",
        "i can help you with", "let me help you with",
        "text-to-speech engine",  # quoting our own system prompt
        "anti-slop rules",        # quoting our own system prompt
        "voice rules",            # quoting our own system prompt
        "would respond",          # "here's how X would respond"
        "the user wants",
        "here's a",               # "here's a simple/basic/quick..."
        "here is a",
        "i'll create", "i will create",  # "I'll create a..."
        "let me create",
        "i'll build", "i will build",
        "i'll make", "i will make",
        "i'll write",             # "I'll write some code..." (not WRITE_NOTEPAD)
        "well-crafted", "well crafted",  # meta-praise of the prompt
        "key components", "action system", "accountability system",
        "prompt for", "prompt design",
        "tts rules", "anti-slop", "conversation flow",
        "directive system", "enforcement", "action tags",
    )
    # Strong signals — a single hit is enough
    _CHARACTER_BREAK_STRONG = (
        "system prompt", "character card", "character configuration",
        "character prompt",
        "i'm claude", "i am claude", "i'm chatgpt", "i am chatgpt",
        "as an ai assistant",
        "based on this document", "based on this prompt",
        "based on the document", "based on the prompt",
        "here's how", "here is how",
        "let me break down",      # meta-analysis opener
        "well-crafted",           # praising the prompt
        "desktop companion application",
        "## ",                    # markdown header in speech = instant break
    )

    # Regex patterns that detect code/structured output (never valid in spoken responses)
    _CODE_OUTPUT_PATTERNS = (
        re.compile(r"```"),                          # code fences
        re.compile(r"^#{1,6}\s+\w", re.MULTILINE),  # markdown headers
        re.compile(r"^\*\*[^*]+\*\*\s*[-—:]", re.MULTILINE),  # **Bold** - description (markdown analysis)
        re.compile(r"^import \w+", re.MULTILINE),    # Python imports
        re.compile(r"^from \w+ import", re.MULTILINE),
        re.compile(r"^def \w+\(", re.MULTILINE),     # Python function defs
        re.compile(r"^class \w+[\(:]", re.MULTILINE), # Python class defs
        re.compile(r"<(?:div|span|html|body|head|script|style|form|input|button|p|h[1-6]|ul|ol|li|table|tr|td|a\s|img\s)[^>]*>", re.IGNORECASE),  # HTML tags
        re.compile(r"^\s*(?:const|let|var|function)\s+\w+", re.MULTILINE),  # JS declarations
        re.compile(r"document\.(?:getElementById|querySelector|createElement)", re.IGNORECASE),  # DOM manipulation
        re.compile(r"\.(?:addEventListener|innerHTML|textContent|appendChild)\b"),  # DOM methods
        re.compile(r"^\s*<\?php", re.MULTILINE),     # PHP
        re.compile(r"(?:console|window)\.(?:log|alert|confirm)\("),  # JS console/window
        re.compile(r"\{\s*\n\s*(?:return|if|for|while)\b", re.MULTILINE),  # code blocks with control flow
        re.compile(r"^\s*[-*]\s+`[^`]+`\s*[-—:]", re.MULTILINE),  # - `tag` — description (docs)
        re.compile(r"^\d+\.\s+\*\*", re.MULTILINE),  # 1. **Bold** (numbered markdown list)
    )

    # Regex to strip meta-analysis preamble before in-character content
    _META_PREAMBLE_RE = re.compile(
        r"^.*?(?:would respond|would say|here's (?:how|what)|in character)\s*[:]\s*\n*",
        re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def _is_character_break(response: str) -> bool:
        """Detect when the model broke character and is meta-analyzing the prompt,
        OR when it outputs code/structured content instead of spoken dialogue."""
        if not response or len(response) < 30:
            return False
        lower = response.lower()
        # Single strong signal is enough
        if any(phrase in lower for phrase in Pipeline._CHARACTER_BREAK_STRONG):
            return True
        # Code output detection — if 2+ code patterns match, it's definitely not speech
        code_hits = sum(1 for pat in Pipeline._CODE_OUTPUT_PATTERNS if pat.search(response))
        if code_hits >= 2:
            return True
        # Two weak signals
        hits = sum(1 for phrase in Pipeline._CHARACTER_BREAK_PHRASES if phrase in lower)
        # A single code pattern + a single phrase = break
        if code_hits >= 1 and hits >= 1:
            return True
        return hits >= 2

    @staticmethod
    def _strip_meta_preamble(response: str) -> str:
        """Strip meta-analysis preamble if the response has in-character content after it."""
        stripped = Pipeline._META_PREAMBLE_RE.sub("", response).strip()
        # Only use stripped version if there's substantial content left
        if stripped and len(stripped) > 20:
            return stripped
        return response

    # Phrases that signal the user is leaving or ending the conversation
    # Only match clear, unambiguous goodbye phrases — not casual acknowledgments
    _USER_END_PHRASES = (
        "goodnight", "good night", "night night", "nighty night",
        "going to sleep", "gonna sleep", "heading to bed", "going to bed",
        "talk to you later", "talk later",
        "goodbye", "ok bye", "alright bye",
        "gotta go", "gonna go", "heading out", "gotta run", "gotta bounce",
        "im out", "i'm out",
    )
    # Phrases in Dash's response that signal she's done talking
    _RESPONSE_END_PHRASES = (
        "goodnight", "good night", "night night", "sleep well", "sleep tight",
        "sweet dreams",
    )

    def _heuristic_convo_end(self, user_text: str, response_text: str) -> bool:
        """Fallback: detect obvious conversation enders when LLM forgets the tag.

        This should be CONSERVATIVE — false negatives (missing an ending) just
        mean the conversation listens for one more round, which is fine.  False
        positives (ending too early) are much worse because the user has to
        re-trigger the wake word.
        """
        u = user_text.lower().strip()
        # Only match if the ENTIRE user message is a goodbye phrase (or very close)
        # Don't substring-match — "I'll do it later" should NOT end the conversation
        u_words = u.split()
        if len(u_words) <= 4:
            if any(u == phrase or u.rstrip("!.") == phrase for phrase in self._USER_END_PHRASES):
                return True
        # For longer messages, only match if they START with a goodbye
        if len(u_words) <= 6:
            if any(u.startswith(phrase) for phrase in ("gotta go", "gonna go", "heading out",
                                                        "going to sleep", "heading to bed")):
                return True
        # Dash's response sounds like a sign-off AND is very short (not a question)
        r = response_text.lower()
        if any(phrase in r for phrase in self._RESPONSE_END_PHRASES):
            if "?" not in response_text and len(response_text.split()) < 12:
                return True
        return False

    def _inject_screen_state(self, user_text: str) -> str:
        """Always inject win32gui window state — zero API cost."""
        if self.screen_monitor is None:
            return user_text
        try:
            state = self.screen_monitor.get_state()
            if not state.foreground:
                return user_text

            fg = state.foreground
            exe = fg.exe_name or "unknown"
            fullscreen = " FULLSCREEN" if fg.is_fullscreen else ""
            dur = f"{state.foreground_duration_s:.0f}s" if state.foreground_duration_s < 60 else f"{state.foreground_duration_s / 60:.0f}m"

            # Compact window list with exe names
            windows = []
            for w in state.open_windows[:15]:
                e = f" [{w.exe_name}]" if w.exe_name else ""
                windows.append(f"{w.title}{e}")

            context = (
                f"[Screen: \"{fg.title}\" ({exe}{fullscreen}) active {dur} | "
                f"Other windows: {', '.join(windows[:8])}]"
            )

            return f"{context}\n\n{user_text}"
        except Exception:
            return user_text

    # ── VC/call app detection (for audio context) ─────────────────────────

    _VC_EXES = {
        "discord.exe", "teams.exe", "zoom.exe", "slack.exe", "skype.exe",
        "telegram.exe", "facetime", "webex.exe", "googlemeetelectron.exe",
    }
    _VC_TITLE_KEYWORDS = [
        "discord", "zoom meeting", "microsoft teams", "slack huddle",
        "skype", "google meet", "facetime", "webex",
    ]

    def _build_audio_context(self, user_text: str) -> str:
        """Inject audio context so the LLM understands WHAT it's hearing.

        Combines speaker verification confidence with screen state to tell
        the LLM whether the transcription is the user talking directly,
        ambient audio from speakers/TV, or a mix.  The LLM gets EVERYTHING —
        nothing is thrown away — but it knows the source.
        """
        parts = []

        # Speaker confidence from transcriber
        confidence = getattr(self.transcriber, "last_speaker_confidence", 1.0)
        has_voice_model = (
            hasattr(self.transcriber, "speaker_verifier")
            and self.transcriber.speaker_verifier is not None
            and self.transcriber.speaker_verifier.enrolled
        )

        # Detect media / VC from screen state
        media_playing = False
        vc_active = False
        media_name = ""
        vc_name = ""
        if self.screen_monitor:
            try:
                state = self.screen_monitor.get_state()
                if state:
                    for w in state.open_windows[:20]:
                        title_lower = (w.title or "").lower()
                        exe_lower = (w.exe_name or "").lower()
                        # Check media
                        if not media_playing:
                            from core.screen_monitor import _is_media_app
                            if _is_media_app(w.exe_name, w.title or ""):
                                media_playing = True
                                media_name = w.title or w.exe_name or "media"
                        # Check VC
                        if not vc_active:
                            if exe_lower in self._VC_EXES:
                                vc_active = True
                                vc_name = w.title or w.exe_name or "voice call"
                            elif any(kw in title_lower for kw in self._VC_TITLE_KEYWORDS):
                                vc_active = True
                                vc_name = w.title or "voice call"
            except Exception:
                pass

        # Build the context annotation
        if has_voice_model:
            if confidence >= 0.85:
                parts.append(f"Speaker: user (confidence {confidence:.0%})")
            elif confidence >= 0.6:
                parts.append(
                    f"Speaker: UNCERTAIN — might be the user or ambient audio "
                    f"(confidence {confidence:.0%}). Read the text carefully "
                    f"and use context to judge."
                )
            else:
                parts.append(
                    f"Speaker: likely NOT the user (confidence {confidence:.0%}). "
                    f"This transcription is probably from speakers, TV, video, "
                    f"or someone else nearby. Do NOT respond as if the user "
                    f"said this — but you can reference it if relevant."
                )

        if media_playing:
            parts.append(f"Media playing: \"{media_name[:80]}\"")
        if vc_active:
            parts.append(
                f"Voice call active: \"{vc_name[:80]}\" — "
                f"some of this transcription may be from the other person on the call"
            )

        if not parts:
            return user_text  # nothing to annotate

        annotation = "[Audio context: " + ". ".join(parts) + ".]"
        return f"{annotation}\n\n{user_text}"

    _SCREEN_KEYWORDS = ("screen", "what do you see", "look", "what's that", "what is that", "what's on")
    _CAMERA_KEYWORDS = ("on camera", "webcam", "camera", "how do i look", "what do i look",
                        "see me", "look at me", "do i look", "am i wearing")

    def _maybe_inject_vision(self, user_text: str) -> str:
        """Inject screen or camera vision based on context.

        - Webcam: ONLY when user explicitly asks (camera keywords).
        - Screen: when user explicitly asks, or randomly for background context.
        - Vision provider controlled by config.vision.screen_vision:
          "api" = main LLM describe_screen, "moondream" = local model.
        """
        text_lower = user_text.lower()

        # Check if user explicitly asked for camera — ONLY way webcam activates
        camera_triggered = any(kw in text_lower for kw in self._CAMERA_KEYWORDS)
        if camera_triggered and self.config.vision.enabled:
            return self._inject_camera_vision(user_text)

        # Check if user explicitly asked about screen — always use main LLM for explicit requests
        screen_triggered = any(kw in text_lower for kw in self._SCREEN_KEYWORDS)
        if screen_triggered and self.config.vision.screen_capture:
            return self._inject_screen(user_text)

        # Background screen context — only if screen capture is enabled
        if not self.config.vision.screen_capture:
            return user_text

        use_moondream = self.config.vision.screen_vision == "moondream"

        if use_moondream and self.moondream and self.moondream.loaded:
            return self._inject_moondream_screen(user_text)

        # API mode: 20% chance per message to keep costs down
        has_vision = self.vision_llm or hasattr(self.llm, "describe_screen")
        if not use_moondream and random.random() < 0.2 and has_vision:
            return self._inject_screen(user_text)

        return user_text

    def _inject_camera_vision(self, user_text: str) -> str:
        """Inject webcam image description into user text."""
        if self.camera is None or not self.camera.available:
            return user_text
        vlm = self.vision_llm
        if vlm is None and not hasattr(self.llm, "describe_image"):
            return user_text

        try:
            jpeg = self.camera.grab()
            if jpeg is None:
                return user_text
            description = vlm.describe_image(jpeg) if vlm else self.llm.describe_image(jpeg)
            if not description:
                return user_text
            logger.info("Camera vision: %s", description)
            self._remember_visual(description)
            return f"[Visual context — what you can currently see: {description}]\n\n{user_text}"
        except Exception as exc:
            logger.warning("Camera vision failed: %s", exc)
            return user_text

    def _inject_screen(self, user_text: str) -> str:
        """Inject screenshot description into user text. Never falls back to webcam."""
        if self.screen is None or not self.screen.available:
            logger.debug("Screen capture not available — skipping.")
            return user_text
        vlm = self.vision_llm
        if vlm is None and not hasattr(self.llm, "describe_screen"):
            logger.debug("LLM has no describe_screen method.")
            return user_text

        try:
            jpeg = self.screen.grab()
            if jpeg is None:
                return user_text
            description = vlm.describe_screen(jpeg) if vlm else self.llm.describe_screen(jpeg)
            if not description:
                return user_text
            logger.info("Screen vision: %s", description)
            self._remember_visual(description)
            return f"[Screen context — what's on the user's screen: {description}]\n\n{user_text}"
        except Exception as exc:
            logger.warning("Screen vision failed: %s", exc)
            return user_text

    def _inject_moondream_screen(self, user_text: str) -> str:
        """Inject cheap local Moondream screen description into every message."""
        if self.moondream is None or not self.moondream.available:
            return user_text
        if self.screen is None or not self.screen.available:
            return user_text
        try:
            jpeg = self.screen.grab()
            if jpeg is None:
                return user_text
            description = self.moondream.describe(jpeg)
            if not description:
                return user_text
            logger.debug("Moondream screen: %s", description)
            self._remember_visual(description)
            return f"[Screen context: {description}]\n\n{user_text}"
        except Exception as exc:
            logger.debug("Moondream screen inject failed: %s", exc)
            return user_text

    def comment_on_screen(self) -> None:
        """Glance at the screen and make a spontaneous comment about what's visible."""
        if self.screen is None or not self.screen.available:
            return

        try:
            # Use Moondream if available, fall back to main LLM
            jpeg = self.screen.grab()
            if jpeg is None:
                return

            description = None
            use_moondream = self.config.vision.screen_vision == "moondream"
            if use_moondream and self.moondream and self.moondream.loaded:
                description = self.moondream.describe(jpeg)
            elif self.vision_llm:
                description = self.vision_llm.describe_screen(jpeg)
            elif hasattr(self.llm, "describe_screen"):
                description = self.llm.describe_screen(jpeg)

            if not description:
                return

            logger.info("Screen glance: %s", description)
            self._remember_visual(description)

            trigger = (
                f"(You glanced at the screen and saw: {description}. "
                f"React in ONE short sentence as {get_character_name()}. Be specific about what you noticed.)"
            )

            # Use chat() so it enters history — Dash remembers what she saw
            raw = self.llm.chat(trigger)
            if not raw:
                return

            from llm.response_parser import parse_response
            parsed = parse_response(raw)
            logger.info("Screen comment: %r", parsed.text)

            if parsed.text:
                self._transition(PipelineState.SPEAK)
                _bubble_shown = False
                def _show_bubble():
                    nonlocal _bubble_shown
                    if _bubble_shown:
                        return
                    _bubble_shown = True
                    if self._on_speech_text:
                        try:
                            self._on_speech_text(parsed.text)
                        except Exception:
                            pass
                # Show bubble immediately — don't wait for TTS callback chain
                _show_bubble()
                from core.tts_queue import PRIORITY_AUTONOMOUS
                self._speak_with_queue(parsed.text, _show_bubble, priority=PRIORITY_AUTONOMOUS, blocking=True)

        except Exception as exc:
            logger.warning("Screen commentary failed: %s", exc)
        finally:
            self._transition(PipelineState.IDLE)

    def _remember_visual(self, description: str) -> None:
        self._visual_memory.append(description)
        if len(self._visual_memory) > 20:
            self._visual_memory.pop(0)

    def _remember_topic(self, text: str) -> None:
        snippet = text.strip()[:60]
        self._recent_topics.append(snippet)
        if len(self._recent_topics) > 20:
            self._recent_topics.pop(0)

    def _extract_user_profile(self, force: bool = False) -> None:
        """Extract user profile facts from conversation.

        Rate-limited to once per 4 hours unless force=True (shutdown).
        When force=True (shutdown), runs synchronously so the process doesn't
        exit before extraction completes.  Otherwise runs in a background thread.
        """
        if not self.llm.has_history():
            return
        now = time.monotonic()
        if not force and (now - self._last_profile_extraction) < _PROFILE_COOLDOWN_S:
            logger.debug("Profile extraction skipped — cooldown active (%.0f min remaining)",
                         (_PROFILE_COOLDOWN_S - (now - self._last_profile_extraction)) / 60)
            return
        try:
            import threading
            from core.user_profile import update_from_conversation
            history = list(getattr(self.llm, "_history", []))
            if len(history) < 2:
                return
            self._last_profile_extraction = now
            print("[Profile] Extracting user profile from conversation...", flush=True)
            if force:
                # Shutdown path — run synchronously so the process doesn't
                # exit before the LLM call finishes
                update_from_conversation(self.llm, history)
            else:
                # Normal path — background thread so it doesn't block wake word
                t = threading.Thread(
                    target=update_from_conversation,
                    args=(self.llm, history),
                    daemon=True,
                )
                t.start()
        except Exception as exc:
            logger.warning("Profile extraction failed: %s", exc)

    @staticmethod
    def _parse_time_estimate(text: str) -> Optional[int]:
        """Parse natural language time estimates. Returns seconds or None."""
        text_l = text.lower()
        # "X minutes" / "X min"
        m = re.search(r'(\d+)\s*(?:minutes?|mins?)', text_l)
        if m:
            return int(m.group(1)) * 60
        # "X hours" / "X hr"
        m = re.search(r'(\d+)\s*(?:hours?|hrs?)', text_l)
        if m:
            return int(m.group(1)) * 3600
        # "half an hour" / "half hour"
        if "half" in text_l and "hour" in text_l:
            return 1800
        # "an hour"
        if "an hour" in text_l:
            return 3600
        # "X seconds" / "X sec"
        m = re.search(r'(\d+)\s*(?:seconds?|secs?)', text_l)
        if m:
            return int(m.group(1))
        # bare number — assume minutes
        m = re.search(r'\b(\d+)\b', text_l)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 180:
                return val * 60
        return None

    def _transition(self, new_state: PipelineState) -> None:
        logger.debug("Pipeline: %s → %s", self.state.name, new_state.name)
        self.state = new_state
        if self._on_state_change:
            try:
                self._on_state_change(new_state.name)
            except Exception:
                pass
