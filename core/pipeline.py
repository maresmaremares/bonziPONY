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
        self._last_end_conversation: bool = False  # LLM signaled conversation over
        self._last_profile_extraction: float = 0.0  # monotonic timestamp
        _PROFILE_COOLDOWN_S = 4 * 3600  # 4 hours between profile extractions

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

        self._last_end_conversation = False
        spoke = self._run_turn(play_ack=True)
        if not spoke and not self._last_end_conversation:
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)
            return

        # If LLM signaled end on first turn (e.g. user said "goodnight")
        if self._last_end_conversation:
            logger.info("LLM signaled end of conversation after first turn.")
            print("[Conversation ended naturally]")
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)
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

            discard_ms = 600 if just_spoke else 0
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

        self._last_end_conversation = False
        if self._on_heard_text:
            try:
                self._on_heard_text(text)
            except Exception:
                pass
        spoke = self._run_turn(play_ack=False, user_text=text)
        if not spoke and not self._last_end_conversation:
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)
            return

        if self._last_end_conversation:
            if self.agent_loop:
                self.agent_loop.set_conversation_active(False)
            if self._on_conversation_end:
                try:
                    self._on_conversation_end()
                except Exception:
                    pass
            self._transition(PipelineState.IDLE)
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
            discard_ms = 600 if just_spoke else 0
            user_text = self.transcriber.listen(speech_start_timeout_s=wait, initial_discard_ms=discard_ms)
            just_spoke = False

            if not user_text or not user_text.strip():
                break

            from stt.transcriber import _is_whisper_hallucination
            if _is_whisper_hallucination(user_text):
                continue

            spoke = self._run_turn(play_ack=False, user_text=user_text)
            if self._last_end_conversation:
                break
            if spoke:
                deadline = time.monotonic() + convo_timeout
                just_spoke = True

        self._extract_user_profile()
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
                if self.config.tts.enabled:
                    self.tts.speak(parsed.text, on_playback_start=_show_bubble)
                _show_bubble()

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

            # Stop keyword detection — clear all directives if user says stop
            if self.agent_loop and self.agent_loop.has_directives:
                _STOP_KW = ("stop", "knock it off", "enough", "quit it", "shut up", "leave me alone", "chill", "cut it out")
                if any(kw in user_text.lower() for kw in _STOP_KW):
                    self.agent_loop.clear_directives()
                    user_text = f"[System: User told you to stop. All your active directives have been cleared.]\n\n{user_text}"

            # ALWAYS inject win32gui screen context (free)
            user_text = self._inject_screen_state(user_text)

            user_text = self._maybe_inject_vision(user_text)

            # Nudge LLM to pick up on tasks/needs, compliance, and conversation flow
            if self.agent_loop:
                hint = (
                    "\n\n[System hint: Pick the RIGHT tag for the situation. "
                    "Something they should do RIGHT NOW (actively neglecting a need) → [DIRECTIVE:goal:urgency]. "
                    "Something they should do LATER today → [DIRECTIVE:goal:urgency:delay_minutes] (delays first nag). "
                    "Something at a SPECIFIC TIME → [TIMER:HH:MM:goal]. "
                    "Something RECURRING → [ROUTINE:...]. "
                    "Something they're just MENTIONING (plans, stories) → NO TAG, just talk about it. "
                    "Only create immediate directives for things they're clearly neglecting RIGHT NOW. "
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
                )

            # Create timer from conversation if LLM used [TIMER:...] tag
            if parsed.timer and self.agent_loop:
                self.agent_loop.add_timer(
                    time_str=parsed.timer.time_str,
                    action=parsed.timer.action,
                )

            # Create recurring routines from conversation if LLM used [ROUTINE:...] tags
            if parsed.routines and self.agent_loop:
                import uuid
                from core.routines import Routine
                for rt in parsed.routines:
                    routine = Routine(
                        id=str(uuid.uuid4())[:8],
                        goal=rt.goal,
                        urgency=rt.urgency,
                        schedule=rt.schedule,
                        time=rt.time,
                        day=rt.day if rt.schedule == "weekly" else None,
                        interval_hours=rt.hours if rt.schedule == "interval" else None,
                        sleep_offset_hours=rt.hours if rt.schedule == "on_sleep" and rt.hours else 8.0,
                    )
                    self.agent_loop.routine_manager.add(routine)
                    logger.info("Routine created from conversation: %s (%s)", rt.goal, rt.schedule)

            # Mark directive as completed if LLM used [DONE] or [DONE:keyword]
            if parsed.done_directive is not None and self.agent_loop and self.agent_loop.has_directives:
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
                if self.config.tts.enabled:
                    self.tts.speak(parsed.text, on_playback_start=_show_bubble)
                # Always ensure bubble was shown (fallback if TTS failed/skipped callback)
                _show_bubble()
                if self._timeline:
                    from core.event_timeline import EventType
                    self._timeline.append(EventType.PONY_SAID,
                                          f'Pony replied: "{parsed.text[:150]}"')
                return True
            return False

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._transition(PipelineState.ERROR)
            logger.exception("Pipeline turn error: %s", exc)
            return False

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
                if self.config.tts.enabled:
                    self.tts.speak(parsed.text, on_playback_start=_show_bubble)
                _show_bubble()

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
        if not force and (now - self._last_profile_extraction) < 4 * 3600:
            logger.debug("Profile extraction skipped — cooldown active (%.0f min remaining)",
                         (4 * 3600 - (now - self._last_profile_extraction)) / 60)
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
