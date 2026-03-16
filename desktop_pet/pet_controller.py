"""Bridges the AI pipeline to the GUI via Qt signals (thread-safe)."""

from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, pyqtSignal

from robot.actions import RobotAction

logger = logging.getLogger(__name__)

# Pipeline state → sprite animation name
_STATE_ANIMATION_MAP = {
    "IDLE": None,           # Clear override, return to roaming
    "ACKNOWLEDGE": "beep",
    "LISTEN": "hover",
    "THINK": "dizzy",
    "SPEAK": "stand",       # Stand still while talking — no movement
    "ERROR": "dizzy",
}

# Robot action → sprite animation name
_ACTION_ANIMATION_MAP = {
    RobotAction.WALK_FORWARD: "walk",
    RobotAction.WALK_BACKWARD: "walk",
    RobotAction.SIT: "sleep",
    RobotAction.STAND: "stand",
    RobotAction.WAVE: "salute",
    RobotAction.SHAKE: "beep",
    RobotAction.SPIN: "dizzy",
    # Desktop control actions
    RobotAction.CLOSE_WINDOW: "salute",
    RobotAction.MINIMIZE_WINDOW: "beep",
    RobotAction.MAXIMIZE_WINDOW: "beep",
    RobotAction.SNAP_WINDOW_LEFT: "beep",
    RobotAction.SNAP_WINDOW_RIGHT: "beep",
    RobotAction.VOLUME_UP: "beep",
    RobotAction.VOLUME_DOWN: "beep",
    RobotAction.VOLUME_MUTE: "beep",
    RobotAction.SCREENSHOT: "dizzy",
}


class PetController(QObject):
    """Thread-safe bridge between the pipeline (background thread) and the GUI (main thread).

    Implements the RobotController interface (execute/shutdown) via duck typing
    to avoid metaclass conflict between QObject and ABC.

    Emits Qt signals from the pipeline thread; GUI widgets connect to them as slots.
    Qt's signal/slot mechanism handles the cross-thread dispatch automatically.
    """

    # Signals emitted from pipeline thread, received on GUI thread
    state_changed = pyqtSignal(str)          # PipelineState name
    speech_text = pyqtSignal(str)            # Text to show in bubble
    heard_text = pyqtSignal(str)             # STT transcription to show
    conversation_started = pyqtSignal()
    conversation_ended = pyqtSignal()
    action_triggered = pyqtSignal(str)       # RobotAction name
    trick_requested = pyqtSignal()           # do a cool trick
    timed_override = pyqtSignal(str, int)    # (animation_name, seconds)
    move_to = pyqtSignal(str)               # screen region name
    grab_run_start = pyqtSignal()           # start grab-cursor run animation
    grab_run_stop = pyqtSignal()            # stop grab-cursor run animation
    countdown_start = pyqtSignal(int)       # start countdown timer (seconds)
    countdown_stop = pyqtSignal()           # hide countdown timer

    def __init__(self) -> None:
        super().__init__()

    # ── RobotController interface (called from pipeline thread) ─────────────

    def execute(self, action: RobotAction) -> None:
        """Execute a robot action by emitting a signal to the GUI thread."""
        logger.info("PetController action: %s", action.name)
        if action == RobotAction.TRICK:
            self.trick_requested.emit()
        else:
            self.action_triggered.emit(action.name)

    def shutdown(self) -> None:
        """Cleanup on shutdown."""
        logger.debug("PetController shutdown")

    # ── Pipeline callbacks (called from pipeline thread) ────────────────────

    def on_state_change(self, state_name: str) -> None:
        """Called when the pipeline transitions state."""
        logger.debug("PetController state: %s", state_name)
        self.state_changed.emit(state_name)

    def on_speech_text(self, text: str) -> None:
        """Called when the pipeline has response text to display."""
        self.speech_text.emit(text)

    def on_heard_text(self, text: str) -> None:
        """Called when the STT transcribes what the user said."""
        self.heard_text.emit(text)

    def on_conversation_start(self) -> None:
        """Called when a conversation begins."""
        self.conversation_started.emit()

    def on_conversation_end(self) -> None:
        """Called when a conversation ends."""
        self.conversation_ended.emit()

    def on_timed_override(self, anim_name: str, seconds: int) -> None:
        """Called when an animation should persist for N seconds."""
        self.timed_override.emit(anim_name, seconds)

    def on_move_to(self, region: str) -> None:
        """Called when the pony should move to a screen region."""
        self.move_to.emit(region)

    # ── Slot helpers for connecting to GUI ──────────────────────────────────

    @staticmethod
    def get_animation_for_state(state_name: str) -> str | None:
        """Return the animation name for a pipeline state, or None to clear override."""
        return _STATE_ANIMATION_MAP.get(state_name)

    @staticmethod
    def get_animation_for_action(action_name: str) -> str | None:
        """Return the animation name for a robot action."""
        try:
            action = RobotAction[action_name]
            return _ACTION_ANIMATION_MAP.get(action)
        except KeyError:
            return None
