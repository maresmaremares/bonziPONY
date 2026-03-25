"""
PonyInstance — bundles all per-pony state for the multi-pony system.

Each pony on the desktop gets one PonyInstance containing its own:
- GUI widgets (PetWindow, SpeechBubble, HeardText)
- Sprite/behavior managers
- LLM provider (own history, shared API config)
- PromptConfig (own system prompt with companion awareness)
- TTS voice slug
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from llm.base import LLMProvider
    from llm.prompt import PromptConfig

logger = logging.getLogger(__name__)


# Keyword aliases for speech routing (name detection)
_NAME_KEYWORDS: dict[str, list[str]] = {
    "rainbow_dash":              ["rainbow dash", "dash", "dashie", "rd"],
    "twilight_sparkle":          ["twilight sparkle", "twilight", "twi", "twily"],
    "princess_twilight_sparkle": ["twilight sparkle", "twilight", "twi", "twily"],
    "pinkie_pie":                ["pinkie pie", "pinkie", "pinks"],
    "rarity":                    ["rarity", "rares", "rari"],
    "fluttershy":                ["fluttershy", "flutter", "shy", "flutters"],
    "applejack":                 ["applejack", "aj", "apple jack"],
    "spike":                     ["spike", "spikey", "spikeywikey"],
    "trixie":                    ["trixie", "trix"],
    "starlight_glimmer":         ["starlight", "glimmer", "starlight glimmer"],
    "princess_celestia":         ["celestia", "princess celestia", "tia"],
    "princess_luna":             ["luna", "princess luna", "lulu", "woona"],
    "princess_cadance":          ["cadance", "cadence", "princess cadance"],
    "discord":                   ["discord"],
}


def _get_keywords_for(slug: str) -> list[str]:
    """Return name keywords for a character slug, longest first."""
    if slug in _NAME_KEYWORDS:
        kws = list(_NAME_KEYWORDS[slug])
    else:
        # Auto-generate from slug: "apple_bloom" → ["apple bloom"]
        display = slug.replace("_", " ")
        kws = [display]
        # Also add first word if multi-word
        parts = display.split()
        if len(parts) > 1:
            kws.append(parts[0])
    # Sort longest first so "rainbow dash" matches before "dash"
    kws.sort(key=len, reverse=True)
    return kws


class PonyInstance:
    """All state for one pony on the desktop."""

    def __init__(
        self,
        slug: str,
        display_name: str,
        is_primary: bool,
        prompt_config: "PromptConfig",
        llm: "LLMProvider",
        pet_window: Any,
        pet_controller: Any,
        speech_bubble: Any,
        heard_text: Any,
        sprite_manager: Any,
        behavior_manager: Any,
        effect_renderer: Any,
        pony_dir: Path,
        name_keywords: list[str] | None = None,
    ) -> None:
        self.slug = slug
        self.display_name = display_name
        self.is_primary = is_primary
        self.prompt_config = prompt_config
        self.llm = llm
        self.pet_window = pet_window
        self.pet_controller = pet_controller
        self.speech_bubble = speech_bubble
        self.heard_text = heard_text
        self.sprite_manager = sprite_manager
        self.behavior_manager = behavior_manager
        self.effect_renderer = effect_renderer
        self.pony_dir = pony_dir
        self.name_keywords = name_keywords or _get_keywords_for(slug)
        self.agent_loop: Any = None  # set externally for primary
        self._destroyed: bool = False

        # Check if this character has a TTS voice
        try:
            from tts.openai_compatible_tts import has_pvt_voice
            self.has_voice = has_pvt_voice(slug)
        except Exception:
            self.has_voice = True  # assume yes if we can't check

    @classmethod
    def create(
        cls,
        slug: str,
        is_primary: bool,
        config: Any,
        ponies_root: Path,
        app_config: Any,
    ) -> "PonyInstance":
        """Full lifecycle creation of a secondary pony.

        Creates all per-pony objects: sprites, behaviors, window, bubble,
        LLM provider (own history), and PromptConfig.

        Parameters
        ----------
        slug : str
            Character slug, e.g. ``"twilight_sparkle"``.
        is_primary : bool
            Whether this is the primary pony (first/main).
        config : AppConfig
            Full app config — used to configure LLM, TTS, etc.
        ponies_root : Path
            Root path to the Ponies/ directory.
        app_config : AppConfig
            Same as *config* (kept for clarity).
        """
        from core.character_registry import get_display_name, slug_to_dir_name
        from desktop_pet.behavior_manager import BehaviorManager
        from desktop_pet.effect_renderer import EffectRenderer
        from desktop_pet.heard_text import HeardText
        from desktop_pet.pet_controller import PetController
        from desktop_pet.pet_window import PetWindow
        from desktop_pet.speech_bubble import SpeechBubble
        from desktop_pet.sprite_manager import SpriteManager
        from llm.factory import get_provider
        from llm.prompt import PromptConfig, get_system_prompt_for

        display_name = get_display_name(slug)

        # ── Per-pony PromptConfig ──
        prompt_config = PromptConfig(
            preset=slug,
            relationship_mode=config.llm.relationship,
            relationship_custom=config.llm.relationship_custom,
        )

        # ── Per-pony LLM provider (own history, shared API config) ──
        llm = get_provider(config)
        llm.system_prompt_fn = lambda: get_system_prompt_for(prompt_config)
        llm.character_name = display_name

        # ── Sprites & behaviors ──
        pony_dir = ponies_root / slug_to_dir_name(slug)
        sprite_manager = SpriteManager(pony_dir, scale=config.desktop_pet.scale)
        behavior_manager = BehaviorManager(pony_dir / "pony.ini")
        behavior_manager.parse()
        sprite_manager.build_sprite_map(behavior_manager)
        sprite_manager.preload_all()
        effect_renderer = EffectRenderer(sprite_manager, behavior_manager)

        # ── GUI widgets ──
        pet_controller = PetController()
        pet_window = PetWindow(sprite_manager, behavior_manager, effect_renderer)
        speech_bubble = SpeechBubble()
        speech_bubble.set_anchor_widget(pet_window)
        heard_text = HeardText()
        heard_text.set_anchor_widget(pet_window)

        # ── Wire Qt signals for secondary ponies ──
        # (Primary pony's signals are wired in main.py with more callbacks;
        #  secondary ponies only need basic animation + bubble support.)
        from PyQt5.QtCore import Qt

        def _on_state_changed(state_name: str) -> None:
            anim = PetController.get_animation_for_state(state_name)
            if anim is None:
                pet_window.clear_override()
            else:
                pet_window.set_override_animation(anim)
            pet_window.set_listening(state_name == "LISTEN")
            if state_name == "THINK" and config.desktop_pet.speech_bubble:
                ax, ay, ah = pet_window.get_anchor_point()
                speech_bubble.show_thinking(ax, ay)
            elif state_name == "IDLE" and config.desktop_pet.speech_bubble:
                if speech_bubble._thinking:
                    speech_bubble.hide_bubble()

        def _on_speech_text(text: str) -> None:
            heard_text.hide_heard()
            if config.desktop_pet.speech_bubble:
                ax, ay, ah = pet_window.get_anchor_point()
                speech_bubble.show_text(text, ax, ay, sprite_h=ah)

        def _on_heard_text_cb(text: str) -> None:
            heard_text.show_heard(text)

        def _on_conversation_ended() -> None:
            pet_window.set_listening(False)
            pet_window.clear_override()
            pet_window.resume_roaming()
            heard_text.hide_heard()

        pet_controller.state_changed.connect(_on_state_changed, Qt.QueuedConnection)
        pet_controller.speech_text.connect(_on_speech_text, Qt.BlockingQueuedConnection)
        pet_controller.heard_text.connect(_on_heard_text_cb, Qt.QueuedConnection)
        pet_controller.conversation_ended.connect(_on_conversation_ended, Qt.QueuedConnection)
        pet_controller.action_triggered.connect(
            lambda action: pet_window.set_override_animation(
                PetController.get_animation_for_action(action)
            ) if PetController.get_animation_for_action(action) else None,
            Qt.QueuedConnection,
        )

        instance = cls(
            slug=slug,
            display_name=display_name,
            is_primary=is_primary,
            prompt_config=prompt_config,
            llm=llm,
            pet_window=pet_window,
            pet_controller=pet_controller,
            speech_bubble=speech_bubble,
            heard_text=heard_text,
            sprite_manager=sprite_manager,
            behavior_manager=behavior_manager,
            effect_renderer=effect_renderer,
            pony_dir=pony_dir,
        )

        logger.info("PonyInstance created: %s (primary=%s)", display_name, is_primary)
        return instance

    def update_companions(self, all_instances: list["PonyInstance"]) -> None:
        """Refresh companion awareness in this pony's PromptConfig."""
        companions = []
        has_twin = False
        for other in all_instances:
            if other is self:
                continue
            companions.append(other.display_name)
            if other.slug == self.slug:
                has_twin = True
        self.prompt_config.companions = companions
        self.prompt_config.is_twin = has_twin

    def get_window_center(self) -> tuple[int, int]:
        """Return (cx, cy) of this pony's PetWindow."""
        w = self.pet_window
        return (w.x() + w.width() // 2, w.y() + w.height() // 2)

    def destroy(self) -> None:
        """Clean up this pony — close windows, stop agent loop, disconnect signals."""
        logger.info("Destroying PonyInstance: %s", self.display_name)
        self._destroyed = True
        if self.agent_loop:
            try:
                self.agent_loop.stop()
            except Exception:
                pass
            self.agent_loop = None
        # Disconnect Qt signals to prevent memory leaks and callbacks on dead objects
        try:
            self.pet_controller.state_changed.disconnect()
            self.pet_controller.speech_text.disconnect()
            self.pet_controller.heard_text.disconnect()
            self.pet_controller.conversation_ended.disconnect()
            self.pet_controller.action_triggered.disconnect()
        except (TypeError, RuntimeError):
            pass  # already disconnected or no connections
        try:
            self.speech_bubble.hide_bubble()
        except Exception:
            pass
        try:
            self.heard_text.hide_heard()
        except Exception:
            pass
        try:
            self.pet_window.close()
        except Exception:
            pass
