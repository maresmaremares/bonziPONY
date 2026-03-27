"""
Desktop Pony Pet — entry point.

Launches a PyQt5 desktop pet with AI voice pipeline integration.
The pet roams freely, responds to wake words and double-clicks,
and shows speech bubbles during conversations.

Usage:
    python main.py
    python main.py --config path/to/config.yaml
"""

from __future__ import annotations

# ── PyAudioWPatch shim ─────────────────────────────────────────────────
# PyAudioWPatch is a drop-in replacement for PyAudio that ships prebuilt
# wheels (no C++ compiler needed), but it installs as "pyaudiowpatch"
# instead of "pyaudio". speech_recognition and other libs do
# `import pyaudio`, so we register it under the expected name.
import sys as _sys
try:
    import pyaudio as _pa  # noqa: F401  — already available, nothing to do
except ImportError:
    try:
        import pyaudiowpatch as _pa
        _sys.modules["pyaudio"] = _pa
    except ImportError:
        pass  # neither installed — mic features will be unavailable
del _sys
try:
    del _pa
except NameError:
    pass

import argparse
import logging
import random
import signal
import sys
import threading
import time
from pathlib import Path


def setup_logging(level: str, log_to_file: bool, log_file: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main() -> None:
    # ── Optional .env loading (pip install python-dotenv) ─────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Desktop Pony Pet")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    from core.config_loader import load_config
    config = load_config(Path(args.config))

    setup_logging(
        config.logging.level,
        config.logging.log_to_file,
        config.logging.log_file,
    )

    logger = logging.getLogger(__name__)

    # Enable faulthandler to capture segfaults / C-level crashes
    import faulthandler
    try:
        # rainbow_dash.log → rainbow_CRASH.log (her mocked nickname)
        log_path = Path(config.logging.log_file)
        crash_name = log_path.stem.rsplit("_", 1)[0] + "_CRASH.log"
        crash_log = log_path.parent / crash_name
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        _crash_fh = open(crash_log, "w")
        faulthandler.enable(file=_crash_fh, all_threads=True)
        logger.info("Crash handler enabled → %s", crash_log)
    except Exception:
        faulthandler.enable()  # fallback: write to stderr

    logger.info("Desktop Pony is waking up!")

    # ── Pre-load torch BEFORE PyQt5 ───────────────────────────────────────────
    # PyQt5 loads OpenGL/platform DLLs that conflict with torch's c10.dll
    # if torch is imported later on a background thread. Import it now.
    try:
        import torch  # noqa: F401
        logger.debug("Pre-loaded torch %s", torch.__version__)
    except ImportError:
        pass

    # ── Scan character registry ──────────────────────────────────────────────
    from core.character_registry import scan_ponies, slug_to_dir_name
    ponies_root = Path(__file__).parent / "Ponies"
    scan_ponies(ponies_root)

    # ── Apply preset ───────────────────────────────────────────────────────────
    from llm.prompt import set_preset, set_relationship
    set_preset(config.llm.preset)
    set_relationship(config.llm.relationship, config.llm.relationship_custom)
    logger.info("Loaded preset: %s", config.llm.preset)

    # ── Build pipeline components ─────────────────────────────────────────────
    from wake_word.detector import WakeWordDetector, get_phrases_for
    from acknowledgement.player import AcknowledgementPlayer
    from stt.transcriber import Transcriber
    from llm.factory import get_provider
    from tts.elevenlabs_tts import ElevenLabsTTS
    from core.pipeline import Pipeline
    from vision.camera import Camera
    from vision.screen import ScreenCapture
    from robot.desktop_controller import DesktopController
    from core.screen_monitor import ScreenMonitor
    from core.agent_loop import AgentLoop

    wake_phrases = get_phrases_for(config.llm.preset, config.wake_word.phrases)
    detector = WakeWordDetector(
        wake_phrases=wake_phrases,
        input_device_index=config.audio.input_device_index,
        language=config.wake_word.language,
        whisper_model=config.wake_word.model,
    )

    ack_player = AcknowledgementPlayer()
    ack_player.set_character(config.llm.preset)

    transcriber = Transcriber(
        model_name=config.whisper.model,
        language=config.whisper.language,
        vad_aggressiveness=config.audio.vad_aggressiveness,
        silence_duration_ms=config.audio.silence_duration_ms,
        input_device_index=config.audio.input_device_index,
    )

    llm_provider = get_provider(config)

    # ── Dedicated vision LLM (optional — uses separate, cheaper model) ────
    vision_llm = None
    vlm_cfg = config.vision_llm
    if vlm_cfg and vlm_cfg.enabled and vlm_cfg.api_keys:
        try:
            from llm.vision_provider import VisionProvider
            from llm.factory import _KNOWN_BASE_URLS
            base_url = vlm_cfg.base_url or _KNOWN_BASE_URLS.get(vlm_cfg.provider.lower())
            # Gemini requires the /openai suffix for OpenAI-compatible API
            if base_url and "generativelanguage.googleapis.com" in base_url and not base_url.rstrip("/").endswith("/openai"):
                base_url = base_url.rstrip("/") + "/openai"
                logger.info("Auto-corrected Gemini base_url to include /openai suffix")
            vision_llm = VisionProvider(
                api_keys=vlm_cfg.api_keys,
                model=vlm_cfg.model,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
                max_requests_per_key_per_day=vlm_cfg.max_requests_per_key_per_day,
                temperature=vlm_cfg.temperature,
                max_tokens=vlm_cfg.max_tokens,
                locate_max_tokens=vlm_cfg.locate_max_tokens,
            )
            logger.info("Dedicated vision LLM: %s (%d keys)", vlm_cfg.model, len(vlm_cfg.api_keys))
        except Exception as exc:
            logger.warning("Failed to init vision LLM, falling back to main LLM: %s", exc)
            vision_llm = None

    # ── TTS provider selection ─────────────────────────────────────────────
    if config.tts.provider == "openai_compatible":
        from tts.openai_compatible_tts import OpenAICompatibleTTS
        tts = OpenAICompatibleTTS(
            base_url=config.tts.base_url,
            model=config.tts.model,
            voice=config.tts.voice,
            response_format=config.tts.response_format,
            sample_rate=config.tts.sample_rate,
            output_device_index=config.audio.output_device_index,
        )
        tts.set_character(config.llm.preset)
        logger.info("TTS: OpenAI-compatible at %s", config.tts.base_url)
    else:
        tts = ElevenLabsTTS(
            api_key=config.elevenlabs.api_key,
            voice_id=config.elevenlabs.voice_id,
            model=config.elevenlabs.model,
            output_format=config.elevenlabs.output_format,
            output_device_index=config.audio.output_device_index,
        )

    # ── TTS thread safety lock ────────────────────────────────────────────
    # Both the pipeline thread and the TTSQueue consumer thread call
    # tts.speak().  Wrap it with a lock so they never overlap.
    _tts_lock = threading.Lock()
    _original_tts_speak = tts.speak
    def _locked_tts_speak(*args, **kwargs):
        with _tts_lock:
            return _original_tts_speak(*args, **kwargs)
    tts.speak = _locked_tts_speak

    # ── TTS Queue (multi-pony: serialised audio playback) ──────────────────
    from core.tts_queue import TTSQueue
    tts_queue = TTSQueue(tts, pause_between=0.4)

    camera = None
    if config.vision.enabled:
        try:
            camera = Camera(device_index=config.vision.device_index)
            if not camera.available:
                logger.warning("Vision enabled but no webcam found — vision disabled.")
                camera = None
        except ImportError as exc:
            logger.warning("Vision disabled: %s", exc)
            camera = None

    screen = None
    if config.vision.screen_capture:
        try:
            screen = ScreenCapture(max_width=config.vision.screen_max_width)
        except ImportError as exc:
            logger.warning("Screen capture disabled: %s", exc)
            screen = None

    moondream = None
    if screen and config.vision.screen_vision == "moondream":
        try:
            from vision.moondream import MoondreamDescriber
            moondream = MoondreamDescriber(use_gpu=config.watch_mode.use_gpu if config.watch_mode else False)
            moondream.start_background_load()
        except Exception as exc:
            logger.warning("Moondream not available: %s", exc)
            moondream = None

    # ── Desktop Pet GUI ───────────────────────────────────────────────────────
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from desktop_pet.pet_controller import PetController
    from desktop_pet.sprite_manager import SpriteManager
    from desktop_pet.behavior_manager import BehaviorManager
    from desktop_pet.effect_renderer import EffectRenderer
    from desktop_pet.pet_window import PetWindow
    from desktop_pet.speech_bubble import SpeechBubble
    from desktop_pet.heard_text import HeardText
    from desktop_pet.countdown_timer import CountdownTimer

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Create the PetController (acts as both RobotController and Qt signal bridge)
    pet_controller = PetController()

    # Load sprites — derive pony_dir from preset slug
    from llm.prompt import get_character_name
    pony_dir = ponies_root / slug_to_dir_name(config.llm.preset)
    sprite_manager = SpriteManager(pony_dir, scale=config.desktop_pet.scale)

    # Parse behaviors
    behavior_manager = BehaviorManager(pony_dir / "pony.ini")
    behavior_manager.parse()

    # Build dynamic sprite map and preload
    sprite_manager.build_sprite_map(behavior_manager)
    sprite_manager.preload_all()

    # Effect renderer
    effect_renderer = EffectRenderer(sprite_manager, behavior_manager)

    # Create pet window
    pet_window = PetWindow(sprite_manager, behavior_manager, effect_renderer)

    # Wire up pony locator for screen capture (captures the correct monitor)
    if screen:
        screen.set_pony_locator(lambda: (pet_window.x() + pet_window.width() // 2,
                                          pet_window.y() + pet_window.height() // 2))

    # Desktop controller (needs pet_window HWND to avoid self-targeting)
    desktop_controller = None
    if config.desktop_control.enabled:
        try:
            pet_hwnd = int(pet_window.winId())
            desktop_controller = DesktopController(config.desktop_control, pet_hwnd=pet_hwnd)
        except ImportError as exc:
            logger.warning("Desktop control disabled: %s", exc)
            desktop_controller = None

    # Scan installed apps/games in background
    if desktop_controller:
        threading.Thread(
            target=desktop_controller.scan_installed_apps,
            daemon=True, name="app-scanner",
        ).start()

    # Screen monitor (free local window tracking)
    screen_monitor = None
    if config.agent.enabled:
        try:
            screen_monitor = ScreenMonitor(pet_hwnd=int(pet_window.winId()), poll_interval=3.0)
        except Exception as exc:
            logger.warning("Screen monitor disabled: %s", exc)
            screen_monitor = None

    # Shared event timeline (bridges Pipeline ↔ AgentLoop context)
    from core.event_timeline import EventTimeline
    timeline = EventTimeline()

    # Agent loop (autonomous brain)
    agent_loop = None
    if config.agent.enabled and screen_monitor:
        agent_loop = AgentLoop(
            config=config.agent,
            screen_monitor=screen_monitor,
            llm=llm_provider,
            tts=tts,
            desktop_controller=desktop_controller,
            robot=pet_controller,
            detector=detector,
            on_speech_text=pet_controller.on_speech_text,
            on_state_change=pet_controller.on_state_change,
            screen_capture=screen,
            transcriber=transcriber,
            tts_config=config.tts,
            moondream=moondream,
            vision_config=config.vision,
            on_grab_cursor=None,  # wired after _on_grab_cursor is defined
            vision_llm=vision_llm,
            timeline=timeline,
        )

    # ── Multi-pony: wrap primary as PonyInstance + create manager ──────────
    from core.pony_instance import PonyInstance, _get_keywords_for
    from core.pony_manager import PonyManager
    from llm.prompt import PromptConfig, get_system_prompt_for

    primary_prompt_config = PromptConfig(
        preset=config.llm.preset,
        relationship_mode=config.llm.relationship,
        relationship_custom=config.llm.relationship_custom,
    )
    # Wire the primary LLM provider to use per-pony prompt (backward-compat:
    # when no companions, get_system_prompt_for produces the same output)
    llm_provider.system_prompt_fn = lambda: get_system_prompt_for(primary_prompt_config)
    llm_provider.character_name = get_character_name()

    primary_pony = PonyInstance(
        slug=config.llm.preset,
        display_name=get_character_name(),
        is_primary=True,
        prompt_config=primary_prompt_config,
        llm=llm_provider,
        pet_window=pet_window,
        pet_controller=pet_controller,
        speech_bubble=None,    # wired later after SpeechBubble creation
        heard_text=None,       # wired later
        sprite_manager=sprite_manager,
        behavior_manager=behavior_manager,
        effect_renderer=effect_renderer,
        pony_dir=pony_dir,
    )
    primary_pony.agent_loop = agent_loop

    mp_cfg = config.multi_pony
    pony_manager = PonyManager(
        config=config,
        ponies_root=ponies_root,
        tts_queue=tts_queue,
        max_ponies=mp_cfg.max_ponies,
        chat_interval_s=mp_cfg.chat_interval_s,
        max_chat_depth=mp_cfg.max_chat_depth,
        piggyback_chance=mp_cfg.piggyback_chance,
    )
    pony_manager.register_primary(primary_pony)
    if screen_monitor:
        pony_manager._screen_monitor = screen_monitor

    # Compact user profile and prune stale events on startup
    try:
        from core.user_profile import prune_events, compact_profile
        compact_profile(llm_provider)
        prune_events(llm_provider)
    except Exception as exc:
        logger.debug("Profile maintenance skipped: %s", exc)

    # Build pipeline with PetController as the robot
    pipeline = Pipeline(
        config=config,
        detector=detector,
        ack_player=ack_player,
        transcriber=transcriber,
        llm_provider=llm_provider,
        tts=tts,
        robot=pet_controller,
        camera=camera,
        screen=screen,
        desktop_controller=desktop_controller,
        agent_loop=agent_loop,
        screen_monitor=screen_monitor,
        moondream=moondream,
        vision_llm=vision_llm,
        timeline=timeline,
    )
    # Wire multi-pony resources into pipeline and agent loop
    pipeline.tts_queue = tts_queue
    pipeline.primary_voice_slug = config.llm.preset
    pipeline.pony_manager = pony_manager
    if agent_loop:
        agent_loop._tts_queue = tts_queue
        agent_loop._primary_voice_slug = config.llm.preset

    # ── Context menu (right-click settings UI) ──────────────────────────────
    from desktop_pet.context_menu import ContextMenuBuilder, _save_yaml_value

    def _on_scale_change(new_scale: float) -> None:
        nonlocal sprite_manager, effect_renderer
        new_sm = SpriteManager(pony_dir, scale=new_scale)
        new_sm.build_sprite_map(behavior_manager)
        new_sm.preload_all()
        sprite_manager = new_sm
        pet_window.sprite_manager = new_sm
        effect_renderer = EffectRenderer(new_sm, behavior_manager)
        pet_window.effect_renderer = effect_renderer
        pet_window._pick_and_start_behavior()

    def _on_character_change(preset_name: str) -> None:
        nonlocal pony_dir, sprite_manager, behavior_manager, effect_renderer
        # 1. Switch LLM persona
        set_preset(preset_name)
        config.llm.preset = preset_name
        _save_yaml_value("llm.preset", preset_name, str(Path(args.config)))

        # 2. Derive new pony directory
        char_name = slug_to_dir_name(preset_name)
        pony_dir = ponies_root / char_name

        # 3. Rebuild behaviors
        behavior_manager = BehaviorManager(pony_dir / "pony.ini")
        behavior_manager.parse()

        # 4. Rebuild sprites
        new_sm = SpriteManager(pony_dir, scale=config.desktop_pet.scale)
        new_sm.build_sprite_map(behavior_manager)
        new_sm.preload_all()
        sprite_manager = new_sm

        # 5. Rebuild effects
        effect_renderer = EffectRenderer(new_sm, behavior_manager)

        # 6. Swap into pet window
        pet_window.sprite_manager = new_sm
        pet_window.behavior_manager = behavior_manager
        pet_window.effect_renderer = effect_renderer
        pet_window._pick_and_start_behavior()

        # 7. Fresh conversation for new character
        llm_provider.reset_history()

        # 8. Swap wake phrases for the new character
        new_phrases = get_phrases_for(preset_name, config.wake_word.phrases)
        detector.set_wake_phrases(new_phrases)

        # 9. Swap acknowledgement sounds for the new character
        ack_player.set_character(preset_name)

        # 10. Switch PVT voice if using OpenAI-compatible TTS
        if hasattr(tts, "set_character"):
            tts.set_character(preset_name)

        # 11. Update primary PonyInstance to reflect new character
        from core.character_registry import get_display_name
        primary_pony.slug = preset_name
        primary_pony.display_name = get_display_name(preset_name)
        primary_pony.pony_dir = pony_dir
        primary_pony.sprite_manager = new_sm
        primary_pony.behavior_manager = behavior_manager
        primary_pony.effect_renderer = effect_renderer
        primary_pony.name_keywords = _get_keywords_for(preset_name)
        primary_pony.prompt_config.preset = preset_name
        primary_pony.prompt_config.relationship_mode = config.llm.relationship
        primary_pony.prompt_config.relationship_custom = config.llm.relationship_custom
        llm_provider.character_name = get_display_name(preset_name)
        try:
            from tts.openai_compatible_tts import has_pvt_voice
            primary_pony.has_voice = has_pvt_voice(preset_name)
        except Exception:
            primary_pony.has_voice = True
        # Update voice slugs for pipeline/agent loop
        pipeline.primary_voice_slug = preset_name
        if agent_loop:
            agent_loop._primary_voice_slug = preset_name
        # Refresh companion lists so other ponies know the new name
        pony_manager._refresh_all_companions()

        logger.info("Character switched to: %s", char_name)

    def _on_provider_change(provider_name: str) -> None:
        nonlocal llm_provider
        from llm.factory import get_provider
        new_provider = get_provider(config)
        # Carry over the per-pony system prompt function
        new_provider.system_prompt_fn = lambda: get_system_prompt_for(primary_prompt_config)
        new_provider.character_name = primary_pony.display_name
        llm_provider = new_provider
        pipeline.llm = new_provider
        if agent_loop:
            agent_loop._llm = new_provider
        menu_builder.llm = new_provider
        primary_pony.llm = new_provider
        # Clear cached model list
        menu_builder._model_choices_cache = None
        # Reset conversation history for fresh start
        new_provider.reset_history()
        # Update secondary ponies too
        for pony in pony_manager.ponies:
            if not pony.is_primary:
                try:
                    secondary_provider = get_provider(config)
                    secondary_provider.character_name = pony.display_name
                    pony.llm = secondary_provider
                except Exception as exc:
                    logger.warning("Failed to update secondary pony %s provider: %s",
                                   pony.display_name, exc)
        logger.info("LLM provider hot-swapped to: %s", provider_name)

    def _on_vision_llm_change() -> None:
        nonlocal vision_llm
        vlm_cfg = config.vision_llm
        if vlm_cfg and vlm_cfg.enabled and vlm_cfg.api_keys:
            try:
                from llm.vision_provider import VisionProvider
                from llm.factory import _KNOWN_BASE_URLS
                base_url = vlm_cfg.base_url or _KNOWN_BASE_URLS.get(vlm_cfg.provider.lower())
                if base_url and "generativelanguage.googleapis.com" in base_url and not base_url.rstrip("/").endswith("/openai"):
                    base_url = base_url.rstrip("/") + "/openai"
                vision_llm = VisionProvider(
                    api_keys=vlm_cfg.api_keys,
                    model=vlm_cfg.model,
                    base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
                    max_requests_per_key_per_day=vlm_cfg.max_requests_per_key_per_day,
                    temperature=vlm_cfg.temperature,
                    max_tokens=vlm_cfg.max_tokens,
                    locate_max_tokens=vlm_cfg.locate_max_tokens,
                )
                logger.info("Vision LLM reloaded: %s (%d keys)", vlm_cfg.model, len(vlm_cfg.api_keys))
            except Exception as exc:
                logger.warning("Failed to reload vision LLM: %s", exc)
                vision_llm = None
        else:
            vision_llm = None
            logger.info("Vision LLM disabled")
        # Update references
        pipeline.vision_llm = vision_llm
        if agent_loop:
            agent_loop._vision_llm = vision_llm
        menu_builder.vision_llm = vision_llm

    menu_builder = ContextMenuBuilder(
        config=config,
        config_path=str(Path(args.config)),
        agent_loop=agent_loop,
        llm_provider=llm_provider,
        on_scale_change=_on_scale_change,
        on_character_change=_on_character_change,
        ack_player=ack_player,
        on_provider_change=_on_provider_change,
        tts=tts,
        vision_llm=vision_llm,
        on_vision_llm_change=_on_vision_llm_change,
        pony_manager=pony_manager,
        pony_instance=primary_pony,
    )
    pet_window.set_menu_builder(menu_builder)

    # Factory for creating context menus on secondary pony windows
    def _make_secondary_menu(instance):
        from desktop_pet.context_menu import ContextMenuBuilder as _CMB
        return _CMB(
            config=config,
            config_path=str(Path(args.config)),
            pony_manager=pony_manager,
            pony_instance=instance,
        )
    pony_manager._menu_builder_factory = _make_secondary_menu

    # Wire pipeline callbacks to PetController methods
    pipeline.set_callbacks(
        on_state_change=pet_controller.on_state_change,
        on_speech_text=pet_controller.on_speech_text,
        on_heard_text=pet_controller.on_heard_text,
        on_conversation_start=pet_controller.on_conversation_start,
        on_conversation_end=pet_controller.on_conversation_end,
    )

    # When recording finishes, immediately show THINK state so mic icon
    # goes away before Whisper transcription runs (which can take seconds)
    transcriber.on_recording_done = lambda: pet_controller.on_state_change("THINK")

    # Create speech bubble
    speech_bubble = SpeechBubble()
    speech_bubble.set_anchor_widget(pet_window)

    heard_text = HeardText()
    heard_text.set_anchor_widget(pet_window)

    # Wire into primary PonyInstance (created before speech_bubble existed)
    primary_pony.speech_bubble = speech_bubble
    primary_pony.heard_text = heard_text

    countdown = CountdownTimer()
    countdown.set_anchor_widget(pet_window)

    # ── Connect signals → slots ──────────────────────────────────────────────

    def _on_state_changed(state_name: str) -> None:
        anim = PetController.get_animation_for_state(state_name)
        if anim is None:
            pet_window.clear_override()
        else:
            pet_window.set_override_animation(anim)
        # Show/hide mic indicator
        pet_window.set_listening(state_name == "LISTEN")
        # Show thinking bubble while LLM is processing — on the correct pony (Fix 11)
        active_bubble = pipeline.active_speech_bubble or speech_bubble
        if state_name == "THINK" and config.desktop_pet.speech_bubble:
            # Show on whatever pony is currently responding
            try:
                target_pony = pipeline._active_responder
                if target_pony and hasattr(target_pony, "pet_window"):
                    ax, ay, ah = target_pony.pet_window.get_anchor_point()
                    active_bubble.show_thinking(ax, ay)
                else:
                    ax, ay, ah = pet_window.get_anchor_point()
                    speech_bubble.show_thinking(ax, ay)
            except Exception:
                ax, ay, ah = pet_window.get_anchor_point()
                speech_bubble.show_thinking(ax, ay)
        elif state_name == "IDLE" and config.desktop_pet.speech_bubble:
            # Only clear thinking bubble — speech bubbles have their own auto-hide timer
            if speech_bubble._thinking:
                speech_bubble.hide_bubble()
            # Also clear on non-primary ponies if they have a bubble
            try:
                for p in pipeline.pony_manager.ponies if pipeline.pony_manager else []:
                    if not p.is_primary and hasattr(p, "speech_bubble") and p.speech_bubble:
                        if getattr(p.speech_bubble, "_thinking", False):
                            p.speech_bubble.hide_bubble()
            except Exception:
                pass

    def _on_heard_text(text: str) -> None:
        """Show what the STT heard below the pony."""
        heard_text.show_heard(text)

    def _on_speech_text(text: str) -> None:
        heard_text.hide_heard()  # Replace with speech bubble
        if config.desktop_pet.speech_bubble:
            ax, ay, ah = pet_window.get_anchor_point()
            speech_bubble.show_text(text, ax, ay, sprite_h=ah)

    def _on_conversation_started() -> None:
        pet_window.pause_roaming()

    def _on_conversation_ended() -> None:
        pet_window.set_listening(False)  # Safety net — always clear mic indicator
        pet_window.clear_override()
        pet_window.resume_roaming()
        # Don't hide the speech bubble here — let SpeechBubble's own auto-hide
        # timer handle it naturally (same as directive bubbles). Hiding it here
        # causes the bubble to vanish before the user can read it, especially
        # when TTS is disabled/fails or for typed conversations with no follow-up loop.
        heard_text.hide_heard()

    def _on_action_triggered(action_name: str) -> None:
        anim = PetController.get_animation_for_action(action_name)
        if anim:
            pet_window.set_override_animation(anim)

    def _on_timed_override(anim_name: str, seconds: int) -> None:
        pet_window.set_timed_override(anim_name, seconds)

    def _on_move_to(region: str) -> None:
        pet_window.move_to_region(region)

    # ── Cursor grab (pony grabs the mouse and runs with it) ────────────

    def _on_grab_cursor(duration: float) -> None:
        """Pony grabs the cursor and runs around with it for `duration` seconds.
        Called from pipeline thread — blocks like mess_with_mouse does.
        Uses signals for animation start/stop (Qt thread safety)."""
        import ctypes
        import time as _time

        # Signal main thread to start grab-run animation
        pet_controller.grab_run_start.emit()
        _time.sleep(0.1)  # let the animation start

        end_time = _time.monotonic() + duration
        while _time.monotonic() < end_time:
            try:
                mx, my = pet_window.get_mouth_position()
                ctypes.windll.user32.SetCursorPos(mx, my)
            except Exception:
                pass
            _time.sleep(0.016)  # ~60fps

        # Signal main thread to stop grab-run animation
        pet_controller.grab_run_stop.emit()

    # Use QueuedConnection for ALL signals — they're emitted from the pipeline
    # thread but the slots manipulate Qt widgets which must run on the main thread.
    pet_controller.state_changed.connect(_on_state_changed, Qt.QueuedConnection)
    # BlockingQueuedConnection: pipeline thread blocks until main thread shows the bubble,
    # guaranteeing the bubble is visible BEFORE audio playback starts.
    # BlockingQueuedConnection so the heard text is guaranteed to be shown
    # before the pipeline continues (prevents LLM fast-path from overriding it)
    pet_controller.heard_text.connect(_on_heard_text, Qt.BlockingQueuedConnection)
    pet_controller.speech_text.connect(_on_speech_text, Qt.BlockingQueuedConnection)
    pet_controller.conversation_started.connect(_on_conversation_started, Qt.QueuedConnection)
    pet_controller.conversation_ended.connect(_on_conversation_ended, Qt.QueuedConnection)
    pet_controller.action_triggered.connect(_on_action_triggered, Qt.QueuedConnection)
    pet_controller.trick_requested.connect(pet_window.do_trick, Qt.QueuedConnection)
    pet_controller.timed_override.connect(_on_timed_override, Qt.QueuedConnection)
    pet_controller.move_to.connect(_on_move_to, Qt.QueuedConnection)
    pet_controller.grab_run_start.connect(pet_window.start_grab_run, Qt.QueuedConnection)
    pet_controller.grab_run_stop.connect(pet_window.stop_grab_run, Qt.QueuedConnection)
    pet_controller.countdown_start.connect(countdown.start_countdown, Qt.QueuedConnection)
    pet_controller.countdown_stop.connect(countdown.stop_countdown, Qt.QueuedConnection)

    # Wire cursor grab callback to agent loop (defined after both exist)
    if agent_loop:
        agent_loop._on_grab_cursor = _on_grab_cursor

    # ── Double-click activation ──────────────────────────────────────────────

    activation_event = threading.Event()
    _pending_text_message: list[str] = []  # thread-safe via GIL; checked in pipeline loop

    def _on_conversation_requested() -> None:
        activation_event.set()

    def _on_text_message(text: str) -> None:
        _pending_text_message.append(text)
        activation_event.set()  # wake the pipeline loop

    pet_window.conversation_requested.connect(_on_conversation_requested)
    pet_window.text_message.connect(_on_text_message)
    pet_window.listen_interrupted.connect(transcriber.interrupt_listening)

    # ── Push-to-talk (PTT) ───────────────────────────────────────────────────
    # Uses pynput (no admin needed) to listen for key press/release.
    # Records immediately on key press (own thread) so no speech is lost.

    _ptt_stop = threading.Event()        # set on key release to stop recording
    _ptt_recording = False               # guard against key-repeat
    _ptt_result_text = None  # completed transcription (str or None)
    _ptt_result_ready = threading.Event()   # signals pipeline that text is ready
    _ptt_last_press: float = 0.0         # monotonic time of last PTT press (suppress wake words)

    ptt_key = config.audio.ptt_key or "f6"
    _ptt_available = False
    try:
        from pynput import keyboard as _pynput_kb

        # Map config key name to pynput Key enum
        _ptt_key_obj = None
        try:
            # Try function keys first (f1-f12)
            _ptt_key_obj = getattr(_pynput_kb.Key, ptt_key.lower(), None)
        except Exception:
            pass
        if _ptt_key_obj is None:
            # Try as a character key
            try:
                _ptt_key_obj = _pynput_kb.KeyCode.from_char(ptt_key.lower())
            except Exception:
                _ptt_key_obj = _pynput_kb.Key.f6  # fallback

        def _ptt_record_thread():
            """Record in background, transcribe when done."""
            nonlocal _ptt_result_text
            try:
                # detector already paused by _ptt_on_press (immediate, before thread spawn)
                text = transcriber.listen_ptt(_ptt_stop)
                _ptt_result_text = text if text and text.strip() else None
            except Exception as exc:
                logger.error("PTT recording failed: %s", exc)
                print(f"[PTT] Recording error: {exc}", flush=True)
                _ptt_result_text = None
            finally:
                _ptt_result_ready.set()
                activation_event.set()  # wake the pipeline loop

        def _ptt_on_press(key):
            nonlocal _ptt_recording, _ptt_last_press
            try:
                if key != _ptt_key_obj:
                    return
            except Exception:
                return
            if _ptt_recording:
                return  # already recording (key repeat)
            _ptt_recording = True
            _ptt_last_press = time.monotonic()
            _ptt_stop.clear()
            _ptt_result_ready.clear()
            # Immediately stop any TTS playback so the user isn't talking
            # over the pony.  This also flushes pending queue items.
            if tts_queue:
                tts_queue.interrupt()
            # Hide speech bubble so it's clear the pony stopped talking
            try:
                speech_bubble.hide_bubble()
            except Exception:
                pass
            # Suppress autonomous speech immediately — don't let the agent
            # interject while Whisper is still transcribing what the user said
            if agent_loop:
                agent_loop.set_conversation_active(True)
            # Pause wake word detector IMMEDIATELY so it doesn't also fire on
            # "hey dash" spoken while PTT is held (race condition: wake word
            # would detect before the recording thread could pause it)
            detector.pause()
            # Interrupt any active agent-initiated listen() so PTT can take
            # the mic.  listen_ptt() also calls interrupt_listening() and
            # acquires the listening lock, but doing it here too means the
            # agent listen starts releasing the mic immediately — before the
            # PTT thread even spawns.
            transcriber.interrupt_listening()
            print(f"[PTT] Recording... (release {ptt_key} to send)", flush=True)
            t = threading.Thread(target=_ptt_record_thread, daemon=True, name="ptt-recorder")
            t.start()

        def _ptt_on_release(key):
            nonlocal _ptt_recording
            try:
                if key != _ptt_key_obj:
                    return
            except Exception:
                return
            if not _ptt_recording:
                return
            _ptt_recording = False
            _ptt_stop.set()
            print("[PTT] Key released — transcribing...", flush=True)

        _ptt_listener = _pynput_kb.Listener(on_press=_ptt_on_press, on_release=_ptt_on_release)
        _ptt_listener.daemon = True
        _ptt_listener.start()
        _ptt_available = True
        print(f"[PTT] Push-to-talk enabled: hold '{ptt_key}' to talk.", flush=True)
        logger.info("Push-to-talk enabled (pynput): hold '%s' to talk.", ptt_key)
    except ImportError:
        print("[PTT] Disabled — install 'pynput' package (pip install pynput).", flush=True)
        logger.info("Push-to-talk disabled — install 'pynput' package to enable.")
    except Exception as exc:
        print(f"[PTT] FAILED to set up: {exc}", flush=True)
        logger.warning("Push-to-talk setup failed: %s", exc)

    # ── Pipeline thread ──────────────────────────────────────────────────────

    _shutdown_requested = threading.Event()
    random_chance = config.conversation.random_speech_chance

    def _pipeline_loop() -> None:
        detector.start()
        if screen_monitor:
            screen_monitor.start()
        logger.info(
            "Listening for wake words: %s",
            ", ".join(detector.wake_phrases),
        )

        try:
            while not _shutdown_requested.is_set():
                # Poll wake word with 1s timeout
                keyword_index = detector.wait_for_wake_word(timeout=1.0)

                if _shutdown_requested.is_set():
                    break

                # Check for double-click activation, typed message, or PTT
                if keyword_index is None and activation_event.is_set():
                    activation_event.clear()
                    # Check if there's a typed message waiting
                    if _pending_text_message:
                        typed = _pending_text_message.pop(0)
                        detector.pause()
                        try:
                            pipeline.run_text_conversation(typed)
                        finally:
                            if not _shutdown_requested.is_set():
                                detector.resume()
                        continue
                    # Push-to-talk: recording already happened on its own thread
                    # (detector was paused by the recording thread)
                    if _ptt_result_ready.is_set():
                        nonlocal _ptt_result_text
                        _ptt_result_ready.clear()
                        ptt_text = _ptt_result_text
                        _ptt_result_text = None  # consume — prevent stale replays
                        if ptt_text and ptt_text.strip():
                            print(f"[PTT] Heard: \"{ptt_text}\"", flush=True)
                            logger.info("PTT heard: %r", ptt_text)
                            try:
                                pipeline.run_conversation_with_text(ptt_text)
                            finally:
                                if not _shutdown_requested.is_set():
                                    detector.resume()
                        else:
                            print("[PTT] No speech detected.", flush=True)
                            # PTT captured nothing — re-enable autonomous behavior
                            if agent_loop:
                                agent_loop.set_conversation_active(False)
                            if not _shutdown_requested.is_set():
                                detector.resume()
                        continue
                    # No PTT and no text message — this was a double-click or
                    # other activation. Only open mic if PTT isn't mid-recording.
                    if _ptt_recording:
                        continue  # PTT in progress, don't open a second mic
                    keyword_index = 0  # Treat as voice conversation trigger

                if keyword_index is None:
                    # Agent loop handles all autonomous behavior (directives, spontaneous speech, screen monitoring)
                    if agent_loop:
                        try:
                            agent_loop.tick()
                        except Exception as exc:
                            logger.debug("Agent tick error: %s", exc)
                    # Inter-pony behavior (runs alongside agent loop)
                    if mp_cfg.inter_pony_chat and len(pony_manager.ponies) > 1:
                        try:
                            # Individual pony remarks (each pony independently)
                            pony_manager.maybe_individual_speech()
                            # Coordinated group conversations (less frequent)
                            pony_manager.maybe_spontaneous_chat()
                        except Exception as exc:
                            logger.debug("Inter-pony chat error: %s", exc)
                    if not agent_loop:
                        # Fallback: old random roll when agent is disabled
                        if random.random() < random_chance:
                            logger.debug("Random speech triggered.")
                            detector.pause()
                            try:
                                pipeline.speak_spontaneously()
                            finally:
                                if not _shutdown_requested.is_set():
                                    detector.resume()
                    continue

                # Suppress wake words: disabled in config, or within 10s of PTT press
                if keyword_index is not None:
                    if not config.wake_word.enabled:
                        continue
                    if (time.monotonic() - _ptt_last_press) < 10.0:
                        continue

                # Wake word or double-click — run full conversation
                detector.pause()

                # Voice verification on wake word — reject if not the user's voice
                if keyword_index is not None and not activation_event.is_set():
                    try:
                        vf = pipeline.transcriber.voice_filter
                        wake_audio = detector.get_wake_audio()
                        if vf and vf.enrolled and wake_audio is not None and not vf.is_user(wake_audio):
                            logger.info("Wake word rejected by voice filter — not the user's voice.")
                            if not _shutdown_requested.is_set():
                                detector.resume()
                            continue
                    except Exception as exc:
                        logger.debug("Wake word voice verification skipped: %s", exc)

                try:
                    pipeline.run_conversation()
                finally:
                    if not _shutdown_requested.is_set():
                        detector.resume()

        except Exception as exc:
            logger.exception("Pipeline thread error: %s", exc)
        finally:
            logger.info("Pipeline thread exiting.")

    pipeline_thread = threading.Thread(target=_pipeline_loop, daemon=True, name="pipeline")

    # ── Graceful shutdown ────────────────────────────────────────────────────

    def _shutdown(*_args) -> None:
        logger.info("Shutdown signal received — cleaning up...")
        _shutdown_requested.set()
        pony_manager._shutting_down = True
        # Stop TTS queue first to avoid audio playing during teardown
        tts_queue.stop()
        # Remove secondary ponies
        for pony in list(pony_manager.ponies):
            if not pony.is_primary:
                pony_manager.remove_pony(pony)
        if screen_monitor:
            screen_monitor.stop()
        detector.stop()
        pipeline.summarize_session()
        pipeline._extract_user_profile(force=True)
        pet_controller.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    menu_builder.on_quit = lambda: _shutdown()

    # ── Auto-load secondary ponies from config ─────────────────────────────
    for slug in mp_cfg.secondary_ponies:
        try:
            pony_manager.add_pony(slug)
        except Exception as exc:
            logger.warning("Failed to auto-load secondary pony %s: %s", slug, exc)

    # ── Launch ───────────────────────────────────────────────────────────────

    pet_window.show()
    pipeline_thread.start()

    pony_count = len(pony_manager.ponies)
    pony_info = f" ({pony_count} ponies)" if pony_count > 1 else ""
    print(
        f"\n{get_character_name()} Desktop Pet is running!{pony_info}\n"
        f"  Wake phrases: {', '.join(detector.wake_phrases)}\n"
        f"  Double-click the pet to start a conversation.\n"
        f"  Right-click for menu. Close to exit.\n"
    )

    exit_code = app.exec()

    # Cleanup after Qt event loop exits (may already be done by _shutdown)
    _shutdown_requested.set()
    try:
        if screen_monitor:
            screen_monitor.stop()
        detector.stop()
    except Exception as exc:
        logger.debug("Post-loop cleanup error (non-fatal): %s", exc)
    logger.info("Desktop Pony signing off. Catch ya later!")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
