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

import argparse
import logging
import random
import signal
import sys
import threading
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
    from llm.prompt import set_preset
    set_preset(config.llm.preset)
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

    # Screen monitor (free local window tracking)
    screen_monitor = None
    if config.agent.enabled:
        try:
            screen_monitor = ScreenMonitor(pet_hwnd=int(pet_window.winId()), poll_interval=3.0)
        except Exception as exc:
            logger.warning("Screen monitor disabled: %s", exc)
            screen_monitor = None

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
        )

    # Prune stale user profile events on startup
    try:
        from core.user_profile import prune_events
        prune_events(llm_provider)
    except Exception as exc:
        logger.debug("Event pruning skipped: %s", exc)

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
    )

    # ── Context menu (right-click settings UI) ──────────────────────────────
    from desktop_pet.context_menu import ContextMenuBuilder, _save_yaml_value

    def _on_scale_change(new_scale: float) -> None:
        nonlocal effect_renderer
        new_sm = SpriteManager(pony_dir, scale=new_scale)
        new_sm.build_sprite_map(behavior_manager)
        new_sm.preload_all()
        pet_window.sprite_manager = new_sm
        effect_renderer = EffectRenderer(new_sm, behavior_manager)
        pet_window.effect_renderer = effect_renderer
        pet_window._pick_and_start_behavior()

    def _on_character_change(preset_name: str) -> None:
        nonlocal pony_dir, behavior_manager, effect_renderer
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

        logger.info("Character switched to: %s", char_name)

    def _on_provider_change(provider_name: str) -> None:
        nonlocal llm_provider
        from llm.factory import get_provider
        new_provider = get_provider(config)
        llm_provider = new_provider
        pipeline.llm = new_provider
        if agent_loop:
            agent_loop._llm = new_provider
        menu_builder.llm = new_provider
        # Clear cached model list
        menu_builder._model_choices_cache = None
        # Reset conversation history for fresh start
        new_provider.reset_history()
        logger.info("LLM provider hot-swapped to: %s", provider_name)

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
    )
    pet_window.set_menu_builder(menu_builder)

    # Wire pipeline callbacks to PetController methods
    pipeline.set_callbacks(
        on_state_change=pet_controller.on_state_change,
        on_speech_text=pet_controller.on_speech_text,
        on_conversation_start=pet_controller.on_conversation_start,
        on_conversation_end=pet_controller.on_conversation_end,
    )

    # Create speech bubble
    speech_bubble = SpeechBubble()

    # ── Connect signals → slots ──────────────────────────────────────────────

    def _on_state_changed(state_name: str) -> None:
        anim = PetController.get_animation_for_state(state_name)
        if anim is None:
            pet_window.clear_override()
        else:
            pet_window.set_override_animation(anim)
        # Show/hide mic indicator
        pet_window.set_listening(state_name == "LISTEN")
        # Show thinking bubble while LLM is processing
        if state_name == "THINK" and config.desktop_pet.speech_bubble:
            ax, ay, ah = pet_window.get_anchor_point()
            speech_bubble.show_thinking(ax, ay)

    def _on_speech_text(text: str) -> None:
        if config.desktop_pet.speech_bubble:
            ax, ay, ah = pet_window.get_anchor_point()
            speech_bubble.show_text(text, ax, ay, sprite_h=ah)

    def _on_conversation_started() -> None:
        pet_window.pause_roaming()

    def _on_conversation_ended() -> None:
        pet_window.set_listening(False)  # Safety net — always clear mic indicator
        pet_window.clear_override()
        pet_window.resume_roaming()
        speech_bubble.hide_bubble()

    def _on_action_triggered(action_name: str) -> None:
        anim = PetController.get_animation_for_action(action_name)
        if anim:
            pet_window.set_override_animation(anim)

    def _on_timed_override(anim_name: str, seconds: int) -> None:
        pet_window.set_timed_override(anim_name, seconds)

    def _on_move_to(region: str) -> None:
        pet_window.move_to_region(region)

    # Use QueuedConnection for ALL signals — they're emitted from the pipeline
    # thread but the slots manipulate Qt widgets which must run on the main thread.
    pet_controller.state_changed.connect(_on_state_changed, Qt.QueuedConnection)
    # BlockingQueuedConnection: pipeline thread blocks until main thread shows the bubble,
    # guaranteeing the bubble is visible BEFORE audio playback starts.
    pet_controller.speech_text.connect(_on_speech_text, Qt.BlockingQueuedConnection)
    pet_controller.conversation_started.connect(_on_conversation_started, Qt.QueuedConnection)
    pet_controller.conversation_ended.connect(_on_conversation_ended, Qt.QueuedConnection)
    pet_controller.action_triggered.connect(_on_action_triggered, Qt.QueuedConnection)
    pet_controller.trick_requested.connect(pet_window.do_trick, Qt.QueuedConnection)
    pet_controller.timed_override.connect(_on_timed_override, Qt.QueuedConnection)
    pet_controller.move_to.connect(_on_move_to, Qt.QueuedConnection)

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

                # Check for double-click activation or typed message
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
                    keyword_index = 0  # Treat as voice conversation trigger

                if keyword_index is None:
                    # Agent loop handles all autonomous behavior (directives, spontaneous speech, screen monitoring)
                    if agent_loop:
                        try:
                            agent_loop.tick()
                        except Exception as exc:
                            logger.debug("Agent tick error: %s", exc)
                    else:
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

    # ── Launch ───────────────────────────────────────────────────────────────

    pet_window.show()
    pipeline_thread.start()

    print(
        f"\n{get_character_name()} Desktop Pet is running!\n"
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
        pipeline.summarize_session()
        pipeline._extract_user_profile(force=True)
    except Exception as exc:
        logger.debug("Post-loop cleanup error (non-fatal): %s", exc)
    logger.info("Desktop Pony signing off. Catch ya later!")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
