"""
PonyManager — coordinator for the multi-pony system.

Manages adding/removing ponies, speech routing, and inter-pony chat
triggers.  Holds the shared resources (detector, transcriber, TTS queue)
and the list of active PonyInstances.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.pony_instance import PonyInstance
    from core.tts_queue import TTSQueue

logger = logging.getLogger(__name__)


class PonyManager:
    """Singleton coordinator for all active ponies."""

    def __init__(
        self,
        config: Any,
        ponies_root: Path,
        tts_queue: "TTSQueue",
        max_ponies: int = 3,
        chat_interval_s: float = 600.0,
        max_chat_depth: int = 6,
        piggyback_chance: float = 0.30,
    ) -> None:
        self.config = config
        self.ponies_root = ponies_root
        self.tts_queue = tts_queue
        self.max_ponies = max_ponies
        self.chat_interval_s = chat_interval_s
        self.max_chat_depth = max_chat_depth
        self.piggyback_chance = piggyback_chance

        self.ponies: list["PonyInstance"] = []
        self._last_inter_chat: float = time.monotonic()
        self._next_individual_speech: dict[str, float] = {}  # slug -> next eligible time
        self._menu_builder_factory: Any = None  # set by main.py
        self._screen_monitor: Any = None  # set by main.py for HWND exclusion
        self._shutting_down: bool = False
        self._convo_in_progress: bool = False   # True while a group convo thread is running
        self._active_convo: Any = None           # current GroupConversation (for user injection)

    @property
    def primary(self) -> Optional["PonyInstance"]:
        """The first pony (main companion)."""
        return self.ponies[0] if self.ponies else None

    # ── Lifecycle ──────────────────────────────────────────────────

    def register_primary(self, instance: "PonyInstance") -> None:
        """Register the primary pony (already constructed by main.py)."""
        instance.is_primary = True
        if self.ponies:
            self.ponies.insert(0, instance)
        else:
            self.ponies.append(instance)
        self._refresh_all_companions()
        logger.info("Primary pony registered: %s", instance.display_name)

    def add_pony(self, slug: str) -> Optional["PonyInstance"]:
        """Add a secondary pony to the desktop.

        Returns the new PonyInstance, or None if at capacity.
        Skips if the slug is already present (prevents duplicates).
        """
        # Prevent duplicates — check if this slug is already loaded
        for existing in self.ponies:
            if existing.slug == slug:
                logger.info("Pony '%s' already loaded — skipping duplicate.", slug)
                return None

        if len(self.ponies) >= self.max_ponies:
            logger.warning("Max ponies (%d) reached — cannot add %s.", self.max_ponies, slug)
            return None

        from core.pony_instance import PonyInstance

        instance = PonyInstance.create(
            slug=slug,
            is_primary=False,
            config=self.config,
            ponies_root=self.ponies_root,
            app_config=self.config,
        )
        self.ponies.append(instance)
        self._refresh_all_companions()

        # Attach a right-click menu to the secondary pony's window
        if self._menu_builder_factory:
            try:
                menu_builder = self._menu_builder_factory(instance)
                instance.pet_window.set_menu_builder(menu_builder)
            except Exception as exc:
                logger.warning("Failed to attach menu to %s: %s", instance.display_name, exc)

        # Exclude secondary pony window from screen monitor observations
        if self._screen_monitor and instance.pet_window:
            try:
                hwnd = int(instance.pet_window.winId())
                self._screen_monitor.exclude_hwnd(hwnd)
            except Exception:
                pass

        # Show the window
        instance.pet_window.show()

        # Offset position so they don't stack on top of each other
        if self.primary:
            px, py = self.primary.get_window_center()
            offset = 200 * (len(self.ponies) - 1)
            instance.pet_window.move(px + offset, py)

        logger.info("Added pony: %s (total: %d)", instance.display_name, len(self.ponies))
        return instance

    def remove_pony(self, instance: "PonyInstance") -> None:
        """Remove a secondary pony from the desktop."""
        if instance.is_primary:
            logger.warning("Cannot remove primary pony.")
            return
        if instance not in self.ponies:
            return
        # Remove from screen monitor exclusions before destroying
        if self._screen_monitor and instance.pet_window:
            try:
                hwnd = int(instance.pet_window.winId())
                self._screen_monitor.include_hwnd(hwnd)
            except Exception:
                pass
        self.ponies.remove(instance)
        instance.destroy()
        self._refresh_all_companions()
        logger.info("Removed pony: %s (remaining: %d)", instance.display_name, len(self.ponies))

    def _refresh_all_companions(self) -> None:
        """Update every pony's companion list after add/remove."""
        for pony in self.ponies:
            pony.update_companions(self.ponies)

    # ── Speech routing ─────────────────────────────────────────────

    def route_user_speech(self, text: str) -> "PonyInstance":
        """Decide which pony should respond to the user's speech.

        1. Check for character name keywords in the text
        2. If multiple matches for same name (duplicates) → random pick
        3. No name found → closest pony to cursor
        """
        target = self._match_by_name(text)
        if target:
            return target
        return self._closest_to_cursor()

    def _match_by_name(self, text: str) -> Optional["PonyInstance"]:
        """Check transcribed text for character name keywords.

        Uses longest-match-first to avoid "dash" matching when user said "rainbow dash".
        """
        text_lower = text.lower()

        # Build sorted list: (keyword, pony_instance), longest keyword first
        candidates: list[tuple[str, "PonyInstance"]] = []
        for pony in self.ponies:
            for kw in pony.name_keywords:
                candidates.append((kw, pony))
        candidates.sort(key=lambda x: len(x[0]), reverse=True)

        for kw, pony in candidates:
            if kw in text_lower:
                # Check for duplicate ponies with same slug
                same_slug = [p for p in self.ponies if p.slug == pony.slug]
                if len(same_slug) > 1:
                    return random.choice(same_slug)
                return pony
        return None

    def _closest_to_cursor(self) -> "PonyInstance":
        """Return the pony closest to the current cursor position."""
        if len(self.ponies) == 1:
            return self.ponies[0]

        try:
            import ctypes
            from ctypes import wintypes

            pt = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            cx, cy = pt.x, pt.y
        except Exception:
            # Can't get cursor — return primary
            return self.ponies[0]

        best = self.ponies[0]
        best_dist = float("inf")
        for pony in self.ponies:
            px, py = pony.get_window_center()
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best = pony
        return best

    def get_pony_by_slug(self, slug: str) -> Optional["PonyInstance"]:
        """Find a pony by slug. Returns first match."""
        for pony in self.ponies:
            if pony.slug == slug:
                return pony
        return None

    # ── Screen context for group chat ───────────────────────────────

    @staticmethod
    def _summarize_screen_for_chat(fg_exe: str, fg_title: str, open_windows: list) -> str:
        """Summarize screen state into a simple description for group chat.

        Instead of dumping raw window titles (which causes ponies to parrot
        error messages and technical junk), describe WHAT the user is doing
        in plain terms ponies can understand.
        """
        # Map exe names to friendly descriptions
        _EXE_MAP = {
            "chrome": "browsing the web", "firefox": "browsing the web",
            "msedge": "browsing the web", "opera": "browsing the web",
            "brave": "browsing the web",
            "code": "writing code in VS Code", "devenv": "writing code in Visual Studio",
            "pycharm": "writing code in PyCharm", "idea": "writing code",
            "notepad++": "editing a text file", "notepad": "editing a text file",
            "sublime_text": "editing code",
            "discord": "chatting on Discord", "slack": "chatting on Slack",
            "telegram": "chatting on Telegram",
            "spotify": "listening to music", "foobar2000": "listening to music",
            "vlc": "watching something", "mpv": "watching something",
            "explorer": "browsing files",
            "steam": "on Steam", "steamwebhelper": "on Steam",
            "cs2": "playing CS2", "valorant": "playing Valorant",
            "minecraft": "playing Minecraft",
            "photoshop": "editing images", "gimp": "editing images",
            "word": "writing a document", "winword": "writing a document",
            "excel": "working on a spreadsheet",
            "powershell": "using the terminal", "cmd": "using the terminal",
            "windowsterminal": "using the terminal",
        }

        exe_lower = (fg_exe or "").lower().replace(".exe", "")
        activity = _EXE_MAP.get(exe_lower)

        if not activity:
            # Try partial matches
            for key, desc in _EXE_MAP.items():
                if key in exe_lower:
                    activity = desc
                    break

        if not activity:
            activity = f"using {fg_exe}" if fg_exe else "on the computer"

        # For browsers, extract the site/page from the title if possible
        if "browsing" in activity and fg_title:
            # Browser titles usually end with " - Chrome" etc, strip that
            import re
            clean_title = re.sub(r'\s*[-–—]\s*(Google Chrome|Mozilla Firefox|Microsoft Edge|Opera|Brave).*$', '', fg_title, flags=re.IGNORECASE).strip()
            # Extract domain-like words but skip technical/error-looking stuff
            if clean_title and len(clean_title) < 80:
                # Skip titles that look like error messages
                error_words = {"error", "failed", "refused", "timeout", "exception", "crash", "404", "500", "503", "denied"}
                title_lower = clean_title.lower()
                if not any(w in title_lower for w in error_words):
                    activity = f"looking at \"{clean_title}\" in the browser"

        # For code editors, try to get the file name
        if "code" in activity and fg_title:
            import re
            # VS Code titles are like "filename.py - ProjectName - Visual Studio Code"
            file_match = re.match(r'^([^\-–—]+)', fg_title)
            if file_match:
                fname = file_match.group(1).strip()
                if fname and len(fname) < 60 and "." in fname:
                    activity = f"editing {fname} in their code editor"

        # Count what kinds of apps are open (without listing titles)
        app_types = set()
        for w in open_windows:
            w_exe = (getattr(w, 'exe_name', '') or '').lower().replace('.exe', '')
            if any(b in w_exe for b in ("chrome", "firefox", "msedge", "opera", "brave")):
                app_types.add("browser tabs")
            elif any(b in w_exe for b in ("code", "pycharm", "devenv", "sublime")):
                app_types.add("code editor")
            elif any(b in w_exe for b in ("discord", "slack", "telegram")):
                app_types.add("chat apps")
            elif any(b in w_exe for b in ("steam", "cs2", "valorant")):
                app_types.add("games")

        summary = f"The user is currently {activity}."
        if app_types:
            also = ", ".join(sorted(app_types))
            summary += f" They also have {also} open."

        return summary

    # ── Inter-pony chat triggers ────────────────────────────────────

    def maybe_spontaneous_chat(self) -> bool:
        """Check if it's time for spontaneous inter-pony banter.

        Called from the main tick loop.  Returns True if a chat was started.
        Group conversations run in a background thread so the main tick loop
        (PTT, idle checks, etc.) is never blocked.
        """
        if self._shutting_down:
            return False
        if len(self.ponies) < 2:
            return False
        if self._convo_in_progress:
            return False  # don't stack conversations

        elapsed = time.monotonic() - self._last_inter_chat
        if elapsed < self.chat_interval_s:
            return False

        # Random chance — don't fire every time the interval elapses
        if random.random() > 0.3:
            self._last_inter_chat = time.monotonic()  # reset anyway to avoid rapid fire
            return False

        self._last_inter_chat = time.monotonic()

        # Pick a random initiator
        initiator = random.choice(self.ponies)
        logger.info("Spontaneous inter-pony chat triggered by %s", initiator.display_name)

        # Get screen context — summarize what the user is DOING, not raw titles.
        # Raw window titles cause ponies to parrot error messages, status text,
        # and technical garbage they don't understand.
        screen_context = ""
        if self._screen_monitor:
            try:
                state = self._screen_monitor.get_state()
                if state and state.foreground:
                    fg_exe = state.foreground.exe_name or "something"
                    fg_title = state.foreground.title or ""
                    # Summarize into a human-readable description
                    screen_context = self._summarize_screen_for_chat(fg_exe, fg_title, state.open_windows[:8])
            except Exception:
                pass

        from core.group_conversation import GroupConversation
        convo = GroupConversation(self, max_depth=self.max_chat_depth)

        def _run():
            self._convo_in_progress = True
            self._active_convo = convo
            try:
                convo.start(initiator, trigger="spontaneous", screen_context=screen_context)
            except Exception as exc:
                logger.error("Spontaneous chat failed: %s", exc)
            finally:
                self._convo_in_progress = False
                self._active_convo = None

        threading.Thread(target=_run, daemon=True).start()
        return True

    def maybe_individual_speech(self) -> bool:
        """Give each pony an independent chance to say something on their own.

        Unlike ``maybe_spontaneous_chat`` (coordinated group conversation),
        this lets individual ponies speak up randomly — a thought, a remark,
        a comment about what's on screen.  Other ponies get a piggyback chance.

        Called from the main tick loop.  Returns True if anyone spoke.
        """
        if self._shutting_down or len(self.ponies) < 2:
            return False

        now = time.monotonic()
        spoke = False

        for pony in self.ponies:
            if getattr(pony, "_destroyed", False):
                continue
            # Primary pony has its own agent_loop for spontaneous speech — skip
            if pony.is_primary:
                continue

            # Per-pony timer: 3-8 min between individual remarks
            key = pony.slug  # use stable identifier instead of id()
            if key not in self._next_individual_speech:
                self._next_individual_speech[key] = now + random.uniform(180.0, 480.0)
            if now < self._next_individual_speech[key]:
                continue

            # Reset timer regardless of outcome
            self._next_individual_speech[key] = now + random.uniform(180.0, 480.0)

            # Generate individual remark
            text = self._generate_individual_remark(pony)
            if not text:
                continue

            spoke = True
            self._speak_individual(pony, text)

            # Offer piggyback to others — run in background to avoid blocking PTT
            if not self._convo_in_progress:
                _pony_ref = pony
                _text_ref = text
                def _run_piggybacks(speaker=_pony_ref, spoken=_text_ref):
                    self._convo_in_progress = True
                    try:
                        for other in list(self.ponies):
                            if self._convo_in_progress is False:
                                break  # interrupted
                            if other is speaker or getattr(other, "_destroyed", False):
                                continue
                            if random.random() > self.piggyback_chance:
                                continue
                            try:
                                from core.group_conversation import GroupConversation
                                convo = GroupConversation(self, max_depth=2)
                                self._active_convo = convo
                                convo.piggyback(
                                    other,
                                    original_speaker=speaker.display_name,
                                    user_text="",
                                    response_text=spoken,
                                )
                            except Exception as exc:
                                logger.debug("Individual piggyback failed for %s: %s",
                                             other.display_name, exc)
                    finally:
                        self._convo_in_progress = False
                        self._active_convo = None
                threading.Thread(target=_run_piggybacks, daemon=True).start()

            # Only one pony speaks per tick to avoid spam
            break

        return spoke

    def _generate_individual_remark(self, pony: "PonyInstance") -> Optional[str]:
        """Generate a short spontaneous remark for a single pony."""
        from core.group_conversation import GroupConversation

        # Build screen context — summarized, not raw titles
        screen_info = ""
        if self._screen_monitor:
            try:
                state = self._screen_monitor.get_state()
                if state and state.foreground:
                    screen_info = self._summarize_screen_for_chat(
                        state.foreground.exe_name or "",
                        state.foreground.title or "",
                        state.open_windows[:8],
                    ) + " "
            except Exception:
                pass

        companions = [p.display_name for p in self.ponies if p is not pony]
        companion_str = ", ".join(companions) if companions else "the user"

        # Pick a random conversation angle so individual remarks aren't always
        # "oh what are you working on"
        angles = [
            "Say something to one of your friends — tease them, ask them something, or start banter.",
            "Share a random thought or opinion that has nothing to do with the screen.",
            "Comment on something the user is doing, but be specific and interesting about it.",
            "Say something funny or sarcastic.",
            "Complain about something or express a strong opinion.",
            "Bring up a random memory, story, or thing that's been on your mind.",
            "Say something that reveals something about your personality — an interest, a pet peeve, a wish.",
        ]
        angle = random.choice(angles)

        recent_warning = GroupConversation._get_recent_topics_warning()

        prompt = (
            f"(You're on the desktop with {companion_str}. {screen_info}"
            f"{angle} Keep it to one sentence.\n"
            f"RULES: Only reference things you actually know. "
            f"Do NOT invent scenery, events, or errors. "
            f"Do NOT parrot technical info from window titles. "
            f"Do NOT comment on the number of open windows.\n"
            f"{recent_warning}"
            f"Be yourself — not a caricature.\n"
            f"Say [PASS] if you have nothing worth saying right now.\n"
            f"Do NOT include any tags like [CONVO:...] — just speak naturally.)"
        )

        try:
            reply = pony.llm.generate_once(prompt, max_tokens=100)
        except Exception as exc:
            logger.debug("Individual speech failed for %s: %s", pony.display_name, exc)
            return None

        return GroupConversation._clean_reply(reply)

    def _speak_individual(self, pony: "PonyInstance", text: str) -> None:
        """Enqueue individual speech for a pony."""
        from core.tts_queue import PRIORITY_SPONTANEOUS_CHAT

        if getattr(pony, "_destroyed", False):
            return

        def _show_bubble():
            if getattr(pony, "_destroyed", False):
                return
            try:
                pony.pet_controller.speech_text.emit(text)
            except Exception:
                pass

        self.tts_queue.enqueue(
            text,
            priority=PRIORITY_SPONTANEOUS_CHAT,
            voice_slug=pony.slug,
            on_start=_show_bubble,
            skip_tts=not getattr(pony, "has_voice", True),
        )
        logger.info("Individual speech by %s: %r", pony.display_name, text[:60])

    def offer_piggyback(
        self,
        responder: "PonyInstance",
        user_text: str,
        response_text: str,
    ) -> None:
        """After a pony responds to the user, offer other ponies a chance to jump in.

        Runs in a background thread to avoid blocking PTT/main thread.
        """
        if self._shutting_down or len(self.ponies) < 2:
            return
        if self._convo_in_progress:
            return  # already in a conversation

        def _run():
            self._convo_in_progress = True
            try:
                for pony in list(self.ponies):
                    if pony is responder:
                        continue
                    if random.random() > self.piggyback_chance:
                        continue
                    try:
                        from core.group_conversation import GroupConversation
                        convo = GroupConversation(self, max_depth=2)
                        self._active_convo = convo
                        convo.piggyback(
                            pony,
                            original_speaker=responder.display_name,
                            user_text=user_text,
                            response_text=response_text,
                        )
                    except Exception as exc:
                        logger.debug("Piggyback failed for %s: %s", pony.display_name, exc)
            finally:
                self._convo_in_progress = False
                self._active_convo = None

        threading.Thread(target=_run, daemon=True).start()
