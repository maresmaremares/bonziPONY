"""Right-click context menu — full in-app settings UI."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QActionGroup, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from core.config_loader import AppConfig
    from core.agent_loop import AgentLoop
    from llm.base import LLMProvider

logger = logging.getLogger(__name__)


# ── YAML persistence (line-level, preserves comments) ──────────────────────

def _save_yaml_value(key_path: str, value, config_path: str = "config.yaml") -> None:
    """Update a single section.key value in config.yaml preserving comments.

    If the section or key doesn't exist yet, it will be appended so that
    new config keys (like ``tts.provider``) persist across restarts.
    """
    try:
        path = Path(config_path)
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines(True)

        parts = key_path.split(".")
        if len(parts) != 2:
            return
        section, key = parts

        # Format value for YAML
        if value is None:
            yaml_val = "null"
        elif isinstance(value, bool):
            yaml_val = "true" if value else "false"
        elif isinstance(value, str):
            yaml_val = f'"{value}"'
        elif isinstance(value, (int, float)):
            yaml_val = str(value)
        else:
            yaml_val = str(value)

        in_section = False
        section_found = False
        section_end = -1  # line index right after the last line in the section

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Entering target section?
            if not stripped.startswith("#") and (stripped == f"{section}:" or stripped.startswith(f"{section}:")):
                in_section = True
                section_found = True
                continue

            # Left section? (next top-level key)
            if in_section and stripped and not line[0].isspace() and ":" in stripped:
                section_end = i
                in_section = False
                continue

            # Found our key inside the section
            if in_section and stripped.startswith(f"{key}:"):
                indent = len(line) - len(line.lstrip())
                prefix = " " * indent + f"{key}: "

                # Preserve inline comment
                comment_idx = line.find("#")
                if comment_idx > 0 and comment_idx > len(prefix):
                    comment = line[comment_idx:].rstrip("\n")
                    new_line = f"{prefix}{yaml_val}"
                    pad = max(1, comment_idx - len(new_line))
                    lines[i] = new_line + " " * pad + comment + "\n"
                else:
                    lines[i] = f"{prefix}{yaml_val}\n"
                break
        else:
            # Key not found — need to add it
            if section_found:
                # Section exists but key doesn't — insert at section boundary
                insert_pos = section_end if section_end >= 0 else len(lines)
                lines.insert(insert_pos, f"  {key}: {yaml_val}\n")
            else:
                # Section doesn't exist — append section + key
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.append(f"\n{section}:\n  {key}: {yaml_val}\n")

        path.write_text("".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save config %s: %s", key_path, exc)


# ── Audio device enumeration ───────────────────────────────────────────────

def _list_audio_devices() -> List[Tuple[int, str, bool]]:
    """List audio devices via PyAudio. Returns [(index, name, is_input), ...]."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        seen = set()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", f"Device {i}")
            max_in = info.get("maxInputChannels", 0)
            max_out = info.get("maxOutputChannels", 0)
            if max_in > 0:
                key = (name, True)
                if key not in seen:
                    seen.add(key)
                    devices.append((i, name, True))
            if max_out > 0:
                key = (name, False)
                if key not in seen:
                    seen.add(key)
                    devices.append((i, name, False))
        pa.terminate()
        return devices
    except Exception:
        return []


# ── Dialogs ────────────────────────────────────────────────────────────────

class _DirectivesDialog(QDialog):
    """Shows current active directives with remove capability."""

    def __init__(self, agent_loop: AgentLoop, parent=None):
        super().__init__(parent)
        self._agent_loop = agent_loop
        self.setWindowTitle("Active Directives")
        self.setMinimumWidth(450)
        self.setMinimumHeight(250)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._refresh()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh(self):
        self._list.clear()
        if not self._agent_loop.directives:
            item = QListWidgetItem("No active directives.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self._list.addItem(item)
            return
        for i, d in enumerate(self._agent_loop.directives):
            age = time.monotonic() - d.created_at
            if age < 60:
                age_str = f"{age:.0f}s"
            elif age < 3600:
                age_str = f"{age / 60:.0f}m"
            else:
                age_str = f"{age / 3600:.1f}h"
            timer = f"  timer:{d.trigger_time}" if d.trigger_time else ""
            fired = " FIRED" if d.triggered else ""
            text = f"[{d.urgency}/10] {d.goal}  ({d.source}, {age_str}{timer}{fired})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            self._list.addItem(item)

    def _remove_selected(self):
        item = self._list.currentItem()
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self._agent_loop.directives):
            self._agent_loop.directives.pop(idx)
            self._refresh()


class _AddDirectiveDialog(QDialog):
    """Simple dialog to add a directive with goal + urgency."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Directive")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        from llm.prompt import get_character_name
        layout.addWidget(QLabel(f"What should {get_character_name()} nag you about?"))
        tip = QLabel(f"(Tip: You can also just ask {get_character_name()} directly in conversation!)")
        tip.setStyleSheet("color: gray; font-size: 11px; font-style: italic;")
        layout.addWidget(tip)
        self._goal = QLineEdit()
        self._goal.setPlaceholderText("e.g. Go eat food, Do homework, Go to sleep...")
        layout.addWidget(self._goal)

        urg_row = QHBoxLayout()
        urg_row.addWidget(QLabel("Urgency (1=chill, 10=nuclear):"))
        self._urgency = QSpinBox()
        self._urgency.setRange(1, 10)
        self._urgency.setValue(5)
        urg_row.addWidget(self._urgency)
        layout.addLayout(urg_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Tuple[str, int]:
        return self._goal.text().strip(), self._urgency.value()


# ── Routines dialogs ────────────────────────────────────────────────────

class _RoutinesDialog(QDialog):
    """Shows recurring routines with add/remove."""

    def __init__(self, agent_loop: "AgentLoop", parent=None):
        super().__init__(parent)
        from core.routines import RoutineManager
        self._agent_loop = agent_loop
        self._rm: RoutineManager = agent_loop.routine_manager
        self.setWindowTitle("Recurring Routines")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        wake_info = self._rm.wake_time
        if wake_info:
            h = self._rm.hours_since_wake
            info = QLabel(f"Last wake-up: {wake_info.strftime('%I:%M %p')} ({h:.1f}h ago)")
            info.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(info)

        self._list = QListWidget()
        self._refresh()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Routine...")
        add_btn.clicked.connect(lambda: self._add_routine())
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        toggle_btn = QPushButton("Enable/Disable")
        toggle_btn.clicked.connect(self._toggle_selected)
        btn_row.addWidget(toggle_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh(self):
        self._list.clear()
        if not self._rm.routines:
            item = QListWidgetItem("No routines set up. Click 'Add Routine...' to create one.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self._list.addItem(item)
            return
        for r in self._rm.routines:
            desc = self._rm.describe_routine(r)
            status = "" if r.enabled else " [DISABLED]"
            last = f"  (last: {r.last_fired_date})" if r.last_fired_date else ""
            text = f"[{r.urgency}/10] {r.goal}  —  {desc}{last}{status}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, r.id)
            self._list.addItem(item)

    def _remove_selected(self):
        item = self._list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.UserRole)
        if rid:
            self._rm.remove(rid)
            self._refresh()

    def _toggle_selected(self):
        item = self._list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.UserRole)
        if rid:
            self._rm.toggle(rid)
            self._refresh()

    def _add_routine(self):
        dlg = _AddRoutineDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            routine = dlg.get_routine()
            if routine:
                self._rm.add(routine)
                self._refresh()


class _AddRoutineDialog(QDialog):
    """Dialog to create a new recurring routine."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Routine")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        from llm.prompt import get_character_name
        layout.addWidget(QLabel(f"What should {get_character_name()} remind you about?"))
        self._goal = QLineEdit()
        self._goal.setPlaceholderText("e.g. Brush your teeth, Drink water, Take meds...")
        layout.addWidget(self._goal)

        # Schedule type
        sched_row = QHBoxLayout()
        sched_row.addWidget(QLabel("Schedule:"))
        self._schedule = QComboBox()
        self._schedule.addItems([
            "on_wake — When I wake up",
            "on_sleep — Before bed (~Xh after waking)",
            "daily — Every day at a specific time",
            "weekly — Once a week at a day+time",
            "interval — Every X hours",
        ])
        self._schedule.currentIndexChanged.connect(self._on_schedule_changed)
        sched_row.addWidget(self._schedule)
        layout.addLayout(sched_row)

        # Time input (for daily/weekly)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time (HH:MM, 24h):"))
        self._time = QLineEdit()
        self._time.setPlaceholderText("e.g. 14:00")
        self._time.setMaximumWidth(80)
        time_row.addWidget(self._time)
        time_row.addStretch()
        layout.addLayout(time_row)
        self._time_row_widgets = [self._time]

        # Day input (for weekly)
        day_row = QHBoxLayout()
        day_row.addWidget(QLabel("Day:"))
        self._day = QComboBox()
        self._day.addItems(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
        day_row.addWidget(self._day)
        day_row.addStretch()
        layout.addLayout(day_row)
        self._day_widget = self._day

        # Hours input (for interval / on_sleep)
        hours_row = QHBoxLayout()
        self._hours_label = QLabel("Hours after waking:")
        hours_row.addWidget(self._hours_label)
        self._hours = QDoubleSpinBox()
        self._hours.setRange(0.5, 24.0)
        self._hours.setValue(8.0)
        self._hours.setSingleStep(0.5)
        hours_row.addWidget(self._hours)
        hours_row.addStretch()
        layout.addLayout(hours_row)

        # Urgency
        urg_row = QHBoxLayout()
        urg_row.addWidget(QLabel("Urgency (1=chill, 10=nuclear):"))
        self._urgency = QSpinBox()
        self._urgency.setRange(1, 10)
        self._urgency.setValue(5)
        urg_row.addWidget(self._urgency)
        layout.addLayout(urg_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_schedule_changed(0)  # initial visibility

    def _on_schedule_changed(self, idx: int):
        sched = self._schedule.currentText().split(" — ")[0]
        self._time.setVisible(sched in ("daily", "weekly"))
        self._day_widget.setVisible(sched == "weekly")
        self._hours.setVisible(sched in ("on_sleep", "interval"))
        self._hours_label.setVisible(sched in ("on_sleep", "interval"))
        if sched == "on_sleep":
            self._hours_label.setText("Hours after waking:")
            self._hours.setValue(8.0)
        elif sched == "interval":
            self._hours_label.setText("Every X hours:")
            self._hours.setValue(2.0)

    def get_routine(self):
        from core.routines import Routine
        import uuid
        goal = self._goal.text().strip()
        if not goal:
            return None
        sched = self._schedule.currentText().split(" — ")[0]
        return Routine(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            urgency=self._urgency.value(),
            schedule=sched,
            time=self._time.text().strip() or None,
            day=self._day.currentText() if sched == "weekly" else None,
            interval_hours=self._hours.value() if sched == "interval" else None,
            sleep_offset_hours=self._hours.value() if sched == "on_sleep" else 8.0,
        )


# ── Context menu builder ──────────────────────────────────────────────────

class _CharacterPickerDialog(QDialog):
    """Searchable dialog to pick from all available characters."""

    def __init__(self, current_slug: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Character")
        self.setMinimumWidth(350)
        self.setMinimumHeight(500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._selected_slug: Optional[str] = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Choose a character:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search characters...")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate
        from core.character_registry import get_all_characters
        self._all_chars = get_all_characters()
        self._populate(current_slug)

    def _populate(self, current_slug: str) -> None:
        self._list.clear()
        scroll_to: Optional[QListWidgetItem] = None
        for info in self._all_chars:
            label = info.display_name
            if info.has_custom_preset:
                label += "  \u2605"  # star for custom presets
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, info.slug)
            self._list.addItem(item)
            if info.slug == current_slug:
                item.setSelected(True)
                scroll_to = item
        if scroll_to:
            self._list.setCurrentItem(scroll_to)
            self._list.scrollToItem(scroll_to)

    def _filter(self, text: str) -> None:
        text_lower = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text_lower not in item.text().lower())

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected_slug = item.data(Qt.UserRole)
        self.accept()

    def _on_accept(self) -> None:
        item = self._list.currentItem()
        if item and not item.isHidden():
            self._selected_slug = item.data(Qt.UserRole)
            self.accept()

    def get_selected_slug(self) -> Optional[str]:
        return self._selected_slug


class _OOCDialog(QDialog):
    """Dialog to send an out-of-character message to the LLM."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from llm.prompt import get_character_name
        self.setWindowTitle(f"OOC Message to {get_character_name()}")
        self.setMinimumWidth(450)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Send an out-of-character instruction to the LLM.\n"
            "Use this to critique writing style, fix mistakes, adjust behavior, etc.\n"
            "The character will read this as a meta-instruction, not as dialogue."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info)

        self._text = QLineEdit()
        self._text.setPlaceholderText("e.g. Stop using so many exclamation marks, be more sarcastic...")
        layout.addWidget(self._text)

        self._response = QLabel("")
        self._response.setWordWrap(True)
        self._response.setStyleSheet("padding: 6px; background: #1a1a2e; border-radius: 4px;")
        self._response.hide()
        layout.addWidget(self._response)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self) -> str:
        return self._text.text().strip()


class ContextMenuBuilder:
    """Holds live-object references and builds the right-click menu on demand."""

    def __init__(
        self,
        config: AppConfig,
        config_path: str = "config.yaml",
        agent_loop: Optional[AgentLoop] = None,
        llm_provider: Optional[LLMProvider] = None,
        on_scale_change: Optional[Callable[[float], None]] = None,
        on_character_change: Optional[Callable[[str], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        ack_player=None,
        on_provider_change: Optional[Callable[[str], None]] = None,
        tts=None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.agent_loop = agent_loop
        self.llm = llm_provider
        self.on_scale_change = on_scale_change
        self.on_character_change = on_character_change
        self.on_quit = on_quit
        self.ack_player = ack_player
        self.on_provider_change = on_provider_change
        self.tts = tts

    # ── Main builder ──────────────────────────────────────────────────────

    def build(self, parent: QWidget) -> QMenu:
        """Build and return the full context menu."""
        menu = QMenu(parent)
        cfg = self.config

        # ── Directives submenu ────────────────────────────────────────
        dir_menu = menu.addMenu("Directives")

        view_act = dir_menu.addAction("View Active...")
        view_act.triggered.connect(lambda: self._show_directives(parent))

        clear_act = dir_menu.addAction("Clear All")
        clear_act.triggered.connect(self._clear_directives)
        has = self.agent_loop and self.agent_loop.has_directives
        clear_act.setEnabled(bool(has))

        add_act = dir_menu.addAction("Add Directive...")
        add_act.triggered.connect(lambda: self._add_directive(parent))
        add_act.setEnabled(self.agent_loop is not None)

        routines_act = dir_menu.addAction("Routines...")
        routines_act.triggered.connect(lambda: self._show_routines(parent))
        routines_act.setEnabled(self.agent_loop is not None)

        ooc_act = menu.addAction("Send OOC Message...")
        ooc_act.triggered.connect(lambda: self._send_ooc(parent))

        menu.addSeparator()

        # ── Conversation Only mode ────────────────────────────────────
        is_convo_only = (
            not cfg.agent.enabled
            and not cfg.agent.self_initiate
            and not cfg.desktop_control.enabled
            and not cfg.vision.screen_capture
            and not cfg.vision.enabled
        )
        self._add_toggle(menu, "Conversation Only", is_convo_only,
                         lambda c: self._set_conversation_only(c))

        menu.addSeparator()

        # ── Features toggles submenu ──────────────────────────────────
        feat_menu = menu.addMenu("Features")

        self._add_toggle(feat_menu, "Autonomous Mode", cfg.agent.enabled,
                         lambda c: self._set("agent", "enabled", c))
        self._add_toggle(feat_menu, "Self-Initiate", cfg.agent.self_initiate,
                         lambda c: self._set("agent", "self_initiate", c))
        self._add_toggle(feat_menu, "Desktop Control", cfg.desktop_control.enabled,
                         lambda c: self._set("desktop_control", "enabled", c))
        self._add_toggle(feat_menu, "TTS (Voice)", cfg.tts.enabled,
                         lambda c: self._set("tts", "enabled", c))
        self._add_toggle(feat_menu, "Speech Bubbles", cfg.desktop_pet.speech_bubble,
                         lambda c: self._set("desktop_pet", "speech_bubble", c))

        feat_menu.addSeparator()

        self._add_toggle(feat_menu, "Screenshots", cfg.vision.screen_capture,
                         lambda c: self._set("vision", "screen_capture", c))
        self._add_toggle(feat_menu, "Webcam", cfg.vision.enabled,
                         lambda c: self._set("vision", "enabled", c))
        self._radio_submenu(feat_menu, "Screen Vision", [
            ("API (LLM)", "api"),
            ("Moondream (Local)", "moondream"),
        ], cfg.vision.screen_vision, lambda v: self._set_screen_vision(v))

        menu.addSeparator()

        # ── Check Interval submenu ────────────────────────────────────
        self._radio_submenu(menu, "Check Interval", [
            ("Hyper (30s)", 30.0),
            ("Fast (2 min)", 120.0),
            ("Normal (5 min)", 300.0),
            ("Relaxed (12 min)", 750.0),
            ("Chill (30 min)", 1800.0),
        ], cfg.agent.base_check_interval_s,
            lambda v: self._set("agent", "base_check_interval_s", v))

        # ── Scale submenu ─────────────────────────────────────────────
        self._radio_submenu(menu, "Scale", [
            ("Tiny (1x)", 1.0),
            ("Normal (2x)", 2.0),
            ("Big (3x)", 3.0),
            ("Huge (4x)", 4.0),
        ], cfg.desktop_pet.scale,
            lambda v: self._apply_scale(v))

        # ── LLM Provider submenu ──────────────────────────────────────
        llm_menu = menu.addMenu("LLM Provider")
        self._radio_submenu_into(llm_menu, [
            ("OpenAI", "openai"),
            ("Anthropic (Claude)", "anthropic"),
            ("Ollama (local)", "ollama"),
            ("LM Studio (local)", "lmstudio"),
            ("OpenRouter", "openrouter"),
            ("Groq", "groq"),
            ("DeepSeek", "deepseek"),
            ("KoboldCpp (local)", "koboldcpp"),
            ("vLLM (local)", "vllm"),
        ], cfg.llm.provider.lower(),
            lambda v: self._apply_provider(v))
        llm_menu.addSeparator()
        key_label = self._mask_key(cfg.llm.api_key)
        set_key_act = llm_menu.addAction(f"API Key: {key_label}...")
        set_key_act.triggered.connect(lambda: self._set_llm_api_key(parent))

        url_label = cfg.llm.base_url or "(default)"
        set_url_act = llm_menu.addAction(f"Base URL: {url_label}...")
        set_url_act.triggered.connect(lambda: self._set_base_url(parent))

        # ── LLM Model submenu (auto-fetched from API) ─────────────────
        model_choices = self._get_model_choices()
        self._radio_submenu(menu, "LLM Model", model_choices, cfg.llm.model,
            lambda v: self._apply_model(v))

        # ── Character picker ──────────────────────────────────────────
        from llm.prompt import get_character_name
        char_act = menu.addAction(f"Character: {get_character_name()}...")
        char_act.triggered.connect(lambda: self._show_character_picker(parent))

        # ── TTS / ElevenLabs submenu ─────────────────────────────────
        tts_menu = menu.addMenu("TTS")
        self._radio_submenu_into(tts_menu, [
            ("ElevenLabs", "elevenlabs"),
            ("Local (ponyvoicetool)", "openai_compatible"),
        ], cfg.tts.provider,
            lambda v: self._set("tts", "provider", v))
        tts_menu.addSeparator()
        el_key_label = self._mask_key(cfg.elevenlabs.api_key)
        el_key_act = tts_menu.addAction(f"ElevenLabs API Key: {el_key_label}...")
        el_key_act.triggered.connect(lambda: self._set_elevenlabs_key(parent))
        el_vid_label = cfg.elevenlabs.voice_id[:8] + "..." if len(cfg.elevenlabs.voice_id) > 8 else cfg.elevenlabs.voice_id or "(not set)"
        el_vid_act = tts_menu.addAction(f"ElevenLabs Voice ID: {el_vid_label}...")
        el_vid_act.triggered.connect(lambda: self._set_elevenlabs_voice_id(parent))

        # ── Audio Devices submenu ─────────────────────────────────────
        audio_menu = menu.addMenu("Audio Devices (restart needed)")
        self._build_audio_submenu(audio_menu)

        # ── Voice Filter ──────────────────────────────────────────────
        voice_menu = menu.addMenu("Voice Filter")
        enroll_act = voice_menu.addAction("Enroll My Voice...")
        enroll_act.triggered.connect(lambda: self._enroll_voice(parent))
        delete_act = voice_menu.addAction("Delete Voice Profile")
        delete_act.triggered.connect(self._delete_voice_profile)

        menu.addSeparator()

        # ── Utilities ─────────────────────────────────────────────────
        menu.addAction("Open Ack Sounds Folder").triggered.connect(
            lambda: self._open_ack_folder())
        menu.addAction("Open Config File").triggered.connect(
            lambda: self._open_file(self.config_path))
        menu.addAction("Open Log File").triggered.connect(
            lambda: self._open_file(cfg.logging.log_file))

        menu.addSeparator()

        update_act = menu.addAction("Check for Updates...")
        update_act.triggered.connect(lambda: self._check_for_updates(parent))

        restart_act = menu.addAction("Restart")
        restart_act.triggered.connect(self._restart)

        quit_act = menu.addAction("Quit")
        quit_act.triggered.connect(self.on_quit if self.on_quit else QApplication.quit)

        return menu

    # ── Widget helpers ────────────────────────────────────────────────────

    def _toggle(self, text: str, checked: bool, callback) -> QAction:
        act = QAction(text)
        act.setCheckable(True)
        act.setChecked(checked)
        act.triggered.connect(callback)
        return act

    @staticmethod
    def _add_toggle(menu: QMenu, text: str, checked: bool, callback) -> QAction:
        """Create a checkable action parented to the menu so it won't be GC'd."""
        act = menu.addAction(text)
        act.setCheckable(True)
        act.setChecked(checked)
        act.triggered.connect(callback)
        return act

    def _radio_submenu(self, parent_menu: QMenu, title: str,
                       options: list, current, callback) -> None:
        sub = parent_menu.addMenu(title)
        self._radio_submenu_into(sub, options, current, callback)

    def _radio_submenu_into(self, sub: QMenu, options: list, current, callback) -> None:
        group = QActionGroup(sub)
        for label, value in options:
            act = QAction(label, sub)
            act.setCheckable(True)
            # Match check: float tolerance or string equality
            if isinstance(value, float) and isinstance(current, (int, float)):
                act.setChecked(abs(float(current) - value) < 0.5)
            else:
                act.setChecked(str(current) == str(value))
            act.triggered.connect(lambda checked, v=value: callback(v))
            group.addAction(act)
            sub.addAction(act)

    # ── Config setters ────────────────────────────────────────────────────

    def _set(self, section: str, key: str, value) -> None:
        """Update live config + persist to YAML."""
        obj = getattr(self.config, section)
        setattr(obj, key, value)
        _save_yaml_value(f"{section}.{key}", value, self.config_path)
        logger.info("Config: %s.%s = %s", section, key, value)

    def _set_screen_vision(self, provider: str) -> None:
        """Switch between API and Moondream screen vision.  Requires restart for Moondream."""
        self._set("vision", "screen_vision", provider)
        if provider == "moondream":
            QMessageBox.information(
                None, "Screen Vision",
                "Moondream (local) selected.\n\n"
                "Requires ~2 GB RAM and the 'transformers' package.\n"
                "Restart the app to load the model.",
            )

    def _set_conversation_only(self, enabled: bool) -> None:
        """Toggle conversation-only mode: disable all computer control features."""
        if enabled:
            # Disable everything except pure conversation
            self._set("agent", "enabled", False)
            self._set("agent", "self_initiate", False)
            self._set("desktop_control", "enabled", False)
            self._set("vision", "screen_capture", False)
            self._set("vision", "enabled", False)
        else:
            # Re-enable all features
            self._set("agent", "enabled", True)
            self._set("agent", "self_initiate", True)
            self._set("desktop_control", "enabled", True)
            self._set("vision", "screen_capture", True)
            self._set("vision", "enabled", True)
        logger.info("Conversation Only mode: %s", "ON" if enabled else "OFF")

    def _get_model_choices(self) -> list[tuple[str, str]]:
        """Fetch available models from the LLM provider API. Cached after first call."""
        if hasattr(self, "_model_choices_cache") and self._model_choices_cache is not None:
            return self._model_choices_cache

        choices: list[tuple[str, str]] = []

        # Non-chat model prefixes to filter out (embeddings, TTS, image, etc.)
        _skip = ("whisper", "tts-", "dall-e", "text-embedding", "text-moderation",
                 "babbage", "davinci", "canary")

        try:
            client = getattr(self.llm, "_client", None)
            if client and hasattr(client, "models"):
                result = client.models.list()
                models_iter = result.data if hasattr(result, "data") else list(result)
                for m in models_iter:
                    mid = m.id if hasattr(m, "id") else str(m)
                    if any(mid.lower().startswith(p) for p in _skip):
                        continue
                    choices.append((mid, mid))
                choices.sort(key=lambda x: x[0].lower())
        except Exception as exc:
            logger.debug("Failed to fetch models from API: %s", exc)

        # Ensure current model is always in the list
        current = self.config.llm.model
        if not any(v == current for _, v in choices):
            choices.insert(0, (current, current))

        # Fallback if nothing was fetched
        if len(choices) <= 1:
            provider = self.config.llm.provider.lower()
            if provider == "anthropic":
                choices = [
                    ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
                    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
                    ("claude-opus-4-6", "claude-opus-4-6"),
                ]
            else:
                choices = [
                    ("gpt-4o-mini", "gpt-4o-mini"),
                    ("gpt-4o", "gpt-4o"),
                    (current, current),
                ]
                # Deduplicate
                seen = set()
                choices = [(l, v) for l, v in choices if v not in seen and not seen.add(v)]

        self._model_choices_cache = choices
        return choices

    @staticmethod
    def _mask_key(key: str) -> str:
        """Show first 4 and last 4 chars of a key, mask the rest."""
        if not key:
            return "(not set)"
        if len(key) <= 10:
            return "****"
        return key[:4] + "..." + key[-4:]

    def _set_llm_api_key(self, parent: QWidget) -> None:
        """Set the LLM API key from the menu."""
        from PyQt5.QtWidgets import QInputDialog
        current = self.config.llm.api_key or ""
        key, ok = QInputDialog.getText(
            parent, "LLM API Key",
            "Enter your LLM API key.\n"
            "This will be saved to config.yaml and take effect immediately.",
            QLineEdit.Normal, current,
        )
        if not ok:
            return
        key = key.strip()
        self.config.llm.api_key = key
        _save_yaml_value("llm.api_key", key, self.config_path)
        # Hot-swap provider so new key takes effect
        if self.on_provider_change:
            self.on_provider_change(self.config.llm.provider)
        logger.info("LLM API key updated.")

    def _set_elevenlabs_key(self, parent: QWidget) -> None:
        """Set the ElevenLabs API key from the menu."""
        from PyQt5.QtWidgets import QInputDialog
        current = self.config.elevenlabs.api_key or ""
        key, ok = QInputDialog.getText(
            parent, "ElevenLabs API Key",
            "Enter your ElevenLabs API key.\n"
            "Takes effect on next TTS call (no restart needed).",
            QLineEdit.Normal, current,
        )
        if not ok:
            return
        key = key.strip()
        self.config.elevenlabs.api_key = key
        _save_yaml_value("elevenlabs.api_key", key, self.config_path)
        if self.tts and hasattr(self.tts, "api_key"):
            self.tts.api_key = key
        logger.info("ElevenLabs API key updated.")

    def _set_elevenlabs_voice_id(self, parent: QWidget) -> None:
        """Set the ElevenLabs voice ID from the menu."""
        from PyQt5.QtWidgets import QInputDialog
        current = self.config.elevenlabs.voice_id or ""
        vid, ok = QInputDialog.getText(
            parent, "ElevenLabs Voice ID",
            "Enter your ElevenLabs voice ID.",
            QLineEdit.Normal, current,
        )
        if not ok:
            return
        vid = vid.strip()
        self.config.elevenlabs.voice_id = vid
        _save_yaml_value("elevenlabs.voice_id", vid, self.config_path)
        if self.tts and hasattr(self.tts, "voice_id"):
            self.tts.voice_id = vid
        logger.info("ElevenLabs voice ID updated to: %s", vid)

    def _set_base_url(self, parent: QWidget) -> None:
        """Let the user type a custom base URL for the LLM provider."""
        from PyQt5.QtWidgets import QInputDialog
        current = self.config.llm.base_url or ""
        url, ok = QInputDialog.getText(
            parent, "LLM Base URL",
            "Enter the base URL for your LLM provider.\n"
            "Leave blank to use the default for the selected provider.",
            QLineEdit.Normal, current,
        )
        if not ok:
            return
        url = url.strip()
        if url:
            self.config.llm.base_url = url
            _save_yaml_value("llm.base_url", url, self.config_path)
        else:
            self.config.llm.base_url = None
            _save_yaml_value("llm.base_url", None, self.config_path)
        # Hot-swap the provider with the new URL
        if self.on_provider_change:
            self.on_provider_change(self.config.llm.provider)
        logger.info("LLM base_url changed to: %s", self.config.llm.base_url)

    def _apply_provider(self, provider: str) -> None:
        """Hot-swap the LLM provider (creates a new client)."""
        self.config.llm.provider = provider
        _save_yaml_value("llm.provider", provider, self.config_path)

        # Auto-set base_url for known providers, clear for cloud providers
        from llm.factory import _KNOWN_BASE_URLS
        if provider in _KNOWN_BASE_URLS:
            self.config.llm.base_url = _KNOWN_BASE_URLS[provider]
            _save_yaml_value("llm.base_url", self.config.llm.base_url, self.config_path)
        elif provider in ("openai", "anthropic"):
            self.config.llm.base_url = None
            _save_yaml_value("llm.base_url", None, self.config_path)

        # Clear cached model list so next menu open fetches from the new provider
        self._model_choices_cache = None

        if self.on_provider_change:
            self.on_provider_change(provider)
        logger.info("LLM provider changed to: %s", provider)

    def _apply_model(self, model_id: str) -> None:
        """Hot-swap the LLM model (no restart needed)."""
        if self.llm and hasattr(self.llm, "model"):
            self.llm.model = model_id
        self.config.llm.model = model_id
        _save_yaml_value("llm.model", model_id, self.config_path)
        logger.info("LLM model changed to: %s", model_id)

    def _apply_scale(self, scale: float) -> None:
        """Change sprite scale (live reload)."""
        self.config.desktop_pet.scale = scale
        _save_yaml_value("desktop_pet.scale", scale, self.config_path)
        if self.on_scale_change:
            self.on_scale_change(scale)
        logger.info("Scale changed to: %.1f", scale)

    def _show_character_picker(self, parent: QWidget) -> None:
        """Open the character picker dialog."""
        from llm.prompt import get_active_preset
        dlg = _CharacterPickerDialog(get_active_preset(), parent)
        if dlg.exec_() == QDialog.Accepted:
            slug = dlg.get_selected_slug()
            if slug and slug != get_active_preset():
                self._apply_character(slug)

    def _apply_character(self, preset_slug: str) -> None:
        """Hot-swap the active character."""
        if self.on_character_change:
            self.on_character_change(preset_slug)

    def _open_ack_folder(self) -> None:
        """Open the current character's acknowledgement sounds folder."""
        if self.ack_player:
            folder = self.ack_player.get_assets_dir()
            folder.mkdir(parents=True, exist_ok=True)
            self._open_file(str(folder))

    # ── Directive actions ─────────────────────────────────────────────────

    def _clear_directives(self) -> None:
        if self.agent_loop:
            self.agent_loop.clear_directives()

    def _show_directives(self, parent: QWidget) -> None:
        if not self.agent_loop:
            return
        dlg = _DirectivesDialog(self.agent_loop, parent)
        dlg.exec_()

    def _add_directive(self, parent: QWidget) -> None:
        if not self.agent_loop:
            return
        dlg = _AddDirectiveDialog(parent)
        if dlg.exec_() == QDialog.Accepted:
            goal, urgency = dlg.get_values()
            if goal:
                self.agent_loop.add_directive(goal, urgency, source="user")

    def _show_routines(self, parent: QWidget) -> None:
        if not self.agent_loop:
            return
        dlg = _RoutinesDialog(self.agent_loop, parent)
        dlg.exec_()

    def _send_ooc(self, parent: QWidget) -> None:
        """Send an out-of-character meta-instruction to the LLM."""
        dlg = _OOCDialog(parent)
        if dlg.exec_() != QDialog.Accepted:
            return
        text = dlg.get_text()
        if not text or not self.llm:
            return
        ooc_msg = (
            f"[OOC — out-of-character instruction from the user. This is NOT dialogue. "
            f"Read this as a meta-note about how to adjust your writing, behavior, or style. "
            f"Acknowledge briefly in-character, then apply it going forward.]\n\n{text}"
        )
        try:
            response = self.llm.chat(ooc_msg)
            logger.info("OOC sent: %s → %s", text, response)
        except Exception as exc:
            logger.warning("OOC message failed: %s", exc)

    # ── Audio devices ─────────────────────────────────────────────────────

    def _build_audio_submenu(self, menu: QMenu) -> None:
        devices = _list_audio_devices()

        # Microphone submenu
        mic_menu = menu.addMenu("Microphone")
        mic_group = QActionGroup(mic_menu)

        act = QAction("Default", mic_menu)
        act.setCheckable(True)
        act.setChecked(self.config.audio.input_device_index == -1)
        act.triggered.connect(lambda: self._set("audio", "input_device_index", -1))
        mic_group.addAction(act)
        mic_menu.addAction(act)

        for idx, name, is_input in devices:
            if not is_input:
                continue
            act = QAction(name, mic_menu)
            act.setCheckable(True)
            act.setChecked(self.config.audio.input_device_index == idx)
            act.triggered.connect(
                lambda checked, i=idx: self._set("audio", "input_device_index", i))
            mic_group.addAction(act)
            mic_menu.addAction(act)

        # Speaker submenu
        spk_menu = menu.addMenu("Speaker")
        spk_group = QActionGroup(spk_menu)

        act = QAction("Default", spk_menu)
        act.setCheckable(True)
        act.setChecked(self.config.audio.output_device_index == -1)
        act.triggered.connect(lambda: self._set("audio", "output_device_index", -1))
        spk_group.addAction(act)
        spk_menu.addAction(act)

        for idx, name, is_input in devices:
            if is_input:
                continue
            act = QAction(name, spk_menu)
            act.setCheckable(True)
            act.setChecked(self.config.audio.output_device_index == idx)
            act.triggered.connect(
                lambda checked, i=idx: self._set("audio", "output_device_index", i))
            spk_group.addAction(act)
            spk_menu.addAction(act)

    # ── Voice filter ────────────────────────────────────────────────────

    def _enroll_voice(self, parent: QWidget) -> None:
        """Record user's voice and create a voice profile."""
        from PyQt5.QtWidgets import QMessageBox, QProgressDialog
        import struct
        import numpy as np

        msg = QMessageBox(parent)
        msg.setWindowTitle("Voice Enrollment")
        msg.setText(
            "This will record 8 seconds of your voice.\n\n"
            "Speak naturally — say anything, just keep talking.\n"
            "After enrollment, the pony will only respond to YOUR voice\n"
            "and ignore YouTube, TV, and other people."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if msg.exec_() != QMessageBox.Ok:
            return

        # Record
        try:
            import pyaudio
            sample_rate = 16000
            frame_size = int(sample_rate * 30 / 1000)
            record_seconds = 8

            pa = pyaudio.PyAudio()
            stream_kwargs = dict(
                format=pyaudio.paInt16, channels=1, rate=sample_rate,
                input=True, frames_per_buffer=frame_size,
            )
            idx = self.config.audio.input_device_index
            if idx >= 0:
                stream_kwargs["input_device_index"] = idx

            progress = QProgressDialog("Recording... speak now!", "Cancel", 0, 100, parent)
            progress.setWindowTitle("Voice Enrollment")
            progress.setMinimumDuration(0)
            progress.show()

            stream = pa.open(**stream_kwargs)
            frames = []
            total_frames = int(record_seconds * sample_rate / frame_size)

            for i in range(total_frames):
                if progress.wasCanceled():
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()
                    return
                raw = stream.read(frame_size, exception_on_overflow=False)
                frames.append(raw)
                progress.setValue(int((i + 1) / total_frames * 100))
                QApplication.processEvents()

            stream.stop_stream()
            stream.close()
            pa.terminate()
            progress.close()

            # Convert and enroll
            audio_bytes = b"".join(frames)
            audio_int16 = struct.unpack(f"{len(audio_bytes) // 2}h", audio_bytes)
            audio_f32 = np.array(audio_int16, dtype=np.float32) / 32768.0

            from stt.voice_filter import VoiceFilter
            vf = VoiceFilter()
            success = vf.enroll(audio_f32)

            result = QMessageBox(parent)
            result.setWindowTitle("Voice Enrollment")
            if success:
                result.setText("Voice profile saved!\nThe pony will now only respond to your voice.")
                result.setIcon(QMessageBox.Information)
            else:
                result.setText("Enrollment failed. Try speaking louder or longer.")
                result.setIcon(QMessageBox.Warning)
            result.exec_()

        except Exception as exc:
            logger.warning("Voice enrollment failed: %s", exc)
            err = QMessageBox(parent)
            err.setWindowTitle("Voice Enrollment Error")
            err.setText(f"Failed: {exc}")
            err.setIcon(QMessageBox.Critical)
            err.exec_()

    @staticmethod
    def _delete_voice_profile() -> None:
        try:
            from stt.voice_filter import VoiceFilter
            vf = VoiceFilter()
            vf.delete_profile()
            logger.info("Voice profile deleted.")
        except Exception as exc:
            logger.warning("Failed to delete voice profile: %s", exc)

    # ── File openers ──────────────────────────────────────────────────────

    @staticmethod
    def _open_file(path: str) -> None:
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        try:
            os.startfile(str(p))
        except Exception:
            try:
                subprocess.Popen(["notepad", str(p)])
            except Exception as exc:
                logger.warning("Failed to open %s: %s", path, exc)

    # ── Restart ───────────────────────────────────────────────────────────

    @staticmethod
    def _restart() -> None:
        """Restart the application."""
        from core.updater import restart_application
        restart_application()

    # ── Self-updater ─────────────────────────────────────────────────────

    def _check_for_updates(self, parent: QWidget) -> None:
        """Check GitHub for updates and offer to install them."""
        from core.updater import check_for_updates, pull_updates, install_new_requirements, restart_application

        # Check phase
        has_updates, status_msg, commits = check_for_updates()

        if not has_updates:
            QMessageBox.information(parent, "bonziPONY Updater", status_msg)
            return

        # Build changelog
        changelog = "\n".join(commits) if commits else "(could not fetch changelog)"
        detail = f"{status_msg}\n\nNew commits:\n{changelog}"

        reply = QMessageBox.question(
            parent,
            "bonziPONY Updater",
            f"{status_msg}\n\nDo you want to update now?\n\n{changelog}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Pull phase
        ok, pull_msg = pull_updates()
        if not ok:
            QMessageBox.warning(parent, "Update Failed", pull_msg)
            return

        # Install new dependencies
        dep_ok, dep_msg = install_new_requirements()
        if not dep_ok:
            logger.warning("Dependency install issue: %s", dep_msg)

        # Ask to restart
        reply = QMessageBox.question(
            parent,
            "Update Complete",
            "Update installed successfully!\n\nRestart now to apply changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            restart_application()
