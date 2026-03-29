"""Transparent frameless window that renders the animated sprite + effects."""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QRectF, QByteArray
from PyQt5.QtGui import QPainter, QCursor, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QAction

from desktop_pet.behavior_manager import BehaviorDef, BehaviorManager, MovementType
from desktop_pet.effect_renderer import EffectRenderer
from desktop_pet.sprite_manager import SpriteAnimation, SpriteManager

logger = logging.getLogger(__name__)

_TICK_MS = 16  # ~60fps


class PetWindow(QWidget):
    """Main transparent always-on-top window for the desktop pet."""

    conversation_requested = pyqtSignal()
    text_message = pyqtSignal(str)  # Emitted when user types a message via double-click
    idle_chatter = pyqtSignal()  # Emitted randomly to trigger a quip bubble
    listen_interrupted = pyqtSignal()  # Emitted when user clicks during LISTEN to stop early

    def __init__(
        self,
        sprite_manager: SpriteManager,
        behavior_manager: BehaviorManager,
        effect_renderer: EffectRenderer,
    ) -> None:
        super().__init__(None)
        self.sprite_manager = sprite_manager
        self.behavior_manager = behavior_manager
        self.effect_renderer = effect_renderer

        # Current state
        self._current_behavior: Optional[BehaviorDef] = None
        self._current_anim: Optional[SpriteAnimation] = None
        self._frame_index = 0
        self._last_frame_time = 0.0
        self._facing_right = True
        self._behavior_start_time = 0.0
        self._behavior_duration = 0.0  # chosen random duration for this behavior

        # Movement
        self._dx = 0  # pixels per tick
        self._dy = 0
        self._roaming = True

        # Override animation (for pipeline states)
        self._override_anim_name: Optional[str] = None
        # Timed override (persists beyond conversation end)
        self._timed_override_until: float = 0.0
        self._timed_anim_name: Optional[str] = None

        # Mic indicator state
        self._show_mic: bool = False
        self._mic_pulse: float = 0.0  # for pulsing animation

        # Context menu builder (set by main.py)
        self._menu_builder = None

        # Cursor tracking (pony faces the mouse when idle)
        self._cursor_check_counter = 0

        # Drag state
        self._dragging = False
        self._drag_offset = QPoint()
        self._was_roaming_before_drag = True

        # Cursor grab state
        self._grab_running = False
        self._grab_run_timer: Optional[QTimer] = None
        # Tab drag walk state
        self._drag_walking = False

        # Pony manager reference for collision avoidance (set externally)
        self._pony_manager_ref = None

        # Setup window
        self._setup_window()

        # Animation timer
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start(_TICK_MS)

        # Start first behavior
        self._pick_and_start_behavior()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(200, 200)  # Will resize dynamically

        # Position at bottom-center of primary screen
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + geom.width() // 2 - 100
            y = geom.y() + geom.height() - 250
            self.move(x, y)

        # Reinforce stay-on-top via Win32 — Qt hint alone loses to browsers
        # (deferred to first tick — HWND isn't valid until show())
        self._topmost_counter = 290  # fires on first tick cycle

    def _ensure_topmost(self) -> None:
        """Use Win32 SetWindowPos to force HWND_TOPMOST."""
        try:
            import ctypes
            hwnd = int(self.winId())
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            HWND_TOPMOST = -1
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    def nativeEvent(self, event_type: QByteArray, message):
        """Pass through native events — no special handling needed."""
        return super().nativeEvent(event_type, message)

    def paintEvent(self, event) -> None:
        if self._current_anim is None or not self._current_anim.frames:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Get the current sprite frame
        frame_idx = self._frame_index % len(self._current_anim.frames)
        pixmap = self._current_anim.frames[frame_idx]

        # Draw effects behind sprite
        effects = self.effect_renderer.tick(
            self.x(), self.y(), pixmap.width(), pixmap.height(),
            facing_right=self._facing_right,
        )
        for eff_pixmap, ex, ey in effects:
            # Convert effect screen coords to widget-local coords
            local_x = ex - self.x()
            local_y = ey - self.y()
            painter.drawPixmap(local_x, local_y, eff_pixmap)

        # Draw the main sprite at (0, 0) in widget coords
        painter.drawPixmap(0, 0, pixmap)

        # Draw mic indicator when listening
        if self._show_mic:
            self._draw_mic_indicator(painter, pixmap.width(), pixmap.height())

        painter.end()

    def _on_tick(self) -> None:
        """Main animation/movement tick (~60fps)."""
        now = time.monotonic()

        # Re-assert topmost every ~5 seconds (300 ticks at 60fps)
        self._topmost_counter += 1
        if self._topmost_counter >= 300:
            self._topmost_counter = 0
            self._ensure_topmost()

        # Check timed override expiry
        if self._timed_override_until and now >= self._timed_override_until:
            self._timed_override_until = 0.0
            self._timed_anim_name = None
            self._override_anim_name = None
            self._roaming = True
            self._pick_and_start_behavior()

        if self._current_anim and self._current_anim.frames:
            # Advance animation frame
            delay = self._current_anim.delays[self._frame_index % len(self._current_anim.delays)]
            if now - self._last_frame_time >= delay / 1000.0:
                self._frame_index = (self._frame_index + 1) % len(self._current_anim.frames)
                self._last_frame_time = now

                # Resize window to match current frame
                pixmap = self._current_anim.frames[self._frame_index]
                if pixmap.width() > 0 and pixmap.height() > 0:
                    self.setFixedSize(pixmap.width(), pixmap.height())

        # Cursor tracking — pony faces the mouse when standing idle
        self._cursor_check_counter += 1
        if self._cursor_check_counter >= 30:  # ~every 500ms at 60fps
            self._cursor_check_counter = 0
            if not self._dragging and self._dx == 0 and self._dy == 0:
                cursor = QCursor.pos()
                center_x = self.x() + self.width() // 2
                should_right = cursor.x() > center_x
                if should_right != self._facing_right:
                    self._facing_right = should_right
                    self._update_facing()

        # Move sprite if roaming and not dragging
        if self._roaming and not self._dragging and self._override_anim_name is None:
            self._move_tick()

            # Check if behavior duration expired
            if self._behavior_duration > 0:
                elapsed = now - self._behavior_start_time
                if elapsed >= self._behavior_duration:
                    self._finish_behavior()

        self.update()

    def _start_behavior(self, behavior: BehaviorDef) -> None:
        """Start a new behavior."""
        self._current_behavior = behavior
        self._behavior_start_time = time.monotonic()
        self._behavior_duration = random.uniform(behavior.min_duration, behavior.max_duration)

        # Load animation from the GIF file specified in pony.ini
        self._current_anim, fallback = self._load_behavior_anim(behavior)
        self._frame_index = 0
        self._last_frame_time = time.monotonic()

        # Set movement direction — zero out if the GIF failed to load (prevents gliding)
        if fallback:
            self._dx, self._dy = 0, 0
        else:
            self._dx, self._dy = self._pick_direction(behavior.movement, behavior.speed)

        # Resize to first frame
        if self._current_anim and self._current_anim.frames:
            px = self._current_anim.frames[0]
            self.setFixedSize(px.width(), px.height())

        # Trigger effects
        if behavior.effects:
            self.effect_renderer.clear()
            self.effect_renderer.trigger_effects(
                behavior.name,
                self._facing_right,
                self.x(), self.y(),
                self.width(), self.height(),
            )

        logger.debug(
            "Started behavior '%s' (%.1fs, speed=%.1f, dx=%d, dy=%d)",
            behavior.name, self._behavior_duration, behavior.speed,
            self._dx, self._dy,
        )

    def _finish_behavior(self) -> None:
        """Behavior finished — chain to linked or pick new random."""
        self.effect_renderer.clear()

        if self._current_behavior:
            linked = self.behavior_manager.get_linked(self._current_behavior)
            if linked:
                self._start_behavior(linked)
                return

        # ~20% chance to pause briefly between behaviors (less predictable)
        if random.random() < 0.20:
            self._dx, self._dy = 0, 0
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(random.randint(1500, 4000), self._pick_and_start_behavior)
            return

        self._pick_and_start_behavior()

    def _pick_and_start_behavior(self) -> None:
        """Pick a random behavior and start it."""
        behavior = self.behavior_manager.pick_behavior()
        # Randomly pick facing direction
        if random.random() < 0.5:
            self._facing_right = not self._facing_right
        self._start_behavior(behavior)
        # ~5% chance to show a random idle quip when switching behaviors
        if random.random() < 0.05:
            self.idle_chatter.emit()

    def _load_behavior_anim(self, behavior: BehaviorDef) -> tuple[SpriteAnimation, bool]:
        """Load the correct animation for a behavior using its pony.ini GIF filenames.

        Returns (animation, used_fallback). If the behavior's GIF fails to load,
        falls back to the stand animation and returns True so the caller can
        zero out movement (prevents the pony gliding without a walk animation).
        """
        gif = behavior.right_image if self._facing_right else behavior.left_image
        anim = self.sprite_manager.get_by_gif(gif)
        if anim.frames:
            return anim, False
        # Fallback to stand — caller should zero out dx/dy
        logger.debug("GIF not found for behavior '%s': %s — falling back to stand", behavior.name, gif)
        return self.sprite_manager.get_animation("stand", self._facing_right), True

    def _pick_direction(self, movement: MovementType, speed: float) -> tuple[int, int]:
        """Pick dx, dy per tick based on movement type."""
        if speed <= 0 or movement in (MovementType.NONE, MovementType.MOUSEOVER,
                                       MovementType.SLEEP, MovementType.DRAGGED):
            return 0, 0

        # Choose random direction sign
        h_sign = 1 if self._facing_right else -1
        v_sign = random.choice([-1, 1])

        def _snap(val: float) -> int:
            """Round to int, but ensure at least 1px in the intended direction."""
            r = int(round(val))
            if r == 0 and val != 0.0:
                return 1 if val > 0 else -1
            return r

        if movement == MovementType.HORIZONTAL_ONLY:
            # Slight vertical drift so ponies don't cluster on same y-axis
            v_drift = random.choice([-0.15, 0, 0, 0, 0.15]) * speed
            return _snap(speed * h_sign), _snap(v_drift)
        elif movement == MovementType.VERTICAL_ONLY:
            return 0, _snap(speed * v_sign)
        elif movement == MovementType.DIAGONAL_VERTICAL:
            return _snap(speed * h_sign * 0.3), _snap(speed * v_sign)
        elif movement == MovementType.DIAGONAL_HORIZONTAL:
            return _snap(speed * h_sign), _snap(speed * v_sign * 0.3)
        elif movement == MovementType.ALL:
            return _snap(speed * h_sign), _snap(speed * v_sign)
        else:
            return 0, 0

    def _move_tick(self) -> None:
        """Apply dx/dy movement, bounce off screen edges, avoid other ponies."""
        if self._dx == 0 and self._dy == 0:
            return

        # ~1% chance per tick to perturb vertical direction (less predictable paths)
        # Only touch dy — changing dx sign without updating facing makes pony walk backwards
        if random.random() < 0.01 and self._dy != 0:
            self._dy += random.choice([-1, 1])
            self._dy = max(-4, min(4, self._dy))

        # Pony-pony collision avoidance
        if self._pony_manager_ref:
            try:
                my_cx = self.x() + self.width() // 2
                my_cy = self.y() + self.height() // 2
                for ox, oy in self._pony_manager_ref.get_other_pony_positions(self):
                    dist_x = my_cx - ox
                    dist_y = my_cy - oy
                    dist_sq = dist_x * dist_x + dist_y * dist_y
                    if dist_sq < 22500:  # within ~150px
                        # Push away from other pony
                        if dist_x != 0:
                            self._dx = abs(self._dx) if dist_x > 0 else -abs(self._dx)
                        if dist_y != 0:
                            self._dy = abs(self._dy) if dist_y > 0 else -abs(self._dy)
                        if dist_x > 0:
                            self._facing_right = True
                        elif dist_x < 0:
                            self._facing_right = False
                        self._update_facing()
                        break
            except Exception:
                pass

        new_x = self.x() + self._dx
        new_y = self.y() + self._dy

        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.virtualGeometry()
            # Bounce off edges of the entire virtual desktop (all monitors)
            if new_x < geom.left():
                new_x = geom.left()
                self._dx = abs(self._dx)
                self._facing_right = True
                self._update_facing()
            elif new_x + self.width() > geom.right():
                new_x = geom.right() - self.width()
                self._dx = -abs(self._dx)
                self._facing_right = False
                self._update_facing()

            if new_y < geom.top():
                new_y = geom.top()
                self._dy = abs(self._dy)
            elif new_y + self.height() > geom.bottom():
                new_y = geom.bottom() - self.height()
                self._dy = -abs(self._dy)

        self.move(new_x, new_y)

    def _update_facing(self) -> None:
        """Switch animation direction to match _facing_right."""
        if self._dragging:
            self._current_anim = self.sprite_manager.get_animation("drag", self._facing_right)
        elif self._override_anim_name:
            self._current_anim = self.sprite_manager.get_animation(
                self._override_anim_name, self._facing_right
            )
        elif self._current_behavior:
            self._current_anim, _ = self._load_behavior_anim(self._current_behavior)
        # Reset frame state to avoid stale index on the new animation
        self._frame_index = 0
        self._last_frame_time = time.monotonic()

    def pause_roaming(self) -> None:
        """Stop roaming movement (used during conversations)."""
        self._roaming = False
        self._dx = 0
        self._dy = 0

    def resume_roaming(self) -> None:
        """Resume roaming movement (unless timed override is active)."""
        if self._timed_override_until and time.monotonic() < self._timed_override_until:
            return
        self._roaming = True
        self._pick_and_start_behavior()

    def set_override_animation(self, name: str) -> None:
        """Force a specific animation (for pipeline states)."""
        self._override_anim_name = name
        self._current_anim = self.sprite_manager.get_animation(name, self._facing_right)
        self._frame_index = 0
        self._last_frame_time = time.monotonic()
        if self._current_anim and self._current_anim.frames:
            px = self._current_anim.frames[0]
            self.setFixedSize(px.width(), px.height())

    def clear_override(self) -> None:
        """Return to the behavior system (unless timed override is active)."""
        if self._timed_override_until and time.monotonic() < self._timed_override_until:
            # Timed override still active — restore timed animation instead of clearing
            if self._timed_anim_name:
                self._override_anim_name = self._timed_anim_name
                self._current_anim = self.sprite_manager.get_animation(
                    self._timed_anim_name, self._facing_right
                )
            return
        self._override_anim_name = None

    def set_timed_override(self, anim_name: str, seconds: int) -> None:
        """Set an animation override that persists for N seconds, even beyond conversation end."""
        self._timed_anim_name = anim_name
        self._timed_override_until = time.monotonic() + seconds
        self._roaming = False
        self._dx = 0
        self._dy = 0
        self.set_override_animation(anim_name)
        logger.info("Timed override: '%s' for %ds", anim_name, seconds)

    # Behaviors that are just normal movement — not interesting as tricks
    _BORING_NAMES = {
        "stand", "walk", "walk_wings", "trot", "fly", "hover", "sleep",
        "drag", "dash", "dash_ground", "gallop", "swim", "coaching",
        "training", "beep", "dizzy",
    }

    def do_trick(self) -> None:
        """Pick a cool/interesting behavior from the current character and play it.

        Only picks visually distinct behaviors: ones with effects, or stationary
        animations that aren't generic walk/stand/sleep. Skips boring movement.
        """
        # Priority 1: behaviors with visual effects (rainboom, crystalspark, etc.)
        with_effects = [
            b for b in self.behavior_manager.behaviors.values()
            if b.effects and b.movement != MovementType.DRAGGED
        ]

        # Priority 2: stationary special animations (gala dress, rage, makerain, etc.)
        specials = []
        for b in self.behavior_manager.behaviors.values():
            name_lower = b.name.lower()
            if name_lower in self._BORING_NAMES:
                continue
            if b.movement == MovementType.DRAGGED:
                continue
            if any(name_lower.startswith(p) for p in ("theme", "conga", "banner", "ride")):
                continue
            # Only stationary or very slow behaviors — no generic running/flying
            if b.speed <= 1.0:
                specials.append(b)

        # Combine both pools, preferring effects (weighted 3x)
        pool = with_effects * 3 + specials
        if not pool:
            pool = specials or with_effects

        if not pool:
            return

        trick = random.choice(pool)
        logger.info("Doing trick: '%s'", trick.name)
        self._start_behavior(trick)

    def move_to_region(self, region: str) -> None:
        """Move the pony to a named screen region on the pony's current monitor."""
        center = self.geometry().center()
        screen = QApplication.screenAt(center) or QApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        margin = 60
        w = self.width()
        h = self.height()

        targets = {
            "top_left": (geom.left() + margin, geom.top() + margin),
            "top_right": (geom.right() - w - margin, geom.top() + margin),
            "bottom_left": (geom.left() + margin, geom.bottom() - h - margin),
            "bottom_right": (geom.right() - w - margin, geom.bottom() - h - margin),
            "center": (geom.center().x() - w // 2, geom.center().y() - h // 2),
            "left": (geom.left() + margin, self.y()),
            "right": (geom.right() - w - margin, self.y()),
            "top": (self.x(), geom.top() + margin),
            "bottom": (self.x(), geom.bottom() - h - margin),
        }

        if region == "aside":
            mid = geom.center().x()
            region = "left" if self.x() + w // 2 < mid else "right"

        target = targets.get(region)
        if target:
            old_x = self.x()
            tx = max(geom.left(), min(target[0] + random.randint(-20, 20), geom.right() - w))
            ty = max(geom.top(), min(target[1] + random.randint(-20, 20), geom.bottom() - h))
            # Update facing BEFORE moving so it doesn't compare new pos to new pos
            if tx > old_x:
                self._facing_right = True
            elif tx < old_x:
                self._facing_right = False
            self.move(tx, ty)
            self._update_facing()
            logger.info("Moved to region '%s': (%d, %d)", region, tx, ty)
        else:
            logger.warning("Unknown MOVETO region: %r", region)

    def set_listening(self, listening: bool) -> None:
        """Show/hide the mic indicator."""
        self._show_mic = listening
        self._mic_pulse = 0.0

    def _draw_mic_indicator(self, painter: QPainter, sprite_w: int, sprite_h: int) -> None:
        """Draw a small pulsing mic icon near the sprite."""
        import math

        # Pulse animation (opacity oscillates)
        self._mic_pulse += 0.15
        pulse = 0.5 + 0.5 * math.sin(self._mic_pulse)

        # Position: top-right corner of sprite
        size = 22
        margin = 4
        cx = sprite_w - size // 2 - margin
        cy = size // 2 + margin

        # Pulsing red circle background
        bg_alpha = int(140 + 80 * pulse)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(220, 40, 40, bg_alpha)))
        painter.drawEllipse(QRectF(cx - size / 2, cy - size / 2, size, size))

        # Mic shape: rounded rect body + stem
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))

        # Mic body (rounded rectangle)
        mic_w, mic_h = 6, 9
        painter.drawRoundedRect(
            QRectF(cx - mic_w / 2, cy - mic_h / 2 - 1, mic_w, mic_h), 3, 3
        )

        # Mic cup (arc below body)
        painter.setBrush(Qt.NoBrush)
        cup_w = 9
        painter.drawArc(
            int(cx - cup_w / 2), int(cy - 1), cup_w, 10, 0, -180 * 16
        )

        # Stem (vertical line from cup to base)
        painter.drawLine(int(cx), int(cy + 4), int(cx), int(cy + 7))

        # Base (small horizontal line)
        painter.drawLine(int(cx - 3), int(cy + 7), int(cx + 3), int(cy + 7))

    def get_anchor_point(self) -> tuple[int, int, int]:
        """Get (center_x, top_y, sprite_height) for speech bubble positioning."""
        return self.x() + self.width() // 2, self.y(), self.height()

    def get_mouth_position(self) -> tuple[int, int]:
        """Get approximate mouth position in screen coordinates.

        MLP Desktop Ponies sprites follow a consistent template where the mouth
        is roughly at 75% width from the back, 65% height from top.
        """
        if self._facing_right:
            mouth_x = self.x() + int(self.width() * 0.75)
        else:
            mouth_x = self.x() + int(self.width() * 0.25)
        mouth_y = self.y() + int(self.height() * 0.65)
        return mouth_x, mouth_y

    def start_grab_run(self) -> None:
        """Force the pony into a fast run/trot animation with random direction changes."""
        self._grab_running = True
        self._roaming = True
        # Cancel any overrides
        self._override_anim_name = None
        self._timed_override_until = 0.0
        self._timed_anim_name = None
        # Find a running/walking behavior for the animation
        walk_behavior = None
        for name in ("trot", "walk", "gallop", "dash", "dash_ground", "walk_wings"):
            if name in self.behavior_manager.behaviors:
                walk_behavior = self.behavior_manager.behaviors[name]
                break
        if walk_behavior:
            self._current_behavior = walk_behavior
            self._behavior_start_time = time.monotonic()
            self._behavior_duration = 999.0  # won't expire during grab
            self._current_anim, _ = self._load_behavior_anim(walk_behavior)
            self._frame_index = 0
            self._last_frame_time = time.monotonic()
            if self._current_anim and self._current_anim.frames:
                px = self._current_anim.frames[0]
                self.setFixedSize(px.width(), px.height())
        else:
            self._pick_and_start_behavior()
            self._behavior_duration = 999.0
        # Fast movement
        speed = 5
        self._dx = speed if self._facing_right else -speed
        self._dy = 0
        # Schedule random direction changes every 1-2s
        if self._grab_run_timer is not None:
            self._grab_run_timer.stop()
            self._grab_run_timer.deleteLater()
        self._grab_run_timer = QTimer(self)
        self._grab_run_timer.timeout.connect(self._grab_run_change_dir)
        self._grab_run_timer.start(random.randint(800, 1800))

    def _grab_run_change_dir(self) -> None:
        """Change direction randomly during a grab run."""
        if not self._grab_running:
            return
        self._facing_right = not self._facing_right
        speed = 5
        self._dx = speed if self._facing_right else -speed
        self._dy = random.choice([-2, -1, 0, 1, 2])
        self._update_facing()
        # Keep behavior alive
        self._behavior_start_time = time.monotonic()
        self._behavior_duration = 999.0
        if self._grab_run_timer:
            self._grab_run_timer.setInterval(random.randint(800, 1800))

    def stop_grab_run(self) -> None:
        """Stop the grab run and return to normal roaming."""
        self._grab_running = False
        if self._grab_run_timer:
            self._grab_run_timer.stop()
            self._grab_run_timer = None
        self._roaming = True
        self._pick_and_start_behavior()

    def start_drag_walk(self) -> None:
        """Start a slow backward walk — pony is 'dragging' something at its mouth.

        The pony walks backward (opposite of facing direction) at a slow pace,
        using a walk/trot animation.  Used for tab-drag behavior.
        """
        self._drag_walking = True
        self._roaming = True
        self._override_anim_name = None
        self._timed_override_until = 0.0
        self._timed_anim_name = None
        # Find a walking behavior
        walk_behavior = None
        for name in ("walk", "trot", "walk_wings"):
            if name in self.behavior_manager.behaviors:
                walk_behavior = self.behavior_manager.behaviors[name]
                break
        if walk_behavior:
            self._current_behavior = walk_behavior
            self._behavior_start_time = time.monotonic()
            self._behavior_duration = 999.0
            self._current_anim, _ = self._load_behavior_anim(walk_behavior)
            self._frame_index = 0
            self._last_frame_time = time.monotonic()
            if self._current_anim and self._current_anim.frames:
                px = self._current_anim.frames[0]
                self.setFixedSize(px.width(), px.height())
        else:
            self._pick_and_start_behavior()
            self._behavior_duration = 999.0
        # Walk BACKWARD (opposite of facing) — slow, like dragging something
        speed = 2
        self._dx = -speed if self._facing_right else speed
        self._dy = 0

    def stop_drag_walk(self) -> None:
        """Stop the drag walk and return to normal roaming."""
        self._drag_walking = False
        self._roaming = True
        self._pick_and_start_behavior()

    # ── Mouse interaction ───────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._show_mic:
            # Clicking while listening → stop recording, process what we have
            self._show_mic = False  # Clear immediately so next click can drag
            self.listen_interrupted.emit()
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = event.globalPos() - self.pos()
            self._was_roaming_before_drag = self._roaming or bool(self._timed_override_until)
            self._roaming = False
            self._dx = 0
            self._dy = 0
            # Cancel any timed override — user grabbed the pony, they're in control now
            if self._timed_override_until:
                self._timed_override_until = 0.0
                self._timed_anim_name = None
                self._override_anim_name = None
            # Switch to drag animation
            self._current_anim = self.sprite_manager.get_animation("drag", self._facing_right)
            self._frame_index = 0
            self._last_frame_time = time.monotonic()
            if self._current_anim and self._current_anim.frames:
                self.setFixedSize(
                    self._current_anim.frames[0].width(),
                    self._current_anim.frames[0].height(),
                )
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            new_pos = event.globalPos() - self._drag_offset
            self.move(new_pos)
            # Update facing based on drag direction
            if event.globalPos().x() > self.x() + self.width() // 2:
                if not self._facing_right:
                    self._facing_right = True
                    self._update_facing()
            else:
                if self._facing_right:
                    self._facing_right = False
                    self._update_facing()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            if self._was_roaming_before_drag:
                self._roaming = True
                self._pick_and_start_behavior()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            # Cancel any drag state from the first press
            self._dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            from PyQt5.QtWidgets import QInputDialog, QLineEdit
            text, ok = QInputDialog.getText(
                self, "Talk to the pony",
                "Type your message (or cancel for voice):",
                QLineEdit.Normal, "",
            )
            if ok and text.strip():
                self.text_message.emit(text.strip())
            elif ok:
                # User hit OK with empty text — treat as voice conversation
                self.conversation_requested.emit()
            else:
                # User cancelled — start voice conversation
                self.conversation_requested.emit()

    def set_menu_builder(self, builder) -> None:
        """Set the context menu builder (called from main.py after all components are ready)."""
        self._menu_builder = builder

    def contextMenuEvent(self, event) -> None:
        if self._menu_builder:
            menu = self._menu_builder.build(self)
        else:
            menu = QMenu(self)
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(QApplication.quit)
            menu.addAction(quit_action)
        menu.exec_(event.globalPos())
