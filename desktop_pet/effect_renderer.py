"""Handles rendering visual effects (Sonic Rainboom, apple trails, mail drops)
as overlay sprites.

Trail effects (non-follow, dont_repeat=False) spawn new instances periodically
at the pony's current position as it moves — creating trails of apples,
letters, sparkles, etc. behind the pony.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PyQt5.QtGui import QPixmap

from desktop_pet.behavior_manager import BehaviorManager, EffectDef
from desktop_pet.sprite_manager import SpriteAnimation, SpriteManager

logger = logging.getLogger(__name__)


@dataclass
class ActiveEffect:
    """A currently-playing effect instance."""

    effect_def: EffectDef
    animation: SpriteAnimation
    frame_index: int = 0
    x: int = 0
    y: int = 0
    start_time: float = 0.0
    last_frame_time: float = 0.0
    origin_x: int = 0  # Where the effect was spawned
    origin_y: int = 0
    facing_right: bool = True


@dataclass
class _TrailState:
    """Tracks spawning state for a repeating trail effect."""
    effect_def: EffectDef
    behavior_name: str
    last_spawn_time: float = 0.0
    spawn_interval: float = 0.0  # derived from delay field


class EffectRenderer:
    """Manages active visual effects that overlay the main sprite."""

    # Maximum trail instances per effect (prevents runaway memory on long behaviors)
    _MAX_TRAIL_INSTANCES = 12

    def __init__(self, sprite_manager: SpriteManager, behavior_manager: BehaviorManager) -> None:
        self.sprite_manager = sprite_manager
        self.behavior_manager = behavior_manager
        self.active_effects: List[ActiveEffect] = []
        # Trail effects: keyed by effect name, tracks periodic re-spawning
        self._trail_states: Dict[str, _TrailState] = {}
        # Currently active behavior name (set by trigger_effects, cleared by clear)
        self._current_behavior: Optional[str] = None

    def trigger_effects(
        self,
        behavior_name: str,
        facing_right: bool,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
    ) -> None:
        """Start all effects associated with a behavior."""
        # Clear trail states from previous behavior
        if behavior_name != self._current_behavior:
            self._trail_states.clear()
        self._current_behavior = behavior_name

        effect_defs = self.behavior_manager.get_effects_for(behavior_name)
        if not effect_defs:
            return

        now = time.monotonic()
        for edef in effect_defs:
            self._spawn_effect(edef, facing_right, sprite_x, sprite_y, sprite_w, sprite_h, now)

            # Set up trail spawning for repeating, non-follow effects
            # These are the "drop items behind me as I move" effects
            if not edef.follow and not edef.dont_repeat:
                interval = max(edef.delay, 0.4)  # at least 0.4s between spawns
                self._trail_states[edef.name] = _TrailState(
                    effect_def=edef,
                    behavior_name=behavior_name,
                    last_spawn_time=now,
                    spawn_interval=interval,
                )

    def _spawn_effect(
        self,
        edef: EffectDef,
        facing_right: bool,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
        now: float,
    ) -> None:
        """Spawn a single effect instance at the given sprite position."""
        img_file = edef.right_image if facing_right else edef.left_image
        anim_key = f"effect_{edef.name}_{'right' if facing_right else 'left'}"
        anim = self.sprite_manager.load_animation(anim_key, img_file)
        if not anim.frames:
            return

        ex, ey = self._calc_position(
            edef, facing_right, sprite_x, sprite_y, sprite_w, sprite_h, anim
        )

        effect = ActiveEffect(
            effect_def=edef,
            animation=anim,
            frame_index=0,
            x=ex,
            y=ey,
            start_time=now,
            last_frame_time=now,
            origin_x=sprite_x + sprite_w // 2,
            origin_y=sprite_y + sprite_h // 2,
            facing_right=facing_right,
        )
        self.active_effects.append(effect)
        logger.debug("Spawned effect '%s' at (%d, %d)", edef.name, ex, ey)

    def tick(
        self,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
        facing_right: bool = True,
    ) -> List[Tuple[QPixmap, int, int]]:
        """Update effects and return list of (pixmap, x, y) to render.

        Also handles periodic re-spawning of trail effects at the pony's
        current position.
        """
        now = time.monotonic()

        # ── Trail re-spawning ──────────────────────────────────────────────
        for trail in self._trail_states.values():
            if now - trail.last_spawn_time >= trail.spawn_interval:
                # Count existing instances of this effect
                count = sum(
                    1 for e in self.active_effects if e.effect_def.name == trail.effect_def.name
                )
                if count < self._MAX_TRAIL_INSTANCES:
                    self._spawn_effect(
                        trail.effect_def, facing_right,
                        sprite_x, sprite_y, sprite_w, sprite_h, now,
                    )
                    trail.last_spawn_time = now

        # ── Update active effects ──────────────────────────────────────────
        results: List[Tuple[QPixmap, int, int]] = []
        still_active: List[ActiveEffect] = []

        for eff in self.active_effects:
            # Skip if delay hasn't elapsed yet
            if now < eff.start_time:
                still_active.append(eff)
                continue

            # Check if duration expired (0 = infinite, lasts until cleared)
            if eff.effect_def.duration > 0:
                elapsed = now - eff.start_time
                if elapsed > eff.effect_def.duration:
                    continue  # Expired, don't keep

            anim = eff.animation
            if not anim.frames:
                continue

            # Advance frame based on delay
            frame_delay = anim.delays[eff.frame_index] / 1000.0
            if now - eff.last_frame_time >= frame_delay:
                eff.frame_index = (eff.frame_index + 1) % len(anim.frames)
                eff.last_frame_time = now

            # Update position if the effect follows the sprite
            if eff.effect_def.follow:
                eff.x, eff.y = self._calc_position(
                    eff.effect_def,
                    eff.facing_right,
                    sprite_x,
                    sprite_y,
                    sprite_w,
                    sprite_h,
                    anim,
                )

            pixmap = anim.frames[eff.frame_index]
            results.append((pixmap, eff.x, eff.y))
            still_active.append(eff)

        self.active_effects = still_active
        return results

    def clear(self) -> None:
        """Remove all active effects and trail states."""
        self.active_effects.clear()
        self._trail_states.clear()
        self._current_behavior = None

    @staticmethod
    def _parse_anchor(placement: str, sprite_x: int, sprite_y: int,
                      sprite_w: int, sprite_h: int) -> Tuple[int, int]:
        """Parse a placement string (may be compound like 'Bottom_Right')
        and return the anchor point on the sprite."""
        p = placement.lower().replace("-", "_")

        # Compound placements (e.g. "bottom_right", "top_left")
        if "_" in p:
            parts = p.split("_")
            # Figure out x and y components
            ax = sprite_x + sprite_w // 2  # default center
            ay = sprite_y + sprite_h // 2
            for part in parts:
                if part == "left":
                    ax = sprite_x
                elif part == "right":
                    ax = sprite_x + sprite_w
                elif part == "top":
                    ay = sprite_y
                elif part == "bottom":
                    ay = sprite_y + sprite_h
                elif part == "center":
                    pass  # keep default
            return ax, ay

        # Simple placements
        if p == "center":
            return sprite_x + sprite_w // 2, sprite_y + sprite_h // 2
        elif p == "right":
            return sprite_x + sprite_w, sprite_y + sprite_h // 2
        elif p == "left":
            return sprite_x, sprite_y + sprite_h // 2
        elif p == "top":
            return sprite_x + sprite_w // 2, sprite_y
        elif p == "bottom":
            return sprite_x + sprite_w // 2, sprite_y + sprite_h
        else:
            return sprite_x + sprite_w // 2, sprite_y + sprite_h // 2

    def _calc_position(
        self,
        edef: EffectDef,
        facing_right: bool,
        sprite_x: int,
        sprite_y: int,
        sprite_w: int,
        sprite_h: int,
        anim: SpriteAnimation,
    ) -> Tuple[int, int]:
        """Calculate effect position based on placement and centering rules.

        Supports compound placements like 'Bottom_Right', 'Top_Left', etc.
        """
        if not anim.frames:
            return sprite_x, sprite_y

        eff_w = anim.frames[0].width()
        eff_h = anim.frames[0].height()

        placement = edef.right_placement if facing_right else edef.left_placement
        centering = edef.right_centering if facing_right else edef.left_centering

        # Get anchor point on the sprite
        anchor_x, anchor_y = self._parse_anchor(
            placement, sprite_x, sprite_y, sprite_w, sprite_h
        )

        # Offset the effect so the correct part aligns to the anchor
        offset_x, offset_y = self._parse_anchor(
            centering, 0, 0, eff_w, eff_h
        )
        ex = anchor_x - offset_x
        ey = anchor_y - offset_y

        return ex, ey
