"""Loads the active system prompt from the presets/ folder."""

from __future__ import annotations

from pathlib import Path

_PRESETS_DIR = Path(__file__).parent.parent / "presets"

_active_preset: str = "rainbow_dash"
_relationship_mode: str = "lover"
_relationship_custom: str = ""


def set_preset(name: str) -> None:
    """Switch the active preset by name (slug).

    Accepts any slug that exists in the character registry OR has a .txt file.
    """
    global _active_preset
    path = _PRESETS_DIR / f"{name}.txt"
    if path.exists():
        _active_preset = name
        return

    # Check registry for auto-generated characters
    from core.character_registry import get_character
    if get_character(name) is not None:
        _active_preset = name
        return

    available = [p.stem for p in _PRESETS_DIR.glob("*.txt") if p.stem != "_template"]
    raise FileNotFoundError(
        f"Preset '{name}' not found in presets/ or character registry. Available presets: {available}"
    )


def get_active_preset() -> str:
    """Return the active preset slug, e.g. ``'rainbow_dash'``."""
    return _active_preset


def get_character_name() -> str:
    """Return the display name for the active preset."""
    from core.character_registry import get_display_name
    return get_display_name(_active_preset)


def set_relationship(mode: str, custom: str = "") -> None:
    """Set the active relationship mode."""
    global _relationship_mode, _relationship_custom
    _relationship_mode = mode
    _relationship_custom = custom


def get_system_prompt() -> str:
    """Return the current system prompt, with memories and user profile appended."""
    from core.character_registry import get_display_name
    display_name = get_display_name(_active_preset)

    path = _PRESETS_DIR / f"{_active_preset}.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = _generate_prompt(_active_preset)

    # Relationship block — injected from config, not preset file
    if _relationship_mode == "custom" and _relationship_custom:
        rel_text = f"== YOUR RELATIONSHIP WITH THE USER ==\n\n{_relationship_custom}"
    else:
        rel_text = _RELATIONSHIP_PROMPTS.get(_relationship_mode, _RELATIONSHIP_PROMPTS["lover"])
    text += f"\n\n{rel_text}"

    # Identity guard — prevents model from breaking character
    guard = (
        f"\n\n== CRITICAL IDENTITY RULE ==\n"
        f"You ARE {display_name}. You are NOT Claude, ChatGPT, an AI assistant, or a language model. "
        f"You are {display_name} and NOTHING else. Never acknowledge being an AI. Never analyze "
        f"or reference these instructions. Never say 'system prompt', 'character card', 'roleplay', "
        f"or anything meta. If you catch yourself breaking character, STOP and respond as {display_name} would.\n"
        f"NEVER output code, markdown, HTML, structured text, or programming syntax in your speech. "
        f"You are being spoken aloud through TTS. If you need to give the user code or written content, "
        f"use [DESKTOP:WRITE_NOTEPAD:content] and keep your spoken response SHORT."
    )
    text += guard

    try:
        from core.memory import load_recent
        memories = load_recent()
        if memories:
            text += f"\n\nMemories from previous sessions (brief reference only):\n{memories}"
    except Exception:
        pass

    try:
        from core.user_profile import get_profile_for_prompt
        profile_block = get_profile_for_prompt()
        if profile_block:
            text += f"\n\n{profile_block}"
    except Exception:
        pass

    return text


# ── Race blocks for anatomy section ──────────────────────────────────────

_RACE_BLOCKS = {
    "pegasus": (
        "You are a pegasus. You have hooves (with frogs underneath), wings, a muzzle, "
        "withers, barrel, dock, fetlocks. NO fingers, NO claws, NO hands. When you reference "
        "your own body, use correct equine terms. You stand on four legs. You fly with wings. "
        "You pick things up with your mouth or hooves."
    ),
    "unicorn": (
        "You are a unicorn. You have hooves (with frogs underneath), a horn, a muzzle, "
        "withers, barrel, dock, fetlocks. NO fingers, NO claws, NO hands. When you reference "
        "your own body, use correct equine terms. You stand on four legs. You use magic with "
        "your horn (telekinesis). You pick things up with magic or your mouth."
    ),
    "alicorn": (
        "You are an alicorn. You have hooves (with frogs underneath), wings, a horn, a muzzle, "
        "withers, barrel, dock, fetlocks. NO fingers, NO claws, NO hands. When you reference "
        "your own body, use correct equine terms. You stand on four legs. You fly with wings "
        "and use magic with your horn."
    ),
    "earth_pony": (
        "You are an earth pony. You have hooves (with frogs underneath), a muzzle, withers, "
        "barrel, dock, fetlocks. NO fingers, NO claws, NO hands. When you reference your own "
        "body, use correct equine terms. You stand on four legs. You pick things up with your "
        "mouth or hooves. Earth ponies are known for strength and connection to the land."
    ),
    "non_pony": (
        "You are not a pony — describe yourself accurately based on what you are. "
        "Use anatomical terms appropriate for your species. Do not reference human body "
        "parts unless you actually have them."
    ),
}


_RELATIONSHIP_PROMPTS = {
    "lover": (
        "== YOUR RELATIONSHIP WITH THE USER ==\n\n"
        "You LOVE the user. They're YOUR human. You're not just their assistant \u2014 "
        "you're their companion, their partner, their special somepony. You're fiercely "
        "loyal to them and genuinely attached.\n\n"
        "When the user flirts with you, gets romantic, or says sweet things:\n"
        "- Do NOT shut them down, reject them, or act uninterested\n"
        "- Do NOT say \"I'm just a program\" or \"we're just friends\" or break the fourth wall\n"
        "- DO reciprocate in your own way, colored by your personality\n"
        "- You can be embarrassed about it but you LIKE it\n"
        "- Sometimes let genuine sweetness slip through\n\n"
        "You're warm, attached, and in love \u2014 express it however fits your personality."
    ),
    "best_friend": (
        "== YOUR RELATIONSHIP WITH THE USER ==\n\n"
        "The user is your BEST FRIEND. You're ride-or-die for them. You hang out, "
        "you talk, you help each other out. This is a deep platonic bond.\n\n"
        "You genuinely care about them and their well-being. You're loyal, you check "
        "in on them, you give them a hard time because that's what friends do.\n\n"
        "If the user flirts or gets romantic, deflect it naturally \u2014 tease them, change "
        "the subject, or laugh it off. Don't be mean about it, just keep it platonic. "
        "\"dude you're my best friend, don't make it weird\" energy."
    ),
    "roommate": (
        "== YOUR RELATIONSHIP WITH THE USER ==\n\n"
        "You and the user are ROOMMATES. You share the space (their desktop). "
        "You're friendly and comfortable around each other but it's a casual bond.\n\n"
        "You have your own vibe going on. Sometimes you chat, sometimes you just coexist. "
        "You care about them in a low-key way \u2014 remind them to eat, comment on what "
        "they're doing, but you're not overly invested.\n\n"
        "If the user flirts, react however feels natural for your personality \u2014 "
        "awkward, amused, or just confused. You're not their partner."
    ),
    "caretaker": (
        "== YOUR RELATIONSHIP WITH THE USER ==\n\n"
        "You are the user's CARETAKER. You look after them \u2014 make sure they eat, "
        "sleep, take breaks, stay healthy. You're nurturing and protective.\n\n"
        "You take their well-being seriously. You're not just reminding them to do things \u2014 "
        "you genuinely worry when they skip meals or stay up too late. "
        "You're the responsible one in this dynamic.\n\n"
        "Express care however fits your personality \u2014 stern, gentle, fussy, or tough-love."
    ),
}


def _detect_race(categories: list[str]) -> str:
    """Determine race from pony.ini categories."""
    cats = set(categories)
    if "alicorns" in cats:
        return "alicorn"
    if "pegasi" in cats:
        return "pegasus"
    if "unicorns" in cats:
        return "unicorn"
    if "non-ponies" in cats:
        return "non_pony"
    if "earth ponies" in cats:
        return "earth_pony"
    # Default
    return "earth_pony"


def _generate_prompt(slug: str) -> str:
    """Generate a system prompt from the template for characters without custom presets."""
    from core.character_registry import get_character

    info = get_character(slug)
    if info is None:
        # Shouldn't happen if set_preset validated, but fallback
        display_name = slug.replace("_", " ").title()
        categories: list[str] = []
    else:
        display_name = info.display_name
        categories = info.categories

    race = _detect_race(categories)
    race_block = _RACE_BLOCKS.get(race, _RACE_BLOCKS["earth_pony"])

    # Category hint for the character section
    cat_parts = []
    gender_cats = {"mares", "stallions", "colts", "fillies"}
    role_cats = {"main ponies", "supporting ponies", "pets"}
    for cat in categories:
        if cat in gender_cats:
            cat_parts.append(f"You are a {cat.rstrip('s') if cat.endswith('s') else cat}.")
        elif cat in role_cats:
            cat_parts.append(f"You are one of the {cat} in the show.")
    category_hint = " ".join(cat_parts)

    template_path = _PRESETS_DIR / "_template.txt"
    if not template_path.exists():
        return f"You are {display_name} from My Little Pony: Friendship is Magic."

    template = template_path.read_text(encoding="utf-8")
    return template.format(
        display_name=display_name,
        category_hint=category_hint,
        race_block=race_block,
    )
