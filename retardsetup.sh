#!/usr/bin/env bash
# bonziPONY one-shot Linux installer (Debian/Ubuntu/Mint).
# Mirrors retardsetup.bat for non-Windows users.
#
# Usage:
#   bash retardsetup.sh              # install deps + run
#   bash retardsetup.sh --no-launch  # install deps, don't start the app
#   bash retardsetup.sh --no-update  # skip the git pull
#
# Re-running is safe: every step is idempotent.

set -u  # fail on unset variables; we handle errors per-step manually

# ── Pretty printing ───────────────────────────────────────────────
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
PINK=$'\033[1;35m'
CYAN=$'\033[1;36m'
RESET=$'\033[0m'
say()  { printf "%s\n" "  $*"; }
ok()   { printf "%s%s%s\n" "$GREEN" "  [OK]   $*" "$RESET"; }
warn() { printf "%s%s%s\n" "$YELLOW" "  [WARN] $*" "$RESET"; }
err()  { printf "%s%s%s\n" "$RED"   "  [ERR]  $*" "$RESET"; }
banner() {
    printf "%s\n" "$PINK"
    printf "%s\n" "  ============================================"
    printf "%s\n" "    bonziPONY v1.69 - One-Click Setup (Linux)"
    printf "%s\n" "  ============================================"
    printf "%s\n" "$RESET"
}

# ── Args ──────────────────────────────────────────────────────────
NO_LAUNCH=0
NO_UPDATE=0
for arg in "$@"; do
    case "$arg" in
        --no-launch) NO_LAUNCH=1 ;;
        --no-update) NO_UPDATE=1 ;;
        -h|--help)
            echo "Usage: bash retardsetup.sh [--no-launch] [--no-update]"
            exit 0 ;;
        *) warn "Unknown argument: $arg (ignoring)" ;;
    esac
done

# ── cd to script directory ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

banner

# ── Refuse to run as root (venvs owned by root are a footgun) ─────
if [[ $EUID -eq 0 ]]; then
    err "Don't run this as root. Run it as your normal user — sudo is invoked"
    err "only for the apt steps. Re-run without sudo."
    exit 1
fi

# ── Detect distro ─────────────────────────────────────────────────
DISTRO=""
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO="${ID:-unknown}"
    DISTRO_LIKE="${ID_LIKE:-}"
else
    err "/etc/os-release not found — can't detect distro."
    exit 1
fi

case "$DISTRO" in
    debian|ubuntu|linuxmint|pop|elementary|zorin|kali|raspbian) APT=1 ;;
    *)
        if [[ "$DISTRO_LIKE" == *"debian"* || "$DISTRO_LIKE" == *"ubuntu"* ]]; then
            APT=1
        else
            APT=0
        fi
        ;;
esac

if [[ "$APT" -ne 1 ]]; then
    warn "Detected distro: $DISTRO (not Debian/Ubuntu)."
    warn "I can't auto-install system packages here. Install the equivalents:"
    say  "    python3-venv python3-dev portaudio19-dev libasound2-dev"
    say  "    libxcb-cursor0 libxkbcommon-x11-0 libgl1 wmctrl xdotool"
    say  "    xprintidle xclip xdg-utils tesseract-ocr ffmpeg git"
    say  ""
    read -r -p "Continue anyway and try to install Python deps? [y/N] " yn
    case "$yn" in [yY]*) ;; *) exit 1 ;; esac
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 1: AUTO-UPDATE if this is a git checkout
# ══════════════════════════════════════════════════════════════════
if [[ "$NO_UPDATE" -ne 1 ]] && [[ -d .git ]] && command -v git >/dev/null 2>&1; then
    say "Checking for updates..."
    if git fetch origin >/dev/null 2>&1; then
        LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "")
        REMOTE=$(git ls-remote origin HEAD 2>/dev/null | awk '{print $1}')
        if [[ -n "$LOCAL" && -n "$REMOTE" && "$LOCAL" != "$REMOTE" ]]; then
            say "Update available — pulling..."
            if git pull --ff-only origin master >/dev/null 2>&1; then
                ok "Updated to latest version."
            else
                warn "ff-only pull failed; trying stash + pull..."
                git stash --include-untracked >/dev/null 2>&1 || true
                if git pull --ff-only origin master >/dev/null 2>&1; then
                    ok "Updated. Local changes were stashed."
                else
                    warn "Auto-pull failed. Continuing with current version."
                    git stash pop >/dev/null 2>&1 || true
                fi
            fi
        else
            ok "Already up to date."
        fi
    else
        warn "Could not check for updates (no internet?). Continuing..."
    fi
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 2: System packages via apt
# ══════════════════════════════════════════════════════════════════
APT_PACKAGES=(
    python3-venv python3-dev python3-pip
    portaudio19-dev libasound2-dev
    libxcb-cursor0 libxkbcommon-x11-0 libgl1
    wmctrl xdotool xprintidle xclip xdg-utils
    tesseract-ocr ffmpeg git
)

if [[ "$APT" -eq 1 ]]; then
    # Build a minimal "missing" list so we don't call sudo unnecessarily.
    MISSING=()
    for pkg in "${APT_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            MISSING+=("$pkg")
        fi
    done
    if [[ "${#MISSING[@]}" -eq 0 ]]; then
        ok "All system packages already present."
    else
        say "Installing system packages: ${MISSING[*]}"
        say "(sudo will prompt for your password)"
        if sudo apt-get update -qq && sudo apt-get install -y "${MISSING[@]}"; then
            ok "System packages installed."
        else
            err "apt install failed. Try re-running, or install the packages by hand:"
            say "    sudo apt install ${MISSING[*]}"
            exit 1
        fi
    fi
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 3: Pick a Python interpreter (3.10 / 3.11 / 3.12 only)
# ══════════════════════════════════════════════════════════════════
PYTHON=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
        case "$ver" in
            3.10|3.11|3.12) PYTHON="$cand"; break ;;
        esac
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "No compatible Python found. PyQt5 and torch don't have wheels for 3.13+."
    err "Install Python 3.12 (or 3.11/3.10) via your package manager:"
    say "    sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
ok "Using $($PYTHON --version 2>&1)"

# ══════════════════════════════════════════════════════════════════
#  STEP 4: venv
# ══════════════════════════════════════════════════════════════════
VENV_DIR="venv"

# Nuke a stale venv built with the wrong Python version
if [[ -x "$VENV_DIR/bin/python" ]]; then
    if ! "$VENV_DIR/bin/python" -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" >/dev/null 2>&1; then
        warn "Existing venv uses an unsupported Python — rebuilding."
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    say "Creating virtualenv in ./$VENV_DIR ..."
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        err "venv creation failed. Make sure python3-venv is installed."
        exit 1
    fi
    ok "Virtualenv created."
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

"$PY" -m pip install --upgrade pip --quiet >/dev/null 2>&1 || warn "pip self-upgrade failed (continuing)."

# ══════════════════════════════════════════════════════════════════
#  STEP 5: Python dependencies
# ══════════════════════════════════════════════════════════════════
say "Installing Python packages from requirements.txt (this can take a few minutes)..."
if "$PIP" install -r requirements.txt; then
    ok "Python packages installed."
else
    err "pip install failed. Scroll up to see which package broke."
    err "Common causes: portaudio19-dev missing → pyaudio fails;"
    err "                python3-dev missing → C extensions fail."
    exit 1
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 6: Config + data directories
# ══════════════════════════════════════════════════════════════════
if [[ ! -f config.yaml ]] && [[ -f config.yaml.example ]]; then
    cp config.yaml.example config.yaml
    ok "Created config.yaml from template."
else
    ok "config.yaml already exists (not overwriting)."
fi

mkdir -p logs diary memory
ok "Data directories ready (logs/ diary/ memory/)."

# ══════════════════════════════════════════════════════════════════
#  STEP 7: Quick X11 sanity check
# ══════════════════════════════════════════════════════════════════
if [[ -n "${WAYLAND_DISPLAY:-}" && -z "${DISPLAY:-}" ]]; then
    warn "You're on Wayland — bonziPONY is X11-only on Linux."
    warn "Stay-on-top, global hotkeys, and wmctrl/xdotool may not work."
    warn "Log out and pick an X11/Xorg session from your login screen."
elif [[ -z "${DISPLAY:-}" ]]; then
    warn "\$DISPLAY is unset — are you in a graphical session?"
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 8: Done — launch unless --no-launch
# ══════════════════════════════════════════════════════════════════
echo
ok "Setup complete!"
echo
say "${CYAN}To run later:${RESET}"
say "    source venv/bin/activate && python main.py"
echo

if [[ "$NO_LAUNCH" -eq 1 ]]; then
    exit 0
fi

say "Launching bonziPONY..."
exec "$PY" main.py
