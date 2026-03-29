# bonziPONY

An AI-powered desktop companion that puts a fully autonomous, voice-interactive pony on your Windows desktop. It listens for wake words, speaks back through TTS, monitors your screen, nags you about your responsibilities, and controls your desktop — all while trotting around as an animated sprite.

This is not a chatbot with a pony skin. It's a persistent agent that runs autonomously between conversations, tracks your tasks, enforces accountability, reacts to what's on your screen, and has opinions about your life choices.

## What it does

**Voice pipeline.** Say the character's name → it listens → transcribes with Whisper (local, offline) → sends to LLM → speaks the response through TTS → shows a speech bubble on screen. Full duplex conversation with configurable timeout. Double-click the sprite to start a conversation without a wake word.

**Autonomous agent loop.** Between conversations, the pony monitors your screen and your behavior. It creates directives (persistent goals) when you mention things you need to do — "I'm hungry", "I should shower", "I have homework" — and will nag you with escalating urgency until you do them. Urgency 1-3 is verbal nagging. 4-5 shakes your windows. 6+ starts closing your distractions and messing with your mouse.

**Enforcement mode.** When you say you're going to do something ("fine I'll go shower"), it asks how long you'll be, then monitors your mouse and keyboard. If you touch the computer while you're supposed to be away, it calls you out immediately.

**Directives, timers, and routines.** The LLM generates structured tags that the agent loop parses and executes:
- `[DIRECTIVE:eat something:5]` — persistent goal at urgency 5
- `[TIMER:21:00:time for bed]` — wall-clock triggered action
- `[ROUTINE:on_wake:drink water:4]` — fires every time you wake up
- `[ROUTINE:interval:stretch:3:2]` — fires every 2 hours
- `[ENFORCE:15]` — monitor for 15 minutes, verify task completion
- `[DONE:shower]` — mark a directive as completed

**Desktop control.** The pony can interact with your desktop through the LLM's output tags:
- Window management: close, minimize, maximize, snap left/right
- Volume control: up, down, mute
- Click at coordinates, type text, press hotkeys
- Open applications, browse URLs, scroll
- Security: configurable allowlist for apps, blocklist for dangerous hotkeys

**Vision.** Optional screen capture and webcam support. The pony can describe what's on your screen and react to it. Vision is sent to the LLM as part of the conversation context.

**Desktop pet.** 311 characters from the Desktop Ponies sprite library. Animated GIF behaviors parsed from pony.ini files — walking, flying, sleeping, hovering, and dozens more per character. Effects rendering, configurable scale, speech bubbles, right-click context menu for everything.

**Mane 6 hotswap.** Switch between Rainbow Dash, Twilight Sparkle, Pinkie Pie, Rarity, Applejack, and Fluttershy at runtime from the right-click menu. Each character has their own personality preset, wake phrases, and sprite set. Swap is instant — new sprites load, LLM history resets, wake word detection switches to the new character's phrases.

**User profile and memory.** The pony builds a persistent profile of the user over time — name, age, location, job, interests, personality traits, whatever comes up naturally in conversation. It also tracks ongoing events (upcoming interviews, exams, deadlines, goals) and follows up on them later. After every conversation, new facts are extracted and saved. Stale events are pruned on startup. The profile is injected into every prompt so the character genuinely remembers who you are across sessions.

**TTS options.** ElevenLabs for cloud TTS, or any OpenAI-compatible TTS endpoint (e.g. a local voice model server) for zero-cost local speech synthesis.

**Any LLM provider.** Anthropic Claude (native SDK), OpenAI, OpenRouter, DeepSeek, Groq, and any OpenAI-compatible API. Local model support for Ollama, LM Studio, llama.cpp, KoboldCPP, text-generation-webui, vLLM, and LocalAI. Bring your own endpoint with `base_url`.

## Architecture

```
main.py                          Entry point, wires everything together
├── core/
│   ├── pipeline.py              State machine: IDLE → ACKNOWLEDGE → LISTEN → THINK → SPEAK
│   ├── agent_loop.py            Autonomous behavior engine (directives, enforcement, screen monitoring)
│   ├── routines.py              Recurring reminder scheduler
│   ├── screen_monitor.py        Win32 window tracking (free, no API calls)
│   ├── memory.py                Session summaries persisted across restarts
│   ├── user_profile.py          Persistent user profile + event tracking
│   ├── config_loader.py         YAML → typed dataclasses + env var overrides
│   └── audio_utils.py           Audio utilities
├── llm/
│   ├── base.py                  Abstract LLMProvider interface
│   ├── factory.py               Provider routing and instantiation
│   ├── anthropic_provider.py    Anthropic Claude (native SDK, retry logic)
│   ├── openai_provider.py       OpenAI-compatible (12+ providers, retry logic, vision fallback)
│   ├── prompt.py                Preset loading, character identity
│   └── response_parser.py       Tag extraction from LLM output
├── desktop_pet/
│   ├── pet_window.py            PyQt5 transparent frameless window, roaming, animation
│   ├── pet_controller.py        Qt signal bridge for thread-safe GUI updates
│   ├── sprite_manager.py        GIF loading, caching, scaling, dynamic sprite mapping
│   ├── behavior_manager.py      pony.ini parsing, behavior selection
│   ├── effect_renderer.py       Visual effects overlay
│   ├── speech_bubble.py         Text bubble widget
│   └── context_menu.py          Right-click menu (character switch, settings, directives)
├── wake_word/
│   └── detector.py              Whisper-based keyword spotting with per-character phrases
├── stt/
│   └── transcriber.py           Speech-to-text (Whisper, local)
├── tts/
│   ├── elevenlabs_tts.py        ElevenLabs TTS → raw PCM → sounddevice playback
│   └── openai_compatible_tts.py OpenAI-compatible TTS endpoint (local voice models)
├── vision/
│   ├── camera.py                Webcam capture
│   └── screen.py                Screenshot capture (mss)
├── robot/
│   ├── desktop_controller.py    Window ops, volume, keyboard/mouse automation
│   └── actions.py               RobotAction enum
├── presets/                     Character system prompts (one .txt per character)
├── Ponies/                      311 sprite directories (pony.ini + GIF animations)
└── config.yaml                  All configuration (gitignored — use config.yaml.example)
```

## Quickstart

```
1. Download/clone this repo
2. Double-click retardsetup.bat
3. Wait for it to finish (it downloads everything for you)
4. Edit config.yaml with your API keys (or right-click the pony to set them)
```

That's it. The setup script handles Python, dependencies, everything. A pony appears on your desktop. Say its name to start talking. Double-click it if you don't have a mic set up yet. Right-click for settings.

## Setup

### Requirements

- Windows 10/11
- **Python 3.10, 3.11, or 3.12** (3.11 recommended — [direct download](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe))
  - **Python 3.13+ does NOT work** — PyQt5 and PyTorch don't have packages for it yet
  - The setup script will download Python 3.11 automatically if you don't have a compatible version
- A microphone (any USB or built-in mic works)
- An LLM API key — pick one:
  - [Anthropic](https://console.anthropic.com/) (Claude) — recommended
  - [OpenAI](https://platform.openai.com/)
  - [OpenRouter](https://openrouter.ai/) — access to multiple models with one key
  - Or run a local model (Ollama, LM Studio, etc.) — no key needed
- For voice output, pick one:
  - [ElevenLabs](https://elevenlabs.io/) API key — best quality cloud TTS
  - A local OpenAI-compatible TTS server — free, runs on your machine

### Step 1: Install

**Easiest way** — just double-click `retardsetup.bat`. It handles everything: downloads the right Python if you don't have it, creates a virtual environment, installs all dependencies, and launches the pony. You're done.

**Manual install** (if you prefer):

```bash
git clone https://github.com/maresmaremares/bonziPONY.git
cd bonziPONY
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Important:** Use Python 3.10-3.12. If `python --version` shows 3.13+, install [Python 3.11](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) first. Check "Add to PATH" in the installer.

### Step 2: Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` in any text editor and fill in your keys:

```yaml
llm:
  provider: "anthropic"       # or openai, openrouter, ollama, lmstudio, etc.
  model: "claude-sonnet-4-6"
  api_key: "your-api-key"
  preset: "rainbow_dash"      # which pony you want (see presets/ folder)
```

If using ElevenLabs for TTS:

```yaml
elevenlabs:
  api_key: "your-elevenlabs-key"
  voice_id: "your-voice-id"
```

If using a local TTS server instead, change the `tts` section:

```yaml
tts:
  provider: "openai_compatible"
  base_url: "http://localhost:8069/v1"
  model: "your-model"
  voice: "default"
```

**Don't want to put keys in a file?** Use environment variables instead — they override config.yaml:

```bash
set BONZI_LLM_API_KEY=your-api-key
set BONZI_ELEVENLABS_API_KEY=your-elevenlabs-key
set BONZI_ELEVENLABS_VOICE_ID=your-voice-id
```

Or use a `.env` file (copy `.env.example` → `.env`, fill it in, and `pip install python-dotenv`).

### Step 3: Pick your audio devices

```bash
python scripts/list_audio_devices.py
```

This prints a numbered list of your microphones and speakers. Set the index numbers in config.yaml:

```yaml
audio:
  input_device_index: 1    # your mic
  output_device_index: 3   # your speakers/headphones
```

Use `-1` for system defaults if you don't care or only have one of each.

### Step 4: Run

```bash
python main.py
```

You should see:

```
Rainbow Dash Desktop Pet is running!
  Wake phrases: hey dash, hey dashie, rainbow dash, dash
  Double-click the pet to start a conversation.
  Right-click for menu. Close to exit.
```

A pony sprite appears on your desktop and starts trotting around.

### How to use it

**Start a conversation:**
- Say the character's wake phrase out loud (e.g. "Hey Dash")
- Or double-click the sprite

**During a conversation:**
- Just talk naturally. The pony listens, thinks, and responds with voice + speech bubble.
- The conversation stays open for a few seconds after each response so you can keep talking.
- When you stop talking, it goes back to idle.

**Give it tasks:**
- "Remind me to eat in 30 minutes"
- "I need to do my homework" — it creates a directive and will nag you until you do it
- "I'm going to shower" — it asks how long, then monitors if you touch the computer early

**Right-click the sprite for:**
- Switching characters (Mane 6 hotswap)
- Viewing/managing directives
- Changing sprite scale
- Switching audio devices
- Quitting

**Keyboard/desktop control:**
- The pony can close, minimize, and shake your windows
- It can open apps, type text, click, and browse URLs
- All controlled by the LLM through structured tags — you don't configure this, it just does it when contextually appropriate
- Safety: `desktop_control.allowed_apps` and `desktop_control.blocked_hotkeys` in config.yaml

### Optional: using a local LLM (no API key)

Install [Ollama](https://ollama.ai/), pull a model, and point bonziPONY at it:

```yaml
llm:
  provider: "ollama"
  model: "llama3"
  api_key: ""
```

Works with any OpenAI-compatible local server (LM Studio, llama.cpp, KoboldCPP, text-generation-webui, vLLM, LocalAI). Just set `provider` and optionally `base_url` if it's not on the default port.

For a totally custom endpoint:

```yaml
llm:
  provider: "custom"
  model: "my-model"
  api_key: ""
  base_url: "http://192.168.1.100:8080/v1"
```

The only requirement is OpenAI chat completions format (`POST /chat/completions`). Vision gracefully degrades if the model doesn't support images.

## Configuration reference

### `wake_word`

| Key | Default | Description |
|-----|---------|-------------|
| `language` | `"en"` | Recognition language |
| `phrases` | `{}` | Per-character phrase overrides. Defaults are built in for all Mane 6. |

### `audio`

| Key | Default | Description |
|-----|---------|-------------|
| `input_device_index` | `-1` | Microphone index (`-1` = system default) |
| `output_device_index` | `-1` | Speaker index (`-1` = system default) |
| `vad_aggressiveness` | `2` | Voice activity detection sensitivity (0-3, higher = more aggressive) |
| `silence_duration_ms` | `800` | Silence duration to end recording |

### `whisper`

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`) |
| `language` | `"en"` | Transcription language |

### `llm`

| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `"openai"` | LLM provider name |
| `model` | `"gpt-4o"` | Model identifier |
| `api_key` | — | API key (or set `BONZI_LLM_API_KEY` env var; leave empty for local models) |
| `temperature` | `0.85` | Response randomness (0.0–1.0) |
| `max_tokens` | `256` | Max response length |
| `max_history_turns` | `10` | Conversation history depth |
| `base_url` | `null` | Custom API endpoint (or set `BONZI_LLM_BASE_URL` env var; auto-detected for known providers) |
| `preset` | `"rainbow_dash"` | Character preset filename (without `.txt`) |

**Supported providers:** `anthropic`, `openai`, `openrouter`, `deepseek`, `groq`, `ollama`, `lmstudio`, `llamacpp`, `koboldcpp`, `textgen`, `vllm`, `localai`, or any OpenAI-compatible endpoint via `base_url`.

### `elevenlabs`

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | — | ElevenLabs API key (or set `BONZI_ELEVENLABS_API_KEY` env var) |
| `voice_id` | — | Voice ID (or set `BONZI_ELEVENLABS_VOICE_ID` env var) |
| `model` | `"eleven_turbo_v2"` | TTS model |
| `output_format` | `"pcm_22050"` | Audio format |

### `tts`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable TTS |
| `provider` | `"elevenlabs"` | TTS provider: `"elevenlabs"` or `"openai_compatible"` |
| `base_url` | `"http://localhost:8069/v1"` | OpenAI-compatible TTS server URL |
| `model` | `"ponyvoicetool"` | Model name for OpenAI-compatible TTS |
| `voice` | `"default"` | Voice name for OpenAI-compatible TTS |
| `response_format` | `"pcm"` | Audio format for OpenAI-compatible TTS |
| `sample_rate` | `24000` | Sample rate for OpenAI-compatible TTS |

### `vision`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable vision capabilities |
| `device_index` | `0` | Webcam index |
| `screen_capture` | `true` | Enable screenshot capture |
| `screen_max_width` | `1280` | Downscale screenshots to this width |

### `conversation`

| Key | Default | Description |
|-----|---------|-------------|
| `timeout_s` | `60` | Seconds to stay in conversation mode after speaking |
| `listen_timeout_s` | `8` | Seconds to wait for follow-up speech |

### `desktop_control`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable desktop automation |
| `allowed_apps` | `[notepad, calculator, explorer, chrome, firefox]` | Apps the pony can open |
| `blocked_hotkeys` | `["ctrl:alt:delete"]` | Blocked keyboard shortcuts |
| `click_enabled` | `true` | Allow mouse clicks |
| `type_enabled` | `true` | Allow keyboard input |

### `agent`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable autonomous agent |
| `self_initiate` | `true` | Allow unprompted speech |
| `max_directives` | `3` | Max concurrent tracked goals |
| `base_check_interval_s` | `120` | Seconds between idle checks |
| `min_check_interval_s` | `30` | Minimum interval at max urgency |
| `self_initiate_interval_s` | `300` | Seconds between autonomous check-ins |
| `spontaneous_speech_min_s` | `120` | Minimum seconds between random commentary |
| `spontaneous_speech_max_s` | `300` | Maximum seconds between random commentary |
| `sustained_focus_threshold_s` | `900` | Flag sustained app focus after this many seconds |
| `distraction_keywords` | `[youtube, reddit, tiktok, ...]` | Window titles that count as distractions |

### `desktop_pet`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Show the desktop pet |
| `scale` | `2.0` | Sprite size multiplier |
| `speech_bubble` | `true` | Show text bubbles with TTS |

## Preset system

Character presets live in `presets/`. Each `.txt` file defines the character's entire personality, speech patterns, and behavior rules for the LLM. The preset is injected as the system prompt.

**Included presets:**

| Preset | File | Species |
|--------|------|---------|
| Rainbow Dash | `rainbow_dash.txt` | Pegasus |
| Twilight Sparkle | `twilight_sparkle.txt` | Alicorn |
| Pinkie Pie | `pinkie_pie.txt` | Earth pony |
| Rarity | `rarity.txt` | Unicorn |
| Applejack | `applejack.txt` | Earth pony |
| Fluttershy | `fluttershy.txt` | Pegasus |

Every preset contains:
- Voice rules (TTS-aware output — spoken words only, no stage directions)
- Anti-slop rules (explicit blocklist of AI-sounding patterns)
- Character personality and speech examples
- Equine anatomy rules (species-correct body references)
- Full tag documentation (actions, directives, timers, routines, enforcement, desktop commands)

**Custom presets.** Drop a `.txt` file in `presets/`, set `llm.preset` to the filename (without extension), and make sure there's a matching sprite directory in `Ponies/` with the title-cased name. The preset slug `my_oc` maps to `Ponies/My Oc/`.

## LLM tag system

The LLM generates structured tags inline with its spoken response. The response parser strips them before TTS and routes them to the appropriate handler.

```
Spoken text goes here. [DIRECTIVE:do homework:6] [CONVO:CONTINUE]
```

| Tag | Purpose | Example |
|-----|---------|---------|
| `[ACTION:X]` | Sprite animation | `[ACTION:WALK_FORWARD]`, `[ACTION:SIT]`, `[ACTION:SPIN]` |
| `[DIRECTIVE:goal:urgency]` | Create persistent goal | `[DIRECTIVE:eat food:5]` |
| `[TIMER:time:action]` | Schedule action at wall-clock time | `[TIMER:21:00:bedtime]` |
| `[ROUTINE:type:goal:urgency:param]` | Recurring reminder | `[ROUTINE:daily:stretch:3:14:00]` |
| `[DONE]` / `[DONE:keyword]` | Complete a directive | `[DONE:shower]` |
| `[ENFORCE:minutes]` | Monitor task completion | `[ENFORCE:15]` |
| `[CONVO:CONTINUE]` / `[CONVO:END]` | Conversation flow control | — |
| `[ACTION:CLOSE_WINDOW]` | Window management | Also: `MINIMIZE`, `MAXIMIZE`, `SNAP_LEFT/RIGHT` |
| `[ACTION:VOLUME_UP/DOWN/MUTE]` | Volume control | — |
| `[DESKTOP:CLICK:x:y]` | Click at coordinates | `[DESKTOP:CLICK:500:300]` |
| `[DESKTOP:TYPE:text]` | Type text | `[DESKTOP:TYPE:hello world]` |
| `[DESKTOP:HOTKEY:k1:k2]` | Press shortcut | `[DESKTOP:HOTKEY:ctrl:s]` |
| `[DESKTOP:OPEN:app]` | Open application | `[DESKTOP:OPEN:notepad]` |
| `[DESKTOP:BROWSE:url]` | Open URL | `[DESKTOP:BROWSE:youtube.com]` |
| `[DESKTOP:SCROLL:n]` | Scroll | `[DESKTOP:SCROLL:-3]` (down) |

## Escalation behavior

When the pony has active directives and the user ignores them, urgency increases over time:

| Urgency | Behavior |
|---------|----------|
| 1–3 | Verbal nagging |
| 4–5 | Shake distracting windows |
| 6 | Shake windows + pause media (YouTube, Spotify) or minimize |
| 7 | Shake + mess with mouse cursor |
| 8+ | Nuclear — shake everything, hijack mouse, close windows |

The pony always shakes before minimizing or closing. Escalation is graduated, not instant.

## Context menu

Right-click the pony sprite to access:

- **Directives** — view, add, or clear active goals
- **Character** — hotswap between Mane 6 (instant sprite + personality switch)
- **Scale** — resize the sprite (0.5x–4.0x)
- **Audio devices** — switch mic/speaker
- **Quit**

## Dependencies

```
PyQt5>=5.15          # Desktop window
Pillow>=10.0         # Sprite processing
SpeechRecognition    # Wake word detection
pyaudio              # Microphone input
elevenlabs>=1.0      # Cloud TTS
sounddevice          # Audio playback
anthropic>=0.20      # Claude API
openai>=1.14         # OpenAI-compatible APIs
opencv-python>=4.8   # Image processing
mss>=9.0             # Screenshot capture
pyautogui            # Desktop automation
pygetwindow          # Window management
pywin32>=306         # Win32 API
PyYAML>=6.0          # Config parsing
numpy>=1.24          # Numerical ops
openai-whisper       # Local STT
```

**Optional:**
```
python-dotenv        # .env file support
scipy                # Signal processing
librosa              # Audio features
```

## Platform

Windows only. Depends on `pywin32` for window manipulation, `win32gui` for idle time detection, and Win32 APIs for desktop control and screen monitoring.
