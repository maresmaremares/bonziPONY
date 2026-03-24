@echo off
title bonziPONY Setup
color 0d
echo.
echo  ============================================
echo     bonziPONY v1.69 - One-Click Setup
echo  ============================================
echo.

:: ── Find the script's own directory and cd into it ───────────
cd /d "%~dp0"

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo.
    echo  1. Download Python from https://www.python.org/downloads/
    echo  2. CHECK "Add Python to PATH" at the bottom of the installer!
    echo  3. Reboot and run this script again.
    echo.
    echo  Recommended: Python 3.11 -- most compatible
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found.

:: ── Check Python version ────────────────────────────────────
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% LSS 3 (
    echo.
    echo  [ERROR] Python 2 is not supported. Install Python 3.10+.
    pause
    exit /b 1
)
if %PYMINOR% LSS 9 (
    echo  [WARN] Python 3.%PYMINOR% is old. Some packages may fail.
    echo         Recommended: Python 3.10, 3.11, or 3.12.
    echo.
)

:: ── Install uv if not available ──────────────────────────────
python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Installing uv package manager...
    python -m pip install uv --quiet
)
python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Failed to install uv. Run: pip install uv
    pause
    exit /b 1
)
echo  [OK] uv ready.

:: ── Create venv if it doesn't exist ──────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo.
    echo  Creating virtual environment...
    python -m uv venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

set PY=venv\Scripts\python.exe

:: ── Stage 1: Try lockfile install (hash-verified) ────────────
echo.
echo  [1/3] Installing dependencies (hash-verified lockfile)...
echo.
python -m uv pip install --require-hashes -r requirements-lock.txt --python %PY% >nul 2>&1
if errorlevel 1 goto :stage2
echo  [OK] Lockfile install succeeded.
goto :installdone

:: ── Stage 2: Prebuilt wheels only (no compilation) ───────────
:stage2
echo  [WARN] Lockfile install failed (normal for newer Python versions).
echo.
echo  [2/3] Installing from requirements (prebuilt wheels only)...
echo.
python -m uv pip install --only-binary :all: -r requirements.txt --python %PY%
if errorlevel 1 goto :stage3
echo  [OK] Prebuilt install succeeded.
goto :installdone

:: ── Stage 3: Allow compilation as last resort ────────────────
:stage3
echo.
echo  [WARN] Some packages missing prebuilt wheels for Python %PYVER%.
echo.
echo  [3/3] Retrying with source compilation allowed...
echo.
python -m uv pip install -r requirements.txt --python %PY%
if errorlevel 1 goto :diagnosefail
echo  [OK] Install succeeded (some packages compiled from source).
goto :installdone

:: ── Diagnose what went wrong ─────────────────────────────────
:diagnosefail
echo.
echo  ============================================
echo     INSTALL FAILED — DIAGNOSING...
echo  ============================================
echo.

set MISSING=0

%PY% -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyQt5 — GUI framework
    set MISSING=1
)

%PY% -c "import pyaudio" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyAudio — microphone input
    echo      ^> Try: %PY% -m pip install PyAudioWPatch
    set MISSING=1
)

%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo  [X] NumPy — math library
    set MISSING=1
)

%PY% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyTorch -- AI/ML, ~2GB download, may have timed out
    echo      ^> Try: %PY% -m pip install torch
    set MISSING=1
)

%PY% -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo  [X] OpenCV — vision/screenshots
    set MISSING=1
)

%PY% -c "import whisper" >nul 2>&1
if errorlevel 1 (
    echo  [X] Whisper — speech-to-text
    set MISSING=1
)

%PY% -c "import yaml" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyYAML — config parser
    set MISSING=1
)

if %MISSING%==0 (
    echo  All critical packages seem installed despite the error.
    echo  The failure may have been a non-critical optional package.
    echo  Attempting to launch anyway...
    echo.
    goto :sanitycheck
)

echo.
echo  ============================================
echo     HOW TO FIX
echo  ============================================
echo.
echo  "Microsoft Visual C++ 14.0 required" error?
echo  That means a package tried to compile from source.
echo  EASIEST FIX: Install Python 3.11 from python.org
echo  (3.11 has the most prebuilt packages available)
echo.
echo  Other common fixes:
echo  - Make sure you have a stable internet connection
echo  - Run this script as Administrator (right-click ^> Run as admin)
echo  - Delete the "venv" folder and run this script again
echo  - If torch timed out, run: %PY% -m pip install torch
echo.
echo  Still stuck? Ask in the Discord or open a GitHub issue
echo  with a screenshot of the error above this message.
echo.
pause
exit /b 1

:installdone

:: ── Sanity check ─────────────────────────────────────────────
:sanitycheck
echo.

%PY% -c "import yaml" >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Core dependency PyYAML missing. Install may have failed.
    echo  Try deleting the "venv" folder and running this script again.
    pause
    exit /b 1
)

:: Quick check for the main problem child
%PY% -c "import pyaudio" >nul 2>&1
if errorlevel 1 (
    echo  [WARN] PyAudio not installed — microphone input won't work.
    echo  Attempting fallback install of PyAudioWPatch...
    python -m uv pip install PyAudioWPatch --python %PY% >nul 2>&1
    %PY% -c "import pyaudio" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] PyAudio still missing. Voice features will be disabled.
        echo  You can try manually: %PY% -m pip install PyAudioWPatch
    ) else (
        echo  [OK] PyAudioWPatch installed as fallback.
    )
    echo.
)

:: ── Copy config if needed ────────────────────────────────────
if not exist "config.yaml" (
    echo.
    echo  No config.yaml found, copying from example...
    copy config.yaml.example config.yaml >nul
    echo  [OK] config.yaml created.
    echo.
    echo  IMPORTANT: Edit config.yaml with your API keys
    echo  Or use the right-click menu after launch to set them.
    echo.
)

:: ── Create empty dirs ────────────────────────────────────────
if not exist "memory" mkdir memory
if not exist "diary" mkdir diary
if not exist "logs" mkdir logs

:: ── Launch ───────────────────────────────────────────────────
echo.
echo  ============================================
echo     Setup complete! Launching bonziPONY...
echo  ============================================
echo.
echo  Right-click the pony for settings.
echo  Double-click the pony to type a message.
echo  Close this window to kill the pony.
echo.

%PY% main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] bonziPONY crashed. Check the error above.
    echo.
    pause
)
