@echo off
setlocal enabledelayedexpansion
title bonziPONY Setup
color 0d
echo.
echo  ============================================
echo     bonziPONY v1.69 - One-Click Setup
echo  ============================================
echo.

:: ── Find the script's own directory and cd into it ───────────
cd /d "%~dp0"

:: ══════════════════════════════════════════════════════════════
::  AUTO-UPDATE: Pull newest version if this is a git repo
:: ══════════════════════════════════════════════════════════════
git --version >nul 2>&1
if errorlevel 1 goto :skip_update
if not exist ".git" goto :skip_update

echo  Checking for updates...
git fetch origin >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Could not check for updates ^(no internet?^). Continuing...
    goto :skip_update
)

:: Check if we're behind the remote - use ls-remote instead of @{u}
:: which breaks batch when no upstream is set
set "LOCAL="
set "REMOTE="
for /f "tokens=*" %%a in ('git rev-parse HEAD 2^>nul') do set "LOCAL=%%a"
for /f "tokens=1" %%a in ('git ls-remote origin HEAD 2^>nul') do set "REMOTE=%%a"
if not defined LOCAL goto :skip_update
if not defined REMOTE goto :skip_update
if "!LOCAL!"=="!REMOTE!" (
    echo  [OK] Already up to date.
    goto :skip_update
)
echo  [!!] Update available! Pulling latest version...
git pull --ff-only origin master >nul 2>&1
if not errorlevel 1 (
    echo  [OK] Updated to latest version!
    goto :skip_update
)
echo  [WARN] Auto-pull failed. Trying stash + pull...
git stash >nul 2>&1
git pull --ff-only origin master >nul 2>&1
if not errorlevel 1 (
    echo  [OK] Updated! Stashed local changes.
    goto :skip_update
)
echo  [WARN] Update failed. Continuing with current version.
git stash pop >nul 2>&1
echo.
:skip_update

:: ══════════════════════════════════════════════════════════════
::  STEP 1: Find Python 3.10, 3.11, or 3.12
::  (3.13+ does NOT work - PyQt5 and torch have no wheels)
:: ══════════════════════════════════════════════════════════════

set "PYTHON="

:: ── Method 1: Windows Python Launcher (py.exe) ──────────────
:: Installed by default with Python. Lets us pick exact versions.
for %%V in (3.11 3.10 3.12) do (
    if not defined PYTHON (
        for /f "tokens=*" %%p in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do (
            set "PYTHON=%%p"
        )
    )
)
if defined PYTHON goto :found_python

:: ── Method 2: Check default install paths ───────────────────
for %%V in (311 310 312) do (
    if not defined PYTHON (
        if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
            set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        )
    )
)
if defined PYTHON goto :found_python

for %%V in (311 310 312) do (
    if not defined PYTHON (
        if exist "C:\Python%%V\python.exe" (
            set "PYTHON=C:\Python%%V\python.exe"
        )
    )
)
if defined PYTHON goto :found_python

:: ── Method 3: Check "python" command, verify version ────────
for /f "tokens=*" %%p in ('python -c "import sys; v=sys.version_info; print(sys.executable) if 10<=v.minor<=12 and v.major==3 else exit(1)" 2^>nul') do (
    set "PYTHON=%%p"
)
if defined PYTHON goto :found_python

:: ══════════════════════════════════════════════════════════════
::  No compatible Python - download it automatically
:: ══════════════════════════════════════════════════════════════
echo  [!] No compatible Python found. Installing Python 3.11...
echo.

:: Pick 64-bit or 32-bit installer
set "PYURL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PYFILE=python-3.11.9-amd64.exe"
if "%PROCESSOR_ARCHITECTURE%"=="x86" if not defined PROCESSOR_ARCHITEW6432 (
    set "PYURL=https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe"
    set "PYFILE=python-3.11.9.exe"
)

echo  Downloading %PYFILE%...
echo  (about 25 MB - sit tight)
echo.
powershell -ExecutionPolicy Bypass -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; try { Invoke-WebRequest -Uri '%PYURL%' -OutFile '%TEMP%\%PYFILE%' -UseBasicParsing } catch { Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 goto :download_failed
if not exist "%TEMP%\%PYFILE%" goto :download_failed
echo  [OK] Downloaded.
echo.

echo  Installing Python 3.11.9...
echo  (a progress bar will appear - don't close it)
echo.
"%TEMP%\%PYFILE%" /passive PrependPath=1 Include_launcher=1 InstallLauncherAllUsers=0 Include_pip=1
if errorlevel 1 (
    echo  [ERROR] Python installer failed or was cancelled.
    echo  Try running this script as Administrator (right-click ^> Run as admin).
    echo.
    pause
    exit /b 1
)
echo  [OK] Python 3.11.9 installed!
echo.
del "%TEMP%\%PYFILE%" >nul 2>&1

:: Find the freshly installed Python
for /f "tokens=*" %%p in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%p"
if defined PYTHON goto :found_python

:: py launcher might not be in PATH yet - check default location
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found_python
)

:: PATH wasn't refreshed - need a restart of the script
echo  [!] Python was installed but this window can't see it yet.
echo  Close this window and double-click setup again. That's it.
echo.
pause
exit /b 1

:download_failed
echo  [ERROR] Could not download Python. Check your internet connection.
echo.
echo  If you're behind a firewall or your internet is weird, you can
echo  install Python manually. Download THIS specific file:
echo.
echo    %PYURL%
echo.
echo  Run the installer. CHECK "Add Python to PATH" at the bottom.
echo  Then run this script again.
echo.
echo  DO NOT download Python 3.13 from the python.org front page.
echo  It does NOT work. You need 3.11.
echo.
pause
exit /b 1

:: ══════════════════════════════════════════════════════════════
::  STEP 2: Set up virtual environment
:: ══════════════════════════════════════════════════════════════
:found_python
for /f "tokens=*" %%v in ('"!PYTHON!" --version 2^>^&1') do echo  [OK] Using %%v

:: ── Check if existing venv was made with an incompatible Python ──
if not exist "venv\Scripts\python.exe" goto :venv_ok
venv\Scripts\python.exe -c "import sys; exit(0 if 10<=sys.version_info.minor<=12 else 1)" >nul 2>&1
if not errorlevel 1 goto :venv_ok
echo  [WARN] Existing venv uses wrong Python version. Rebuilding...
rmdir /s /q venv >nul 2>&1
:venv_ok

:: ── Install/check uv ────────────────────────────────────────
set "USE_UV=0"
"!PYTHON!" -m uv --version >nul 2>&1
if not errorlevel 1 set "USE_UV=1" & goto :uv_done
echo.
echo  Installing uv package manager...
"!PYTHON!" -m pip install uv --quiet >nul 2>&1
"!PYTHON!" -m uv --version >nul 2>&1
if not errorlevel 1 set "USE_UV=1" & goto :uv_done
echo  [WARN] uv not available. Using pip (slower but fine).
:uv_done
if "!USE_UV!"=="1" echo  [OK] uv ready.

:: ── Create venv ─────────────────────────────────────────────
if exist "venv\Scripts\python.exe" echo  [OK] Virtual environment exists. & goto :venv_ready
echo.
echo  Creating virtual environment...
if not "!USE_UV!"=="1" goto :venv_pip
"!PYTHON!" -m uv venv venv
goto :venv_check
:venv_pip
"!PYTHON!" -m venv venv
:venv_check
if errorlevel 1 echo  [ERROR] Failed to create venv. Delete "venv" folder and rerun. & pause & exit /b 1
echo  [OK] Virtual environment created.
:venv_ready

set "PY=venv\Scripts\python.exe"

:: ══════════════════════════════════════════════════════════════
::  STEP 3: Install dependencies (3-stage fallback)
:: ══════════════════════════════════════════════════════════════

:: ── Stage 1: Hash-verified lockfile (fastest, most reliable) ─
echo.
echo  [1/3] Installing dependencies (lockfile)...
if not "!USE_UV!"=="1" goto :s1_pip
"!PYTHON!" -m uv pip install --require-hashes -r requirements-lock.txt --python %PY% >nul 2>&1
goto :s1_check
:s1_pip
%PY% -m pip install -r requirements-lock.txt --quiet >nul 2>&1
:s1_check
if not errorlevel 1 echo  [OK] Lockfile install succeeded. & goto :installdone

:: ── Stage 2: Prebuilt wheels only (no C++ compiler needed) ──
echo  [WARN] Lockfile didn't match your Python - trying prebuilt packages...
echo.
echo  [2/3] Installing from requirements (prebuilt wheels)...
echo.
if not "!USE_UV!"=="1" goto :s2_pip
"!PYTHON!" -m uv pip install --only-binary :all: -r requirements.txt --python %PY%
goto :s2_check
:s2_pip
%PY% -m pip install --only-binary :all: -r requirements.txt
:s2_check
if not errorlevel 1 echo  [OK] Prebuilt install succeeded. & goto :installdone

:: ── Stage 3: Allow source compilation (needs C++ build tools) ─
echo.
echo  [WARN] Some packages need to compile from source...
echo.
echo  [3/3] Retrying with compilation allowed...
echo.
if not "!USE_UV!"=="1" goto :s3_pip
"!PYTHON!" -m uv pip install -r requirements.txt --python %PY%
goto :s3_check
:s3_pip
%PY% -m pip install -r requirements.txt
:s3_check
if not errorlevel 1 echo  [OK] Install succeeded. & goto :installdone

:: ══════════════════════════════════════════════════════════════
::  INSTALL FAILED - figure out what's missing
:: ══════════════════════════════════════════════════════════════
echo.
echo  ============================================
echo     INSTALL FAILED - DIAGNOSING...
echo  ============================================
echo.

set MISSING=0
set CORE_OK=1

%PY% -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyQt5 - the window/graphics library
    set MISSING=1
    set CORE_OK=0
)

%PY% -c "import yaml" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyYAML - config file reader
    set MISSING=1
    set CORE_OK=0
)

%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo  [X] NumPy - math library
    set MISSING=1
    set CORE_OK=0
)

%PY% -c "import pyaudio" >nul 2>&1
if errorlevel 1 (
    %PY% -c "import pyaudiowpatch" >nul 2>&1
    if errorlevel 1 (
        echo  [X] PyAudio - microphone input
        set MISSING=1
    )
)

%PY% -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo  [X] OpenCV - vision/screenshots
    set MISSING=1
)

%PY% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo  [X] PyTorch - AI engine (~2GB, may have timed out on slow internet)
    set MISSING=1
)

%PY% -c "import whisper" >nul 2>&1
if errorlevel 1 (
    echo  [X] Whisper - speech recognition (needs PyTorch)
    set MISSING=1
)

:: If only torch/whisper failed, the pony can still run (just no local STT)
if "!CORE_OK!"=="1" if !MISSING!==1 (
    echo.
    echo  Core packages are installed. Only optional packages failed.
    echo  bonziPONY should still work (local speech recognition may not).
    echo.
    echo  If torch timed out, you can install it later:
    echo    %PY% -m pip install torch
    echo.
    goto :sanitycheck
)

if !MISSING!==0 (
    echo  Everything looks installed despite the error above.
    echo  Attempting to launch anyway...
    echo.
    goto :sanitycheck
)

echo.
echo  ============================================
echo     HOW TO FIX
echo  ============================================
echo.
echo  MOST LIKELY FIX:
echo    1. Delete the "venv" folder in this directory
echo    2. Run this script again
echo.
echo  If that doesn't work:
echo    - Make sure you have internet
echo    - Try right-click ^> Run as Administrator
echo    - If torch keeps timing out (it's a 2GB download), just keep
echo      running the script - it'll pick up where it left off
echo.
echo  Still stuck? Post a screenshot of this window in the Discord.
echo.
pause
exit /b 1

:: ══════════════════════════════════════════════════════════════
::  STEP 4: Final checks and launch
:: ══════════════════════════════════════════════════════════════
:installdone

:sanitycheck
echo.

:: ── PyYAML is the one thing we absolutely need ──────────────
%PY% -c "import yaml" >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Critical dependency missing. Delete "venv" folder and rerun.
    pause
    exit /b 1
)

:: ── PyAudio fallback ────────────────────────────────────────
:: PyAudioWPatch installs as "pyaudiowpatch" not "pyaudio" - check both
%PY% -c "import pyaudio" >nul 2>&1
if not errorlevel 1 goto :pyaudio_ok
%PY% -c "import pyaudiowpatch" >nul 2>&1
if not errorlevel 1 echo  [OK] PyAudioWPatch already installed. & goto :pyaudio_ok
echo  Installing PyAudioWPatch...
if not "!USE_UV!"=="1" goto :pyaudio_pip
"!PYTHON!" -m uv pip install PyAudioWPatch --python %PY% >nul 2>&1
goto :pyaudio_verify
:pyaudio_pip
%PY% -m pip install PyAudioWPatch --quiet >nul 2>&1
:pyaudio_verify
%PY% -c "import pyaudiowpatch" >nul 2>&1
if not errorlevel 1 echo  [OK] PyAudioWPatch installed. & goto :pyaudio_ok
echo  [WARN] PyAudio install failed. Voice input won't work,
echo         but everything else will. You can install it later:
echo         %PY% -m pip install PyAudioWPatch
echo.
:pyaudio_ok

:: ── Copy config if needed ───────────────────────────────────
if not exist "config.yaml" (
    if exist "config.yaml.example" (
        echo  No config.yaml found - copying from example...
        copy config.yaml.example config.yaml >nul
        echo  [OK] config.yaml created.
        echo.
        echo  IMPORTANT: You need API keys to use this.
        echo  Right-click the pony after it launches to set them,
        echo  or edit config.yaml in any text editor.
        echo.
    )
)

:: ── Create empty dirs ───────────────────────────────────────
if not exist "memory" mkdir memory
if not exist "diary" mkdir diary
if not exist "logs" mkdir logs

:: ── Launch ──────────────────────────────────────────────────
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
    echo  ============================================
    echo     bonziPONY crashed!
    echo  ============================================
    echo.
    echo  Common fixes:
    echo  - Delete "venv" folder and run setup again
    echo  - Make sure config.yaml has valid API keys
    echo  - Check the error message above
    echo.
    echo  Post a screenshot of this window in the Discord if stuck.
    echo.
    pause
)
