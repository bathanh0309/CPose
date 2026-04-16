@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "ROOT=%~dp0.."
set "VENV=%ROOT%\venv"
call "%VENV%\Scripts\activate.bat"

echo [CPose] Starting Phase 1 Recorder...
python tools\run_phase1_recorder.py ^
    --config configs\phase1.yaml ^
    --output-dir data\raw_videos

pause
