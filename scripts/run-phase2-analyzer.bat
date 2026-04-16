@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "ROOT=%~dp0.."
set "VENV=%ROOT%\venv"
call "%VENV%\Scripts\activate.bat"

echo [CPose] Starting Phase 2 Analyzer...
python tools\run_phase2_analyzer.py ^
    --input-dir data\raw_videos ^
    --output-dir data\output_labels ^
    --model-path models\yolo11n.pt

pause
