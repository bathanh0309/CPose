@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "ROOT=%~dp0.."
set "VENV=%ROOT%\venv"
call "%VENV%\Scripts\activate.bat"

set ADL_MODE=%1
if "%ADL_MODE%"=="" set ADL_MODE=rule

echo [CPose] Starting Phase 3 Recognizer (Mode: %ADL_MODE%)...
python tools\run_phase3_recognizer.py ^
    --input-dir data\raw_videos ^
    --output-dir data\output_pose ^
    --pose-model models\yolo11n-pose.pt ^
    --adl-mode %ADL_MODE% ^
    --adl-cfg configs\ctr_gcn\ctr_gcn_adl_cpose.py ^
    --adl-ckpt models\ctr_gcn_adl_cpose.pth

pause
