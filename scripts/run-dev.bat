@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

set "ROOT=%~dp0.."
set "VENV=%ROOT%\venv"
call "%VENV%\Scripts\activate.bat"

:: Set environment to dev
set CPOSER_MODE=dev
set LOG_LEVEL=DEBUG

echo [CPose] Starting in DEV mode (Hot-Reload enabled)...
python -u main.py --config configs\dev.yaml
pause
