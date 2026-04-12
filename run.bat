@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [CPose] Creating virtual environment...
    python -m venv "%VENV%"
)

call "%VENV%\Scripts\activate.bat"

pip show flask >nul 2>&1
if errorlevel 1 (
    echo [CPose] Installing dependencies...
    pip install -r "%ROOT%requirements.txt" --quiet
)

mkdir "%ROOT%data\config" 2>nul
mkdir "%ROOT%data\raw_videos" 2>nul
mkdir "%ROOT%data\output_labels" 2>nul
mkdir "%ROOT%data\output_pose" 2>nul
mkdir "%ROOT%models" 2>nul

echo [CPose] Starting dashboard at http://localhost:5000
python "%ROOT%main.py"
