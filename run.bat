@echo off
chcp 65001 >nul
setlocal

:: CPose - startup script for Windows
set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

echo.
echo ============================================================
echo   CPose - Camera Pose ^& ADL Demo
echo   Dashboard: http://localhost:5000
echo ============================================================
echo.

:: --- Check system Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

:: --- Create venv if missing ---
if not exist "%PYTHON%" (
    echo [CPose] Creating virtual environment...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

:: --- Activate venv ---
call "%VENV%\Scripts\activate.bat"

:: --- Install dependencies if ultralytics is missing ---
"%PYTHON%" -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [CPose] Installing dependencies from requirements.txt...
    "%PIP%" install -r "%ROOT%requirements.txt" -q
    if errorlevel 1 (
        echo [ERROR] Installation failed. Check internet connection or requirements.txt.
        pause
        exit /b 1
    )
    echo [CPose] Dependencies installed OK.
)

:: --- Create required data folders ---
mkdir "%ROOT%data\config"        2>nul
mkdir "%ROOT%data\raw_videos"    2>nul
mkdir "%ROOT%data\output_labels" 2>nul
mkdir "%ROOT%data\output_pose"   2>nul
mkdir "%ROOT%data\multicam"      2>nul
mkdir "%ROOT%models"             2>nul

:: --- Warn if model files are missing ---
if not exist "%ROOT%models\yolov8n.pt" (
    echo [WARN] Missing models\yolov8n.pt  (needed for Phase 1 recording)
)
if not exist "%ROOT%models\yolo11n.pt" (
    echo [WARN] Missing models\yolo11n.pt  (needed for Phase 2 analysis)
)
if not exist "%ROOT%models\yolo11n-pose.pt" (
    echo [WARN] Missing models\yolo11n-pose.pt  (needed for Phase 3 pose/ADL)
)

:: --- Create default resources.txt if missing ---
if not exist "%ROOT%data\config\resources.txt" (
    echo # Add RTSP camera URLs below, one per line> "%ROOT%data\config\resources.txt"
    echo # Example: rtsp://admin:pass@192.168.1.10:554/stream>> "%ROOT%data\config\resources.txt"
)

:: --- Launch (suppress EventletDeprecationWarning via -W flag) ---
echo [CPose] Starting... Press Ctrl+C to stop.
echo.
"%PYTHON%" -W ignore::DeprecationWarning "%ROOT%main.py"

echo.
echo [CPose] Server stopped.
pause
