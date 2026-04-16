@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:: CPose - startup script for Windows (cmd-only, no PowerShell profiles)
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
if not exist "%ROOT%data\config"         mkdir "%ROOT%data\config"
if not exist "%ROOT%data\raw_videos"     mkdir "%ROOT%data\raw_videos"
if not exist "%ROOT%data\output_labels"  mkdir "%ROOT%data\output_labels"
if not exist "%ROOT%data\output_pose"    mkdir "%ROOT%data\output_pose"
if not exist "%ROOT%data\output_process" mkdir "%ROOT%data\output_process"
if not exist "%ROOT%data\multicam"       mkdir "%ROOT%data\multicam"
if not exist "%ROOT%models"              mkdir "%ROOT%models"

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
    (
        echo # Add RTSP camera URLs below, one per line
        echo # Example: rtsp://admin:pass@192.168.1.10:554/stream
    ) > "%ROOT%data\config\resources.txt"
)

:: --- Launch Flask app (run main.py directly; pushd ensures correct working dir)
echo [CPose] Starting... Press Ctrl+C to stop.
echo.
call "%VENV%\Scripts\activate.bat"
pushd "%ROOT%"
"%PYTHON%" -W ignore::DeprecationWarning -u main.py
set EXITCODE=%ERRORLEVEL%
popd

if %EXITCODE% NEQ 0 (
    echo.
    echo [CPose] Server exited with code %EXITCODE%.
)

echo.
echo [CPose] Server stopped.
pause
