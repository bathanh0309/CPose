@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title HavenNet - Setup

cls
echo.
echo ========================================
echo   HAVENNET - Initial Setup
echo ========================================
echo.

REM Load common utilities
call "%~dp0scripts\_common.bat"

REM Step 0: Verify Python
call :verify_python
if errorlevel 1 (
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"

REM ===============================================
REM Step 1: Create Virtual Environment
REM ===============================================
call :log_separator "Step 1/4: Creating Virtual Environment"

if exist ".venv" (
    call :log_info "Virtual environment already exists"
    goto :skip_venv_creation
)

echo Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 (
    call :log_error "Failed to create virtual environment"
    echo.
    echo Solution:
    echo   - Check Python installation: python --version
    echo   - Try: python -m venv --help
    pause
    exit /b 1
)
call :log_success "Virtual environment created at .venv"

:skip_venv_creation

REM ===============================================
REM Step 2: Upgrade pip
REM ===============================================
call :log_separator "Step 2/4: Upgrading pip"

echo Activating virtual environment...
call :activate_venv
if errorlevel 1 (
    pause
    exit /b 1
)

echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 (
    call :log_error "Failed to upgrade pip"
    pause
    exit /b 1
)
call :log_success "pip upgraded"

REM ===============================================
REM Step 3: Install Requirements
REM ===============================================
call :log_separator "Step 3/4: Installing Dependencies"

cd /d "%BACKEND_PATH%"

if not exist "requirements.txt" (
    call :log_error "requirements.txt not found in %BACKEND_PATH%"
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    call :log_error "Failed to install dependencies"
    echo.
    echo Solution:
    echo   - Check internet connection
    echo   - Try: pip install --upgrade pip
    echo   - Check requirements.txt syntax
    pause
    exit /b 1
)
call :log_success "All dependencies installed"

REM ===============================================
REM Step 4: Download YOLO Models
REM ===============================================
call :log_separator "Step 4/4: Downloading YOLO Models"

echo Downloading YOLOv8m detection model...
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); print('  Model downloaded to ~/.cache/ultralytics')" >nul 2>&1
if errorlevel 1 (
    call :log_error "Failed to download YOLO model"
    echo.
    echo Note: This is optional. Restart this step if needed.
) else (
    call :log_success "YOLO models ready"
)

REM ===============================================
REM Final Summary
REM ===============================================
echo.
call :log_separator "Setup Complete!"

echo.
echo Next steps:
echo   1. Run: init.bat           (Initialize database)
echo   2. Run: demo.bat          (Start full demo)
echo.
echo Or run individually:
echo   - run.bat                 (Processing pipeline only)
echo   - web.bat                 (Web server only)
echo.

pause
