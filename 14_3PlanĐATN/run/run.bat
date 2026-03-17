@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title HavenNet - Processing Pipeline

cls
echo.
echo ========================================
echo   HAVENNET - Processing Pipeline
echo ========================================
echo.

REM Load common utilities
call "%~dp0scripts\_common.bat"

REM Check virtual environment
call :check_venv
if errorlevel 1 (
    echo Solution: Run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call :activate_venv
if errorlevel 1 (
    pause
    exit /b 1
)

cd /d "%BACKEND_PATH%"
set PYTHONPATH=%BACKEND_PATH%\src

REM ===============================================
REM Step 1: Verify Configuration
REM ===============================================
call :log_separator "Step 1/3: Verifying Configuration"

echo Validating configuration...
python -c "from core.config import Config; Config.load('config.yaml'); print('  Cameras configured: OK')" >nul 2>&1
if errorlevel 1 (
    call :log_error "Configuration validation failed"
    echo.
    echo Check:
    echo   - config.yaml exists and is valid
    echo   - Core config module loads correctly
    pause
    exit /b 1
)
call :log_success "Configuration verified"

REM ===============================================
REM Step 2: Verify Database
REM ===============================================
call :log_separator "Step 2/3: Verifying Database"

if not exist "database\haven.db" (
    call :log_error "Database not found"
    echo.
    echo Solution: Run init.bat first to initialize database
    pause
    exit /b 1
)
call :log_success "Database found and ready"

REM ===============================================
REM Step 3: Start Pipeline
REM ===============================================
call :log_separator "Step 3/3: Starting Processing Pipeline"

echo.
echo Processing Details:
echo   - Capturing from configured cameras
echo   - Running YOLO person detection
echo   - Storing results in database
echo.
echo Controls:
echo   - Press Ctrl+C to stop the pipeline
echo.
echo ========================================
echo.

REM Run the main application
python -u src\app.py

REM Cleanup on exit
echo.
call :log_separator "Pipeline Stopped"

echo.
echo Output files:
echo   - Logs: logs\havennet.log
echo   - Database: database\haven.db
echo.

pause
