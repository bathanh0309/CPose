@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title HavenNet - Full Demo Setup

cls
echo.
echo ========================================
echo   HAVENNET - Full Demo Orchestration
echo ========================================
echo.
echo This script will automatically:
echo   1. Setup virtual environment
echo   2. Initialize database
echo   3. Start processing pipeline
echo   4. Start web server (in separate terminal)
echo.
echo System Requirements:
echo   - Python 3.9+
echo   - 8GB RAM (16GB recommended)
echo   - Internet connection (for model download)
echo.
echo Timeline: ~3-5 minutes first run
echo.
pause

REM Load common utilities
call "%~dp0scripts\_common.bat"

REM Verify Python before starting
call :verify_python
if errorlevel 1 (
    pause
    exit /b 1
)

REM ===============================================
REM Phase 1: Setup
REM ===============================================
echo.
call :log_separator "PHASE 1: Setup (2-3 min)"

call setup.bat
if errorlevel 1 (
    call :log_error "Setup failed. Fix errors and try again."
    pause
    exit /b 1
)

REM ===============================================
REM Phase 2: Initialize Database
REM ===============================================
echo.
call :log_separator "PHASE 2: Database Initialization"

call init.bat
if errorlevel 1 (
    call :log_error "Database initialization failed"
    pause
    exit /b 1
)

REM ===============================================
REM Phase 3: Launch Services
REM ===============================================
echo.
call :log_separator "PHASE 3: Starting Services"

echo.
echo Opening two terminals:
echo   1. Processing Pipeline (run.bat)
echo   2. Web Server (web.bat)
echo.
echo This will allow:
echo   - Run.bat terminal: Real-time processing logs
echo   - Web.bat terminal: API logs + hot-reload
echo.
echo Press ENTER to continue...
pause

REM Start processing pipeline in new window
echo Starting processing pipeline...
start "HavenNet - Pipeline" cmd /k "cd /d %PROJECT_ROOT% && call run.bat"

REM Wait for pipeline to initialize
timeout /t 3 /nobreak

REM Start web server in new window
echo Starting web server...
start "HavenNet - Web Server" cmd /k "cd /d %PROJECT_ROOT% && call web.bat"

REM Wait for server to start
timeout /t 3 /nobreak

REM ===============================================
REM Final Instructions
REM ===============================================
echo.
call :log_separator "DEMO READY!"

echo.
echo ========================================
echo   ACCESS YOUR SYSTEM
echo ========================================
echo.
echo Dashboard:         http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check:      http://localhost:8000/api/health
echo.
echo ========================================
echo   WHAT'S RUNNING
echo ========================================
echo.
echo Terminal 1: Processing Pipeline
echo   - Captures video from cameras
echo   - Runs YOLO person detection
echo   - Stores results in database
echo.
echo Terminal 2: Web Server
echo   - FastAPI serving dashboard
echo   - WebSocket for real-time updates
echo   - Hot-reload enabled (changes auto-reflect)
echo.
echo ========================================
echo   STOP THE DEMO
echo ========================================
echo.
echo To stop everything:
echo   1. Close Terminal 1 (pipeline) - Ctrl+C
echo   2. Close Terminal 2 (web server) - Ctrl+C
echo   3. Close this window
echo.
echo ========================================
echo.

REM Keep main window open
pause
