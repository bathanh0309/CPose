@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title HavenNet - Web Server

cls
echo.
echo ========================================
echo   HAVENNET - Web Server
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

REM Check port availability
call :check_port 8000
if errorlevel 1 (
    echo.
    echo Solution: Close the other application using port 8000
    echo   Or change port in web.bat and src/api/main.py
    pause
    exit /b 1
)

cd /d "%BACKEND_PATH%"
set PYTHONPATH=%BACKEND_PATH%\src

REM ===============================================
REM Step 1: Verify Dependencies
REM ===============================================
call :log_separator "Step 1/2: Verifying Dependencies"

echo Checking FastAPI and uvicorn...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing FastAPI dependencies...
    pip install fastapi uvicorn python-multipart -q
    if errorlevel 1 (
        call :log_error "Failed to install web dependencies"
        pause
        exit /b 1
    )
)
call :log_success "Web dependencies ready"

REM ===============================================
REM Step 2: Start Web Server
REM ===============================================
call :log_separator "Step 2/2: Starting Web Server"

echo.
echo Dashboard & API Endpoints:
echo   - Dashboard:  http://localhost:8000
echo   - API Docs:   http://localhost:8000/docs
echo   - Health:     http://localhost:8000/api/health
echo.
echo Controls:
echo   - Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Run FastAPI server with hot reload
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

REM Cleanup on exit
echo.
call :log_separator "Web Server Stopped"

echo.
echo Next: Run run.bat in another terminal to start processing pipeline
echo.

pause
