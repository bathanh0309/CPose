@echo off
REM ===============================================
REM COMMON UTILITIES - Shared by all batch files
REM ===============================================

REM Detect project root from script location
for %%I in ("%~dp0.") do set PROJECT_ROOT=%%~fI
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

REM Define key paths
set VENV_PATH=%PROJECT_ROOT%\.venv
set BACKEND_PATH=%PROJECT_ROOT%\backend
set PYTHONPATH=%BACKEND_PATH%\src

REM ===============================================
REM FUNCTION: Check if venv exists
REM ===============================================
:check_venv
if not exist "%VENV_PATH%" (
    echo [ERROR] Virtual environment not found
    echo   Path: %VENV_PATH%
    echo.
    echo Solution: Run 'setup.bat' first
    exit /b 1
)
exit /b 0

REM ===============================================
REM FUNCTION: Activate venv
REM ===============================================
:activate_venv
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activate script not found
    exit /b 1
)
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)
exit /b 0

REM ===============================================
REM FUNCTION: Check if port is available
REM Usage: call :check_port 8000
REM ===============================================
:check_port
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%1 "') do (
    if not "%%a"=="" (
        echo [ERROR] Port %1 is already in use
        echo   Process ID: %%a
        echo.
        echo Solution: taskkill /PID %%a /F
        exit /b 1
    )
)
exit /b 0

REM ===============================================
REM FUNCTION: Verify Python installation
REM ===============================================
:verify_python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Solution: Install Python 3.9+ from https://www.python.org
    exit /b 1
)
exit /b 0

REM ===============================================
REM LOGGING FUNCTIONS
REM ===============================================
:log_info
echo [INFO] %~1
exit /b 0

:log_success
echo [SUCCESS] %~1
exit /b 0

:log_error
echo [ERROR] %~1
exit /b 1

:log_separator
echo.
echo ========================================
echo   %~1
echo ========================================
echo.
exit /b 0
