@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title HavenNet - Database Initialization

cls
echo.
echo ========================================
echo   HAVENNET - Database Initialization
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

if not exist "config.yaml" (
    call :log_error "config.yaml not found in %BACKEND_PATH%"
    echo.
    echo Solution: Create config.yaml with camera settings
    pause
    exit /b 1
)

echo Validating config.yaml syntax...
python -c "import yaml; yaml.safe_load(open('config.yaml'))" >nul 2>&1
if errorlevel 1 (
    call :log_error "config.yaml has invalid YAML syntax"
    echo.
    echo Solution: Check config.yaml formatting
    pause
    exit /b 1
)
call :log_success "Configuration valid"

REM ===============================================
REM Step 2: Clean Previous Database
REM ===============================================
call :log_separator "Step 2/3: Cleaning Previous Data"

if exist "database" (
    if exist "database\haven.db" (
        echo Removing old database...
        del /q "database\haven.db"
        call :log_success "Old database deleted"
    ) else (
        call :log_info "No previous database found"
    )
) else (
    echo Creating database directory...
    mkdir database
    call :log_success "Database directory created"
)

REM ===============================================
REM Step 3: Initialize Database
REM ===============================================
call :log_separator "Step 3/3: Initializing Database"

echo Running database initialization...
python src\core\database.py
if errorlevel 1 (
    call :log_error "Database initialization failed"
    echo.
    echo Check:
    echo   - database.py file exists at src\core\database.py
    echo   - SQLite is properly configured
    pause
    exit /b 1
)
call :log_success "Database initialized"

REM ===============================================
REM Verification
REM ===============================================
echo.
echo Verifying database...

if exist "database\haven.db" (
    call :log_success "Database file created"
) else (
    call :log_error "Database file was not created"
    pause
    exit /b 1
)

REM ===============================================
REM Final Summary
REM ===============================================
echo.
call :log_separator "Database Ready!"

echo.
echo Next steps:
echo   1. Run: run.bat           (Start processing pipeline)
echo   2. Run: web.bat           (Start web server, separate terminal)
echo.
echo Or run both together:
echo   - demo.bat               (Full demo setup)
echo.

pause
