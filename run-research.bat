@echo off
setlocal EnableExtensions EnableDelayedExpansion
title CPose - Research Report

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if exist "!PYTHON_EXE!" (
    set "PYTHON_PATH=!PYTHON_EXE!"
) else (
    where "!PYTHON_EXE!" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found in PATH and .venv is unavailable.
        pause
        exit /b 1
    )
    set "PYTHON_PATH=!PYTHON_EXE!"
)

if not exist "research_outputs" mkdir "research_outputs"
if exist "research_outputs\latest_run.txt" del /Q "research_outputs\latest_run.txt" >nul 2>nul

set "RESEARCH_PYTHON=!PYTHON_PATH!"
"!PYTHON_PATH!" "scripts\research_report.py" --project-root "." --output-dir "research_outputs" %* > "research_outputs\latest_console.log" 2>&1
set "RC=%ERRORLEVEL%"
type "research_outputs\latest_console.log"

if exist "research_outputs\latest_run.txt" (
    set /p "RUN_DIR="<"research_outputs\latest_run.txt"
    if exist "!RUN_DIR!\console.log" (
        copy /Y "research_outputs\latest_console.log" "!RUN_DIR!\console.log" >nul 2>nul
    )
)

if not "%RC%"=="0" (
    echo [ERROR] Research pipeline failed with exit code %RC%.
    pause
    exit /b %RC%
)

echo [CPose Research] Output: !RUN_DIR!
echo [OK] Research pipeline completed.
exit /b 0
