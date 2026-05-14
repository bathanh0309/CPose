@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo Starting CPose web backend with cmd...
echo URL: http://127.0.0.1:8000
echo Press Ctrl+Z in this window to stop.
echo.

"%PYTHON_EXE%" apps\run_web_cmd.py

endlocal
