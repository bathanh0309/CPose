@echo off
chcp 65001 >nul 2>&1
title HAVEN Web Server

echo ========================================
echo    HAVEN Web Interface
echo    http://localhost:8000
echo ========================================
echo.

REM Activate venv
call D:\HavenNet\.venv\Scripts\activate.bat

REM Set Python path
set PYTHONPATH=D:\HavenNet\backend\src

REM Change to backend directory
cd /d D:\HavenNet\backend

echo [1/2] Checking dependencies...
python -m pip install fastapi uvicorn python-multipart jinja2 --quiet
echo   Dependencies: OK
echo.

echo [2/2] Starting web server...
echo.
echo   Dashboard:  http://localhost:8000
echo   Video Feed: http://localhost:8000/video_feed
echo   API Status: http://localhost:8000/api/status
echo.
echo   Press Ctrl+C to stop the server.
echo ========================================
echo.

python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

pause
