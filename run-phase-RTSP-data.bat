@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "SHARED_PYTHON=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%SHARED_PYTHON%" (
  echo Missing shared Python environment: %SHARED_PYTHON%
  echo Create it once at repo root and install: python -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

set "APP_URL=http://127.0.0.1:8000/"
set "HEALTH_URL=http://127.0.0.1:8000/api/health"
set "PYTHONPATH=%ROOT_DIR%\feat-realtime-data\backend"
set "APP_ENV=production"
set "API_PORT=8000"
set "FRONTEND_ORIGIN=http://127.0.0.1:8000"
set "RECORDINGS_DIR=%ROOT_DIR%\data\raw_videos"
set "PROCESSED_VIDEOS_DIR=%ROOT_DIR%\data\processed_videos"
set "DB_PATH=%ROOT_DIR%\data\rtsp_monitor.db"
set "DETECTOR_MODEL=yolo_nano"
set "OPENCV_LOG_LEVEL=ERROR"
set "OPENCV_VIDEOIO_DEBUG=0"
set "OPENCV_FFMPEG_LOGLEVEL=-8"
set "PYTHONUTF8=1"

if not exist "%RECORDINGS_DIR%" mkdir "%RECORDINGS_DIR%"
if not exist "%PROCESSED_VIDEOS_DIR%" mkdir "%PROCESSED_VIDEOS_DIR%"

powershell -NoProfile -Command "try { $resp = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { } exit 1" >nul 2>&1
if not errorlevel 1 (
  start "" "%APP_URL%"
  exit /b 0
)

start "Phase 1 - RTSP Collector" /D "%ROOT_DIR%\feat-realtime-data\backend" "%SHARED_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

powershell -NoProfile -Command "$deadline = (Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { try { $resp = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { } Start-Sleep -Milliseconds 500 }; exit 1" >nul 2>&1
if errorlevel 1 (
  echo Phase 1 backend did not become ready at %APP_URL%
  exit /b 1
)

start "" "%APP_URL%"
exit /b 0
