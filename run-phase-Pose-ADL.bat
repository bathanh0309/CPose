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

set "PYTHONPATH=%ROOT_DIR%\feat-realtime-data\backend"
set "RECORDINGS_DIR=%ROOT_DIR%\data\raw_videos"
set "PROCESSED_VIDEOS_DIR=%ROOT_DIR%\data\processed_videos"
set "DB_PATH=%ROOT_DIR%\data\rtsp_monitor.db"
set "PYTHONUTF8=1"

if not exist "%RECORDINGS_DIR%" mkdir "%RECORDINGS_DIR%"
if not exist "%PROCESSED_VIDEOS_DIR%" mkdir "%PROCESSED_VIDEOS_DIR%"

pushd "%ROOT_DIR%\feat-realtime-data\backend"
"%SHARED_PYTHON%" -m app.workers.pose_adl_gui_server %*
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%

