@REM Script bootstrap môi trường backend và khởi chạy hệ thống trên Windows.
@echo off
setlocal

set "ROOT_DIR=%~dp0"
pushd "%ROOT_DIR%"

set "VENV_DIR=%ROOT_DIR%backend\.venv"
set "PYTHON_EXE=%ROOT_DIR%backend\.venv\Scripts\python.exe"
set "REQUIREMENTS_FILE=%ROOT_DIR%backend\requirements.txt"
set "RECORDINGS_DIR=%ROOT_DIR%data\raw_videos"
set "DB_PATH=%ROOT_DIR%backend\events.db"
set "YOLO_CONFIG_DIR=%ROOT_DIR%backend\.ultralytics"
set "APP_URL=http://127.0.0.1:8000/"
set "HEALTH_URL=http://127.0.0.1:8000/api/health"
set "PYTHON_READY="
set "VENV_EXISTS="

if exist "%PYTHON_EXE%" (
  set "VENV_EXISTS=1"
  "%PYTHON_EXE%" -c "from pydantic import Field; from fastapi import FastAPI; import aiosqlite; import cv2; import multipart; import pydantic_settings; import ultralytics" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_READY=1"
  )
)

if not defined PYTHON_READY (
  where python >nul 2>&1
  if not errorlevel 1 (
    python -c "from pydantic import Field; from fastapi import FastAPI; import aiosqlite; import cv2; import multipart; import pydantic_settings; import ultralytics" >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_EXE=python"
      set "PYTHON_READY=1"
    )
  )
)

if not defined PYTHON_READY (
  if defined VENV_EXISTS (
    echo Repairing backend virtual environment...
  ) else (
    echo Creating backend virtual environment...
    where py >nul 2>&1
    if errorlevel 1 (
      where python >nul 2>&1
      if errorlevel 1 (
        echo Python was not found in PATH.
        popd
        exit /b 1
      )
      python -m venv "%VENV_DIR%"
    ) else (
      py -3 -m venv "%VENV_DIR%"
    )

    if errorlevel 1 (
      echo Failed to create backend virtual environment.
      popd
      exit /b 1
    )
  )

  set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

  "%PYTHON_EXE%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo Failed to upgrade pip.
    popd
    exit /b 1
  )

  "%PYTHON_EXE%" -m pip install --upgrade --force-reinstall -r "%REQUIREMENTS_FILE%"
  if errorlevel 1 (
    echo Failed to prepare backend dependencies.
    popd
    exit /b 1
  )
)

if not exist "%RECORDINGS_DIR%" (
  mkdir "%RECORDINGS_DIR%"
)

if not exist "%YOLO_CONFIG_DIR%" (
  mkdir "%YOLO_CONFIG_DIR%"
)

set "APP_ENV=production"
set "API_PORT=8000"
set "RECORDINGS_DIR=%RECORDINGS_DIR%"
set "DB_PATH=%DB_PATH%"
set "DETECTOR_MODEL=yolo_nano"
set "YOLO_CONFIG_DIR=%YOLO_CONFIG_DIR%"
set "OPENCV_LOG_LEVEL=ERROR"
set "OPENCV_VIDEOIO_DEBUG=0"
set "OPENCV_FFMPEG_LOGLEVEL=-8"

powershell -NoProfile -Command "try { $resp = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { } exit 1" >nul 2>&1
if not errorlevel 1 (
  start "" "%APP_URL%"
  set "EXIT_CODE=0"
  popd
  exit /b %EXIT_CODE%
)

start "RTSP Backend" /D "%ROOT_DIR%backend" "%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

powershell -NoProfile -Command "$deadline = (Get-Date).AddSeconds(30); while ((Get-Date) -lt $deadline) { try { $resp = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { } Start-Sleep -Milliseconds 500 }; exit 1" >nul 2>&1
if errorlevel 1 (
  echo Backend did not become ready at %APP_URL%
  set "EXIT_CODE=1"
) else (
  start "" "%APP_URL%"
  set "EXIT_CODE=0"
)

popd
exit /b %EXIT_CODE%
