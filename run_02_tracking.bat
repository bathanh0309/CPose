@echo off
REM ============================================================
REM  CPose — Module 2: Human Tracking
REM  Reads detections from run_01 output.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set INPUT=data-test
set OUTPUT=dataset\outputs\2_tracking
set DETECTION_DIR=dataset\outputs\1_detection
set CONF=0.5
set CONFIG=configs\model_registry.demo_i5.yaml

echo ============================================================
echo  CPose Human Tracking
echo  Input        : %INPUT%
echo  Detection dir: %DETECTION_DIR%
echo  Output       : %OUTPUT%
echo ============================================================

.venv\Scripts\python.exe -m src.human_tracking.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --detection-dir "%DETECTION_DIR%" ^
  --conf %CONF% ^
  --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Tracking failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Tracking complete. Results in: %OUTPUT%
endlocal
