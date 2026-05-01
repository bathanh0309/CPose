@echo off
REM ============================================================
REM  CPose — Module 3: Pose Estimation
REM  Reads tracks from run_02 output.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set INPUT=data-test
set OUTPUT=dataset\outputs\3_pose
set TRACK_DIR=dataset\outputs\2_tracking
set CONF=0.45
set CONFIG=configs\model_registry.demo_i5.yaml

echo ============================================================
echo  CPose Pose Estimation
echo  Input     : %INPUT%
echo  Track dir : %TRACK_DIR%
echo  Output    : %OUTPUT%
echo ============================================================

.venv\Scripts\python.exe -m src.pose_estimation.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --track-dir "%TRACK_DIR%" ^
  --conf %CONF% ^
  --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Pose estimation failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Pose estimation complete. Results in: %OUTPUT%
endlocal
