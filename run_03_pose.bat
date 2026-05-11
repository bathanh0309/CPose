@echo off
REM ============================================================
REM  CPose - Module 3: Pose Estimation
REM  Reads tracks from run_02 output.
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

set "INPUT=data-test"
set "OUTPUT=dataset\outputs\3_pose"
set "TRACK_DIR=dataset\outputs\2_tracking"
set "CONF=0.45"
set "CONFIG=configs\model_registry.demo_i5.yaml"

echo ============================================================
echo  CPose Pose Estimation
echo  Input     : %INPUT%
echo  Track dir : %TRACK_DIR%
echo  Output    : %OUTPUT%
echo ============================================================

"%PYTHON%" -m src.modules.pose_estimation.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --track-dir "%TRACK_DIR%" ^
  --conf %CONF% ^
  --config "%CONFIG%" ^
  --preview

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Pose estimation failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Pose estimation complete. Results in: %OUTPUT%
popd
endlocal
