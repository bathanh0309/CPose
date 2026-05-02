@echo off
REM ============================================================
REM  CPose - Module 4: ADL Recognition
REM  Reads keypoints from run_03 output.
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

set "POSE_DIR=dataset\outputs\3_pose"
set "VIDEO_DIR=data-test"
set "OUTPUT=dataset\outputs\4_adl"
set "WINDOW=30"
set "CONFIG=configs\model_registry.demo_i5.yaml"

echo ============================================================
echo  CPose ADL Recognition
echo  Pose dir  : %POSE_DIR%
echo  Video dir : %VIDEO_DIR%
echo  Output    : %OUTPUT%
echo  Window    : %WINDOW% frames
echo ============================================================

"%PYTHON%" -m src.modules.adl_recognition.main ^
  --pose-dir "%POSE_DIR%" ^
  --video-dir "%VIDEO_DIR%" ^
  --output "%OUTPUT%" ^
  --window-size %WINDOW% ^
  --config "%CONFIG%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] ADL recognition failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo ADL recognition complete. Results in: %OUTPUT%
popd
endlocal
