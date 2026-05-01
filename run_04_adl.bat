@echo off
REM ============================================================
REM  CPose — Module 4: ADL Recognition
REM  Reads keypoints from run_03 output.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set POSE_DIR=dataset\outputs\3_pose
set VIDEO_DIR=data-test
set OUTPUT=dataset\outputs\4_adl
set WINDOW=30
set CONFIG=configs\model_registry.demo_i5.yaml

echo ============================================================
echo  CPose ADL Recognition
echo  Pose dir  : %POSE_DIR%
echo  Video dir : %VIDEO_DIR%
echo  Output    : %OUTPUT%
echo  Window    : %WINDOW% frames
echo ============================================================

.venv\Scripts\python.exe -m src.adl_recognition.main ^
  --pose-dir "%POSE_DIR%" ^
  --video-dir "%VIDEO_DIR%" ^
  --output "%OUTPUT%" ^
  --window-size %WINDOW% ^
  --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] ADL recognition failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ADL recognition complete. Results in: %OUTPUT%
endlocal
