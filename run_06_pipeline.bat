@echo off
REM ============================================================
REM  CPose — Live Pipeline (Detection + Tracking + Pose + ADL)
REM  Unified frame-by-frame: shows ONE window with all overlays.
REM  Press Q or Esc in the preview window to stop.
REM  Ctrl+C in this terminal also stops cleanly.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set INPUT=data-test
set OUTPUT=dataset\outputs\live
set DET_CONF=0.5
set POSE_CONF=0.45
set KEYPOINT_CONF=0.30
set ADL_WINDOW=30

echo ============================================================
echo  CPose Live Pipeline
echo  Input      : %INPUT%
echo  Output     : %OUTPUT%
echo  Det conf   : %DET_CONF%
echo  Pose conf  : %POSE_CONF%
echo  ADL window : %ADL_WINDOW% frames
echo  Controls   : Q / Esc  = close preview window
echo               Ctrl+C   = stop everything
echo ============================================================

.venv\Scripts\python.exe -m src.pipeline.live_pipeline ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --det-conf %DET_CONF% ^
  --pose-conf %POSE_CONF% ^
  --keypoint-conf %KEYPOINT_CONF% ^
  --adl-window %ADL_WINDOW%

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Live pipeline failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Live pipeline complete. Results in: %OUTPUT%
endlocal
