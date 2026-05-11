@echo off
REM CPose - Module 4: ADL Recognition
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "POSE_DIR=dataset\outputs\3_pose"
set "VIDEO_DIR=data-test"
set "OUTPUT=dataset\outputs\4_adl"
set "WINDOW=30"
set "CONFIG=configs\profiles\dev.yaml"
echo [CPose] ADL Recognition - Pose: %POSE_DIR%, Video: %VIDEO_DIR%, Output: %OUTPUT%
"%PYTHON%" -m src.modules.adl_recognition.main ^
  --pose-dir "%POSE_DIR%" ^
  --video-dir "%VIDEO_DIR%" ^
  --output "%OUTPUT%" ^
  --window-size %WINDOW% ^
  --config "%CONFIG%"
popd
  --config "%CONFIG%" ^
  --preview

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
