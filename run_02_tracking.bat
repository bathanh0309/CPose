@echo off
REM CPose - Module 2: Human Tracking
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "INPUT=data-test"
set "OUTPUT=dataset\outputs\2_tracking"
set "DETECTION_DIR=dataset\outputs\1_detection"
set "CONF=0.5"
set "CONFIG=configs\profiles\dev.yaml"
set "COMPARISON_DIR=dataset\outputs\6_comparison"
echo [CPose] Human Tracking - Input: %INPUT%, Detection: %DETECTION_DIR%, Output: %OUTPUT%
"%PYTHON%" -m src.modules.tracking.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --detection-dir "%DETECTION_DIR%" ^
  --conf %CONF% ^
  --config "%CONFIG%"
popd
  --output "%OUTPUT%" ^
  --detection-dir "%DETECTION_DIR%" ^
  --conf %CONF% ^
  --config "%CONFIG%" ^
  --preview ^
  --make-comparison ^
  --compare-count 2 ^
  --comparison-dir "%COMPARISON_DIR%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Tracking failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Tracking complete. Results in: %OUTPUT%
popd
endlocal
