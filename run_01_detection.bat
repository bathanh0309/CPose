@echo off
REM CPose - Module 1: Person Detection
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "INPUT=data-test"
set "OUTPUT=dataset\outputs\1_detection"
set "CONF=0.5"
set "CONFIG=configs\profiles\dev.yaml"
set "COMPARISON_DIR=dataset\outputs\6_comparison"
echo [CPose] Person Detection - Input: %INPUT%, Output: %OUTPUT%
"%PYTHON%" -m src.modules.detection.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --conf %CONF% ^
  --config "%CONFIG%"
popd
  --preview ^
  --make-comparison ^
  --compare-count 2 ^
  --comparison-dir "%COMPARISON_DIR%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Detection failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Detection complete. Results in: %OUTPUT%
popd
endlocal
