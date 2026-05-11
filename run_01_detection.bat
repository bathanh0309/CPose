@echo off
REM ============================================================
REM  CPose - Module 1: Person Detection
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

set "INPUT=data-test"
set "OUTPUT=dataset\outputs\1_detection"
set "CONF=0.5"
set "CONFIG=configs\model_registry.demo_i5.yaml"
set "COMPARISON_DIR=dataset\outputs\6_comparison"

echo ============================================================
echo  CPose Person Detection
echo  Input      : %INPUT%
echo  Output     : %OUTPUT%
echo  Crops      : %OUTPUT%\*/crops\
echo  Comparison : %COMPARISON_DIR%
echo  Config     : %CONFIG%
echo ============================================================

"%PYTHON%" -m src.modules.detection.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --conf %CONF% ^
  --config "%CONFIG%" ^
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
