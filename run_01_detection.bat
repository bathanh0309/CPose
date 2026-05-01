@echo off
REM ============================================================
REM  CPose — Module 1: Person Detection
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set INPUT=data-test
set OUTPUT=dataset\outputs\1_detection
set MODEL=
set CONF=0.5
set CONFIG=configs\model_registry.demo_i5.yaml

echo ============================================================
echo  CPose Person Detection
echo  Input  : %INPUT%
echo  Output : %OUTPUT%
echo  Config : %CONFIG%
echo ============================================================

.venv\Scripts\python.exe -m src.human_detection.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --conf %CONF% ^
  --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Detection failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Detection complete. Results in: %OUTPUT%
endlocal
