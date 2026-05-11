@echo off
REM ============================================================
REM  CPose - Live Combined Pipeline
REM  Shows Detection + Tracking + Pose + ADL + ReID on every frame.
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

set "INPUT=data-test"
set "OUTPUT=dataset\runs"
set "MODELS=configs\model_registry.demo_i5.yaml"
set "TOPOLOGY=configs\camera_topology.yaml"
set "RUN_ID=live_combined"

echo ============================================================
echo  CPose Live Combined Pipeline
echo  Input    : %INPUT%
echo  Output   : %OUTPUT%
echo  Models   : %MODELS%
echo  Topology : %TOPOLOGY%
echo  Run ID   : %RUN_ID%
echo  Overlay  : Detection + Tracking + Pose + ADL + ReID
echo ============================================================

"%PYTHON%" -m src.pipeline.live_pipeline ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --models "%MODELS%" ^
  --topology "%TOPOLOGY%" ^
  --run-id "%RUN_ID%" ^
  --preview

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Live combined pipeline failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Live combined pipeline complete. Results in: %OUTPUT%
popd
endlocal
