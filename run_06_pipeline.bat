@echo off
REM CPose - Live Combined Pipeline
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "INPUT=data-test"
set "OUTPUT=dataset\runs"
set "MODELS=configs\profiles\dev.yaml"
set "TOPOLOGY=configs\camera\topology.yaml"
set "RUN_ID=live_combined"
echo [CPose] Live Pipeline - Input: %INPUT%, Output: %OUTPUT%, Run ID: %RUN_ID%
"%PYTHON%" -m src.pipeline.live_pipeline ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --models "%MODELS%" ^
  --topology "%TOPOLOGY%" ^
  --run-id %RUN_ID%
popd
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
