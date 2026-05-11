@echo off
REM CPose - Module 5: Global ReID (TFCS-PAR)
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "INPUT=data-test"
set "OUTPUT=dataset\outputs\5_reid"
set "POSE_DIR=dataset\outputs\3_pose"
set "ADL_DIR=dataset\outputs\4_adl"
set "MANIFEST=configs\camera\multicam_manifest.json"
set "TOPOLOGY=configs\camera\topology.yaml"
set "CONFIG=configs\profiles\dev.yaml"
echo [CPose] Global ReID - Input: %INPUT%, Pose: %POSE_DIR%, ADL: %ADL_DIR%, Output: %OUTPUT%
"%PYTHON%" -m src.modules.global_reid.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --pose-dir "%POSE_DIR%" ^
  --adl-dir "%ADL_DIR%" ^
  --manifest "%MANIFEST%" ^
  --topology "%TOPOLOGY%" ^
  --config "%CONFIG%"
popd
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --pose-dir "%POSE_DIR%" ^
  --adl-dir "%ADL_DIR%" ^
  --manifest "%MANIFEST%" ^
  --topology "%TOPOLOGY%" ^
  --config "%CONFIG%" ^
  --preview

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] ReID failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo ReID complete. Results in: %OUTPUT%
popd
endlocal
