@echo off
REM ============================================================
REM  CPose — Module 5: Global ReID (TFCS-PAR)
REM  Reads pose + ADL from previous module outputs.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

set INPUT=data-test
set OUTPUT=dataset\outputs\5_reid
set POSE_DIR=dataset\outputs\3_pose
set ADL_DIR=dataset\outputs\4_adl
set MANIFEST=configs\multicam_manifest.json
set TOPOLOGY=configs\camera_topology.yaml
set CONFIG=configs\model_registry.demo_i5.yaml

echo ============================================================
echo  CPose Global ReID (TFCS-PAR)
echo  Input    : %INPUT%
echo  Pose dir : %POSE_DIR%
echo  ADL dir  : %ADL_DIR%
echo  Output   : %OUTPUT%
echo  Manifest : %MANIFEST%
echo  Topology : %TOPOLOGY%
echo ============================================================

.venv\Scripts\python.exe -m src.global_reid.main ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --pose-dir "%POSE_DIR%" ^
  --adl-dir "%ADL_DIR%" ^
  --manifest "%MANIFEST%" ^
  --topology "%TOPOLOGY%" ^
  --config "%CONFIG%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] ReID failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ReID complete. Results in: %OUTPUT%
endlocal
