@echo off
REM CPose - Run tests by module
REM Usage: run_tests.bat [module]
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "MODULE=%~1"
if "%MODULE%"=="" (
    echo [CPose] Run all tests
    "%PYTHON%" -m pytest tests/ -v
) else (
    echo [CPose] Run tests for module: %MODULE%
    "%PYTHON%" -m pytest tests/test_%MODULE%.py -v
)
popd
    goto python_ready
)

echo [ERROR] Python not found. Create the venv first:
echo         py -m venv .venv
echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
popd
exit /b 1

:python_ready
set "MODULE=%~1"
if "%MODULE%"=="" set "MODULE=all"

set "TARGET=tests"
set "KNOWN=0"
if /I "%MODULE%"=="all" (
    set "TARGET=tests"
    set "KNOWN=1"
)
if /I "%MODULE%"=="smoke" (
    set "TARGET=tests\test_common_smoke.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="detection" (
    set "TARGET=tests\test_detection_schema.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="tracking" (
    set "TARGET=tests\test_tracking_schema.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="pose" (
    set "TARGET=tests\test_pose_schema.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="adl" (
    set "TARGET=tests\test_adl_schema.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="reid" (
    set "TARGET=tests\test_reid_schema.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="topology" (
    set "TARGET=tests\test_topology.py"
    set "KNOWN=1"
)
if /I "%MODULE%"=="manifest" (
    set "TARGET=tests\test_manifest.py"
    set "KNOWN=1"
)

if "%KNOWN%"=="0" (
    echo [ERROR] Unknown module "%MODULE%".
    echo Valid: all, smoke, detection, tracking, pose, adl, reid, topology, manifest
    popd
    exit /b 2
)

echo ============================================================
echo  CPose Tests
echo  Python : %PYTHON%
echo  Target : %TARGET%
echo ============================================================

"%PYTHON%" -m pytest "%TARGET%" -q
set EXITCODE=%ERRORLEVEL%

popd
endlocal & exit /b %EXITCODE%
