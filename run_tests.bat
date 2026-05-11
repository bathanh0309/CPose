@echo off
REM ============================================================
REM  CPose - Run tests by module
REM
REM  Usage:
REM    run_tests.bat              all tests
REM    run_tests.bat detection    Module 1 schema tests
REM    run_tests.bat tracking     Module 2 schema tests
REM    run_tests.bat pose         Module 3 schema tests
REM    run_tests.bat adl          Module 4 schema tests
REM    run_tests.bat reid         Module 5 schema tests
REM    run_tests.bat topology     camera topology tests
REM    run_tests.bat manifest     manifest tests
REM    run_tests.bat smoke        shared smoke tests
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto python_ready

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON=py"
    goto python_ready
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON=python"
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
