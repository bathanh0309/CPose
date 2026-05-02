@echo off
REM ============================================================
REM  CPose - Module 7: Benchmark & Paper Tables
REM  Run after individual module runs or a pipeline run.
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

REM Set to the specific run dir you want to benchmark.
set "RUN_DIR=dataset\outputs"

echo ============================================================
echo  CPose Benchmark + Paper Table Generation
echo  Run dir : %RUN_DIR%
echo ============================================================

"%PYTHON%" -m src.pipeline.benchmark_all --run-dir "%RUN_DIR%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Benchmark failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Benchmark complete.

"%PYTHON%" -m src.reports.make_paper_tables --run-dir "%RUN_DIR%"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Paper table generation failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo Paper tables written to: %RUN_DIR%\08_paper_report\
popd
endlocal
