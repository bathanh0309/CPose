@echo off
REM ============================================================
REM  CPose - Module 7: Benchmark
REM  Run after individual module runs or a pipeline run.
REM
REM  Default keeps output light: JSON summary only.
REM  Optional:
REM    run_07_benchmark.bat csv    writes benchmark_summary.csv too
REM    run_07_benchmark.bat paper  writes paper CSV tables too
REM ============================================================
setlocal
pushd "%~dp0" >nul

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"

REM Set to the specific run dir you want to benchmark.
set "RUN_DIR=dataset\outputs"
set "SAVE_CSV="
set "MAKE_PAPER=0"
if /I "%~1"=="csv" set "SAVE_CSV=--save-csv"
if /I "%~1"=="paper" (
    set "SAVE_CSV=--save-csv"
    set "MAKE_PAPER=1"
)

echo ============================================================
echo  CPose Benchmark
echo  Run dir : %RUN_DIR%
echo ============================================================

"%PYTHON%" -m src.pipeline.benchmark_all --run-dir "%RUN_DIR%" %SAVE_CSV%

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
    echo [ERROR] Benchmark failed with exit code %EXITCODE%
    popd
    exit /b %EXITCODE%
)

echo.
echo Benchmark complete.

if "%MAKE_PAPER%"=="0" (
    echo Paper CSV tables skipped. Use: run_07_benchmark.bat paper
    popd
    endlocal
    exit /b 0
)

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
