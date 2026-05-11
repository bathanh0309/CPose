@echo off
REM CPose - Module 7: Benchmark
REM Usage: run_07_benchmark.bat [csv|paper]
setlocal
pushd "%~dp0" >nul
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py"
set "RUN_DIR=dataset\outputs"
set "SAVE_CSV="
set "MAKE_PAPER=0"
if /I "%~1"=="csv" set "SAVE_CSV=--save-csv"
if /I "%~1"=="paper" (
        set "SAVE_CSV=--save-csv"
        set "MAKE_PAPER=1"
)
echo [CPose] Benchmark - Run dir: %RUN_DIR%, Save CSV: %SAVE_CSV%, Paper: %MAKE_PAPER%
"%PYTHON%" -m src.benchmark_all ^
    --run-dir "%RUN_DIR%" ^
    %SAVE_CSV% ^
    --make-paper %MAKE_PAPER%
popd

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
