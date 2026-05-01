@echo off
REM ============================================================
REM  CPose — Module 7: Benchmark & Paper Tables
REM  Run AFTER run_06_pipeline.bat (or any pipeline run).
REM  Aggregates all metrics and writes paper-ready CSV/MD.
REM  CLAUDE.md §6 CLI convenience script
REM ============================================================
setlocal

REM Set to the specific pipeline run dir you want to benchmark.
REM Default: scan latest run under dataset/outputs/pipeline/
set RUN_DIR=dataset\outputs

echo ============================================================
echo  CPose Benchmark + Paper Table Generation
echo  Run dir : %RUN_DIR%
echo ============================================================

.venv\Scripts\python.exe -m src.pipeline.benchmark_all --run-dir "%RUN_DIR%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Benchmark failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Benchmark complete.

REM Also generate paper tables for the run directory
.venv\Scripts\python.exe -m src.reports.make_paper_tables --run-dir "%RUN_DIR%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Paper table generation failed.
    exit /b %ERRORLEVEL%
)

echo Paper tables written to: %RUN_DIR%\08_paper_report\
endlocal
