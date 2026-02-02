@echo off
REM ============================================================
REM HAVEN Final - 4 Camera Sequential
REM ============================================================
REM cam1 (SHOW) -> cam2 (MASTER) -> cam3 (SLAVE) -> cam4 (SLAVE)
REM ============================================================

echo ============================================================
echo HAVEN Final - 4 Camera Sequential
echo ============================================================
echo.
echo   CAM1: Show only
echo   CAM2: MASTER (create Global IDs, Pose, ADL)
echo   CAM3: SLAVE (match only)
echo   CAM4: SLAVE (match only)
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Controls:
echo   N = Skip to next video
echo   G = Start/Stop recording (saves to database)
echo   Q = Quit
echo.

REM Run
python backend\src\run_final.py --config configs\unified_config.yaml

echo.
echo ============================================================
echo Done! Recordings saved to: backend\database\recordings\
echo Database: backend\database\haven_reid.db
echo ============================================================
pause
