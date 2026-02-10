@echo off
REM ========================================
REM HAVEN - Complete Multi-Camera Pipeline
REM Verification + Execution
REM ========================================

echo.
echo ========================================
echo    HAVEN Multi-Camera System
echo    Full Pipeline: Pose + ADL + ReID
echo ========================================
echo.

cd /d D:\HavenNet\backend

REM Activate virtual environment
call D:\HavenNet\.venv\Scripts\activate.bat

REM Set Python path
set PYTHONPATH=D:\HavenNet\backend\src

REM ========================================
REM STEP 1: System Verification
REM ========================================
echo.
echo [Step 1/7] Verifying Core Infrastructure...
python -c "from storage.persistence import PersistenceManager; print('  Persistence: OK')"
if errorlevel 1 goto error
python -c "from storage.vector_db import VectorDatabase; print('  VectorDB: OK')"
if errorlevel 1 goto error
python -c "from core.global_id_manager import GlobalIDManager; print('  GlobalIDManager: OK')"
if errorlevel 1 goto error

echo.
echo [Step 2/7] Verifying ReID Components...
python -c "from reid import EnhancedReID; print('  EnhancedReID: OK')"
if errorlevel 1 goto error

echo.
echo [Step 3/7] Verifying ADL Detector...
python -c "from adl import classify_posture, ADLConfig; print('  ADL: OK')"
if errorlevel 1 goto error

echo.
echo [Step 4/7] Verifying Pose Model...
python -c "from ultralytics import YOLO; print('  YOLO Pose: OK')"
if errorlevel 1 goto error

echo.
echo [Step 5/7] Checking Camera Configuration...
if exist "src\config.yaml" (
    echo   Config file: OK
    echo.
    echo   Cameras configured:
    type src\config.yaml | findstr "name:"
    echo.
    echo   ReID Settings:
    type src\config.yaml | findstr "threshold:"
    type src\config.yaml | findstr "confirm_frames:"
    type src\config.yaml | findstr "min_keypoints:"
) else (
    echo   ERROR: config.yaml not found!
    pause
    exit /b 1
)

REM ========================================
REM STEP 2: Clean Previous State
REM ========================================
echo.
echo [Step 6/7] Cleaning Previous State...
if exist "database\haven_state.db" (
    del /q "database\haven_state.db"
    echo   Deleted: haven_state.db
)
if exist "database\embeddings.npy" (
    del /q "database\embeddings.npy"
    echo   Deleted: embeddings.npy
)
echo   Ready for fresh run (G1, G2, G3...)

REM ========================================
REM STEP 3: Run Multi-Camera Processing
REM ========================================
echo.
echo [Step 7/7] Starting Multi-Camera Processing...
echo.
echo ========================================
echo   Processing cameras sequentially
echo   Press Ctrl+C to stop
echo ========================================
echo.
echo Controls: SPACE=Pause, N=Next, Q=Quit, G=Record MP4
echo.

python src/run.py

REM ========================================
REM STEP 4: Summary
REM ========================================
echo.
echo ========================================
echo   Processing Complete!
echo ========================================
echo.
echo Output files:
echo   - database/haven_reid.db (event database)
echo   - database/haven_state.db (ReID state)
echo.
goto end

:error
echo.
echo ========================================
echo   ERROR: System verification failed!
echo ========================================
echo.
echo Please check:
echo   1. Virtual environment activated
echo   2. All dependencies installed
echo   3. Python path configured
echo.
pause
exit /b 1

:end
pause
