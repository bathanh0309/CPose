# DEPLOYMENT STRATEGY — Batch Files & Startup Orchestration

## Overview

Hiện tại project có 2 file `.bat` chính:
- `run.bat` — Chạy pipeline xử lý camera (CLI)
- `web.bat` — Chạy web server (FastAPI)

**VẤNĐỀ:** Cả hai đều hardcoded paths (D:\HavenNet\...) và không có error handling tốt. Khi deploy trên Raspberry Pi hoặc máy khác sẽ bị lỗi.

**GIẢI PHÁP:** Tách thành **nhiều file `.bat` nhỏ theo thứ tự** với error handling mạnh mẽ.

---

## Recommended Architecture: Sequential Batch Files

### Option 1: RECOMMENDED (Multi-file với orchestration chính)

```
root/
├── setup.bat              # 1️⃣ Setup venv, install dependencies
├── init.bat              # 2️⃣ Initialize database
├── run.bat               # 3️⃣ Run processing pipeline (main)
├── web.bat               # 4️⃣ Run web server (separate terminal)
├── demo.bat              # 5️⃣ Run full demo (chạy lần lượt: setup → init → run + web)
└── scripts/
    ├── _common.bat        # Shared utilities (paths, error handling)
    ├── _check-env.bat     # Check environment
    └── _cleanup.bat       # Clean temp files
```

**ƯU ĐIỂM:**
- ✅ Mỗi file có 1 trách nhiệm (separation of concern)
- ✅ Dễ debug từng bước
- ✅ Có thể chạy riêng lẻ hoặc tuần tự
- ✅ Portable (không hardcode paths)
- ✅ Reusable across Windows/Linux (with .sh variant)

**NHƯỢC ĐIỂM:**
- Cần user chạy lần lượt (nhưng `demo.bat` giải quyết)

---

### Option 2: Single file với step-by-step (current approach)

```
run.bat  # All steps in 1 file
```

**ƯU ĐIỂM:**
- ✅ Đơn giản, user chỉ click 1 lần

**NHƯỢC ĐIỂM:**
- ❌ Khó debug (phải scroll dài dài)
- ❌ Không thể chạy riêng các bước
- ❌ Hardcoded paths → lỗi khi deploy
- ❌ Error handling phức tạp

---

## RECOMMENDATION: CHỌN OPTION 1

Vì sao:
1. **MVP có 2 phần riêng:**
   - Backend pipeline (camera processing)
   - Frontend API (web dashboard)
   
   Cần 2 terminal riêng để chạy lồng lẫn nhau.

2. **Demo cần tự động hóa:**
   - `demo.bat` orchestrate tất cả (tạo terminal mới nếu cần)
   - Dev team chỉ double-click 1 file

3. **Deployment linh hoạt:**
   - Testing: chạy `init.bat` → `run.bat` (test pipeline)
   - Demo: chạy `demo.bat` (full system)
   - Production: tùy custom (có thể skip steps nào đó)

---

## File Structure Details

### `_common.bat` (Shared Utilities)

```batch
@echo off
REM ===============================================
REM COMMON UTILITIES - Shared by all batch files
REM ===============================================

REM Detect project root (script location)
for %%I in ("%~dp0.") do set PROJECT_ROOT=%%~fI
set VENV_PATH=%PROJECT_ROOT%\.venv
set BACKEND_PATH=%PROJECT_ROOT%\backend
set PYTHONPATH=%BACKEND_PATH%\src

REM Function: Check if venv exists
:check_venv
if not exist "%VENV_PATH%" (
    echo ERROR: Virtual environment not found at %VENV_PATH%
    echo Run 'setup.bat' first
    pause
    exit /b 1
)
exit /b 0

REM Function: Activate venv
:activate_venv
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
exit /b 0

REM Function: Check if port is in use
:check_port
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%1') do (
    if not "%%a"=="" (
        echo ERROR: Port %1 is already in use
        echo Use: taskkill /PID %%a /F
        exit /b 1
    )
)
exit /b 0

REM Function: Log message
:log_info
echo [INFO] %1
exit /b 0

:log_error
echo [ERROR] %1
exit /b 1

:log_success
echo [SUCCESS] %1
exit /b 0
```

### `setup.bat` (First-time setup)

```batch
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ========================================
echo   HAVENNET - Initial Setup
echo ========================================
echo.

REM Load common utilities
call "%~dp0scripts\_common.bat"

cd /d "%PROJECT_ROOT%"

REM Step 1: Create venv
echo [1/4] Creating Python virtual environment...
if exist ".venv" (
    echo   Virtual environment already exists
) else (
    python -m venv .venv
    if errorlevel 1 (
        call :log_error "Failed to create virtual environment"
        pause
        exit /b 1
    )
    call :log_success "Virtual environment created"
)
echo.

REM Step 2: Activate and upgrade pip
echo [2/4] Upgrading pip...
call "%VENV_PATH%\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
call :log_success "Pip upgraded"
echo.

REM Step 3: Install requirements
echo [3/4] Installing dependencies...
cd /d "%BACKEND_PATH%"
pip install -r requirements.txt
if errorlevel 1 (
    call :log_error "Failed to install requirements"
    pause
    exit /b 1
)
call :log_success "Dependencies installed"
echo.

REM Step 4: Download YOLO models
echo [4/4] Downloading YOLO models...
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" >nul 2>&1
call :log_success "YOLO models downloaded"
echo.

echo ========================================
echo   Setup Complete!
echo.
echo Next step: Run 'init.bat' to initialize database
echo ========================================
pause
```

### `init.bat` (Database initialization)

```batch
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ========================================
echo   Database Initialization
echo ========================================
echo.

call "%~dp0scripts\_common.bat"

REM Check venv
call :check_venv
if errorlevel 1 pause & exit /b 1

REM Activate venv
call :activate_venv
if errorlevel 1 pause & exit /b 1

cd /d "%BACKEND_PATH%"
set PYTHONPATH=%BACKEND_PATH%\src

echo [1/3] Checking configuration...
if not exist "config.yaml" (
    call :log_error "config.yaml not found"
    pause
    exit /b 1
)
call :log_success "Configuration found"
echo.

echo [2/3] Cleaning previous database...
if exist "database\haven.db" (
    del /q "database\haven.db"
    call :log_success "Old database deleted"
) else (
    call :log_info "No previous database found"
)
echo.

echo [3/3] Initializing new database...
python src\core\database.py
if errorlevel 1 (
    call :log_error "Database initialization failed"
    pause
    exit /b 1
)
call :log_success "Database initialized"
echo.

echo ========================================
echo   Database Ready!
echo   Next: Run 'run.bat' or 'demo.bat'
echo ========================================
pause
```

### `run.bat` (Processing pipeline)

```batch
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title HavenNet - Processing Pipeline

echo ========================================
echo   HAVENNET - Processing Pipeline
echo ========================================
echo.

call "%~dp0scripts\_common.bat"

REM Check venv
call :check_venv
if errorlevel 1 pause & exit /b 1

REM Activate venv
call :activate_venv
if errorlevel 1 pause & exit /b 1

cd /d "%BACKEND_PATH%"
set PYTHONPATH=%BACKEND_PATH%\src

echo [1/3] Verifying configuration...
python -c "from core.config import Config; Config.load('config.yaml'); print('  Config: OK')" >nul 2>&1
if errorlevel 1 (
    call :log_error "Configuration validation failed"
    pause
    exit /b 1
)
call :log_success "Configuration verified"
echo.

echo [2/3] Verifying database...
if not exist "database\haven.db" (
    call :log_error "Database not found. Run 'init.bat' first"
    pause
    exit /b 1
)
call :log_success "Database found"
echo.

echo [3/3] Starting processing pipeline...
echo   Press Ctrl+C to stop
echo.

python -u src\app.py

REM Cleanup
echo.
call :log_info "Pipeline stopped"
pause
```

### `web.bat` (Web server - separate terminal)

```batch
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title HavenNet - Web Server

echo ========================================
echo   HAVENNET - Web Server
echo ========================================
echo.

call "%~dp0scripts\_common.bat"

REM Check venv
call :check_venv
if errorlevel 1 pause & exit /b 1

REM Activate venv
call :activate_venv
if errorlevel 1 pause & exit /b 1

REM Check port availability
call :check_port 8000
if errorlevel 1 (
    pause
    exit /b 1
)

cd /d "%BACKEND_PATH%"
set PYTHONPATH=%BACKEND_PATH%\src

echo [1/2] Verifying dependencies...
python -c "import fastapi; import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo   Missing dependencies, installing...
    pip install fastapi uvicorn python-multipart >nul 2>&1
)
call :log_success "Dependencies verified"
echo.

echo [2/2] Starting web server...
echo   Dashboard:  http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo   Press Ctrl+C to stop
echo.

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

echo.
call :log_info "Web server stopped"
pause
```

### `demo.bat` (Full demo orchestration) ⭐

```batch
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

echo ========================================
echo   HAVENNET - Full Demo Setup
echo ========================================
echo.
echo This will:
echo   1. Setup virtual environment
echo   2. Initialize database
echo   3. Start processing pipeline + web server
echo.
echo Requirements: Python 3.9+, 8GB RAM
echo.
pause

call "%~dp0scripts\_common.bat"

REM Step 1: Setup
echo.
echo ========================================
echo   STEP 1: Setup (this may take 2-3 min)
echo ========================================
call setup.bat
if errorlevel 1 exit /b 1

REM Step 2: Initialize
echo.
echo ========================================
echo   STEP 2: Database Initialization
echo ========================================
call init.bat
if errorlevel 1 exit /b 1

REM Step 3: Launch both services
echo.
echo ========================================
echo   STEP 3: Starting Services
echo ========================================
echo.
echo Opening TWO terminals:
echo   - Terminal 1: Processing pipeline (run.bat)
echo   - Terminal 2: Web server (web.bat)
echo.
pause

REM Start run.bat in new window
start "HavenNet - Pipeline" cmd /k "cd /d %PROJECT_ROOT% && call run.bat"

REM Small delay to let first terminal start
timeout /t 2 /nobreak

REM Start web.bat in new window
start "HavenNet - Web Server" cmd /k "cd /d %PROJECT_ROOT% && call web.bat"

echo.
echo ========================================
echo   Services Starting...
echo   Wait 10 seconds for initialization
echo ========================================
timeout /t 10 /nobreak

echo.
echo ========================================
echo   DEMO READY!
echo ========================================
echo.
echo Dashboard:  http://localhost:8000
echo API Docs:   http://localhost:8000/docs
echo.
echo Close this window or press Ctrl+C to stop
echo both services (they run in separate terminals)
echo.
pause
```

### `_check-env.bat` (Environment validation)

```batch
@echo off
REM Validate Python, CUDA, paths, etc.

echo Checking environment...
python --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

echo.
echo All checks passed!
```

---

## Usage Flow

### First-time Setup:
```
1. Double-click: setup.bat        (creates venv, installs packages)
2. Double-click: init.bat          (creates database)
3. Double-click: run.bat + web.bat (test individually, or...)
4. Double-click: demo.bat          (run full demo with 2 terminals)
```

### Daily Development:
```
Just run: demo.bat    (handles everything)
```

### Debugging:
```
Run each step separately:
- setup.bat           (only if changing requirements.txt)
- init.bat            (only if changing config.yaml)
- run.bat             (test pipeline alone)
- web.bat             (test web server alone)
```

---

## Error Handling Improvements

### Before (current run.bat):
```batch
python -c "from storage.vector_db import VectorDatabase"
if errorlevel 1 goto error    ❌ Generic error
```

### After (new approach):
```batch
python -c "from core.config import Config; Config.load('config.yaml')"
if errorlevel 1 (
    call :log_error "Configuration validation failed"
    echo   Check: config.yaml exists and is valid YAML
    pause
    exit /b 1
)
```

**Better error messages** → faster debugging

---

## Portable Paths (No D:\ hardcoding)

### Before:
```batch
set PYTHONPATH=D:\HavenNet\backend\src  ❌ Only works on this PC
```

### After:
```batch
for %%I in ("%~dp0.") do set PROJECT_ROOT=%%~fI
set PYTHONPATH=%PROJECT_ROOT%\backend\src  ✅ Works anywhere
```

**Auto-detects** where script is located.

---

## Cross-Platform (Windows + Linux)

For Raspberry Pi and Linux deployment, create `.sh` variants:

```bash
# setup.sh, init.sh, run.sh, web.sh, demo.sh (same logic, bash syntax)
```

**Key difference:**
- Windows: `call :function`
- Bash: `function_name`

---

## Summary

| Aspect | Single File (run.bat) | Multi-File (Recommended) |
|--------|----------------------|--------------------------|
| Complexity | Low | Medium |
| Debuggability | Hard (scroll long) | Easy (per-file) |
| Reusability | Limited | High |
| Portability | Hardcoded paths | Auto-detect |
| Error Messages | Generic | Specific |
| Demo Automation | Manual steps | Single `demo.bat` |
| CI/CD Ready | No | Yes |

---

## DECISION

**✅ IMPLEMENT OPTION 1: Multi-file with demo.bat orchestration**

This is the MVP-ready approach that scales to production.

