# BATCH FILES USAGE GUIDE

## Overview

HavenNet project uses a **multi-file batch system** (not single run.bat) for better organization, error handling, and portability.

```
setup.bat  → init.bat  → run.bat + web.bat  (or demo.bat for all at once)
```

---

## Quick Start (Recommended)

### For First-Time Users: 1 Command
```cmd
Double-click: demo.bat
```

That's it! demo.bat will:
1. Setup virtual environment
2. Initialize database
3. Launch 2 terminals (pipeline + web server)

### For Daily Use: 1 Command
```cmd
Double-click: demo.bat
```

---

## Individual Files Reference

### 1. setup.bat — Virtual Environment & Dependencies

**When to run:**
- First time installation
- After updating requirements.txt
- When Python packages are missing

**What it does:**
```
✓ Creates .venv virtual environment
✓ Upgrades pip/setuptools/wheel
✓ Installs packages from requirements.txt
✓ Downloads YOLO models
```

**Expected output:**
```
========================================
   HAVENNET - Initial Setup
========================================

[INFO] Virtual environment created at .venv
[SUCCESS] pip upgraded
[SUCCESS] All dependencies installed
[SUCCESS] YOLO models ready

Next steps:
   1. Run: init.bat
   2. Run: demo.bat
```

**Time:** ~2-3 minutes (first run), ~30 seconds (subsequent runs)

---

### 2. init.bat — Database Initialization

**When to run:**
- First time after setup.bat
- After changing config.yaml
- Before running run.bat

**What it does:**
```
✓ Validates config.yaml syntax
✓ Removes old database (fresh start)
✓ Creates new database schema
✓ Verifies database file was created
```

**Expected output:**
```
========================================
   HAVENNET - Database Initialization
========================================

[SUCCESS] Configuration verified
[SUCCESS] Old database deleted
[SUCCESS] Database initialized
[SUCCESS] Database file created

Next steps:
   1. Run: run.bat
   2. Run: web.bat (separate terminal)
```

**Time:** ~10 seconds

---

### 3. run.bat — Processing Pipeline (MAIN BACKEND)

**When to run:**
- When you want to process video from cameras
- In one terminal while web.bat runs in another

**What it does:**
```
✓ Validates configuration
✓ Verifies database exists
✓ Starts camera workers (RTSP capture)
✓ Starts inference worker (YOLO detection)
✓ Writes results to database
```

**Expected output:**
```
========================================
   HAVENNET - Processing Pipeline
========================================

[SUCCESS] Configuration verified
[SUCCESS] Database found and ready

Processing Details:
   - Capturing from configured cameras
   - Running YOLO person detection
   - Storing results in database

Controls: Press Ctrl+C to stop the pipeline

[INFO] Frame 1: 1 person detected (0.95 conf)
[INFO] Frame 2: 1 person detected (0.94 conf)
...
```

**Time:** Runs indefinitely until you press Ctrl+C

---

### 4. web.bat — Web Server (DASHBOARD API)

**When to run:**
- In a separate terminal while run.bat runs
- To access the dashboard at http://localhost:8000

**What it does:**
```
✓ Verifies FastAPI dependencies
✓ Checks if port 8000 is available
✓ Starts FastAPI server with hot-reload
✓ Serves dashboard and API endpoints
```

**Expected output:**
```
========================================
   HAVENNET - Web Server
========================================

[SUCCESS] Web dependencies ready

Dashboard & API Endpoints:
   - Dashboard:  http://localhost:8000
   - API Docs:   http://localhost:8000/docs
   - Health:     http://localhost:8000/api/health

Controls: Press Ctrl+C to stop the server

INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Time:** Runs indefinitely until you press Ctrl+C

---

### 5. demo.bat — Full Demo (ALL AT ONCE)

**When to run:**
- First time: full setup from scratch
- Demo day: automated everything
- Testing: verify full system works

**What it does:**
```
✓ Calls setup.bat (virtual environment)
✓ Calls init.bat (database)
✓ Launches run.bat in Terminal 1
✓ Launches web.bat in Terminal 2
✓ Shows success message + dashboard URL
```

**Expected flow:**
```
1. setup.bat runs (2-3 min)
   └─ Creates venv, installs packages

2. init.bat runs (10 sec)
   └─ Creates database

3. Prompts: "Press ENTER to continue"
   └─ User confirms before launching terminals

4. Terminal 1 opens (run.bat)
   └─ Processing pipeline starts

5. Terminal 2 opens (web.bat)
   └─ Web server starts

6. Main window shows:
   Dashboard: http://localhost:8000
   Ctrl+C in either terminal to stop
```

**Time:** ~3-5 minutes total (including waits)

---

## Common Scenarios

### Scenario 1: First-Time Setup (Easiest)
```
1. Double-click: demo.bat
2. Wait ~3 minutes
3. Open browser: http://localhost:8000
```

### Scenario 2: Developer Testing Components
```
1. Double-click: setup.bat (once)
2. Double-click: init.bat (once)
3. Double-click: run.bat (test pipeline)
4. Close terminal when done
5. Double-click: web.bat (test server separately)
```

### Scenario 3: Demo/Presentation
```
1. Run setup.bat + init.bat beforehand
2. When ready: double-click: demo.bat
3. Wait 1 minute for terminals to spawn
4. Point browser to: http://localhost:8000
```

### Scenario 4: Debugging Pipeline
```
1. Double-click: setup.bat (once)
2. Double-click: init.bat
3. Double-click: run.bat
4. Watch real-time logs in terminal
5. Press Ctrl+C to stop and see output summary
6. Check logs/havennet.log for detailed errors
```

### Scenario 5: Debugging Web Server
```
1. Double-click: setup.bat (once)
2. Double-click: init.bat
3. Double-click: web.bat
4. Watch server startup logs
5. Open browser, click around
6. Watch API response logs in terminal
7. Press Ctrl+C to stop
```

---

## Troubleshooting

### Error: "Virtual environment not found"
```
Solution: Run setup.bat first
```

### Error: "config.yaml has invalid YAML syntax"
```
Solution: Check config.yaml formatting
  - Use spaces (not tabs)
  - Proper indentation
  - Valid key-value pairs
```

### Error: "Database not found"
```
Solution: Run init.bat before run.bat
```

### Error: "Port 8000 is already in use"
```
Solution:
  1. Close other application using port 8000
  2. Or modify web.bat to use different port (e.g., 8001)
  3. taskkill /F /IM process_name.exe (if you know the process)
```

### Error: "YOLO model not found"
```
Solution: Re-run setup.bat (model download will retry)
  - Or download manually:
    python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Processing pipeline runs but no output
```
Solutions:
  1. Check config.yaml has camera settings
  2. Check logs/havennet.log for errors
  3. Verify RTSP camera URL is correct
  4. Check internet connection (for first model load)
```

### Web server starts but dashboard is blank
```
Solutions:
  1. Check browser console for JavaScript errors
  2. Check web server logs for API errors
  3. Verify pipeline is running (check run.bat terminal)
  4. Clear browser cache (Ctrl+Shift+Del)
```

---

## Advanced Usage

### Running Components Separately (For Debugging)

**Test just the pipeline:**
```cmd
run.bat
(Ctrl+C to stop)
```

**Test just the API:**
```cmd
web.bat
(Open http://localhost:8000 in browser)
(Ctrl+C to stop)
```

### Restarting After Errors

```cmd
1. Close any failed terminals
2. Run: init.bat (to reset database)
3. Run: demo.bat (or run.bat + web.bat individually)
```

### Modifying Batch Files

All batch files are in the root directory:
```
setup.bat
init.bat
run.bat
web.bat
demo.bat
scripts/_common.bat  (shared utilities)
```

Common modifications:
- **Change port:** Edit web.bat, find "8000", change to desired port
- **Change detector model:** Edit init.bat comment, modify YOLO model name
- **Add debugging:** Add `@echo on` at top of any .bat file

---

## Architecture

### Portable Path Detection

All .bat files automatically detect the project root (no D:\ hardcoding):

```batch
for %%I in ("%~dp0.") do set PROJECT_ROOT=%%~fI
```

This means batch files work on **any computer**, any drive letter.

### Error Handling

Each batch file has these standard error checks:

```batch
call :check_venv      # Is virtual environment present?
call :activate_venv   # Can we activate it?
call :check_port 8000 # Is port available?
call :verify_python   # Is Python installed?
```

If any check fails, the batch file shows:
1. What went wrong (specific error)
2. How to fix it (solution)
3. Example commands (if applicable)

### Shared Utilities

Common functions are in `scripts/_common.bat`:

```batch
:check_venv       # Check venv exists
:activate_venv    # Activate venv
:check_port       # Check port availability
:verify_python    # Check Python installation
:log_info         # Log info message
:log_error        # Log error message
:log_success      # Log success message
:log_separator    # Log separator line
```

Every .bat file sources this: `call "%~dp0scripts\_common.bat"`

---

## Summary

| Task | Command | Duration |
|------|---------|----------|
| First-time setup | `demo.bat` | 3-5 min |
| Daily startup | `demo.bat` | 1 min |
| Test pipeline only | `setup.bat` → `init.bat` → `run.bat` | 3 min + running |
| Test web server | `setup.bat` → `init.bat` → `web.bat` | 3 min + running |
| Reset database | `init.bat` | 10 sec |
| Debug component | Individual `.bat` files | Varies |

---

## Next Steps

After batch files are working:

1. **Customize config.yaml** with your camera settings
2. **Test camera connection** before running pipeline
3. **Check dashboard** at http://localhost:8000
4. **Review logs** at logs/havennet.log
5. **Integrate with your cameras** (RTSP URLs)

---

**Status:** Production-ready
**Last Updated:** 2026-03-14
**Questions?** Check SKILL.md or DEPLOYMENT_STRATEGY.md
