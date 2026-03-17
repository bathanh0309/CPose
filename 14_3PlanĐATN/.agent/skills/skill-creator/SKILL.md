# HavenNet Development SKILL Guide

This is the canonical rulebook for all code generation in the HavenNet project. All developers and AI assistants MUST follow these rules when creating, modifying, or reviewing code.

---

## 1. FOUNDATION: The 5-Part Audit Framework

All code MUST align with:

- **PHẦN A (Gap Analysis):** Current system issues, limitations, constraints
- **PHẦN B (Target MVP Architecture):** System design, threading model, component boundaries
- **PHẦN C (Repo Restructure Plan):** File organization, module layout, code locations
- **PHẦN D (Implementation Phase Plan):** Sprint breakdown, daily tasks, acceptance criteria
- **PHẦN E (Quick Start Guide):** Code examples, setup instructions, component patterns

**Rule:** Before writing ANY code, answer: "Which phase does this belong to? What does the audit say about this component?"

---

## 2. ARCHITECTURE RULES (From PHẦN B)

### 2.1 Threading Model
- **One thread per camera** for RTSP capture
- **One thread for inference** (shared YOLO worker)
- **Main thread** orchestrates all components
- **NO blocking operations** in camera threads (use queues)
- All threads must be daemon threads that gracefully shut down

**Example:**
```python
# CORRECT: Non-blocking camera thread
camera_thread = threading.Thread(target=camera_worker, daemon=True, args=(camera_id,))
camera_thread.start()

# WRONG: Blocking main thread waiting for frame
frame = camera.capture()  # <-- blocks everything
```

### 2.2 Event-Driven Architecture
- All inter-module communication uses **standardized Event schema**
- Events are JSON-serializable and immutable
- Event queue is thread-safe and bounded (max 5000 events)
- NO direct function calls between modules

**Event Schema (REQUIRED):**
```python
@dataclass
class Event:
    timestamp: float           # Unix timestamp
    camera_id: str            # "camera_0", "camera_1", etc.
    event_type: str           # "person_detected", "frame_captured", "error"
    confidence: float         # 0.0-1.0 for detections
    frame_index: int          # Sequential frame number per camera
    data: dict                # Additional context (bbox, posture, etc.)
    
    def to_json(self) -> dict:
        return asdict(self)
```

### 2.3 Bounded Queues
- Frame queue: max 5 frames per camera (auto-drops oldest)
- Event queue: max 5000 events (global)
- **NO unlimited lists** - causes memory leaks

**Example:**
```python
from collections import deque

frame_queue = deque(maxlen=5)  # Auto-drops oldest when full
frame_queue.append(frame)       # Safe - never grows > 5
```

### 2.4 Configuration System
- All config from YAML files (backend/config.yaml)
- Config loaded once at startup, immutable after
- Environment variables override YAML values
- Profile support: `desktop` vs `pi5` mode

**Rule:** NO hardcoded values except for constants. All configurable values MUST have a YAML entry.

### 2.5 Parallel Processing Strategy
```
Input Layer (Thread per camera)
    ↓ frame queue
Processing Layer (Single inference worker)
    ↓ event queue
Output Layer (API + Storage)
```

---

## 3. MODULE STRUCTURE (From PHẦN C)

### 3.1 Required Modules

```
backend/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration system
│   │   ├── logger.py           # Logging utilities
│   │   └── events.py           # Event schema + queue
│   ├── input/
│   │   ├── __init__.py
│   │   └── camera_worker.py    # RTSP camera thread
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLO wrapper
│   │   └── inference_worker.py # Main inference loop
│   ├── storage/
│   │   ├── __init__.py
│   │   └── database.py         # SQLite operations
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   ├── routes.py           # REST endpoints
│   │   └── websocket.py        # WebSocket handlers
│   └── app.py                  # Main entry point
├── config.yaml                 # Configuration
├── models/                     # YOLO weights
└── tests/
    ├── test_camera.py
    ├── test_detector.py
    ├── test_events.py
    └── test_api.py
```

**Rule:** Files MUST go in the correct module. Use grep to verify no file is orphaned.

### 3.2 Module Boundaries (NO Cross-Module Imports)

```
✓ ALLOWED:
input/camera_worker.py → core/events.py (put event on queue)
processing/detector.py → core/events.py (put event on queue)
api/routes.py → storage/database.py (query data)

✗ FORBIDDEN:
input/camera_worker.py → processing/detector.py (direct call)
api/routes.py → input/camera_worker.py (bypass events)
storage/database.py → api/routes.py (reverse dependency)
```

---

## 4. CODING STANDARDS

### 4.1 Python Version & Style
- Python 3.9+
- PEP 8 compliant
- Type hints on all functions (no `Any` unless justified)
- Docstrings on all public functions

**Example:**
```python
def process_frame(frame: np.ndarray, camera_id: str) -> Event:
    """
    Process a single frame and return detection events.
    
    Args:
        frame: RGB numpy array (H, W, 3)
        camera_id: Camera identifier
        
    Returns:
        Event object with detection results
    """
    pass
```

### 4.2 Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Classes | PascalCase | `CameraWorker`, `YOLODetector` |
| Functions | snake_case | `capture_frame()`, `get_detections()` |
| Constants | UPPER_SNAKE_CASE | `MAX_QUEUE_SIZE`, `FRAME_TIMEOUT_SEC` |
| Private | _leading_underscore | `_load_weights()` |
| Modules | snake_case | `camera_worker.py`, `inference_worker.py` |
| Variables | snake_case | `frame_id`, `detected_persons` |

### 4.3 Error Handling

**MUST:**
- Catch specific exceptions (NOT bare `except:`)
- Log errors with full context
- Gracefully degrade (log + continue, not crash)
- Include retry logic for network operations

**Example:**
```python
try:
    frame = camera.capture(timeout=5.0)
except cv2.error as e:
    logger.error(f"Camera {camera_id} capture failed: {e}", exc_info=True)
    return None
except TimeoutError:
    logger.warning(f"Camera {camera_id} timeout, retrying...")
    return None
```

### 4.4 Logging Standards

- Use `logger` from `core.logger` module
- Log level: DEBUG, INFO, WARNING, ERROR
- Include context: camera_id, frame_id, timestamp
- NO print() statements (use logger only)

**Example:**
```python
logger.info(f"Camera {camera_id} connected", extra={"camera_id": camera_id})
logger.warning(f"Frame dropped: {camera_id}", extra={"frame_id": frame_id})
logger.error(f"Detector failed: {e}", extra={"error": str(e)})
```

### 4.5 Type Hints Everywhere

```python
# ✓ CORRECT
def get_frame(camera_id: str, timeout: float = 5.0) -> np.ndarray:
    pass

# ✗ WRONG
def get_frame(camera_id, timeout=5.0):
    pass

# ✗ WRONG
def process(data: Any) -> Any:
    pass
```

---

## 5. COMPONENT PATTERNS (From PHẦN E)

### 5.1 Camera Worker Pattern
```python
class CameraWorker:
    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.frame_queue = deque(maxlen=5)
        self.is_running = False
        self.cap = None
        
    def connect(self) -> bool:
        """Open RTSP connection with retries."""
        for attempt in range(3):
            try:
                self.cap = cv2.VideoCapture(self.config.rtsp_url)
                if self.cap.isOpened():
                    logger.info(f"Camera {self.camera_id} connected")
                    return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return False
    
    def run(self):
        """Main capture loop (runs in thread)."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.append(frame)
            else:
                self.reconnect()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get oldest frame from queue (thread-safe)."""
        try:
            return self.frame_queue.popleft()
        except IndexError:
            return None
```

### 5.2 Detector Pattern
```python
class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)
        
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on frame.
        
        Returns:
            List of dicts: {class, confidence, bbox, mask}
        """
        results = self.model(frame, conf=0.5, device=0)  # device=0 for GPU
        
        detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Person class
                    detections.append({
                        "class": "person",
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    })
        return detections
```

### 5.3 Event Queue Pattern
```python
class EventQueue:
    def __init__(self, max_size: int = 5000):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def put(self, event: Event):
        """Thread-safe put."""
        with self.lock:
            self.queue.append(event)
            
    def get_all(self) -> list[Event]:
        """Get all events since last call (thread-safe)."""
        with self.lock:
            events = list(self.queue)
            self.queue.clear()
            return events
```

### 5.4 FastAPI Pattern
```python
@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "cameras": get_camera_status(),
        "uptime_seconds": get_uptime(),
    }

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        events = event_queue.get_all()
        if events:
            await websocket.send_json([e.to_json() for e in events])
        await asyncio.sleep(0.1)
```

---

## 6. DATABASE RULES (From PHẦN A/C)

### 6.1 SQLite Only (NO Vector DB)
- SQLite for ALL data (events, logs, detections)
- Simple schema (NO complex vector operations)
- WAL mode enabled for concurrent access
- Automatic vacuum enabled

**Schema:**
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    camera_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    confidence REAL,
    data TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_camera_timestamp ON events(camera_id, timestamp);
```

### 6.2 Query Patterns
```python
# ✓ CORRECT: Parameterized query
cursor.execute(
    "SELECT * FROM events WHERE camera_id = ? AND timestamp > ?",
    (camera_id, since_timestamp)
)

# ✗ WRONG: SQL injection risk
cursor.execute(f"SELECT * FROM events WHERE camera_id = '{camera_id}'")
```

---

## 7. TESTING RULES (From PHẦN D)

### 7.1 Test Coverage Requirements
- Unit tests for all core modules (detector, config, events)
- Integration tests for camera + inference pipeline
- Performance benchmarks (FPS, latency, memory)
- Stability test (30-minute runtime)

### 7.2 Test File Locations
```
backend/tests/
├── test_config.py        # Config loading + validation
├── test_detector.py      # YOLO detection accuracy
├── test_events.py        # Event queue thread-safety
├── test_camera.py        # Camera worker with mock
├── test_api.py           # FastAPI endpoints
└── test_integration.py   # End-to-end pipeline
```

### 7.3 Test Template
```python
import pytest
from backend.src.core.config import Config

def test_config_loads_yaml():
    """Config should load from YAML file."""
    config = Config.from_yaml("backend/config.yaml")
    assert config.cameras is not None
    assert len(config.cameras) > 0

def test_config_env_override():
    """Environment variables should override YAML."""
    import os
    os.environ["NUM_CAMERAS"] = "2"
    config = Config.from_yaml("backend/config.yaml")
    assert config.num_cameras == 2
```

---

## 8. GIT & BRANCHING RULES

### 8.1 Branch Naming
```
main              # Production branch (stable)
dev               # Development branch
feature/...       # Feature branches
bugfix/...        # Bug fix branches
spike/...         # Exploration/research
```

### 8.2 Commit Messages
```
format: <type>: <description>

types:
  feat:    New feature
  fix:     Bug fix
  refactor: Code restructure (no behavior change)
  docs:    Documentation update
  test:    Test additions
  chore:   Dependencies, config, tooling

examples:
  feat: Add RTSP camera worker with auto-reconnect
  fix: Fix frame queue memory leak in camera worker
  refactor: Move event schema to core module
  test: Add 30-minute stability test for camera pipeline
```

### 8.3 PR Review Checklist
- [ ] Follows architecture from PHẦN B?
- [ ] Located in correct module from PHẦN C?
- [ ] Implements sprint task from PHẦN D?
- [ ] All functions have type hints?
- [ ] Tests included?
- [ ] No hardcoded values?
- [ ] No breaking changes to main?

---

## 9. CONFIGURATION RULES

### 9.1 YAML Structure (backend/config.yaml)
```yaml
app:
  name: "HavenNet"
  version: "1.0.0"
  profile: "desktop"  # or "pi5"
  
logging:
  level: "INFO"       # DEBUG, INFO, WARNING, ERROR
  dir: "logs"
  
cameras:
  - id: "camera_0"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    resolution: [1920, 1080]
    fps: 30
  - id: "camera_1"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    resolution: [1280, 720]
    fps: 24

inference:
  model: "yolov8n.pt"
  confidence: 0.5
  device: 0  # 0=GPU, "cpu"=CPU
  
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

### 9.2 Config Loading Pattern
```python
@dataclass
class CameraConfig:
    id: str
    rtsp_url: str
    resolution: tuple[int, int]
    fps: int

class Config:
    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config(**data)
```

---

## 10. DEPLOYMENT RULES

### 10.1 Docker Pattern
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY backend/ .

# Download YOLO model at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8000

CMD ["python", "src/app.py"]
```

### 10.2 Startup Script (run.sh)
```bash
#!/bin/bash
cd $(dirname $0)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python backend/src/app.py
```

---

## 11. COMMON PITFALLS & FIXES

### ❌ Pitfall 1: Hardcoded Paths
```python
# WRONG
cap = cv2.VideoCapture("rtsp://192.168.1.100:554/stream")

# CORRECT
cap = cv2.VideoCapture(self.config.cameras[0].rtsp_url)
```

### ❌ Pitfall 2: Blocking Main Thread
```python
# WRONG
for frame in camera.continuous_capture():
    frame = detector.detect(frame)
    
# CORRECT (use queues + threads)
camera_thread = threading.Thread(target=camera_worker)
inference_thread = threading.Thread(target=inference_worker)
```

### ❌ Pitfall 3: Unlimited List Growth
```python
# WRONG
self.frames = []
self.frames.append(frame)  # grows forever → memory leak

# CORRECT
from collections import deque
self.frames = deque(maxlen=5)
self.frames.append(frame)  # auto-drops oldest
```

### ❌ Pitfall 4: Bare Exception Catches
```python
# WRONG
try:
    frame = capture()
except:
    pass  # hides all errors!

# CORRECT
try:
    frame = capture()
except TimeoutError:
    logger.warning("Timeout on capture")
except cv2.error as e:
    logger.error(f"OpenCV error: {e}", exc_info=True)
```

### ❌ Pitfall 5: Missing Type Hints
```python
# WRONG
def process(data):
    return result

# CORRECT
def process(data: np.ndarray) -> Event:
    return event
```

### ❌ Pitfall 6: Direct Module Imports (Breaking Boundaries)
```python
# WRONG
from backend.src.input.camera_worker import CameraWorker
worker = CameraWorker()
worker.run()  # direct call, should use events!

# CORRECT
camera_thread = threading.Thread(target=camera_worker_main)
# Output only via event queue
event_queue.put(Event(...))
```

### ❌ Pitfall 7: No Graceful Shutdown
```python
# WRONG
while True:
    frame = capture()  # never stops
    
# CORRECT
while self.is_running:
    frame = capture()
    
# Clean shutdown
signal.signal(signal.SIGINT, cleanup_handler)
```

---

## 12. VERIFICATION CHECKLIST

Before committing code, verify:

- [ ] **Architecture:** Follows PHẦN B design?
- [ ] **Location:** Correct module from PHẦN C?
- [ ] **Task:** Completes PHẦN D sprint item?
- [ ] **Code Quality:**
  - [ ] Type hints on all functions
  - [ ] Docstrings on public functions
  - [ ] Error handling (specific exceptions, logging)
  - [ ] NO hardcoded values (use config)
  - [ ] NO print() (use logger)
  - [ ] NO unlimited lists (use bounded queues)
  - [ ] Thread-safe (use locks for shared state)
- [ ] **Testing:**
  - [ ] Unit tests for new code
  - [ ] Integration tests pass
  - [ ] No regressions on existing tests
- [ ] **Git:**
  - [ ] Descriptive commit message
  - [ ] Single feature per commit
  - [ ] No debug prints left
  - [ ] Passes pre-commit hooks

---

## 10. DEPLOYMENT & BATCH FILES (From DEPLOYMENT_STRATEGY.md)

### 10.1 File Structure (Multiple Files, NOT Single run.bat)

```
root/
├── setup.bat              # Virtual env + dependencies
├── init.bat               # Database initialization
├── run.bat                # Processing pipeline (main backend)
├── web.bat                # FastAPI web server (separate terminal)
├── demo.bat               # Full demo (orchestrates everything) ⭐
└── scripts/
    └── _common.bat        # Shared utilities, error handling
```

**RULE:** Each `.bat` file has ONE responsibility. NO single giant run.bat.

### 10.2 Common Utilities (_common.bat)

All batch files MUST source `_common.bat` for:
- Portable path detection (no D:\ hardcoding)
- Error handling functions
- Environment validation

**Example:**
```batch
call "%~dp0scripts\_common.bat"
call :check_venv      # Validates venv exists
call :activate_venv   # Activates venv
call :log_error "message"  # Logs errors
```

### 10.3 Error Handling Standards

**BEFORE (bad):**
```batch
python src\run.py
if errorlevel 1 goto error    ❌ Generic error
```

**AFTER (good):**
```batch
python src\app.py
if errorlevel 1 (
    call :log_error "Processing pipeline failed"
    echo   Check: config.yaml is valid YAML
    echo   Check: Database initialized with init.bat
    pause
    exit /b 1
)
```

**RULE:** Every error MUST have:
1. Specific error message
2. Suggested fix (not generic "ERROR")
3. Actionable steps for user

### 10.4 Portable Paths (Auto-detection)

**BEFORE (bad):**
```batch
set PYTHONPATH=D:\HavenNet\backend\src  ❌ Fails on other PCs
```

**AFTER (good):**
```batch
for %%I in ("%~dp0.") do set PROJECT_ROOT=%%~fI
set PYTHONPATH=%PROJECT_ROOT%\backend\src  ✅ Works anywhere
```

**RULE:** All paths MUST be relative to script location, not hardcoded.

### 10.5 Usage Patterns

**First-time setup:**
```
1. setup.bat  (creates venv)
2. init.bat   (initializes database)
3. demo.bat   (runs full system)
```

**Daily use:**
```
Just run: demo.bat  (orchestrates everything)
```

**Debugging one component:**
```
run.bat      (pipeline alone)
web.bat      (server alone)
```

**RULE:** Users should be able to run `demo.bat` once and have everything work.

---

## 13. QUICK REFERENCE: Which Phase Does This Belong To?

| Question | Phase | Reference |
|----------|-------|-----------|
| "What does the system look like?" | B | PHẦN_B_TARGET_MVP_ARCHITECTURE.md |
| "Where does this file go?" | C | PHẦN_C_REPO_RESTRUCTURE_PLAN.md |
| "What should I code this sprint?" | D | PHẦN_D_IMPLEMENTATION_PHASE_PLAN.md |
| "Show me code examples" | E | PHẦN_E_QUICK_START_IMPLEMENTATION_GUIDE.md |
| "Why is this broken?" | A | PHẦN_A_GAP_ANALYSIS.md |

---

## 14. EMERGENCY CONTACTS / ESCALATION

**Code Review Questions:** Reference PHẦN B (architecture)
**File Location Issues:** Reference PHẦN C (structure)  
**Implementation Blockers:** Reference PHẦN D (sprint plan)
**Code Patterns:** Reference PHẦN E (examples)
**Design Issues:** Reference PHẦN A (gap analysis)

---

**Last Updated:** 2026-03-14
**Status:** ACTIVE - All developers MUST follow this guide
**Questions?** Check the audit documents (PHẦN A-E) for detailed explanations
