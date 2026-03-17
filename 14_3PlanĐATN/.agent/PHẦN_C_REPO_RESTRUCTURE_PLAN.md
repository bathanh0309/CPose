# PHẦN C — REPO RESTRUCTURE PLAN
## Cấu Trúc Lại Repository Cho MVP Thực Tế

**Document Version:** 1.0  
**Status:** Planning Phase  
**Target Completion:** Before Phase 2 coding  
**Branch:** `audit-and-refactor-project`

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Current vs Target Structure](#current-vs-target-structure)
3. [Module-by-Module Refactor Plan](#module-by-module-refactor-plan)
4. [File-Level Changes](#file-level-changes)
5. [Implementation Sequence](#implementation-sequence)
6. [Migration Strategy](#migration-strategy)
7. [Testing & Validation Checklist](#testing--validation-checklist)

---

## EXECUTIVE SUMMARY

### Current Problems

```
backend/src/
├── run.py              ❌ Sequential processing (kills realtime)
├── app.py              ❌ No WebSocket, minimal API
├── reid.py             ❌ Fragile color histogram ReID
├── adl.py              ❌ Untested posture detection
├── core/
│   └── global_id_manager.py  ❌ Overkill for demo scope
└── storage/            ❌ Complex vector DB not needed
```

### Target Architecture (Clean MVP)

```
backend/
├── src/
│   ├── main.py              # New: Single entry point
│   ├── config.py            # Extracted: Config management
│   ├── logger.py            # New: Unified logging
│   ├── camera/              # New: Camera workers
│   │   ├── __init__.py
│   │   ├── worker.py        # RTSP capture thread
│   │   ├── rtsp_client.py   # RTSP connection mgmt
│   │   └── frame_queue.py   # Bounded queue
│   ├── processing/          # New: Inference pipeline
│   │   ├── __init__.py
│   │   ├── detector.py      # YOLO wrapper
│   │   ├── processor.py     # Main inference loop
│   │   └── models.py        # Model loading utils
│   ├── events/              # New: Event system
│   │   ├── __init__.py
│   │   ├── schema.py        # Event TypedDict definitions
│   │   └── manager.py       # Event dedup & publishing
│   ├── storage/             # Refactored: SQLite only
│   │   ├── __init__.py
│   │   ├── db.py            # SQLite connection pool
│   │   ├── models.py        # Table schemas
│   │   └── queries.py       # CRUD operations
│   ├── api/                 # Refactored: FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── server.py        # FastAPI app setup
│   │   ├── routes.py        # REST endpoints
│   │   └── websocket.py     # WebSocket handler
│   ├── web/                 # Frontend assets
│   │   ├── static/
│   │   │   └── index.html   # Refactored dashboard
│   │   └── public/
│   │       ├── css/
│   │       └── js/
│   └── utils/               # Shared utilities
│       ├── __init__.py
│       ├── time.py          # Timestamp utils
│       ├── path.py          # Path management (replace hardcoded D:/)
│       └── snapshot.py      # Snapshot saving
├── config.yaml             # Single unified config
├── requirements.txt        # Cleaned up dependencies
└── pyproject.toml          # New: Modern Python project file

frontend/                   # NEW: Separate React app (optional Phase 2)
├── package.json
├── src/
│   ├── App.tsx
│   ├── pages/
│   │   └── Dashboard.tsx
│   └── ...
└── public/

tests/                      # New: Proper test structure
├── __init__.py
├── test_camera_worker.py
├── test_detector.py
├── test_events.py
├── test_api.py
└── conftest.py

docs/                       # NEW: Implementation guides
├── SETUP.md               # Installation & quick start
├── ARCHITECTURE.md        # Architecture deep dive
├── API.md                 # API reference
└── DEPLOYMENT.md          # Pi 5 deployment guide

scripts/                    # Reorganized
├── setup.py              # Auto-download models
├── cleanup.py            # Cleanup old snapshots
├── test_rtsp.py          # Debug RTSP connectivity
└── benchmark.py          # FPS/latency profiling
```

---

## CURRENT VS TARGET STRUCTURE

### Mapping: Old → New

| Old File/Module | Issue | New Location | New Approach |
|---|---|---|---|
| `run.py` (SequentialRunner) | Sequential processing, hard to extend | → `processing/processor.py` | Threaded inference loop |
| `app.py` (Flask/FastAPI) | No WebSocket, minimal endpoints | → `api/server.py` + `api/routes.py` | Full REST + WebSocket |
| `reid.py` (EnhancedReID) | Color histogram fragile | → `DELETE` (for MVP scope) | Per-camera tracking only |
| `adl.py` (ADL classifier) | Untested, not in demo scope | → `DELETE` (defer to Phase 2) | Posture via simple rules (optional) |
| `global_id_manager.py` | Overkill for 2-4 camera demo | → `DELETE` | Local camera ID manager sufficient |
| `storage/vector_db.py` | Not needed, adds complexity | → `DELETE` | SQLite only for events |
| `storage/persistence.py` | Complex state serialization | → Simplify to `storage/db.py` | Just CRUD on SQLite |
| `visualize.py` | Good, but move to frontend | → Keep but refactor for web | React dashboard replaces it |
| `config.yaml` (scattered) | Multiple config files | → `backend/config.yaml` (single) | YAML → dataclass in Python |
| `requirements.txt` | Likely bloated | → Cleaned up | Remove unused deps |

### Why This Restructure?

1. **Clarity:** Each module has 1 clear responsibility
2. **Testability:** Separate concerns = easy unit tests
3. **Maintainability:** New developer can navigate easily
4. **Scalability:** Easy to add cameras, new event types, etc.
5. **Demo Ready:** No dead code or over-engineering

---

## MODULE-BY-MODULE REFACTOR PLAN

### 1. CAMERA MODULE (`backend/src/camera/`)

**Purpose:** Isolated RTSP capture, no dependencies on processing/storage

#### Files to Create

**`camera/rtsp_client.py`**
```python
class RTSPClient:
    """Low-level RTSP connection management"""
    
    def __init__(self, rtsp_url: str, timeout: int = 10):
        self.rtsp_url = rtsp_url
        self.timeout = timeout
        self.cap = None
    
    def connect(self) -> bool:
        """Connect to RTSP stream, return True if success"""
        # Use cv2.VideoCapture(rtsp_url)
        # Return success/failure
    
    def is_connected(self) -> bool:
        """Check if stream is alive (heartbeat)"""
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame, return None if error"""
    
    def disconnect(self):
        """Close connection gracefully"""
```

**`camera/frame_queue.py`**
```python
class BoundedFrameQueue:
    """Fixed-size queue, drops old frames when full"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def put(self, frame: np.ndarray, timestamp: float):
        """Add frame (auto-drops oldest if full)"""
    
    def get_latest(self) -> Optional[tuple[np.ndarray, float]]:
        """Get newest frame without removing (non-blocking)"""
    
    def size(self) -> int:
        """Current queue size"""
```

**`camera/worker.py`**
```python
class CameraWorker:
    """One thread per camera, handles RTSP capture"""
    
    def __init__(self, camera_id: str, rtsp_url: str, config: dict):
        self.camera_id = camera_id
        self.rtsp_client = RTSPClient(rtsp_url)
        self.frame_queue = BoundedFrameQueue(max_size=5)
        self.status = "DISCONNECTED"
        self.thread = None
        self.running = False
        self.last_frame_time = None
        self.reconnect_attempts = 0
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop gracefully"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _capture_loop(self):
        """Main capture loop (runs in thread)"""
        while self.running:
            if self.status == "DISCONNECTED":
                if self._try_reconnect():
                    self.status = "CONNECTED"
                    # Emit CAMERA_CONNECTED event
                else:
                    time.sleep(5)  # Retry every 5s
                    continue
            
            # Capture frame
            frame = self.rtsp_client.read_frame()
            if frame is None:
                # Timeout
                self.status = "OFFLINE"
                # Emit CAMERA_OFFLINE event
                continue
            
            timestamp = time.time()
            self.frame_queue.put(frame, timestamp)
            self.last_frame_time = timestamp
    
    def _try_reconnect(self) -> bool:
        """Attempt connection with backoff"""
        self.reconnect_attempts += 1
        # Emit CAMERA_RECONNECTING event
        success = self.rtsp_client.connect()
        if success:
            self.reconnect_attempts = 0
            # Emit CAMERA_RECONNECTED event
        return success
    
    def get_status(self) -> dict:
        """Return current status for API/UI"""
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "queue_size": self.frame_queue.size(),
            "last_frame_time": self.last_frame_time,
            "reconnect_attempts": self.reconnect_attempts
        }
```

#### What Gets Deleted
- ❌ No RTSP-related code stays in `run.py`
- ❌ No hardcoded stream URLs

---

### 2. PROCESSING MODULE (`backend/src/processing/`)

**Purpose:** YOLO inference + event generation (single shared worker)

#### Files to Create

**`processing/models.py`**
```python
class YOLODetector:
    """Wrapper around YOLO model"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect persons in frame
        Returns: [Detection(box, conf, class), ...]
        """
        results = self.model.predict(frame, conf=0.5, classes=[0])  # class 0 = person
        return results[0].boxes
    
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Batch detect (if available in YOLO)"""
```

**`processing/processor.py`**
```python
class ProcessingWorker:
    """Inference pipeline (single thread)"""
    
    def __init__(self, camera_workers: list[CameraWorker], config: dict):
        self.camera_workers = camera_workers
        self.detector = YOLODetector(model_path)
        self.event_manager = EventManager()
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _inference_loop(self):
        """Main inference loop"""
        while self.running:
            # Collect latest frames from all cameras
            frames_to_process = {}
            for worker in self.camera_workers:
                frame_data = worker.frame_queue.get_latest()
                if frame_data:
                    frames_to_process[worker.camera_id] = frame_data
            
            if not frames_to_process:
                time.sleep(0.01)
                continue
            
            # Run detection on all available frames
            for camera_id, (frame, timestamp) in frames_to_process.items():
                detections = self.detector.detect(frame)
                person_count = len([d for d in detections if d.conf > 0.5])
                
                # Generate event
                event = {
                    "camera_id": camera_id,
                    "event_type": "PERSON_DETECTED",
                    "person_count": person_count,
                    "timestamp": timestamp,
                    "boxes": [box.to_dict() for box in detections],
                    "frame": frame  # For snapshot
                }
                
                # Publish event (to storage, API, etc.)
                self.event_manager.emit(event)
```

#### What Gets Deleted
- ❌ `run.py` (SequentialRunner) — replaced by processor.py
- ❌ Complex scheduling logic
- ❌ Global ID manager integration

---

### 3. EVENTS MODULE (`backend/src/events/`)

**Purpose:** Standardized event schema + publishing

#### Files to Create

**`events/schema.py`**
```python
from typing import TypedDict, Optional
from datetime import datetime

class PersonDetectionEvent(TypedDict):
    id: str                    # evt_camera0_20250314143022
    timestamp: float           # Unix timestamp
    camera_id: str            # camera_0, camera_1, etc.
    event_type: str           # PERSON_DETECTED, CAMERA_OFFLINE, etc.
    person_count: int         # Number of people detected
    message: str              # Human readable message
    boxes: list[dict]         # Detection boxes
    snapshot_path: Optional[str]  # Path to saved frame

class CameraStatusEvent(TypedDict):
    id: str
    timestamp: float
    camera_id: str
    event_type: str           # CAMERA_CONNECTED, OFFLINE, RECONNECTING, etc.
    status: str               # CONNECTED, OFFLINE, CONNECTING
    last_frame_time: Optional[float]
    reconnect_attempts: int
```

**`events/manager.py`**
```python
class EventManager:
    """Central event dispatch + deduplication"""
    
    def __init__(self):
        self.event_queue = queue.Queue()
        self.dedup_manager = EventDeduplicationManager()
        self.subscribers = []  # API, DB, WebSocket, etc.
    
    def emit(self, event: dict):
        """Publish event to all subscribers"""
        if self.dedup_manager.should_emit(event):
            self.event_queue.put(event)
            for subscriber in self.subscribers:
                subscriber(event)
    
    def subscribe(self, callback: callable):
        """Register event consumer"""
        self.subscribers.append(callback)
    
    def get_queue(self) -> queue.Queue:
        """For API/storage to consume events"""
        return self.event_queue

class EventDeduplicationManager:
    """Prevent event spam (e.g., PERSON_DETECTED every frame)"""
    
    def __init__(self, cooldown_seconds: float = 2.0):
        self.last_event = {}  # camera_id -> timestamp
        self.cooldown = cooldown_seconds
    
    def should_emit(self, event: dict) -> bool:
        camera_id = event["camera_id"]
        event_type = event["event_type"]
        now = time.time()
        last = self.last_event.get((camera_id, event_type), 0)
        
        if now - last > self.cooldown:
            self.last_event[(camera_id, event_type)] = now
            return True
        return False
```

---

### 4. STORAGE MODULE (`backend/src/storage/`)

**Purpose:** SQLite event logging (simple CRUD)

#### Files to Create

**`storage/models.py`**
```python
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EventRecord(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String(100), unique=True)
    timestamp = Column(Float)
    camera_id = Column(String(50), index=True)
    event_type = Column(String(50), index=True)
    person_count = Column(Integer)
    message = Column(Text)
    snapshot_path = Column(String(500))
    boxes = Column(JSON)  # YOLO detection boxes

class CameraStatus(Base):
    __tablename__ = "camera_status"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String(50), unique=True)
    status = Column(String(20))  # CONNECTED, OFFLINE, etc.
    last_seen = Column(Float)
    reconnect_count = Column(Integer, default=0)
```

**`storage/db.py`**
```python
class Database:
    """SQLite connection pool + CRUD"""
    
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def log_event(self, event: dict):
        """Insert event into DB"""
        session = self.Session()
        try:
            record = EventRecord(
                event_id=event["id"],
                timestamp=event["timestamp"],
                camera_id=event["camera_id"],
                # ... other fields
            )
            session.add(record)
            session.commit()
        finally:
            session.close()
    
    def get_recent_events(self, limit: int = 100) -> list[dict]:
        """Fetch last N events"""
        session = self.Session()
        try:
            records = session.query(EventRecord) \
                .order_by(EventRecord.timestamp.desc()) \
                .limit(limit) \
                .all()
            return [r.to_dict() for r in records]
        finally:
            session.close()
    
    def get_events_by_camera(self, camera_id: str, limit: int = 50) -> list[dict]:
        """Fetch events for specific camera"""
        session = self.Session()
        try:
            records = session.query(EventRecord) \
                .filter_by(camera_id=camera_id) \
                .order_by(EventRecord.timestamp.desc()) \
                .limit(limit) \
                .all()
            return [r.to_dict() for r in records]
        finally:
            session.close()
```

#### What Gets Deleted
- ❌ `storage/vector_db.py` (not needed for demo)
- ❌ `storage/persistence.py` (complex serialization)
- ✓ Keep: Simple SQLite logging

---

### 5. API MODULE (`backend/src/api/`)

**Purpose:** FastAPI server with REST + WebSocket

#### Files to Create

**`api/server.py`**
```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="HavenNet API")

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Store for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Send event to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
```

**`api/routes.py`**
```python
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

@router.get("/health")
async def health():
    """System health check"""
    return {"status": "ok"}

@router.get("/cameras")
async def list_cameras():
    """List all cameras + status"""
    # Return camera list with current status
    pass

@router.post("/cameras")
async def add_camera(config: dict):
    """Add new camera dynamically"""
    pass

@router.get("/events")
async def get_events(limit: int = 50):
    """Get recent events (paginated)"""
    pass

@router.get("/events/camera/{camera_id}")
async def get_camera_events(camera_id: str, limit: int = 50):
    """Get events for specific camera"""
    pass

@router.get("/stream/{camera_id}.mjpeg")
async def mjpeg_stream(camera_id: str):
    """Stream MJPEG video from specific camera"""
    # Yield frames from camera_worker.frame_queue
    pass

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for realtime events"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or handle incoming messages
    except:
        manager.disconnect(websocket)
```

#### What Gets Deleted
- ❌ Basic Flask templates
- ✓ Keep: FastAPI core

---

### 6. CONFIG MODULE (`backend/src/config.py`)

**Purpose:** Load YAML → Python dataclass

#### Files to Create

**`config.py`**
```python
from dataclasses import dataclass
import yaml

@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    timeout_seconds: int = 10
    reconnect_interval_seconds: int = 5
    target_fps: int = 30
    resolution: str = "1280x720"
    enabled: bool = True

@dataclass
class ProcessingConfig:
    model_path: str
    inference_fps: int = 20
    confidence_threshold: float = 0.5
    enable_pose: bool = False
    enable_posture: bool = False

@dataclass
class StorageConfig:
    db_path: str = "backend/outputs/events.db"
    snapshot_dir: str = "backend/outputs/snapshots"
    max_snapshots: int = 1000
    snapshot_ttl_days: int = 7

@dataclass
class HavenNetConfig:
    cameras: list[CameraConfig]
    processing: ProcessingConfig
    storage: StorageConfig
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    
    @classmethod
    def from_yaml(cls, path: str) -> "HavenNetConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Parse into dataclasses
        return cls(...)
```

**`backend/config.yaml` (unified)**
```yaml
api:
  host: "0.0.0.0"
  port: 8000

processing:
  model_path: "backend/models/yolo11n.pt"
  inference_fps: 20
  confidence_threshold: 0.5
  enable_pose: false
  enable_posture: false

storage:
  db_path: "backend/outputs/events.db"
  snapshot_dir: "backend/outputs/snapshots"
  max_snapshots: 1000
  snapshot_ttl_days: 7

cameras:
  - id: "camera_0"
    name: "Front Gate"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    timeout_seconds: 10
    reconnect_interval_seconds: 5
    target_fps: 30
    enabled: true
  
  - id: "camera_1"
    name: "Back Entrance"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    timeout_seconds: 10
    reconnect_interval_seconds: 5
    target_fps: 30
    enabled: true
```

#### What Gets Deleted
- ❌ `backend/src/config.yaml` (old file)
- ❌ `configs/unified_config.yaml` (unnecessary duplication)

---

### 7. MAIN ENTRY POINT (`backend/src/main.py`)

**Purpose:** Orchestrate all components

#### Files to Create

**`main.py`**
```python
import logging
import time
from config import HavenNetConfig
from camera.worker import CameraWorker
from processing.processor import ProcessingWorker
from events.manager import EventManager
from storage.db import Database
from api.server import app, manager as ws_manager
import uvicorn

logger = logging.getLogger(__name__)

class HavenNetSystem:
    def __init__(self, config_path: str):
        self.config = HavenNetConfig.from_yaml(config_path)
        self.db = Database(self.config.storage.db_path)
        self.event_manager = EventManager()
        self.camera_workers = []
        self.processing_worker = None
    
    def start(self):
        """Start all components"""
        logger.info("Starting HavenNet system...")
        
        # Start camera workers
        for cam_config in self.config.cameras:
            if cam_config.enabled:
                worker = CameraWorker(cam_config.id, cam_config.rtsp_url, cam_config)
                worker.start()
                self.camera_workers.append(worker)
                logger.info(f"Started camera worker: {cam_config.id}")
        
        # Start processing worker
        self.processing_worker = ProcessingWorker(self.camera_workers, self.config)
        self.processing_worker.start()
        logger.info("Started processing worker")
        
        # Subscribe event handlers
        self.event_manager.subscribe(self._on_event_db)
        self.event_manager.subscribe(self._on_event_websocket)
    
    def stop(self):
        """Stop all components gracefully"""
        logger.info("Stopping HavenNet system...")
        
        if self.processing_worker:
            self.processing_worker.stop()
        
        for worker in self.camera_workers:
            worker.stop()
        
        logger.info("HavenNet stopped")
    
    def _on_event_db(self, event: dict):
        """Store event in database"""
        self.db.log_event(event)
    
    def _on_event_websocket(self, event: dict):
        """Broadcast event to WebSocket clients"""
        # This will be called asynchronously from FastAPI context
        # Use asyncio.run_coroutine_threadsafe if needed
        pass

# Global system instance
system = None

@app.on_event("startup")
async def startup_event():
    global system
    system = HavenNetSystem("backend/config.yaml")
    system.start()

@app.on_event("shutdown")
async def shutdown_event():
    global system
    if system:
        system.stop()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

## FILE-LEVEL CHANGES

### DELETE (Already Analyzed - Not Needed for MVP)

```
❌ backend/src/run.py                    # Replaced by processing/processor.py
❌ backend/src/reid.py                   # Remove fragile global ReID
❌ backend/src/adl.py                    # Defer posture to Phase 2
❌ backend/src/core/                     # Delete entire folder
❌ backend/src/storage/vector_db.py      # Remove embedding DB
❌ backend/src/storage/persistence.py    # Simplify to storage/db.py
❌ configs/unified_config.yaml           # Merge to backend/config.yaml
❌ backend/src/templates/index.html      # Refactor for web/static/
```

### MODIFY (Keep but Refactor)

```
✏️ backend/src/app.py
   - Merge into api/server.py + api/routes.py
   - Add full REST endpoint coverage
   - Add WebSocket support
   - Remove old code

✏️ backend/src/visualize.py
   - Keep visualization utilities for debugging
   - Move to utils/visualization.py
   - Not used in web API (for dev only)

✏️ requirements.txt
   - Remove unused deps: torch, torchvision (just YOLO will pull what's needed)
   - Remove obsolete: vector-db libraries, etc.
   - Add: fastapi, uvicorn, pydantic, sqlalchemy
   - See cleanup section below

✏️ backend/src/config.yaml
   - Migrate to structured YAML in backend/config.yaml
   - Remove hardcoded D:/ paths
   - Use relative paths
```

### CREATE (New MVP Components)

```
✨ backend/src/camera/                   # New module
✨ backend/src/processing/               # New module
✨ backend/src/events/                   # New module
✨ backend/src/api/                      # Refactored module
✨ backend/src/config.py                 # New config module
✨ backend/src/logger.py                 # Unified logging
✨ backend/src/main.py                   # New entry point
✨ backend/src/utils/                    # Shared utilities
✨ backend/config.yaml                   # Unified config
✨ backend/pyproject.toml                # Modern Python project
✨ tests/                                # Full test suite
✨ docs/                                 # Implementation guides
```

---

## CLEANED-UP REQUIREMENTS.txt

```plaintext
# Core CV/ML
opencv-python==4.8.1.78
ultralytics==8.0.200  # YOLO (will pull torch/torchvision automatically if needed)
numpy==1.24.3

# Backend
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
sqlite3  # Built-in, but explicitly listed

# Config
pyyaml==6.0.1

# Utility
python-dotenv==1.0.0
requests==2.31.0

# Development (optional, for testing)
pytest==7.4.3
pytest-asyncio==0.21.1

# Deployment (optional, for Raspberry Pi)
gunicorn==21.2.0
```

**What's Removed:**
- ❌ `torch`, `torchvision` (YOLO will auto-install if needed)
- ❌ `faiss`, `vector-db` libraries
- ❌ `scipy` (if not used)
- ❌ `pandas` (only for logging CSV, use sqlite instead)
- ❌ unused visualization libs

---

## IMPLEMENTATION SEQUENCE

### Phase 1: Core Infrastructure (Week 1)

**Priority 1: Bootable System**
1. Create folder structure (camera/, processing/, events/, storage/, api/)
2. Implement `config.py` + load YAML
3. Implement `camera/rtsp_client.py` + `camera/frame_queue.py`
4. Implement `camera/worker.py` (test with real RTSP)
5. Implement `processing/models.py` + `processing/processor.py`
6. Implement `events/schema.py` + `events/manager.py`
7. Implement `storage/models.py` + `storage/db.py`
8. Create `main.py` orchestrator

**Deliverable:** System boots, captures from 2 cameras, logs detections to SQLite

### Phase 2: Web API (Week 1-2)

**Priority 2: Demo Dashboard**
1. Implement `api/server.py` + `api/routes.py`
2. Add REST endpoints (cameras, events, health)
3. Add MJPEG stream endpoint
4. Add WebSocket support
5. Build simple React dashboard
6. Connect frontend to WebSocket (realtime updates)

**Deliverable:** Web UI shows 2 cameras live with detection logs

### Phase 3: Polish & Testing (Week 2-3)

**Priority 3: Production Ready**
1. Write unit tests (tests/)
2. Write integration tests
3. Document architecture (docs/)
4. Benchmark FPS/latency
5. Test camera reconnect logic
6. Create startup script
7. Validate on Raspberry Pi 5

**Deliverable:** Stable 30-minute run on localhost

### Phase 4: Deployment (Week 3+)

**Priority 4: Field Ready**
1. Create Pi 5 deployment scripts
2. Test with travel router setup
3. Package as Docker (optional)
4. Create Pi 5 image (optional)

**Deliverable:** Deploy to Pi 5, run in field

---

## MIGRATION STRATEGY

### How to Migrate Without Breaking Things

**Approach: Parallel Implementation**

1. **Branch: `feature/mvp-refactor`**
   - Implement all new modules alongside old code
   - Don't delete anything yet

2. **Phase A (Days 1-3): Core modules**
   ```bash
   git checkout -b feature/mvp-refactor
   mkdir -p backend/src/{camera,processing,events,storage,api,utils}
   # Create all new files
   ```

3. **Phase B (Days 4-5): Integration**
   - Get system working with new code
   - Keep old code as fallback
   - Test thoroughly

4. **Phase C (Day 6): Cutover**
   - Flip `main.py` to use new code
   - Delete old code (reid.py, adl.py, etc.)
   - Push to `audit-and-refactor-project` branch

5. **Phase D (Day 7+): Testing**
   - Run demo with new system
   - Validate all features work

### Rollback Plan

If new code has issues:
```bash
git revert <commit>  # Go back to parallel state
Switch back to old run.py as fallback
Diagnose issue
Re-implement and retry
```

---

## TESTING & VALIDATION CHECKLIST

### Unit Tests

- [ ] `test_camera_worker.py`
  - [ ] RTSP connection (mock)
  - [ ] Frame queue (bounded)
  - [ ] Reconnect logic
  - [ ] Status reporting

- [ ] `test_detector.py`
  - [ ] YOLO inference
  - [ ] Batch processing
  - [ ] Empty frame handling

- [ ] `test_events.py`
  - [ ] Event schema validation
  - [ ] Deduplication logic
  - [ ] Publishing to subscribers

- [ ] `test_storage.py`
  - [ ] Database creation
  - [ ] Event logging
  - [ ] Query operations
  - [ ] Retention policy

### Integration Tests

- [ ] `test_api.py`
  - [ ] `/api/health` endpoint
  - [ ] `/api/cameras` listing
  - [ ] `/api/events` pagination
  - [ ] WebSocket connection + broadcast
  - [ ] MJPEG stream serving

- [ ] `test_system_integration.py`
  - [ ] End-to-end: camera → inference → event → DB → API
  - [ ] Concurrent camera capture
  - [ ] Camera reconnect scenario
  - [ ] Event deduplication
  - [ ] Memory leaks (long run)

### Performance Tests

- [ ] `benchmark.py`
  - [ ] Capture FPS per camera
  - [ ] Inference latency
  - [ ] E2E latency (camera → UI)
  - [ ] Memory footprint
  - [ ] CPU usage (2 cameras)
  - [ ] CPU usage (4 cameras)

### Field Validation (Demo)

- [ ] 2-4 RTSP cameras online simultaneously
- [ ] Detection accuracy >85% (person detection)
- [ ] Live dashboard updates in <1s
- [ ] System runs stable for 30+ minutes
- [ ] Camera reconnect works (pull network cable, reconnect)
- [ ] SQLite log grows correctly (1000+ events)
- [ ] No memory leaks after 30 minutes

---

## SUCCESS CRITERIA

By end of refactor, system must:

✅ **Functional**
- Capture from 2-4 RTSP cameras in parallel
- Detect persons in realtime (>20 FPS)
- Stream to web dashboard with <1s latency
- Log events to SQLite (no data loss)

✅ **Maintainable**
- New developer can understand architecture in 1 hour
- Clear module boundaries
- Testable components (unit + integration)
- Well-documented with examples

✅ **Reliable**
- Handles camera disconnects gracefully
- Auto-reconnect without user intervention
- Runs stable for 30+ minutes
- No memory leaks

✅ **Expandable**
- Easy to add 3rd or 4th camera (YAML config only)
- Easy to add new event types
- Easy to swap YOLO models
- Easy to add features (pose, posture, etc.) in Phase 2

---

## NEXT STEPS

1. **This Week:** Approve PHẦN C structure
2. **Next:** Create PHẦN D — Implementation Phase Plan (detailed sprints)
3. **Then:** Code implementation begins

