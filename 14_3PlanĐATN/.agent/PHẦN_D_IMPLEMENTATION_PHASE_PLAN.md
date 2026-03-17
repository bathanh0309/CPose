# PHẦN D — IMPLEMENTATION PHASE PLAN
## Chi Tiết Từng Sprint Cho Phát Triển MVP

**Document Version:** 1.0  
**Status:** Ready for Development  
**Total Duration:** 3-4 weeks  
**Branch:** `audit-and-refactor-project` → feature branches per sprint

---

## TABLE OF CONTENTS

1. [Sprint Overview](#sprint-overview)
2. [Sprint 1: Foundation & Core Infrastructure](#sprint-1-foundation--core-infrastructure)
3. [Sprint 2: Processing Pipeline & Events](#sprint-2-processing-pipeline--events)
4. [Sprint 3: API & Dashboard](#sprint-3-api--dashboard)
5. [Sprint 4: Testing & Deployment](#sprint-4-testing--deployment)
6. [Daily Standup Template](#daily-standup-template)
7. [Risk Management & Rollback](#risk-management--rollback)

---

## SPRINT OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│ SPRINT 1 (Days 1-7): Foundation & Bootable System               │
│ ✓ Config system          ✓ Camera worker    ✓ Basic YOLO setup   │
│ ✓ Logger module          ✓ Frame queue      ✓ SQLite DB          │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│ SPRINT 2 (Days 8-14): Processing & Events                       │
│ ✓ Inference pipeline     ✓ Event schema    ✓ Deduplication       │
│ ✓ Detector wrapper       ✓ Event manager   ✓ Storage CRUD        │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│ SPRINT 3 (Days 15-21): API & Dashboard                          │
│ ✓ FastAPI setup          ✓ REST endpoints  ✓ WebSocket           │
│ ✓ Health checks          ✓ Stream UI       ✓ Metrics endpoint    │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│ SPRINT 4 (Days 22-28): Testing & Deployment                     │
│ ✓ Unit tests             ✓ Integration tests  ✓ Benchmarking      │
│ ✓ Error handling         ✓ Docker image       ✓ Deployment docs   │
└─────────────────────────────────────────────────────────────────┘
```

---

## SPRINT 1: Foundation & Core Infrastructure
### Days 1-7 | Goal: Bootable system with parallel camera workers

### Day 1-2: Configuration & Logging System

**Files to Create:**
- `backend/src/config.py` — Config dataclass system
- `backend/src/logger.py` — Unified logging
- `backend/config.yaml` — Updated YAML config

**Tasks:**
1. Parse `config.yaml` → Python dataclasses (using `dataclasses` module)
2. Support environment variable overrides
3. Create unified logger with both file + console output
4. Add colored console output for development

**Code Skeleton:**
```python
# backend/src/config.py
from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    resolution: tuple = (1280, 720)
    fps: int = 30
    buffer_size: int = 5

@dataclass
class ProcessingConfig:
    model_path: str = "models/yolov8m.pt"
    confidence: float = 0.45
    max_workers: int = 1
    device: str = "cuda"  # or "cpu"

@dataclass
class AppConfig:
    cameras: List[CameraConfig]
    processing: ProcessingConfig
    db_path: str = "data/havennet.db"
    log_level: str = "INFO"
```

**Acceptance Criteria:**
- [ ] Load `backend/config.yaml` without errors
- [ ] CLI can override config values: `--log-level DEBUG`
- [ ] Logs appear in `logs/havennet.log`
- [ ] Config validation catches missing required fields

---

### Day 3-4: Camera Worker & Frame Queue

**Files to Create:**
- `backend/src/camera/rtsp_client.py` — RTSP connection wrapper
- `backend/src/camera/frame_queue.py` — Bounded queue implementation
- `backend/src/camera/worker.py` — Camera capture thread

**Tasks:**
1. Wrap OpenCV VideoCapture for RTSP URLs
2. Implement thread-safe bounded queue (max 5 frames)
3. Create worker thread that captures frames continuously
4. Handle disconnections + exponential backoff reconnect

**Code Skeleton:**
```python
# backend/src/camera/frame_queue.py
from queue import Queue
import threading

class FrameQueue:
    def __init__(self, max_size=5):
        self.queue = Queue(maxsize=max_size)
    
    def put(self, frame, block=False):
        try:
            self.queue.put_nowait(frame)
        except:
            # Drop oldest frame if full
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except:
                pass
    
    def get(self):
        return self.queue.get()

# backend/src/camera/worker.py
class CameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, frame_queue: FrameQueue, logger):
        self.config = config
        self.queue = frame_queue
        self.logger = logger
        self.running = True
        self.rtsp_client = RTSPClient(config.rtsp_url)
    
    def run(self):
        while self.running:
            frame = self.rtsp_client.read()
            if frame is not None:
                self.queue.put(frame)
            else:
                # Handle reconnect
                self.rtsp_client.reconnect()
```

**Acceptance Criteria:**
- [ ] Camera worker captures 30 FPS from RTSP stream
- [ ] Frame queue never exceeds 5 frames
- [ ] Handles network disconnects gracefully
- [ ] Worker thread can be stopped cleanly

---

### Day 5-6: SQLite Database Setup

**Files to Create:**
- `backend/src/storage/db.py` — SQLite connection pool
- `backend/src/storage/models.py` — Table schemas
- `backend/src/storage/queries.py` — CRUD operations

**Tasks:**
1. Create SQLite schema with 3 tables: `detections`, `events`, `camera_health`
2. Implement connection pool using `sqlite3`
3. Write CRUD methods for each table
4. Add schema versioning (migrations)

**Code Skeleton:**
```python
# backend/src/storage/models.py
import sqlite3
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    frame_num INTEGER,
    person_count INTEGER,
    postures TEXT,  -- JSON: {"standing": 5, "sitting": 2}
    confidence_avg REAL,
    snapshot_path TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    event_type TEXT,  -- "person_detected", "anomaly", etc.
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    details TEXT,  -- JSON
    UNIQUE(camera_id, event_type, timestamp)
);

CREATE TABLE IF NOT EXISTS camera_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    last_frame_time DATETIME,
    status TEXT,  -- "online", "offline"
    error_count INTEGER DEFAULT 0,
    UNIQUE(camera_id)
);
"""

# backend/src/storage/queries.py
class DetectionQueries:
    @staticmethod
    def insert_detection(db: sqlite3.Connection, camera_id, person_count, postures):
        db.execute("""
            INSERT INTO detections (camera_id, person_count, postures)
            VALUES (?, ?, ?)
        """, (camera_id, person_count, postures))
        db.commit()
    
    @staticmethod
    def get_latest_detection(db: sqlite3.Connection, camera_id):
        return db.execute("""
            SELECT * FROM detections WHERE camera_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (camera_id,)).fetchone()
```

**Acceptance Criteria:**
- [ ] Schema creates 3 tables on first run
- [ ] Can insert + retrieve detections
- [ ] Connection pool handles concurrent writes
- [ ] Database file created at `data/havennet.db`

---

### Day 7: Integration & Testing

**Tasks:**
1. Create `backend/src/main.py` that boots all components
2. Test config loading → camera workers → DB insert
3. Run 5 minutes with real RTSP stream (or simulated)
4. Verify logs appear, frames queue up, DB fills

**Test Script:**
```python
# Quick integration test
if __name__ == "__main__":
    config = load_config("backend/config.yaml")
    logger = setup_logger(config.log_level)
    
    # Boot camera workers
    workers = []
    frame_queues = {}
    for cam_cfg in config.cameras:
        queue = FrameQueue(max_size=5)
        worker = CameraWorker(cam_cfg, queue, logger)
        worker.start()
        workers.append(worker)
        frame_queues[cam_cfg.id] = queue
    
    # Read frames for 30 seconds
    import time
    start = time.time()
    while time.time() - start < 30:
        for cam_id, queue in frame_queues.items():
            frame = queue.get(timeout=1)
            logger.info(f"Got frame from {cam_id}: {frame.shape}")
            # Store to DB
```

**Acceptance Criteria:**
- [ ] System boots without errors
- [ ] Logs show "Camera X connected"
- [ ] Database has 50+ detection records after 5 min
- [ ] No memory leaks (monitor RAM)

---

## SPRINT 2: Processing Pipeline & Events
### Days 8-14 | Goal: Single inference worker + event system

### Day 8-9: YOLO Detector Wrapper

**Files to Create:**
- `backend/src/processing/detector.py` — YOLO wrapper
- `backend/src/processing/models.py` — Model loading

**Tasks:**
1. Load YOLOv8 model (person detection)
2. Implement inference on single frame
3. Extract person bounding boxes + confidences
4. Run pose estimation (separate model or same?)
5. Estimate posture (standing/sitting/lying) from skeleton

**Code Skeleton:**
```python
# backend/src/processing/detector.py
from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, model_path, device="cuda"):
        self.model = YOLO(model_path)
        self.model.to(device)
    
    def detect(self, frame):
        """
        Returns: List[Detection]
        """
        results = self.model(frame, conf=0.45)
        detections = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Class 0 = person
                    detections.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf),
                        "keypoints": result.keypoints  # If available
                    })
        
        return detections

class PostureClassifier:
    @staticmethod
    def classify(keypoints):
        """
        Returns: "standing" | "sitting" | "lying"
        Using y-coordinates of skeleton keypoints
        """
        if keypoints is None:
            return "unknown"
        
        # Get hip and shoulder height
        hip_y = keypoints[11, 1]  # COCO keypoint #11
        shoulder_y = keypoints[5, 1]
        ankle_y = keypoints[15, 1]
        
        # Heuristic: if hip-to-ankle distance > shoulder-to-hip, person is standing
        if ankle_y - hip_y > hip_y - shoulder_y:
            return "standing"
        else:
            return "sitting"  # Simplified
```

**Acceptance Criteria:**
- [ ] Detect persons in sample image
- [ ] Return 5-10 detections with confidence > 0.45
- [ ] Classify posture for each person
- [ ] Inference time < 100ms per frame

---

### Day 10-11: Inference Pipeline & Event Manager

**Files to Create:**
- `backend/src/processing/processor.py` — Inference loop
- `backend/src/events/schema.py` — Event TypedDict
- `backend/src/events/manager.py` — Event deduplication + publishing

**Tasks:**
1. Create inference worker that pulls frames from all queues
2. Run detection on each frame
3. Generate events (person_detected, count_change, etc.)
4. Deduplicate events (don't spam same event 30x/sec)
5. Publish events to storage + event subscribers

**Code Skeleton:**
```python
# backend/src/events/schema.py
from typing import TypedDict

class DetectionEvent(TypedDict):
    event_id: str
    camera_id: str
    timestamp: str
    event_type: str  # "person_detected", "count_change"
    person_count: int
    postures: dict  # {"standing": 5, "sitting": 2}
    confidence: float

# backend/src/events/manager.py
from collections import defaultdict
from datetime import datetime, timedelta

class EventManager:
    def __init__(self, dedup_window_sec=3):
        self.recent_events = defaultdict(lambda: None)
        self.dedup_window = timedelta(seconds=dedup_window_sec)
        self.subscribers = []
    
    def publish(self, event: DetectionEvent):
        """
        Only emit event if it's different from last event for this camera
        """
        cam_id = event["camera_id"]
        last_event = self.recent_events[cam_id]
        
        # Check if event is "new"
        if last_event is None or self._is_different(last_event, event):
            self.recent_events[cam_id] = event
            
            # Notify subscribers
            for subscriber in self.subscribers:
                subscriber(event)
    
    def subscribe(self, callback):
        self.subscribers.append(callback)
    
    @staticmethod
    def _is_different(old_event, new_event):
        # Consider different if person count changed by ≥2
        return abs(old_event["person_count"] - new_event["person_count"]) >= 2

# backend/src/processing/processor.py
class InferenceProcessor(threading.Thread):
    def __init__(self, detector, event_manager, frame_queues, logger):
        self.detector = detector
        self.event_manager = event_manager
        self.frame_queues = frame_queues  # Dict[camera_id -> queue]
        self.logger = logger
        self.running = True
    
    def run(self):
        while self.running:
            for camera_id, queue in self.frame_queues.items():
                try:
                    frame = queue.get(timeout=0.1)
                    detections = self.detector.detect(frame)
                    person_count = len(detections)
                    
                    # Build event
                    event = DetectionEvent(
                        event_id=f"{camera_id}_{time.time()}",
                        camera_id=camera_id,
                        timestamp=datetime.now().isoformat(),
                        event_type="person_detected",
                        person_count=person_count,
                        postures={...},
                        confidence=np.mean([d["confidence"] for d in detections])
                    )
                    
                    # Publish (will deduplicate)
                    self.event_manager.publish(event)
                except:
                    pass
```

**Acceptance Criteria:**
- [ ] Inference loop processes all camera queues
- [ ] Events are deduplicated (max 1 event per 3 sec per camera)
- [ ] Event schema matches TypedDict
- [ ] Events persist to database via subscriber

---

### Day 12-13: Event Subscribers & Storage Integration

**Tasks:**
1. Create database subscriber that logs events
2. Create camera health subscriber (tracks connectivity)
3. Test event flow: detection → event → database

**Code Skeleton:**
```python
# Integration
def create_db_subscriber(db_connection, logger):
    def on_event(event: DetectionEvent):
        DetectionQueries.insert_detection(
            db=db_connection,
            camera_id=event["camera_id"],
            person_count=event["person_count"],
            postures=json.dumps(event["postures"]),
            confidence=event["confidence"]
        )
        logger.info(f"Stored event: {event['event_id']}")
    
    return on_event

# In main.py
event_manager.subscribe(create_db_subscriber(db, logger))
```

**Acceptance Criteria:**
- [ ] Each event is stored to database
- [ ] No duplicate events in DB (deduplication works)
- [ ] Camera health table updated on connection events

---

### Day 14: Integration & Load Testing

**Tasks:**
1. Run full pipeline: cameras → detection → events → database
2. Stress test with 2-4 simulated cameras
3. Monitor latency (frame → detection → event ≤ 500ms)
4. Check memory usage (should be < 500 MB)

**Metrics to Capture:**
- FPS achieved per camera
- Detection latency (ms)
- Event throughput (events/sec)
- Memory usage (MB)
- CPU usage (%)

**Acceptance Criteria:**
- [ ] Process 2+ cameras simultaneously
- [ ] Detection latency < 100ms
- [ ] End-to-end latency < 500ms
- [ ] Memory usage < 500 MB

---

## SPRINT 3: API & Dashboard
### Days 15-21 | Goal: Web interface for realtime monitoring

### Day 15-16: FastAPI Setup & REST Endpoints

**Files to Create:**
- `backend/src/api/server.py` — FastAPI app
- `backend/src/api/routes.py` — REST endpoints
- `backend/src/api/models.py` — Pydantic schemas

**Tasks:**
1. Create FastAPI application
2. Add endpoints:
   - `GET /api/cameras` — List all cameras
   - `GET /api/cameras/{id}/status` — Camera health
   - `GET /api/detections/{camera_id}?limit=100` — Latest detections
   - `POST /api/cameras/{id}/snapshot` — Capture frame
   - `GET /api/metrics` — System metrics
3. Add CORS for frontend

**Code Skeleton:**
```python
# backend/src/api/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HavenNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# backend/src/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")

@router.get("/cameras")
async def list_cameras():
    """Return all camera configurations"""
    return {
        "cameras": [
            {
                "id": "camera_1",
                "name": "Entrance",
                "rtsp_url": "...",
                "status": "online"
            }
        ]
    }

@router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """Return current status + latest detection"""
    return {
        "camera_id": camera_id,
        "status": "online",
        "last_frame": "2024-03-14T10:30:00Z",
        "person_count": 5,
        "postures": {"standing": 3, "sitting": 2}
    }

@router.get("/detections/{camera_id}")
async def get_detections(camera_id: str, limit: int = 100):
    """Return latest detections from database"""
    detections = DetectionQueries.get_latest(db, camera_id, limit)
    return {"detections": detections}

@router.post("/cameras/{camera_id}/snapshot")
async def capture_snapshot(camera_id: str):
    """Capture current frame from camera"""
    frame = frame_queues[camera_id].get()
    path = save_snapshot(frame, camera_id)
    return {"snapshot_url": f"/snapshots/{path}"}

@router.get("/metrics")
async def get_metrics():
    """Return system metrics"""
    return {
        "cameras_online": 4,
        "total_persons_detected": 127,
        "memory_usage_mb": 256,
        "cpu_percent": 45.2
    }
```

**Acceptance Criteria:**
- [ ] All endpoints return valid JSON
- [ ] Camera status updates in real-time
- [ ] Detections queryable by limit + camera
- [ ] Snapshot endpoint saves frame to disk

---

### Day 17-18: WebSocket for Realtime Updates

**Files to Create:**
- `backend/src/api/websocket.py` — WebSocket handler

**Tasks:**
1. Create WebSocket endpoint at `/ws/stream`
2. Subscribe to event manager
3. Broadcast events to connected clients
4. Handle connection/disconnection

**Code Skeleton:**
```python
# backend/src/api/websocket.py
from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# In main.py, subscribe event manager to broadcast
def websocket_subscriber(event: DetectionEvent):
    import asyncio
    asyncio.create_task(manager.broadcast(event))

event_manager.subscribe(websocket_subscriber)

# In server.py
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except:
        manager.disconnect(websocket)
```

**Acceptance Criteria:**
- [ ] Clients can connect to WebSocket
- [ ] Events broadcast to all connected clients
- [ ] Browser console shows event stream
- [ ] No memory leak on long connections

---

### Day 19-20: Dashboard UI

**Files to Create:**
- `backend/src/web/static/index.html` — Main dashboard
- `backend/src/web/static/css/style.css` — Styling
- `backend/src/web/static/js/dashboard.js` — Frontend logic

**Tasks:**
1. Create dashboard with:
   - Camera grid (2x2 or 1x4 layout)
   - Live snapshot feed
   - Real-time person count per camera
   - Event log (last 20 events)
   - System metrics sidebar
2. Use WebSocket to update in realtime
3. Add responsive design for mobile

**Code Skeleton (HTML):**
```html
<!-- backend/src/web/static/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>HavenNet Dashboard</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>HavenNet Monitor</h1>
            <div id="metrics" class="metrics">
                <span>Cameras: <strong id="cameras-online">0</strong>/4</span>
                <span>Persons: <strong id="total-persons">0</strong></span>
            </div>
        </header>
        
        <div class="camera-grid" id="camera-grid">
            <!-- Camera tiles generated by JS -->
        </div>
        
        <div class="event-log">
            <h3>Recent Events</h3>
            <table id="event-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Camera</th>
                        <th>Event</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody id="events"></tbody>
            </table>
        </div>
    </div>
    
    <script src="js/dashboard.js"></script>
</body>
</html>
```

**Code Skeleton (JavaScript):**
```javascript
// backend/src/web/static/js/dashboard.js
const ws = new WebSocket("ws://localhost:8000/ws/stream");
const eventLog = [];

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateCameraDisplay(data);
    updateEventLog(data);
};

function updateCameraDisplay(event) {
    const cameraId = event.camera_id;
    const tile = document.querySelector(`[data-camera="${cameraId}"]`);
    
    if (tile) {
        tile.querySelector(".person-count").textContent = event.person_count;
        tile.querySelector(".postures").textContent = 
            `Standing: ${event.postures.standing}, Sitting: ${event.postures.sitting}`;
    }
}

function updateEventLog(event) {
    eventLog.unshift(event);
    if (eventLog.length > 20) eventLog.pop();
    
    const tbody = document.querySelector("#events");
    tbody.innerHTML = eventLog.map(e => `
        <tr>
            <td>${new Date(e.timestamp).toLocaleTimeString()}</td>
            <td>${e.camera_id}</td>
            <td>${e.event_type}</td>
            <td>${e.person_count} persons</td>
        </tr>
    `).join("");
}

// Load cameras on startup
fetch("/api/cameras")
    .then(r => r.json())
    .then(data => {
        const grid = document.querySelector("#camera-grid");
        grid.innerHTML = data.cameras.map(cam => `
            <div class="camera-tile" data-camera="${cam.id}">
                <h3>${cam.name}</h3>
                <div class="snapshot" id="snap-${cam.id}"></div>
                <div class="person-count">0</div>
                <div class="postures"></div>
                <button onclick="captureSnapshot('${cam.id}')">Snapshot</button>
            </div>
        `).join("");
    });
```

**Acceptance Criteria:**
- [ ] Dashboard loads without errors
- [ ] WebSocket connects and receives events
- [ ] Camera tiles update in real-time
- [ ] Event log populated with latest 20 events
- [ ] Responsive on mobile (tested in Chrome DevTools)

---

### Day 21: API Testing & Documentation

**Tasks:**
1. Test all API endpoints with Postman or curl
2. Generate OpenAPI docs (automatic with FastAPI)
3. Write quick deployment guide
4. Test WebSocket from browser console

**Acceptance Criteria:**
- [ ] All 5 REST endpoints return valid responses
- [ ] WebSocket test successful
- [ ] OpenAPI docs accessible at `/docs`
- [ ] Dashboard loads in browser

---

## SPRINT 4: Testing & Deployment
### Days 22-28 | Goal: Production-ready system

### Day 22-23: Unit Tests

**Files to Create:**
- `tests/test_config.py` — Config loading tests
- `tests/test_detector.py` — YOLO detector tests
- `tests/test_camera_worker.py` — Camera worker tests
- `tests/test_events.py` — Event deduplication tests
- `tests/conftest.py` — Pytest fixtures

**Test Coverage:**
- Config loading (valid + invalid YAML)
- Frame queue (bounded, FIFO)
- Detector output (valid bboxes + confidences)
- Event deduplication (spam filtering)
- Database CRUD (insert + query)

**Code Skeleton:**
```python
# tests/conftest.py
import pytest
from backend.src.config import load_config
from backend.src.processing.detector import PersonDetector

@pytest.fixture
def config():
    return load_config("backend/config.yaml")

@pytest.fixture
def detector():
    return PersonDetector("models/yolov8m.pt", device="cpu")

@pytest.fixture
def sample_frame():
    import cv2
    import numpy as np
    return np.zeros((720, 1280, 3), dtype=np.uint8)

# tests/test_detector.py
def test_detector_output_format(detector, sample_frame):
    result = detector.detect(sample_frame)
    assert isinstance(result, list)
    for det in result:
        assert "bbox" in det
        assert "confidence" in det

def test_detector_confidence_threshold(detector, sample_frame):
    result = detector.detect(sample_frame)
    for det in result:
        assert det["confidence"] >= 0.45
```

**Run Tests:**
```bash
pytest tests/ -v --cov=backend/src
```

**Acceptance Criteria:**
- [ ] 15+ unit tests written
- [ ] All tests pass
- [ ] Code coverage > 70%
- [ ] CI/CD configured (GitHub Actions)

---

### Day 24-25: Integration Tests

**Files to Create:**
- `tests/test_integration_e2e.py` — End-to-end flow test
- `tests/test_api_endpoints.py` — API endpoint tests

**Test Scenarios:**
1. Boot system → capture frame → detect → event → database
2. API: GET /api/cameras → valid JSON
3. API: POST /api/cameras/{id}/snapshot → file saved
4. WebSocket: Connect → receive 5 events → disconnect
5. Stress test: 2 cameras, 30 sec, no crashes

**Code Skeleton:**
```python
# tests/test_integration_e2e.py
@pytest.mark.integration
def test_full_pipeline(config, detector, db, event_manager):
    """Test: Frame → Detection → Event → Database"""
    
    # Create event
    event = DetectionEvent(
        event_id="test_1",
        camera_id="camera_1",
        timestamp=datetime.now().isoformat(),
        event_type="person_detected",
        person_count=3,
        postures={"standing": 3},
        confidence=0.85
    )
    
    # Publish
    event_manager.publish(event)
    
    # Verify in database
    stored = DetectionQueries.get_latest(db, "camera_1")
    assert stored["person_count"] == 3

# tests/test_api_endpoints.py
@pytest.mark.integration
def test_api_camera_status(client):
    response = client.get("/api/cameras/camera_1/status")
    assert response.status_code == 200
    assert "person_count" in response.json()
```

**Acceptance Criteria:**
- [ ] 5+ integration tests written
- [ ] Full E2E pipeline tested
- [ ] WebSocket test passes
- [ ] Stress test runs 30 sec without crash

---

### Day 26: Performance Benchmarking

**Tasks:**
1. Measure inference latency (target: < 100ms)
2. Measure end-to-end latency (frame → display)
3. Measure memory usage under load
4. Measure CPU usage

**Benchmark Script:**
```python
# benchmark.py
import time
import psutil

def benchmark_detector(detector, frame, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detector.detect(frame)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    print(f"Inference latency (ms): {np.mean(times):.2f} ± {np.std(times):.2f}")
    print(f"Max: {np.max(times):.2f}, Min: {np.min(times):.2f}")

def benchmark_memory():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {mem:.1f} MB")

def benchmark_cpu():
    cpu = psutil.cpu_percent(interval=1)
    print(f"CPU usage: {cpu:.1f}%")
```

**Acceptance Criteria:**
- [ ] Detection latency < 100ms
- [ ] End-to-end latency < 500ms
- [ ] Memory usage < 500 MB
- [ ] CPU usage < 80%

---

### Day 27: Docker & Deployment

**Files to Create:**
- `Dockerfile` — Container image
- `docker-compose.yml` — Local dev environment
- `DEPLOYMENT.md` — Deployment guide

**Tasks:**
1. Create Dockerfile with Python + OpenCV + YOLO
2. Create docker-compose for local testing
3. Test container runs end-to-end
4. Create Raspberry Pi deployment docs

**Dockerfile Skeleton:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/src ./src
COPY backend/config.yaml .

CMD ["python", "-m", "src.main"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./backend/config.yaml:/app/config.yaml
      - ./data:/app/data
    environment:
      - LOG_LEVEL=INFO
```

**Acceptance Criteria:**
- [ ] Docker image builds successfully
- [ ] Container runs and accepts requests
- [ ] Volume mounts work (config + data persist)
- [ ] Deployment guide written

---

### Day 28: Final Testing & Handoff

**Tasks:**
1. Full 30-minute stability test with 2-4 cameras
2. Verify all metrics + logs
3. Create handoff documentation
4. Test cleanup: remove debug code

**Stability Test Checklist:**
- [ ] All cameras stay online
- [ ] No memory leaks (RAM constant)
- [ ] Database size reasonable (<100 MB)
- [ ] WebSocket clients don't crash
- [ ] Logs are clean (no spam)

**Handoff Documentation:**
- [ ] README with setup instructions
- [ ] API documentation (OpenAPI link)
- [ ] Troubleshooting guide
- [ ] Performance baseline numbers
- [ ] Known limitations + Phase 2 roadmap

**Acceptance Criteria:**
- [ ] System runs stable for 30 minutes
- [ ] All documentation complete
- [ ] Code reviewed and approved
- [ ] Ready for demo + Phase 2 features

---

## DAILY STANDUP TEMPLATE

**Format:** 10-15 minutes, same time daily

```
STANDUP REPORT - Day X

✅ COMPLETED YESTERDAY:
  - Task 1 (file1.py)
  - Task 2 (file2.py)

🚀 PLANNED TODAY:
  - Task 3 (file3.py)
  - Task 4 (test_file4.py)

⚠️ BLOCKERS:
  - [If any] Describe issue + action plan

📊 METRICS:
  - Tests passing: 15/15
  - Code coverage: 72%
  - Estimation accuracy: ±20%
```

---

## RISK MANAGEMENT & ROLLBACK

### High-Risk Areas

| Risk | Mitigation | Rollback Plan |
|------|-----------|---------------|
| RTSP stream disconnects | Implement backoff + reconnect | Fall back to video files in Sprint 1 test |
| YOLO inference too slow | Test on actual hardware early | Use lighter model (YOLOv8n) |
| WebSocket broadcast storms | Implement event dedup + rate limit | Queue-based system instead |
| Database locks | Use connection pool + WAL mode | Switch to in-memory + file write buffer |
| Memory leak | Test with `memory_profiler` weekly | Kill + restart worker on memory threshold |

### Branch Strategy

```
main (stable, tag releases)
  ↑
  └─── audit-and-refactor-project (parent)
         ├─── sprint-1-foundation (Days 1-7)
         ├─── sprint-2-processing (Days 8-14)
         ├─── sprint-3-api (Days 15-21)
         └─── sprint-4-testing (Days 22-28)
```

Each sprint gets a PR with full test coverage before merge.

---

## SUCCESS CRITERIA FOR MVP

```
✓ System boots without error
✓ Processes 2-4 RTSP cameras in parallel
✓ Detects persons with >80% confidence
✓ Runs stable for 30 minutes
✓ Dashboard loads and updates in real-time
✓ All logs clean (no error spam)
✓ < 500 MB memory usage
✓ < 80% CPU usage
✓ < 500ms end-to-end latency
✓ 70%+ test coverage
✓ Deployable via Docker
```

---

## APPENDIX: Code Snippets by File

### Configuration Management
See Day 1-2 section above.

### Threading Best Practices
- Always use `threading.Lock()` for shared state
- Use `queue.Queue` for thread-safe communication
- Implement graceful shutdown: `self.running = False` flag
- Test with `threading.enumerate()` to verify all threads stopped

### Error Handling Pattern
```python
try:
    # Operation
except SpecificException as e:
    logger.error(f"Specific error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
finally:
    # Cleanup
```

### Testing Pattern
```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
])
def test_function(input, expected):
    assert function(input) == expected
```

---

**Document Created:** 2024-03-14  
**Next Review:** After Sprint 1 completion  
**Owner:** Development Team  
**Status:** Ready for Implementation
