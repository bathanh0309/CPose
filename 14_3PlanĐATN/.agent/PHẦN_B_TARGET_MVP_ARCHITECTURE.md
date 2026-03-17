# PHẦN B — TARGET MVP ARCHITECTURE
## HavenNet Demo Surveillance System (Multi-Camera RTSP Monitoring)

**Document Version:** 1.0  
**Status:** Design Phase  
**Target Environment:** localhost → Raspberry Pi 5 4GB  
**Demo Timeline:** Phase 1 (2 cameras), Phase 2+ (3-4 cameras)

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Components](#architecture-components)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Threading & Concurrency Model](#threading--concurrency-model)
6. [Event System](#event-system)
7. [Configuration Strategy](#configuration-strategy)
8. [Performance Characteristics](#performance-characteristics)
9. [Scalability Path (2→4 cameras)](#scalability-path-2→4-cameras)
10. [Raspberry Pi Adaptation Strategy](#raspberry-pi-adaptation-strategy)

---

## EXECUTIVE SUMMARY

### What We're Building
A **realtime multi-camera RTSP monitoring dashboard** that:
- Connects 2-4 IP cameras via RTSP protocol
- Detects humans in realtime using YOLO lightweight models
- Displays a live dashboard with camera feeds and detection logs
- Handles camera disconnects gracefully with auto-reconnect
- Runs on localhost first, then ports to Raspberry Pi 5

### What We're NOT Doing (This Phase)
- ❌ Global multi-camera person ReID (too complex, fragile)
- ❌ Fall detection (needs more data, untested)
- ❌ Zone intrusion detection (not critical for demo)
- ❌ Face recognition / age/gender classification
- ❌ Over-engineered tracking systems

### Key Design Principles
1. **Demo-First** — Every design decision prioritizes demo stability
2. **Realtime-First** — Parallel camera workers, frame prioritization
3. **Simplicity-First** — No microservices, no distributed complexity
4. **Expandable** — Easy to add cameras without rewrite

---

## SYSTEM OVERVIEW

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       HAVENNET DEMO SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

                    NETWORK LAYER (LAN)
                    ┌────────────────────┐
                    │   2-4 RTSP Cameras │
                    │   (IP addresses)   │
                    └─────────┬──────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
        ┌───────▼────────┐         ┌────────▼────────┐
        │  Camera Worker │         │ Camera Worker   │
        │   (Thread 1)   │         │  (Thread N)     │
        │  - Connect     │         │ - Connect       │
        │  - Capture     │         │ - Capture       │
        │  - Queue Mgmt  │         │ - Queue Mgmt    │
        └────────┬───────┘         └────────┬────────┘
                 │                         │
                 └─────────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Processing Layer   │
                    │ (Shared Inference)  │
                    │  - YOLO Detection   │
                    │  - Pose (optional)  │
                    │  - Event Creation   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼───────┐  ┌──▼────┐  ┌────▼──────┐
        │  Event Queue  │  │SQLite │  │Snapshots  │
        │  (Realtime)   │  │ Log   │  │(outputs/) │
        └───────┬───────┘  └───────┘  └───────────┘
                │
        ┌───────▼──────────────┐
        │   WebSocket Server   │
        │  (FastAPI + uvicorn) │
        │ - Push live events   │
        │ - Camera status      │
        │ - Detection logs     │
        └───────┬──────────────┘
                │
        ┌───────▼──────────────┐
        │   REST API Endpoints │
        │ - Camera management  │
        │ - Config retrieval   │
        │ - Log history        │
        └───────┬──────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
 ┌──▼──┐  ┌─────▼──┐  ┌────▼────────┐
 │ Web │  │ MJPEG  │  │  WebSocket  │
 │ UI  │  │ Streams│  │  Events     │
 └─────┘  └────────┘  └─────────────┘

Frontend: React/Vue (localhost:3000)
Backend: FastAPI (localhost:8000)
```

---

## ARCHITECTURE COMPONENTS

### 1. INPUT LAYER — Camera Workers

**Purpose:** Capture RTSP streams independently without blocking others

**Architecture:**
```
Camera Worker Thread (per camera)
├─ RTSPConnection Manager
│  ├─ Connect with retry logic
│  ├─ Heartbeat monitor
│  ├─ Auto-reconnect on timeout
│  └─ RTSP frame streaming
├─ BoundedQueue (max 5 frames)
│  ├─ Drop old frames when full
│  ├─ Signal processing latency
│  └─ Prevent unbounded memory
└─ Status Publisher
   ├─ CONNECTED / OFFLINE / CONNECTING
   └─ Reconnect count
```

**Key Behavior:**
- Each camera runs in isolated thread (not sequential loop)
- Queue is bounded to 5 frames max
- If queue is full, oldest frame is dropped (prefer freshness)
- Timeout after 10 seconds of no data → mark OFFLINE
- Automatic reconnect every 5 seconds if disconnected
- Frame metadata includes capture timestamp

**Configuration per Camera:**
```yaml
camera_0:
  id: "camera_0"
  name: "Front Gate"
  rtsp_url: "rtsp://192.168.1.100:554/stream"
  timeout_seconds: 10
  reconnect_interval_seconds: 5
  target_fps: 30  # max FPS we want to process
  resolution: "1280x720"  # capture resolution
  enabled: true
```

---

### 2. PROCESSING LAYER — Inference Pipeline

**Purpose:** Run detection models without blocking camera capture

**Architecture:**
```
Shared Inference Pipeline
├─ Frame Intake
│  ├─ Get latest frame from all queues
│  ├─ Skip stale frames
│  └─ Pack batch (if 2-4 cameras)
├─ YOLO Person Detection
│  ├─ Input: frame 640×480 (default)
│  ├─ Model: YOLOv8n or YOLOv5s (nano/small)
│  ├─ Output: bbox + confidence
│  └─ Latency target: <50ms (desktop)
├─ Optional Pose (if enabled)
│  ├─ HRNet or MediaPipe
│  ├─ Output: 17-keypoint skeleton
│  └─ Latency: <100ms
├─ Optional Posture (if enabled)
│  ├─ Rule-based or simple classifier
│  ├─ Output: standing/sitting/lying
│  └─ Latency: <50ms
└─ Event Generation
   └─ Emit PERSON_DETECTED, POSTURE_CHANGED, etc.
```

**Design Decisions:**
- **Single inference worker**, not per-camera (saves GPU memory)
- **Batch processing** if 2+ cameras have new frames
- **Frame skipping** — process at most N FPS, not all frames
- **Resolution downscaling** — work with 640×480 or 416×416 not full HD
- **Model selection** — YOLOv8 nano or YOLOv5s (runs on CPU in <100ms)

**Processing Loop (Pseudocode):**
```python
while running:
    # Collect latest frame from each camera queue
    frames = {}
    for cam_id in camera_ids:
        frame = camera_queues[cam_id].get_latest()  # non-blocking
        if frame:
            frames[cam_id] = frame
    
    if not frames:
        time.sleep(0.01)  # No new frames, wait
        continue
    
    # Run detection on all available frames
    detections = yolo_detector.detect_batch(frames.values())
    
    # Generate events
    for cam_id, boxes in detections.items():
        person_count = len([b for b in boxes if b.confidence > 0.5])
        emit_event(
            type="PERSON_DETECTED",
            camera_id=cam_id,
            person_count=person_count,
            frame=frames[cam_id]
        )
```

---

### 3. EVENT LAYER — Standardized Event Schema

**Purpose:** Decouple components and enable WebSocket broadcasting

**Event Types:**

| Event Type | Trigger | Payload |
|-----------|---------|---------|
| PERSON_DETECTED | ≥1 person in frame | {camera_id, count, timestamp, bbox_list, snapshot_path} |
| PERSON_COUNT_CHANGED | Count changed from N to M | {camera_id, old_count, new_count, timestamp} |
| CAMERA_CONNECTED | RTSP connection established | {camera_id, timestamp, rtsp_url} |
| CAMERA_OFFLINE | Connection timeout | {camera_id, timestamp, last_frame_time} |
| CAMERA_RECONNECTING | Attempting reconnect | {camera_id, attempt_number, timestamp} |
| CAMERA_RECONNECTED | Connection restored | {camera_id, timestamp, downtime_seconds} |
| POSTURE_CHANGED | (optional) pose state change | {camera_id, person_id, old_posture, new_posture, timestamp} |
| FALL_DETECTED | (optional) fall event | {camera_id, person_id, timestamp, confidence, snapshot_path} |

**Event Deduplication / Cooldown:**
```python
# Avoid spam logging every frame
class EventDedupManager:
    def __init__(self, cooldown_seconds=2.0):
        self.last_event = {}  # camera_id -> timestamp
        self.cooldown = cooldown_seconds
    
    def should_emit(self, camera_id: str) -> bool:
        now = time.time()
        last = self.last_event.get(camera_id, 0)
        if now - last > self.cooldown:
            self.last_event[camera_id] = now
            return True
        return False
```

---

### 4. STORAGE LAYER — Minimal Data Persistence

**Purpose:** Log events and store snapshots without overcomplicating

**Storage Targets:**

1. **Event Log (SQLite)**
   ```sql
   CREATE TABLE events (
       id INTEGER PRIMARY KEY,
       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
       camera_id TEXT NOT NULL,
       event_type TEXT NOT NULL,
       person_count INTEGER,
       message TEXT,
       snapshot_path TEXT,
       metadata JSON
   );
   
   CREATE INDEX idx_camera_timestamp 
   ON events(camera_id, timestamp DESC);
   ```

2. **Snapshot Storage**
   ```
   backend/outputs/
   ├─ snapshots/
   │  ├─ camera_0_20250314_143022.jpg
   │  ├─ camera_1_20250314_143035.jpg
   │  └─ ...
   └─ logs/
      └─ events.db (SQLite)
   ```

3. **Retention Policy**
   - Keep last 1000 events in DB
   - Keep snapshots for 7 days (or until 2GB limit)
   - Cleanup script runs every hour

---

### 5. API LAYER — Backend Services

**Technology:** FastAPI + uvicorn (lightweight, ASGI-native)

**REST Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | System health check |
| GET | `/api/cameras` | List all cameras + status |
| POST | `/api/cameras` | Add new camera (RTSP URL) |
| GET | `/api/cameras/{id}` | Get camera details |
| PUT | `/api/cameras/{id}` | Update camera config |
| DELETE | `/api/cameras/{id}` | Remove camera |
| GET | `/api/cameras/{id}/status` | Real-time camera status |
| GET | `/api/events` | Get recent events (paginated) |
| GET | `/api/events/camera/{id}` | Events for specific camera |
| GET | `/api/stream/{id}.mjpeg` | MJPEG video stream |
| WS | `/ws` | WebSocket for realtime events |

**WebSocket Events (Outbound):**
```json
{
  "type": "PERSON_DETECTED",
  "data": {
    "camera_id": "camera_0",
    "timestamp": "2025-03-14T14:30:22Z",
    "person_count": 2,
    "event_id": "evt_12345"
  }
}
```

---

### 6. FRONTEND LAYER — Web Dashboard

**Framework:** React or Vue.js (simple, fast, good for realtime)

**Layout Structure:**

```
┌─────────────────────────────────────────────────────┐
│          HavenNet Surveillance Dashboard            │
├─────────────────────┬───────────────────────────────┤
│                     │                               │
│   CAMERA GRID       │      RIGHT SIDEBAR            │
│   (2-4 tiles)       │   ┌─────────────────────┐    │
│                     │   │ SYSTEM STATUS       │    │
│  ┌─────────────┐   │   │ - Cameras online: 2 │    │
│  │  Camera 0   │   │   │ - Avg FPS: 28       │    │
│  │  LIVE       │   │   │ - Total persons: 5  │    │
│  │  🔴 [feed]  │   │   └─────────────────────┘    │
│  │  [2 persons]│   │                               │
│  └─────────────┘   │   DETECTION LOG (realtime)   │
│                     │   ┌─────────────────────┐    │
│  ┌─────────────┐   │   │ 14:30:22 Cam0: 2p   │    │
│  │  Camera 1   │   │   │ 14:30:19 Cam0: 1p   │    │
│  │  LIVE       │   │   │ 14:30:15 Cam1: 3p   │    │
│  │  🔴 [feed]  │   │   │ 14:30:10 Cam0: 0p   │    │
│  │  [3 persons]│   │   │ [...]               │    │
│  └─────────────┘   │   └─────────────────────┘    │
│                     │                               │
└─────────────────────┴───────────────────────────────┘
```

**Key Features:**
- **Camera Grid:** 2 cameras = 1×2 layout, 3-4 cameras = 2×2 layout
- **Each Tile Shows:**
  - Camera name / ID
  - Live/Offline status + badge
  - MJPEG stream
  - Current person count
- **Right Sidebar:**
  - System status (online cameras, avg FPS)
  - Detection log (last 30 events)
  - Clear log button
  - Settings toggle (if needed)
- **Realtime Updates** via WebSocket (no polling)

**State Management:**
- Use React Context or Vuex (simple, no Redux overkill)
- Camera list (names, URLs, status)
- Current detections (count per camera)
- Recent events (last 30)
- Connection status

---

## DATA FLOW DIAGRAMS

### Nominal Flow (Person Detection)

```
Camera Worker                Processing                 Frontend
(Thread 1)                   (Thread 2)
│                            │                         │
├─ Read RTSP frame ─────────>│                         │
│                            │                         │
│  (Frame in queue)          │                         │
│                            │                         │
│                   Frame out of queue                 │
│                            │                         │
│                      Run YOLO detection              │
│                            │                         │
│                     Generate PERSON_                 │
│                     DETECTED event                   │
│                            │                         │
│                      Store in DB ──────────>│        │
│                            │                │        │
│                    Emit WebSocket event ───────────> │
│                            │                         │
│                            │                   Update UI
│                            │                  - person count
│                            │                  - log entry
│                            │                  - timestamp
```

### Camera Reconnect Flow

```
Camera Worker (Thread)           Main System           User
│                                │                    │
├─ Connect to RTSP ─────────────>│                    │
│  (success)                     │                    │
│                                │                    │
├─ Emit CAMERA_CONNECTED ──────> │                    │
│                                ├─ WebSocket ─────>  │
│                                │                 Update:
│                                │              "Front Gate: LIVE"
│
├─ Heartbeat interval
├─ No data for >10s
│
├─ Mark status OFFLINE ─────────>│                  │
│                                ├─ WebSocket ─────>│
│                                │            Update: "OFFLINE"
│
├─ Retry connection
│  (sleep 5s, try again)
├─ Retry #1: fail
├─ Retry #2: fail
├─ Retry #3: success ──────────>│                    │
│                                ├─ WebSocket ─────>│
│                                │           Emit: CAMERA_RECONNECTED
│                                │           "reconnect successful,
│                                │            downtime: 18s"
│                                │                    │
│                                │                 Update UI: LIVE
```

---

## THREADING & CONCURRENCY MODEL

### Thread Architecture

```
┌──────────────────────────────────────────────────┐
│              HAVENNET PROCESS (1)                │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────┐        ┌──────────────────┐   │
│  │   Main      │        │  API Server      │   │
│  │  (thread)   │        │  (FastAPI)       │   │
│  │             │        │  Threads: N+10   │   │
│  │ - Config    │        │  (uvicorn pool)  │   │
│  │ - Monitor   │        │                  │   │
│  └─────────────┘        └──────────────────┘   │
│                                                  │
│  ┌──────────────────────┐                       │
│  │ Camera Workers       │                       │
│  │ (Threads: 1 per cam) │                       │
│  │                      │                       │
│  │ ┌──────────────────┐ │                       │
│  │ │ CameraWorker(0)  │ │                       │
│  │ │ RTSP connection  │ │                       │
│  │ │ Frame capture    │ │                       │
│  │ │ Queue management │ │                       │
│  │ └──────────────────┘ │                       │
│  │                      │                       │
│  │ ┌──────────────────┐ │                       │
│  │ │ CameraWorker(N)  │ │                       │
│  │ │ RTSP connection  │ │                       │
│  │ │ Frame capture    │ │                       │
│  │ │ Queue management │ │                       │
│  │ └──────────────────┘ │                       │
│  └──────────────────────┘                       │
│           ↓ (queue data)                         │
│  ┌──────────────────────┐                       │
│  │ Processing Worker    │                       │
│  │ (Thread: 1)          │                       │
│  │ YOLO inference       │                       │
│  │ Event generation     │                       │
│  │ DB logging           │                       │
│  └──────────────────────┘                       │
│           ↓ (events)                             │
│  ┌──────────────────────┐                       │
│  │ WebSocket Broadcaster│                       │
│  │ (Built into FastAPI) │                       │
│  │ Push to all clients  │                       │
│  └──────────────────────┘                       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Concurrency Model

- **Camera Workers:** Python `threading.Thread` (one per camera)
  - Simple, no GIL issue since RTSP read is I/O-bound
  - Each has own queue, no lock contention

- **Processing Worker:** Single thread, reads all camera queues
  - Avoids double-locking
  - Scales easily to 4+ cameras (queues are lock-free)

- **API Server:** FastAPI + uvicorn (ASGI)
  - Built-in async handling
  - WebSocket support out-of-box
  - Uses thread pool for CPU-bound inference (optional)

- **Event Queue:** Thread-safe `queue.Queue` (Python standard)
  - Simple, proven, fast

---

## EVENT SYSTEM

### Event Flow Architecture

```
Event Sources          Event Queue            Event Consumers
(Camera Workers)       (thread-safe)          (API + DB)
│                          │                        │
├─ PERSON_DETECTED ───────>│                        │
├─ CAMERA_OFFLINE ────────>│                        ├─ Store in SQLite
├─ RECONNECT_ATTEMPT ─────>│                        │
│                          │                        ├─ WebSocket push
│                          │───── Event batch ─────>│
│                          │                        ├─ Snapshot save
│                          │                        │
```

### Event Schema (Python TypedDict)

```python
class DetectionEvent(TypedDict):
    id: str                    # evt_xxx_timestamp
    timestamp: datetime        # UTC ISO 8601
    camera_id: str            # camera_0, camera_1, ...
    event_type: str           # PERSON_DETECTED, POSTURE_CHANGED, etc.
    person_count: int         # 0, 1, 2, ...
    message: str              # Human readable: "2 persons detected"
    snapshot_path: str | None # /outputs/snapshots/...
    boxes: list[dict]         # [{"x": 10, "y": 20, "w": 50, "h": 80, "conf": 0.95}]
    metadata: dict            # FPS, latency, etc.
```

### Event Publishing

```python
# Somewhere in processing worker
event = {
    "id": f"evt_{camera_id}_{int(time.time()*1000)}",
    "timestamp": datetime.utcnow().isoformat(),
    "camera_id": "camera_0",
    "event_type": "PERSON_DETECTED",
    "person_count": 2,
    "message": "2 persons detected",
    "snapshot_path": "/outputs/snapshots/camera_0_20250314_143022.jpg",
    "boxes": [{"x": 100, "y": 150, "w": 50, "h": 100, "conf": 0.92}],
    "metadata": {"inference_ms": 45, "queue_len": 2, "fps": 28}
}

# 1. Store in DB
event_db.insert(event)

# 2. Broadcast to WebSocket clients
ws_broadcaster.send_to_all("detection", event)

# 3. Append to in-memory log (last 100 events)
recent_events.append(event)
```

---

## CONFIGURATION STRATEGY

### Config File Structure

**File:** `backend/shared_config.py`

```python
from dataclasses import dataclass
from typing import List, Dict
import yaml
import os

@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    timeout_seconds: int = 10
    reconnect_interval_seconds: int = 5
    target_fps: int = 30
    resolution: str = "1280x720"
    snapshot_on_detect: bool = True

@dataclass
class SystemConfig:
    mode: str  # "desktop" or "pi5"
    cameras: List[CameraConfig]
    detection_confidence: float = 0.5
    enable_pose: bool = True
    enable_posture: bool = True
    enable_fall: bool = False
    snapshot_retention_days: int = 7
    max_events_in_memory: int = 1000
    event_cooldown_seconds: float = 2.0
    outputs_dir: str = "./backend/outputs"
    models_dir: str = "./backend/models"

def load_config(config_file: str = "backend/config.yaml") -> SystemConfig:
    with open(config_file) as f:
        data = yaml.safe_load(f)
    
    cameras = [CameraConfig(**c) for c in data.get("cameras", [])]
    return SystemConfig(
        mode=data.get("mode", "desktop"),
        cameras=cameras,
        **{k: v for k, v in data.items() if k != "cameras"}
    )
```

**File:** `backend/config.yaml`

```yaml
mode: "desktop"  # or "pi5"

cameras:
  - id: "camera_0"
    name: "Front Gate"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    enabled: true
    timeout_seconds: 10
    reconnect_interval_seconds: 5
    target_fps: 30
    resolution: "1280x720"
    snapshot_on_detect: true

  - id: "camera_1"
    name: "Back Garden"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    enabled: true
    timeout_seconds: 10
    reconnect_interval_seconds: 5
    target_fps: 30
    resolution: "1280x720"
    snapshot_on_detect: true

detection:
  confidence_threshold: 0.5

features:
  pose: true
  posture: true
  fall_detection: false

paths:
  outputs: "./backend/outputs"
  models: "./backend/models"
  logs: "./backend/outputs/logs"

retention:
  events_max: 1000
  snapshots_days: 7
  event_cooldown_seconds: 2.0
```

### Runtime Configuration Updates

Via REST API:
```python
@app.post("/api/cameras")
async def add_camera(camera: CameraConfig):
    config.cameras.append(camera)
    # Update config.yaml
    save_config(config)
    # Restart camera worker
    camera_manager.restart(camera.id)
    return {"status": "ok"}
```

---

## PERFORMANCE CHARACTERISTICS

### Target Metrics (Desktop Mode)

| Metric | Target | Notes |
|--------|--------|-------|
| Capture FPS (per camera) | 30 | RTSP frame rate |
| Detection FPS (all cameras) | 20 | YOLO processing |
| Detection Latency | <100ms | Frame → inference → output |
| E2E Latency (input → display) | <500ms | Acceptable for demo |
| Memory Usage | <500 MB | All cameras + models |
| CPU Usage | <60% (2 core avg) | On i5/i7 desktop |
| Queue Length | <3 | Frames waiting in queue |
| Reconnect Time | <5 sec | OFFLINE → LIVE |

### Measurement Points

```python
class PerformanceMetrics:
    def __init__(self):
        self.capture_fps = 0      # frames captured per sec
        self.processed_fps = 0    # frames processed per sec
        self.inference_ms = 0     # YOLO latency
        self.queue_lengths = {}   # per camera
        self.memory_mb = 0
        self.cpu_percent = 0
        self.reconnect_count = {} # per camera
    
    def report(self):
        return {
            "capture_fps": self.capture_fps,
            "processed_fps": self.processed_fps,
            "inference_latency_ms": self.inference_ms,
            "queue_lengths": self.queue_lengths,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
        }
```

---

## SCALABILITY PATH (2→4 CAMERAS)

### Current Design Supports Multi-Camera

**No rewrite needed because:**
- Each camera has independent worker thread
- Processing layer reads all queues, not sequential
- Event system is decoupled
- Database has camera_id foreign key
- Frontend uses dynamic grid layout

### Scaling from 2 to 4

**Step 1: Add 2 more cameras in config.yaml**
```yaml
cameras:
  - id: "camera_0"
    # ...
  - id: "camera_1"
    # ...
  - id: "camera_2"  # NEW
    name: "Side Entrance"
    rtsp_url: "rtsp://192.168.1.102:554/stream"
    # ...
  - id: "camera_3"  # NEW
    name: "Loading Dock"
    rtsp_url: "rtsp://192.168.1.103:554/stream"
    # ...
```

**Step 2: Restart backend (no code changes)**
```bash
pkill -f "python app.py"
python app.py  # Reads all 4 cameras from config
```

**Step 3: Frontend automatically adapts**
- Layout changes from 1×2 to 2×2
- Each new camera tile added to grid
- WebSocket pushes status for new cameras

### Performance Scaling

- **Capture:** Each camera worker adds ~5% CPU
- **Detection:** Batch processing scales nearly linearly (4 cameras ≈ 1.8× slower than 2)
- **Memory:** ~100MB per camera model cache, shared among all cameras
- **Network:** RTSP bandwidth = 4 × bitrate (multiplicative)

**Optimization if needed:**
- Reduce resolution per camera (640×480 → 416×416)
- Skip frames (process every 2nd frame on Pi5)
- Disable pose estimation on Pi5
- Use YOLOv8 nano instead of small

---

## RASPBERRY PI ADAPTATION STRATEGY

### "Lite Mode" Configuration Profile

**File:** `backend/config_pi5.yaml`

```yaml
mode: "pi5"

cameras:
  - id: "camera_0"
    name: "Front Gate"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    enabled: true
    timeout_seconds: 15  # More generous
    reconnect_interval_seconds: 10
    target_fps: 10  # Lower FPS
    resolution: "640x480"  # Downscale
    snapshot_on_detect: false  # Save disk I/O

  - id: "camera_1"
    name: "Back Garden"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    enabled: true
    timeout_seconds: 15
    reconnect_interval_seconds: 10
    target_fps: 10
    resolution: "640x480"
    snapshot_on_detect: false

detection:
  confidence_threshold: 0.6  # Higher threshold = faster

features:
  pose: false  # Disable on Pi5
  posture: false  # Disable on Pi5
  fall_detection: false

optimization:
  # Force single-threaded processing
  max_inference_threads: 1
  batch_processing: false
  # Reduce memory footprint
  model_precision: "int8"  # Quantized
```

### Resource Budget for Pi5 4GB

| Component | Allocation | Notes |
|-----------|-----------|-------|
| OS + system | 500 MB | Ubuntu Server |
| Python runtime | 100 MB | venv |
| YOLO model | 50-100 MB | YOLOv8n quantized |
| Frame buffers | 100 MB | 2 cameras × 640×480×3 |
| SQLite DB | 50 MB | Event log |
| Snapshots (RAM) | 0 MB | Only save to disk |
| Free headroom | 1000 MB | Safety margin |
| **Total** | **~2 GB** | Comfortable fit |

### Deployment Script

**File:** `scripts/deploy_pi5.sh`

```bash
#!/bin/bash
set -e

echo "🎯 Deploying HavenNet to Raspberry Pi 5"

# 1. Copy code
rsync -av --exclude venv --exclude __pycache__ \
  /path/to/HavenNet pi@raspberrypi.local:/home/pi/havennet

# 2. Create venv on Pi
ssh pi@raspberrypi.local << 'EOF'
  cd ~/havennet
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements_pi5.txt
EOF

# 3. Start background service
ssh pi@raspberrypi.local << 'EOF'
  cd ~/havennet
  nohup bash -c 'source venv/bin/activate && \
    python backend/app.py --config backend/config_pi5.yaml' \
    > /tmp/havennet.log 2>&1 &
  echo $! > /tmp/havennet.pid
EOF

echo "✅ Deployed! Access dashboard at http://raspberrypi.local:8000"
```

### Network Setup for Mobile Demo

**Scenario:** Bring Pi5 + 2 cameras + travel router to demo location

**Setup Steps:**
1. **Create LAN:**
   ```
   Travel Router (5GHz: HavenNet / 2.4GHz: backup)
   ├─ Raspberry Pi 5 (IP: 192.168.4.10)
   ├─ Camera 0 (IP: 192.168.4.20)
   └─ Camera 1 (IP: 192.168.4.21)
   ```

2. **Configure cameras (beforehand):**
   ```
   Camera 0: RTSP URL = rtsp://admin:password@192.168.4.20:554/stream
   Camera 1: RTSP URL = rtsp://admin:password@192.168.4.21:554/stream
   ```

3. **Prepare config file:**
   ```yaml
   # backend/config_mobile.yaml
   cameras:
     - id: "camera_0"
       rtsp_url: "rtsp://admin:password@192.168.4.20:554/stream"
     - id: "camera_1"
       rtsp_url: "rtsp://admin:password@192.168.4.21:554/stream"
   ```

4. **Start on Pi:**
   ```bash
   python backend/app.py --config backend/config_mobile.yaml
   ```

5. **Access from laptop/tablet:**
   ```
   http://raspberrypi.local:8000
   or
   http://192.168.4.10:8000
   ```

---

## SUMMARY TABLE: Architecture Layer Mapping

| Layer | Technology | Implementation | Responsibility |
|-------|-----------|-----------------|-----------------|
| **Input** | Python threading | CameraWorker(s) | Capture RTSP, bounded queue |
| **Processing** | YOLO + OpenCV | InferenceWorker | Detection, pose (optional) |
| **Events** | Python queue.Queue | EventBroker | Standardize, deduplicate |
| **Storage** | SQLite + FS | EventStore | Log events, snapshots |
| **API** | FastAPI + uvicorn | APIServer | REST + WebSocket |
| **Frontend** | React/Vue | Dashboard | UI, realtime updates |

---

## NEXT STEPS

Once this architecture is approved, proceed to:
1. **PHẦN C** — Repo restructure plan
2. **PHẦN D** — Implementation phases
3. **PHẦN E** — Detailed task list
4. **PHẦN F** — API design spec
5. **PHẦN G** — Frontend component spec

---

**END OF PHẦN B**
