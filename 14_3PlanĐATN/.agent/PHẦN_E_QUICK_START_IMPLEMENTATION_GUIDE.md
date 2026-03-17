# PHẦN E — QUICK START IMPLEMENTATION GUIDE
## Hướng Dẫn Thực Hành Bắt Đầu Phát Triển MVP

**Document Version:** 1.0  
**Status:** Ready for Development  
**Target Audience:** Developers (chỉ cần 1-2 RTSP cameras để kiểm tra)

---

## TABLE OF CONTENTS

1. [Setup Environment](#setup-environment)
2. [Directory Structure](#directory-structure)
3. [Component Implementation Guide](#component-implementation-guide)
4. [Testing Each Component](#testing-each-component)
5. [Running the Full System](#running-the-full-system)
6. [Troubleshooting](#troubleshooting)

---

## SETUP ENVIRONMENT

### Prerequisites
```bash
# Check Python version (need 3.10+)
python --version  # Should output 3.10.x or higher

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip ffmpeg

# (macOS)
brew install python@3.11 ffmpeg

# (Windows - use WSL2 or native Python 3.11)
```

### Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Dependencies
```bash
cd /path/to/HavenNet

# Minimal setup (development)
pip install fastapi uvicorn pydantic pyyaml opencv-python ultralytics torch

# Full setup with all extras
pip install -r requirements.txt

# Verify installations
python -c "import cv2, torch, ultralytics; print('✓ All deps OK')"
```

### Get YOLO Model
```bash
# Download YOLOv8m (person detection + pose)
# First time will auto-download from Ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Models stored in: ~/.cache/ultralytics/
```

---

## DIRECTORY STRUCTURE

### Target Layout (after refactoring)
```
HavenNet/
├── backend/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                    # Entry point (YOLO + FastAPI)
│   │   ├── config.py                  # Config system (dataclass-based)
│   │   ├── logger.py                  # Logging utility
│   │   ├── database.py                # SQLite + ORM setup
│   │   │
│   │   ├── camera/
│   │   │   ├── __init__.py
│   │   │   ├── rtsp_worker.py         # RTSP capture + frame queue
│   │   │   ├── frame_queue.py         # Thread-safe bounded queue
│   │   │   └── exceptions.py
│   │   │
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py            # YOLO wrapper + inference
│   │   │   ├── pose_classifier.py     # Posture detection (standing/sitting/lying)
│   │   │   └── worker.py              # Inference loop
│   │   │
│   │   ├── events/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py              # Event dataclass + validation
│   │   │   ├── manager.py             # Event deduplication + aggregation
│   │   │   └── broadcaster.py         # WebSocket broadcasting
│   │   │
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── models.py              # SQLAlchemy ORM models
│   │   │   ├── repository.py          # CRUD operations
│   │   │   └── migrations/            # Alembic schema versioning
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── app.py                 # FastAPI app setup
│   │   │   ├── routes.py              # REST endpoints (/cameras, /events, etc)
│   │   │   ├── websocket.py           # WebSocket handler
│   │   │   └── middleware.py          # CORS, auth, etc
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── visualization.py       # Draw boxes, poses
│   │       └── metrics.py             # Performance monitoring
│   │
│   ├── config.yaml                    # Camera + processing config
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js                     # Main app logic
│       ├── websocket.js               # WebSocket connection
│       └── ui.js                      # Dashboard rendering
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_camera.py
│   │   ├── test_detector.py
│   │   └── test_events.py
│   │
│   └── integration/
│       ├── test_camera_detector_flow.py
│       ├── test_api.py
│       └── test_websocket.py
│
├── scripts/
│   ├── init_db.py                     # Initialize SQLite
│   ├── generate_test_frames.py        # For testing without real cameras
│   ├── benchmark.py                   # Performance measurement
│   └── deploy_pi.sh                   # Raspberry Pi deployment
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── API.md
│
└── docker-compose.yml
```

---

## COMPONENT IMPLEMENTATION GUIDE

### 1. Config System (`backend/src/config.py`)

**Purpose:** YAML → Python dataclass conversion with type validation

```python
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml
import os

@dataclass
class CameraConfig:
    """Single camera configuration"""
    id: str
    name: str
    rtsp_url: str
    resolution: tuple = (1280, 720)
    fps: int = 30
    buffer_size: int = 5  # Frames to keep in queue
    
    def validate(self):
        """Ensure RTSP URL is valid"""
        if not self.rtsp_url.startswith(('rtsp://', 'rtmp://', 'http://')):
            raise ValueError(f"Invalid URL: {self.rtsp_url}")

@dataclass
class ProcessingConfig:
    """YOLO and inference settings"""
    model_path: str = "yolov8m.pt"
    confidence: float = 0.45
    device: str = "cuda"  # "cpu" for non-GPU
    max_inference_workers: int = 1  # Always 1 for single YOLO
    iou_threshold: float = 0.5

@dataclass
class DatabaseConfig:
    """SQLite settings"""
    db_path: str = "data/havennet.db"
    echo_sql: bool = False  # Log SQL queries

@dataclass
class LogConfig:
    """Logging settings"""
    level: str = "INFO"
    file: str = "logs/havennet.log"
    console: bool = True

@dataclass
class AppConfig:
    """Root configuration"""
    cameras: List[CameraConfig] = field(default_factory=list)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """Load from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Apply env var overrides
        cameras = [CameraConfig(**c) for c in data.get('cameras', [])]
        
        config = cls(
            cameras=cameras,
            processing=ProcessingConfig(**data.get('processing', {})),
            database=DatabaseConfig(**data.get('database', {})),
            logging=LogConfig(**data.get('logging', {}))
        )
        
        # Validate all cameras
        for cam in config.cameras:
            cam.validate()
        
        return config
```

**config.yaml:**
```yaml
# Camera configurations
cameras:
  - id: cam0
    name: "Front Entrance"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    resolution: [1280, 720]
    fps: 30
    buffer_size: 5
  
  - id: cam1
    name: "Back Door"
    rtsp_url: "rtsp://192.168.1.101:554/stream"
    resolution: [1280, 720]
    fps: 30
    buffer_size: 5

# YOLO detection settings
processing:
  model_path: "yolov8m.pt"
  confidence: 0.45
  device: "cuda"  # Change to "cpu" if no GPU
  max_inference_workers: 1
  iou_threshold: 0.5

# SQLite database
database:
  db_path: "data/havennet.db"
  echo_sql: false

# Logging
logging:
  level: "INFO"
  file: "logs/havennet.log"
  console: true
```

---

### 2. Logger Module (`backend/src/logger.py`)

**Purpose:** Unified logging to file + console

```python
import logging
import logging.handlers
from pathlib import Path

def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    """Setup logger with file + console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Create log directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    fh = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(getattr(logging, level))
    
    # Console handler (colored)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Usage in app
logger = setup_logger("havennet", "logs/havennet.log", "INFO")
```

---

### 3. Frame Queue (`backend/src/camera/frame_queue.py`)

**Purpose:** Thread-safe bounded queue (drops oldest on overflow)

```python
from queue import Queue, Empty
from dataclasses import dataclass
import threading
import numpy as np

@dataclass
class Frame:
    """Frame container"""
    camera_id: str
    frame: np.ndarray  # OpenCV image
    timestamp: float
    frame_id: int

class BoundedFrameQueue:
    """Thread-safe queue that drops oldest frames on overflow"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.queue = Queue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def put(self, frame: Frame, block: bool = False) -> bool:
        """
        Put frame in queue. If full, drop oldest.
        Returns True if queued, False if dropped.
        """
        try:
            self.queue.put_nowait(frame)
            return True
        except:
            # Queue full - drop oldest
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
                return True
            except:
                return False
    
    def get(self, timeout: float = 0.1) -> Frame | None:
        """Get next frame (non-blocking)"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def size(self) -> int:
        """Current queue size"""
        with self.lock:
            return self.queue.qsize()
```

---

### 4. RTSP Camera Worker (`backend/src/camera/rtsp_worker.py`)

**Purpose:** Capture frames from RTSP stream in background thread

```python
import cv2
import threading
import time
from datetime import datetime
from .frame_queue import BoundedFrameQueue, Frame
from ..logger import setup_logger

logger = setup_logger("camera_worker", "logs/havennet.log")

class RTSPCameraWorker:
    """Background thread for RTSP capture"""
    
    def __init__(self, camera_id: str, rtsp_url: str, queue: BoundedFrameQueue, 
                 fps: int = 30, resolution: tuple = (1280, 720)):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.queue = queue
        self.fps = fps
        self.resolution = resolution
        self.running = False
        self.thread = None
        self.frame_id = 0
        self.connection_failures = 0
        self.max_connection_failures = 5
    
    def start(self):
        """Start capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.camera_id} worker started")
    
    def stop(self):
        """Stop capture thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"Camera {self.camera_id} worker stopped")
    
    def _capture_loop(self):
        """Main capture loop (runs in thread)"""
        cap = None
        frame_delay = 1.0 / self.fps
        
        while self.running:
            # Connect/reconnect to camera
            if cap is None:
                cap = self._connect_camera()
                if cap is None:
                    time.sleep(2)  # Wait before retry
                    continue
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read from {self.camera_id}")
                cap.release()
                cap = None
                self.connection_failures += 1
                continue
            
            # Reset failure counter on success
            self.connection_failures = 0
            
            # Resize to target resolution
            frame = cv2.resize(frame, self.resolution)
            
            # Create Frame object
            frame_obj = Frame(
                camera_id=self.camera_id,
                frame=frame,
                timestamp=time.time(),
                frame_id=self.frame_id
            )
            
            # Put in queue (drops if full)
            self.queue.put(frame_obj)
            self.frame_id += 1
            
            # Maintain target FPS
            time.sleep(frame_delay)
        
        if cap:
            cap.release()
    
    def _connect_camera(self) -> cv2.VideoCapture | None:
        """Attempt to connect to RTSP stream"""
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single frame buffer
            
            # Test connection
            ret, _ = cap.read()
            if ret:
                logger.info(f"Connected to {self.camera_id}")
                return cap
            else:
                cap.release()
                return None
        except Exception as e:
            logger.error(f"Connection error for {self.camera_id}: {e}")
            return None
```

---

### 5. YOLO Detector (`backend/src/processing/detector.py`)

**Purpose:** Wrapper around YOLOv8 for inference

```python
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    """Single person detection"""
    class_id: int  # Always 0 for person
    class_name: str  # "person"
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    keypoints: np.ndarray  # Pose (17 points)
    pose_score: float

class YOLODetector:
    """Wrapper around YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8m.pt", device: str = "cuda"):
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.45) -> list[Detection]:
        """
        Run YOLO detection on frame
        Returns list of Detection objects (person only)
        """
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Filter for person class (0)
                if int(box.cls) != 0:
                    continue
                
                # Get keypoints if available
                keypoints = None
                pose_score = 0.0
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.xy[0].numpy()
                    if hasattr(result.keypoints, 'conf'):
                        pose_score = result.keypoints.conf[0].mean().item()
                
                detection = Detection(
                    class_id=0,
                    class_name="person",
                    confidence=box.conf.item(),
                    bbox=tuple(box.xyxy[0].numpy()),
                    keypoints=keypoints,
                    pose_score=pose_score
                )
                detections.append(detection)
        
        return detections
```

---

### 6. Event Schema (`backend/src/events/schema.py`)

**Purpose:** Standardized event format for all detections

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

class EventType(str, Enum):
    PERSON_DETECTED = "person_detected"
    PERSON_LOST = "person_lost"
    POSTURE_CHANGE = "posture_change"

class PostureType(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    UNKNOWN = "unknown"

@dataclass
class Event:
    """Standardized event schema"""
    event_id: str  # Unique ID
    timestamp: datetime
    camera_id: str
    event_type: EventType
    person_id: str | None = None
    posture: PostureType | None = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON (for WebSocket)"""
        return json.dumps({
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id,
            'event_type': self.event_type.value,
            'person_id': self.person_id,
            'posture': self.posture.value if self.posture else None,
            'confidence': self.confidence,
            'metadata': self.metadata
        })
```

---

### 7. FastAPI Setup (`backend/src/api/app.py`)

**Purpose:** REST API + WebSocket server

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI(title="HavenNet", version="0.1.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST Endpoints
@app.get("/api/health")
async def health():
    """System health check"""
    return {"status": "ok", "version": "0.1.0"}

@app.get("/api/cameras")
async def get_cameras():
    """List all cameras and their status"""
    # Will be populated by main app
    return {"cameras": []}

@app.get("/api/events")
async def get_events(limit: int = 100):
    """Get recent events from database"""
    # Will be populated by main app
    return {"events": []}

# WebSocket for realtime updates
@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for realtime event broadcast"""
    await websocket.accept()
    try:
        while True:
            # Will be populated by main app
            data = await websocket.receive_text()
    except:
        pass
    finally:
        await websocket.close()

# Serve frontend
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## TESTING EACH COMPONENT

### 1. Test Config Loading
```python
# test_config.py
from backend.src.config import AppConfig

def test_config_loading():
    config = AppConfig.from_yaml("backend/config.yaml")
    
    assert len(config.cameras) > 0
    assert config.processing.device in ["cuda", "cpu"]
    assert config.database.db_path.endswith(".db")
    
    print("✓ Config loaded successfully")
    print(f"  - {len(config.cameras)} cameras")
    print(f"  - Device: {config.processing.device}")

if __name__ == "__main__":
    test_config_loading()
```

### 2. Test YOLO Detector
```python
# test_detector.py
import cv2
import numpy as np
from backend.src.processing.detector import YOLODetector

def test_detector():
    detector = YOLODetector(device="cpu")  # Use CPU for testing
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(frame, conf_threshold=0.45)
    
    print(f"✓ Detector initialized")
    print(f"  - Model: yolov8m.pt")
    print(f"  - Device: cpu")
    print(f"  - Detections on empty frame: {len(detections)}")

if __name__ == "__main__":
    test_detector()
```

### 3. Test Frame Queue
```python
# test_queue.py
import numpy as np
import time
from backend.src.camera.frame_queue import BoundedFrameQueue, Frame

def test_frame_queue():
    queue = BoundedFrameQueue(max_size=5)
    
    # Put 10 frames (queue will drop oldest 5)
    for i in range(10):
        frame = Frame(
            camera_id="cam0",
            frame=np.zeros((720, 1280, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=i
        )
        queue.put(frame)
    
    # Should have 5 frames (last 5)
    assert queue.size() == 5
    
    # Get a frame
    f = queue.get()
    assert f.frame_id >= 5  # Should be from last 5
    
    print("✓ Frame queue working correctly")
    print(f"  - Max size: 5")
    print(f"  - Current size: {queue.size()}")

if __name__ == "__main__":
    test_frame_queue()
```

---

## RUNNING THE FULL SYSTEM

### Step 1: Setup
```bash
# Navigate to project
cd HavenNet

# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install deps
pip install -r requirements.txt

# Download YOLO
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Step 2: Update config.yaml
```bash
# Edit backend/config.yaml with YOUR camera IPs/RTSP URLs
# Example with test files instead of real cameras:
```

**For Testing (without real cameras):**
```yaml
cameras:
  - id: test_cam
    name: "Test Camera"
    rtsp_url: "rtsp://example.com/stream"  # Will fail gracefully
    resolution: [1280, 720]
    fps: 10
    buffer_size: 5
```

### Step 3: Initialize Database
```bash
# Create SQLite DB and schema
python backend/src/database.py
```

### Step 4: Run System
```bash
# Start FastAPI + YOLO processing
python backend/src/main.py
```

Expected output:
```
2024-01-15 10:30:45 - havennet - INFO - Loading config from backend/config.yaml
2024-01-15 10:30:45 - havennet - INFO - Config loaded: 1 cameras
2024-01-15 10:30:45 - havennet - INFO - Initializing YOLO detector (device: cuda)
2024-01-15 10:30:50 - havennet - INFO - YOLO ready
2024-01-15 10:30:50 - havennet - INFO - Camera test_cam worker started
2024-01-15 10:30:51 - havennet - INFO - Starting FastAPI on http://0.0.0.0:8000
2024-01-15 10:30:51 - havennet - INFO - Processing loop started
```

### Step 5: Access Dashboard
```
Frontend: http://localhost:8000
API Docs: http://localhost:8000/docs
```

---

## TROUBLESHOOTING

### Issue 1: "YOLO model not found"
```bash
# Solution: Download model explicitly
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Check location:
ls ~/.cache/ultralytics/
```

### Issue 2: "CUDA not available"
```bash
# Check CUDA setup
python -c "import torch; print(torch.cuda.is_available())"

# If False, change config.yaml device to "cpu"
```

### Issue 3: "Cannot connect to RTSP camera"
```bash
# Test camera connection directly
ffmpeg -rtsp_transport tcp -i rtsp://YOUR_CAMERA_IP:554/stream -t 5 -f null -

# Common RTSP URLs:
# Hikvision: rtsp://USER:PASS@IP:554/Streaming/Channels/101
# Dahua: rtsp://USER:PASS@IP:554/stream/sub
# Generic: rtsp://IP:554/stream
```

### Issue 4: "Out of memory"
```bash
# Reduce batch size or model size
# In config.yaml:
processing:
  model_path: "yolov8s.pt"  # Smaller than yolov8m.pt
  device: "cpu"  # Use CPU instead of CUDA
```

### Issue 5: "Low FPS (< 10)"
```bash
# Check GPU utilization
nvidia-smi  # NVIDIA cards

# If GPU not used:
# 1. Verify CUDA installation
# 2. Check config.yaml device: "cuda"
# 3. Reduce resolution in camera config

# If CPU maxed:
# 1. Reduce FPS in camera config
# 2. Use smaller YOLO model (s or n instead of m)
```

---

## NEXT STEPS

After completing this setup:

1. **Verify Components Work:**
   - Run unit tests in `tests/unit/`
   - Check logs in `logs/havennet.log`

2. **Get Real Camera Working:**
   - Find RTSP URL for your camera
   - Test connection with ffmpeg
   - Update config.yaml

3. **Build Dashboard UI:**
   - Create `frontend/index.html`
   - Add WebSocket listener for events
   - Display live camera feeds + event log

4. **Deploy to Raspberry Pi:**
   - Use `scripts/deploy_pi.sh`
   - Set device: "cpu" in config
   - Reduce resolution and FPS

See **PHẦN B, C, D** for more details on architecture and implementation phases.

---

**Questions or Issues?** Check the full audit documents:
- PHẦN A: Gap Analysis (current issues)
- PHẦN B: Target MVP Architecture (system design)
- PHẦN C: Repo Restructure (file organization)
- PHẦN D: Implementation Plan (sprint details)
