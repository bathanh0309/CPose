# PHẦN A — GAP ANALYSIS (Chi Tiết Tổng Quan Dự Án)

**Mục đích:** Audit toàn bộ codebase hiện tại, chỉ ra mismatch giữa tham vọng design / docs vs khả năng demo thực tế, và đề xuất chiến lược rút gọn scope cho milestone bảo vệ.

---

## 1. TÌNH TRẠNG HIỆN TẠI

### 1.1. Cấu Trúc Codebase

```
HavenNet/
├── backend/src/
│   ├── app.py                      # FastAPI web server
│   ├── run.py                      # SequentialRunner (main pipeline)
│   ├── reid.py                     # EnhancedReID (color histogram + shape)
│   ├── adl.py                      # ADL classification (posture detection)
│   ├── visualize.py                # Visualization functions
│   ├── core/
│   │   └── global_id_manager.py    # Master-Slave ID assignment
│   ├── storage/
│   │   ├── persistence.py          # State persistence
│   │   └── vector_db.py            # Vector DB for embedding search
│   ├── config.yaml                 # Local config
│   └── templates/
│       └── index.html              # Basic web dashboard
├── configs/
│   └── unified_config.yaml         # Unified config attempt
├── backend/models/
│   ├── yolo11n-pose.pt             # YOLO11 Nano Pose
│   └── yolo11n.pt                  # YOLO11 Nano Object Detection
├── backend/database/               # SQLite DB + embeddings
├── backend/outputs/                # CSV logs
├── data/                           # Test videos
├── scripts/
│   ├── mp4_to_gif.py
│   └── run.bat / web.bat
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

---

## 2. MISMATCH GIỮA DESIGN VÀ REALITY

### 2.1. **Mismatch #1: RTSP Multi-Camera vs Video File Sequential Processing**

**Documented Goal (README):**
- Hệ thống hỗ trợ "Real-time Multi-Camera Monitoring" 
- Có thể nhập RTSP/IP camera streams
- Parallel processing cho realtime

**Actual Implementation:**
```python
# backend/src/run.py: SequentialRunner
# - Xử lý từng camera tuần tự (sequential)
# - Input: video file paths (D:/HavenNet/data/multi-camera/cam1/video1.mp4)
# - Không có RTSP support
# - Không có reconnect logic
# - Không có camera health check
```

**Impact for Demo:**
- ❌ Không thể demo với thực camera IP/RTSP
- ❌ Chỉ có thể demo với video files đã quay sẵn (limited)
- ❌ Không thể cấu hình camera trực tiếp từ giao diện web
- ❌ Demás tiến độ bị tie-up trên video files

**Severity:** CRITICAL for P1 requirement

---

### 2.2. **Mismatch #2: Realtime Dashboard vs Batch Video Processing**

**Documented Goal:**
- Web dashboard hiển thị "realtime" video streams
- WebSocket push events
- Live camera status

**Actual Implementation:**
```python
# backend/src/app.py
# - FastAPI server nhưng chỉ có 1 thread chạy SequentialRunner
# - Video feed là MJPEG từ runner_thread
# - Không có WebSocket
# - Status endpoint chỉ return runner object state
# - Không có per-camera heartbeat / health monitoring
# - Không có camera online/offline detection
```

**Impact for Demo:**
- ❌ Cannot demo live camera switching
- ❌ Cannot demo camera offline/reconnect (no network handling)
- ❌ UI không thể cập nhật per-camera status realtime
- ❌ Log realtime không reliable

**Severity:** CRITICAL for demo UX

---

### 2.3. **Mismatch #3: 4-Camera Scalability vs Master-Slave Hardcoding**

**Documented Goal:**
- Hỗ trợ 2-4 camera linh hoạt
- UI responsive với số camera biến động

**Actual Implementation:**
```python
# configs/unified_config.yaml
# Hardcoded cam1, cam2, cam3, cam4
# Master-Slave logic:
#   - Cam1/Cam2 = Master (registration)
#   - Cam3/Cam4 = Slave
# SequentialRunner loops through enabled cameras
# Nhưng không có dynamic add/remove camera feature
```

**Impact for Demo:**
- ⚠️ Can add 4 cameras nhưng không thể:
  - Add camera at runtime từ UI
  - Remove camera gracefully
  - Change camera RTSP URL động
  - Restart individual camera

**Severity:** MEDIUM — works for static setup but not flexible

---

### 2.4. **Mismatch #4: ReID "Robust" vs Simple Color Histogram**

**Documented Claim:**
- "Định danh người (ReID)" với "Cross-camera ID tracking"
- "Master-Slave" identity assignment
- Sophisticated global_id_manager implementation

**Actual Implementation:**
```python
# backend/src/reid.py: EnhancedReID
# Features:
#   - HSV histogram (3-part: head/body/legs)
#   - Hu moments
#   - Aspect ratio
# Matching: cosine similarity threshold 0.48-0.65
# NO deep features, NO embeddings, NO face recognition
# Color-based matching is environment-sensitive (lighting changes break it)
```

**Reality Check:**
```python
# backend/src/core/global_id_manager.py
# Temporal voting: track_id -> {global_id: count}
# Yêu cầu 15 frames confirmation trước khi assign ID
# Nhưng nếu camera mất connection → tất cả vote bị reset
# UNK (unknown) handling là complex nhưng không robust
```

**Impact for Demo:**
- ⚠️ ReID works for controlled environments (studio demo)
- ❌ Will FAIL with:
  - Lighting changes
  - Camera angle changes
  - New clothes
  - Occlusion
- 🚨 Risk: ReID fail → entire demo looks broken (hội đồng thấy "người A" → "người B" → "người A" lại)

**Severity:** CRITICAL for credibility

---

### 2.5. **Mismatch #5: "ADL Event Detection" vs Limited Posture Classification**

**Documented Features:**
- Pose Detection (Đứng, Đi, Ngồi, Nằm)
- Fall Detection + ADL Events
- Spatial Zone Intrusion
- Dangerous Object Detection

**Actual Implementation:**
```python
# backend/src/adl.py: TrackState + classify_posture()
# Supports:
#   - STANDING (từ keypoint angles)
#   - WALKING (movement + pose)
#   - SITTING (knee angle)
#   - LYING (aspect ratio + angle)
# Fall Detection: Available nhưng not thoroughly tested
# ADL Events: Only basic state transitions logged
# Zone Intrusion: KHÔNG IMPLEMENT
# Object Detection: KHÔNG IMPLEMENT (chỉ có YOLO model import)
```

**Reality Check:**
```python
# Config chứa references đến:
# dangerous_objects: classes [34, 38, 43]
# Nhưng không có actual object detection pipeline
# Chỉ có person detection (YOLO bbox)
```

**Impact for Demo:**
- ⚠️ Basic posture detection works
- ❌ Fall detection không tested → unreliable
- ❌ ADL events incomplete
- ❌ Dangerous object detection không built
- ❌ Zone intrusion không built

**Severity:** MEDIUM — core features present but incomplete

---

### 2.6. **Mismatch #6: Web Dashboard vs Minimal HTML Template**

**Documented Goal:**
- "Dashboard" với camera tiles, detection logs, status panel

**Actual Implementation:**
```html
<!-- backend/src/templates/index.html -->
Basic single-video MJPEG stream display
No grid layout for 2-4 cameras
No detection log panel
No camera status indicators
No websocket integration
```

**Impact for Demo:**
- ❌ UI không showcase demo thấy rõ
- ❌ Người hội đồng không thấy detection logs realtime
- ❌ Cannot see multi-camera layout
- ❌ Looks "incomplete/prototype"

**Severity:** CRITICAL for presentation

---

### 2.7. **Mismatch #7: Config System Chaos**

**Current State:**
- `backend/src/config.yaml` — references D:/ paths (hardcoded Windows paths)
- `configs/unified_config.yaml` — newer version, also D:/ paths
- Both have conflicting settings
- No environment-based config loading
- No .env support for RTSP URLs or paths

**Actual code:**
```python
# backend/src/run.py
if config_path is None:
    config_path = SRC_DIR / "config.yaml"
```

**Impact for Demo:**
- ❌ Cannot run on different machines without editing config files
- ❌ Cannot demo on Raspberry Pi (paths hardcoded)
- ❌ Cannot easily swap between localhost/home/travel networks
- ⚠️ Non-portable

**Severity:** MEDIUM — works locally but not portable

---

### 2.8. **Mismatch #8: "Headless Mode" vs GUI-First Design**

**Documentation Claims:**
- Web server mode (headless=True)
- FastAPI backend for UI

**Actual Design:**
```python
# backend/src/run.py
# SequentialRunner designed to display CV2 windows (GUI)
# Headless mode exists but is incomplete:
#   - Still needs to process frames
#   - MJPEG stream generation is expensive
#   - No frame queue management
```

**Impact for Demo:**
- ⚠️ Works on desktop with display
- ❌ Will be slow on Raspberry Pi (GUI overhead + MJPEG encoding)
- ❌ Web UI is afterthought, not first-class

**Severity:** HIGH for Pi deployment

---

## 3. BOTTLENECK + RỦI RO HIỆN TẠI

### Critical Issues for Demo Success:

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| **No RTSP support** | CRITICAL | Cannot use real cameras | Code doesn't exist |
| **Sequential processing** | CRITICAL | FPS will be terrible | Architecture issue |
| **No camera health monitoring** | CRITICAL | Demo crash if stream fails | Not implemented |
| **Minimal web UI** | CRITICAL | Demo looks unfinished | Template exists but bare |
| **ReID fragile** | CRITICAL | Identity tracking unreliable | Algorithm dependent on lighting |
| **Hardcoded paths** | HIGH | Not portable | Windows D:/ paths |
| **No WebSocket** | HIGH | UI doesn't update realtime | Missing |
| **Config chaos** | MEDIUM | Hard to reconfigure | Two conflicting configs |
| **Fall detection untested** | MEDIUM | May not work during demo | Incomplete |
| **No object detection pipeline** | MEDIUM | Dangerous object feature non-functional | Model exists, no code |

---

## 4. ĐẬU RA MỚI TỪ USER REQUIREMENTS

### P0 Requirements (Bắt Buộc cho Demo):
1. ✅ **2-4 RTSP camera support** — MISSING
2. ✅ **Real-time dashboard** — PARTIAL (GUI only)
3. ✅ **Detection logging** — PARTIAL (CSV only)
4. ✅ **Camera online/offline status** — MISSING
5. ✅ **Auto-reconnect** — MISSING
6. ✅ **Stable 15-30 min demo** — RISKY (no error handling)

### P1 Requirements (Nice to Have):
1. ⚠️ **Pose estimation** — DONE but not optimized
2. ⚠️ **Basic ADL events** — DONE but incomplete
3. ❌ **Snapshot logging** — NOT DONE
4. ⚠️ **SQLite persistence** — DONE but not integrated with UI

### P2/P3 (Can Skip for Now):
1. ❌ **ReID across cameras** — TOO FRAGILE
2. ❌ **Fall detection** — NOT TESTED
3. ❌ **Object detection** — NOT BUILT
4. ❌ **Zone intrusion** — NOT BUILT

---

## 5. DIAGNOSE: ĐỐI TƯỢNG PHÁT TRIỂN SỬ DỤNG HIỆN TẠI

### Current Design Pattern:
```
Researcher's Notebook → Academic Paper Claims
         ↓
     AI Engineer (me)
         ↓
Started as research project (many features, lots of ambition)
         ↓
Master-Slave logic, face-based ReID, ADL framework
         ↓
PROBLEM: Too many features, not integrated into working system
```

### What Happened:
1. **DESIGN IS ACADEMIC**: Features are documented (README), not implemented
2. **CODE IS PARTIAL**: Core CV pipelines exist (YOLO, pose, ReID) but orchestration is missing
3. **ARCHITECTURE IS SEQUENTIAL**: Single loop per camera, no concurrency
4. **NETWORK IS MISSING**: No RTSP, no streaming, no connection handling
5. **UI IS AFTERTHOUGHT**: Web server exists but dashbv dashboard is skeleton

---

## 6. PHÂN LOẠI CODE HIỆN TẠI

### 🟢 KEEP (Robust + Usable)
- `backend/src/adl.py` — Posture classification logic is solid
- `backend/src/reid.py` — Color histogram extraction works (even if matching is fragile)
- `backend/src/visualize.py` — Drawing utilities are fine
- YOLO models (yolo11n-pose.pt, yolo11n.pt) — Proven models
- `backend/src/storage/persistence.py` — DB logic seems OK
- Batch CSV logging — Works fine

### 🟡 REFACTOR (Works but Incomplete)
- `backend/src/run.py` — SequentialRunner needs to become:
  - Multi-camera with per-camera workers
  - Per-camera frame queues (bounded)
  - Frame skipping logic
- `backend/src/core/global_id_manager.py` — Master-Slave logic is over-engineered:
  - Simplify to per-camera tracking for P0
  - Global ID sharing can come later (P3)
- `backend/src/app.py` — FastAPI structure OK but:
  - Need proper async workers per camera
  - Add WebSocket for realtime updates
  - Add camera management API
- `configs/` — Consolidate to single config
  - Support .env loading
  - Remove hardcoded paths
  - Make relative/environment-based

### 🔴 REMOVE or DEPRIORITIZE (Not Demo-Critical)
- Face enrollment pipeline — NOT NEEDED for P0 (constraints: no face recognition)
- Complex ReID matching — TOO FRAGILE for open demo
- Fall detection — NOT TESTED, risky
- Zone intrusion — NOT IMPLEMENTED, can skip
- Dangerous object detection — Model imported but no pipeline
- Vector DB (FAISS-like) — Overkill for simple color matching

---

## 7. PROPOSED MINIMAL VIABLE ARCHITECTURE (MVA) CHO DEMO

### Philosophy: 
**DEMO-FIRST, STABLE-FIRST, REALTIME-FIRST**

Remove complexity that doesn't directly support live 2-4 camera monitoring with person detection.

### Layered Architecture:

```
┌─────────────────────────────────────────────────────────┐
│           FRONTEND (Web Dashboard)                       │
│  - Grid layout (2/3/4 camera tiles)                      │
│  - Real-time log sidebar (WebSocket)                     │
│  - Camera config panel                                   │
│  - Status indicators                                     │
└─────────────────────────────────────────────────────────┘
                        ↕ REST API / WebSocket
┌─────────────────────────────────────────────────────────┐
│             BACKEND (FastAPI)                            │
│  /api/cameras (GET, POST, DELETE, UPDATE)               │
│  /api/status (camera health, person count)              │
│  /ws/events (WebSocket stream)                          │
│  /stream/{camera_id} (MJPEG)                            │
└─────────────────────────────────────────────────────────┘
                        ↕ Manager
┌─────────────────────────────────────────────────────────┐
│        CAMERA MANAGER (Orchestration)                    │
│  - Per-camera worker threads                            │
│  - Health monitoring + auto-reconnect                   │
│  - Event aggregation                                     │
│  - Metrics collection                                   │
└─────────────────────────────────────────────────────────┘
                        ↕
┌──────────────────┬──────────────────┬──────────────────┐
│  Camera Worker 1 │  Camera Worker 2 │  Camera Worker N │
│  (RTSP/Video)    │  (RTSP/Video)    │  (RTSP/Video)    │
│  - Capture frame │  - Capture frame │  - Capture frame │
│  - Bounded queue │  - Bounded queue │  - Bounded queue │
│  - Frame skip    │  - Frame skip    │  - Frame skip    │
│  - Reconnect     │  - Reconnect     │  - Reconnect     │
└──────────────────┴──────────────────┴──────────────────┘
         ↓                ↓                ↓
┌─────────────────────────────────────────────────────────┐
│         INFERENCE WORKERS (Pool)                        │
│  - YOLO person detection                                │
│  - Pose estimation (optional)                           │
│  - Posture classification (optional)                    │
│  - Simple ReID (per-camera only for P0)                 │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│            EVENT GENERATION                             │
│  - PERSON_DETECTED                                      │
│  - PERSON_COUNT_CHANGED                                 │
│  - CAMERA_CONNECTED / DISCONNECTED                      │
│  - Frame timestamps + metadata                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┬──────────────────┬──────────────────┐
│   SQLite Log     │  CSV Audit Log   │  Snapshot Cache  │
│  (Events)        │  (Raw data)      │  (JPEG images)   │
└──────────────────┴──────────────────┴──────────────────┘
```

### Key Design Decisions:

1. **Per-Camera Workers** (Not Sequential):
   - Each camera runs in independent thread/async task
   - Own frame queue (bounded, e.g., 10 frames)
   - Own reconnect timer
   - One camera's problem doesn't block others

2. **Bounded Frame Queues**:
   - Max 10 frames per camera
   - If backend is slow → drop old frames, always process newest
   - No infinite queue growth

3. **Shared Inference Pool** (Optional):
   - Single YOLO model (shared) to save memory
   - Multiple inference tasks queued
   - Lock-free or async inference queue

4. **Event-Driven Architecture**:
   - Standardized events (PERSON_DETECTED, etc.)
   - Debouncing/cooldown to prevent log spam
   - Each event published to WebSocket

5. **Simple ReID (Per-Camera Only for P0)**:
   - Track person by bbox trajectory within single camera
   - Don't try cross-camera ID matching initially
   - Avoids fragile global ID complexity
   - Can add cross-camera ReID as P3 (Phase 3)

6. **Config-as-Code**:
   - Single config.yaml (no D:/ hardcoding)
   - Environment variables for RTSP URLs, paths
   - Support localhost, home Wi-Fi, travel network modes

---

## 8. QUICK WINS (Minimal Effort, High Impact)

1. **Add RTSP support** (2-3 hours):
   - Modify camera worker to accept RTSP URLs
   - Add cv2.VideoCapture() with timeout/reconnect
   - Test with mock RTSP server or real camera

2. **Create proper web dashboard** (4-6 hours):
   - Bootstrap 5 grid layout (2/3/4 camera responsive)
   - WebSocket for realtime updates
   - Detection log sidebar
   - Camera add/remove form

3. **Add WebSocket event push** (2-3 hours):
   - FastAPI websocket endpoint
   - Event aggregator publishes events
   - Frontend subscribes

4. **Camera health check** (1-2 hours):
   - Per-camera heartbeat timer
   - Detect stalled frame capture
   - Auto-reconnect with exponential backoff

5. **Config portability** (1-2 hours):
   - Single config.yaml with env variable substitution
   - Remove hardcoded D:/ paths
   - Load camera list from config dynamically

---

## 9. SCOPE CHANGES RECOMMENDED

### **Remove from P0 Scope:**
- ❌ Multi-camera ReID (global ID tracking) → Too complex, too fragile
  - **Alternative**: Track locally per camera only
  - **Rationale**: ReID across cameras needs robust embedding or face data; color histogram is too environment-sensitive
  
- ❌ Fall detection (untested) → Risk of demo failure
  - **Alternative**: Keep pose detection, skip fall event
  - **Rationale**: Posture classification works; fall detection needs more testing

- ❌ Zone intrusion detection → Not implemented
  - **Alternative**: Skip for now
  - **Rationale**: Can add in P2 if time permits

- ❌ Dangerous object detection → Model exists, no pipeline
  - **Alternative**: Remove for now
  - **Rationale**: YOLO object detection separate feature; scope bloat

### **Keep in P0 Scope:**
- ✅ 2-4 RTSP camera streaming
- ✅ Person detection (YOLO)
- ✅ Per-camera local tracking
- ✅ Posture estimation (standing/sitting/lying)
- ✅ Realtime dashboard with detection logs
- ✅ Camera online/offline status
- ✅ Auto-reconnect

### **Defer to P2/P3:**
- ⏳ Global ReID (cross-camera identity)
- ⏳ Fall detection
- ⏳ Advanced ADL events
- ⏳ Raspberry Pi optimization

---

## 10. ACCEPTANCE CRITERIA REVISION

### For P0 (Localhost 2-Camera Demo):

#### Functional:
- [ ] Can configure 2 RTSP/video camera URLs via web form
- [ ] Both cameras stream live in grid layout (2 tiles)
- [ ] Person detection runs on each camera independently
- [ ] Detection log updates in sidebar (timestamp + camera + person count)
- [ ] When camera disconnects → UI shows OFFLINE
- [ ] System auto-attempts reconnect every 5 sec
- [ ] When camera reconnects → UI shows LIVE
- [ ] No crashes after 30 min continuous demo
- [ ] Each camera FPS ≥ 5 (minimum for visible motion)

#### Non-Functional:
- [ ] Latency: detection log update < 2 sec after detection
- [ ] Memory: < 1 GB RAM usage (single machine)
- [ ] CPU: reasonable usage on modern desktop (not >80%)
- [ ] Queue: no unbounded memory growth (bounded queues only)

#### Code Quality:
- [ ] Configuration is portable (no hardcoded D:/ paths)
- [ ] Logging is structured, easy to debug
- [ ] Camera workers are independent (one failure doesn't crash others)
- [ ] Code is modular, easy to add 3rd/4th camera

---

## 11. NEXT STEPS (Transition to PHẦN B)

Based on this analysis:
1. **Remove complexity**: Skip global ReID, fall detection, zones for now
2. **Build core**: RTSP support, per-camera workers, event system
3. **Polish UI**: Real dashboard with proper layout
4. **Ensure stability**: Proper error handling, reconnect, timeouts
5. **Document**: Clear paths for Raspberry Pi porting

---

## SUMMARY TABLE

| Component | Current Status | Verdict | Action |
|-----------|---|---|---|
| **RTSP Streaming** | Not implemented | NEED TO BUILD | Add cv2 RTSP support + reconnect |
| **Multi-camera Workers** | Sequential, single-threaded | REFACTOR | Parallel workers per camera |
| **Web Dashboard** | Minimal HTML template | REBUILD | Bootstrap grid + WebSocket |
| **WebSocket Events** | Not implemented | NEED TO BUILD | FastAPI WebSocket + event aggregator |
| **Camera Health Check** | Not implemented | NEED TO BUILD | Heartbeat + auto-reconnect |
| **YOLO Inference** | Works | KEEP | Optimize for FPS/latency |
| **Posture Classification** | Works | KEEP | Integrate into pipeline |
| **ReID (Global)** | Complex, fragile | DOWNGRADE | Skip for P0, do per-camera only |
| **Fall Detection** | Untested | SKIP | Mark as P2 |
| **Config System** | Hardcoded paths | FIX | Make portable, env-based |
| **Pose Estimation** | Works | KEEP | Optional, can toggle |
| **CSV/SQLite Logging** | Works | KEEP | Use as-is |
| **Dangerous Objects** | Model only, no pipeline | SKIP | Mark as P2 |
| **Zone Intrusion** | Not implemented | SKIP | Mark as P2 |

---

**PHẦN A Complete. Ready for PHẦN B — TARGET MVP ARCHITECTURE.**
