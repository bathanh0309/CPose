# Pipeline Kết Hợp CPose × HavenNet — Thiết Kế Workflow Tối Ưu

> **Tài liệu thiết kế** kết hợp hai hệ thống: **CPose** (thu thập dữ liệu RTSP + Pose/ADL offline) và **HavenNet/HAVEN** (nhận diện người đa camera realtime), nhằm tạo ra một pipeline vừa có **demo xử lý tuần tự** (offline) vừa **chạy được realtime RTSP**.

---

## 1. Phân Tích Điểm Giao Nhau

### 1.1 Những gì CPose có mà HAVEN cần

| Tài nguyên từ CPose | Vai trò trong pipeline kết hợp |
|---|---|
| `phase1_recorder.py` — RTSP multi-cam ghi clip | Source video cho HAVEN khi không có camera live |
| `phase3_recognizer.py` — YOLOv11-pose → keypoints + ADL | Bổ sung ADL layer cho HAVEN (HAVEN dùng rules đơn giản hơn) |
| `pose_utils.py` — `rule_based_adl()`, `calc_angle()`, `draw_skeleton()` | Tái sử dụng trực tiếp trong HAVEN ADL engine |
| `unified_config.yaml` + `pose_adl.yaml` | Chuẩn hoá config thay vì 2 config riêng biệt |
| `StorageManager.enforce_storage_limit()` | HAVEN chưa có storage management |
| SPA Frontend (5 tab, dark theme) | HAVEN chỉ có Flask stream đơn giản |

### 1.2 Những gì HavenNet/HAVEN có mà CPose cần

| Tài nguyên từ HavenNet/HAVEN | Vai trò trong pipeline kết hợp |
|---|---|
| `global_id_manager.py` — GlobalID hierarchy (Master/Slave/Strict) | CPose không có identity tracking xuyên camera |
| `reid.py` — Color histogram Re-ID | CPose không track người qua nhiều cam |
| `src/detectors/face.py` — YuNet + SFace | CPose không nhận diện mặt |
| `src/detectors/body.py` — YOLOv8 + DeepSort | CPose dùng ByteTrack, DeepSort tốt hơn cho Re-ID |
| SQLite embedding database | CPose chỉ dùng filesystem |
| Zone intrusion + intruder alert | CPose không có security layer |

---

## 2. Kiến Trúc Pipeline Đề Xuất: **HavenPose**

Pipeline kết hợp được đặt tên **HavenPose**, hoạt động theo **hai chế độ song song** trên cùng codebase:

```
                    ┌─────────────────────────────────────────┐
                    │              HavenPose System           │
                    │                                         │
          ┌─────────┴──────────┐          ┌────────────────── ┴────────┐
          │   OFFLINE MODE     │          │      REALTIME MODE         │
          │  (Sequential Demo) │          │     (RTSP Live Stream)     │
          │                    │          │                            │
          │  Video File Input  │          │  N IP Cameras (RTSP)       │
          │        ↓           │          │        ↓                   │
          │  Phase1: Record    │          │  CameraManager (threads)   │
          │  Phase2: Detect    │          │        ↓                   │
          │  Phase3: Pose+ADL  │          │  YOLOv8n detect + track    │
          │        ↓           │          │        ↓                   │
          │  Output: files     │          │  GlobalIDManager (Master→  │
          │  keypoints/adl.txt │          │  Slave→Strict hierarchy)   │
          │                    │          │        ↓                   │
          └─────────┬──────────┘          │  FaceDetector / ReIDEngine │
                    │                     │        ↓                   │
                    │    SHARED CORE      │  ADLAnalyzer (Pose + rules)│
                    │   ┌─────────────┐   │        ↓                   │
                    └──►│ PoseADL     │◄──┘  MJPEG stream + SocketIO   │
                        │ Engine      │                                │
                        │ (yolo11-    │                                │
                        │  pose.pt)   │                                │
                        └──────┬──────┘                                │
                               │                                       │
                        ┌──────▼──────┐                                │
                        │  pose_utils │                                │
                        │  rule_based │                                │
                        │  _adl()     │                                │
                        └─────────────┘                                │
                    └─────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Thư Mục Kết Hợp

```
HavenPose/                          ← Root project
├── app/
│   ├── __init__.py                 # Flask factory (từ CPose)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # TẤT CẢ endpoints — Phase1/2/3 + ReID + ADL Live
│   └── services/
│       ├── __init__.py
│       ├── phase1_recorder.py      # CPose RecorderManager (giữ nguyên)
│       ├── phase2_analyzer.py      # CPose Analyzer (giữ nguyên)
│       ├── phase3_recognizer.py    # CPose PoseADLRecognizer (giữ nguyên)
│       │
│       ├── reid_engine.py          # TỪ HAVEN — reid.py refactor thành class
│       ├── camera_manager.py       # TỪ HAVEN — camera.py (CameraStream + CameraManager)
│       └── identity_manager.py     # TỪ HAVEN — global_id_manager.py
│
├── core/                           # NEW — Shared engines (không phụ thuộc Flask)
│   ├── __init__.py
│   ├── pose_adl_engine.py          # Unified: yolo11-pose + rule_based_adl()
│   │                               # Dùng cho CẢ offline (Phase3) lẫn realtime
│   ├── face_detector.py            # TỪ HavenNet — YuNet + SFace
│   └── body_detector.py            # TỪ HavenNet — YOLOv8n + DeepSort
│
├── app/utils/                      # TỪ CPose
│   ├── file_handler.py             # StorageManager
│   ├── stream_probe.py             # probe_rtsp() (thread-safe)
│   └── pose_utils.py               # calc_angle, rule_based_adl, draw_skeleton
│
├── configs/
│   ├── unified_config.yaml         # MỞ RỘNG — gộp CPose + HAVEN config
│   ├── pose_adl.yaml               # ADL thresholds (từ CPose)
│   └── cameras.yaml                # Camera hierarchy config (từ HAVEN)
│
├── data/
│   ├── config/
│   │   └── resources.txt           # RTSP URLs
│   ├── raw_videos/                 # Clip Phase1 [git-ignored]
│   ├── output_labels/              # Phase2 output [git-ignored]
│   ├── output_pose/                # Phase3 output [git-ignored]
│   └── embeddings.db               # SQLite: face/body embeddings [git-ignored]
│
├── models/
│   ├── yolov8n.pt                  # Phase1 + body detect
│   ├── yolo11n.pt                  # Phase2
│   ├── yolo11n-pose.pt             # Phase3 + realtime ADL
│   ├── face_detection_yunet.onnx   # Face detect
│   └── face_recognition_sface.onnx # Face embedding
│
├── static/
│   ├── index.html                  # SPA 6 tab (mở rộng từ CPose 5 tab)
│   ├── css/style.css
│   └── js/app.js
│
├── main.py
├── run.bat
└── requirements.txt
```

---

## 4. Hai Chế Độ Hoạt Động (Dual-Mode Architecture)

### Mode A — Offline Sequential (Demo)

Dùng lại **toàn bộ 3 phase của CPose** không thay đổi. Phù hợp cho:
- Demo đồ án, báo cáo
- Thu thập dataset keypoints/ADL labels
- Chạy trên máy không có camera IP

```
resources.txt
    ↓ (hoặc upload video file thủ công)
Phase 1: RecorderManager
  - Đọc RTSP → YOLOv8n detect person → ghi clip MP4
  - Output: data/raw_videos/YYYYMMDD_HHMMSS_camXX.mp4

Phase 2: Analyzer
  - YOLOv11n offline trên clip → PNG frames + labels.txt
  - Output: data/output_labels/<clip_stem>/

Phase 3: PoseADLRecognizer
  - yolo11n-pose.pt → keypoints 17 COCO + rule_based_adl()
  - Output: data/output_pose/<clip_stem>/_keypoints.txt + _adl.txt
  - Optional: _overlay_NNNN.png (skeleton visualization)
```

**Trigger**: Tab "Analysis" → "Pose & ADL" → chọn folder → Start

---

### Mode B — Realtime RTSP (Live)

Dùng kiến trúc camera hierarchy từ **HAVEN**, tích hợp ADL engine từ CPose.

```
resources.txt (RTSP URLs)
    ↓
CameraManager khởi tạo N CameraThread (daemon threads)
    ↓
Với mỗi frame, phân nhánh theo camera role:

CAM-01 (MASTER — Face Registration):
  frame → FaceDetector.detect() [YuNet]
        → FaceDetector.extract_embedding() [SFace]
        → IdentityManager.register_face() → GlobalID

CAM-02/03 (SLAVE — Monitor):
  frame → BodyDetector.detect() [YOLOv8n]
        → BodyDetector.extract_embedding() [DeepSort features]
        → ReIDEngine.match_body() → GlobalID / UNKNOWN
        → PoseADLEngine.process(frame, bbox) → ADL label
        → visualize(bbox, GlobalID, ADL, confidence)

CAM-04 (STRICT — Security):
  Giống SLAVE + nếu UNKNOWN:
        → trigger intruder_alert()
        → SocketIO emit("security_event", {...})
        → Bounding box nhấp nháy đỏ

Tất cả cameras:
  Mỗi N frame → encode JPEG → emit("stream_frame") → only to subscribers
```

**Trigger**: Tab "Live Monitor" → Start Realtime → chọn cam để xem

---

## 5. Shared Core: `PoseADLEngine`

Đây là module quan trọng nhất — **dùng chung cho cả hai mode**:

```python
# core/pose_adl_engine.py

class PoseADLEngine:
    """
    Unified Pose + ADL engine.
    - Offline mode: được gọi từ Phase3Recognizer, xử lý clip frame-by-frame
    - Realtime mode: được gọi từ CameraThread, xử lý live frame
    
    Interface thống nhất: process(frame, bbox=None) → PoseResult
    """

    def __init__(self, model_path: str, config: dict):
        self.model = YOLO(model_path)   # yolo11n-pose.pt
        self.config = config
        self.kp_buffers: dict[int, deque] = {}  # person_id → sliding window

    def process(self, frame: np.ndarray, person_id: int = 0,
                bbox: tuple = None) -> "PoseResult":
        """
        Nếu bbox được cung cấp: chỉ crop vùng người → pose inference (nhanh hơn)
        Nếu không có bbox: full-frame inference (dùng khi không có detector trước)
        
        Returns PoseResult:
          .keypoints: np.ndarray [17, 3]  (x, y, conf)
          .adl_label: str
          .adl_conf:  float
          .skeleton_frame: np.ndarray | None  (nếu draw_overlay=True)
        """

    def reset_buffer(self, person_id: int):
        """Xoá sliding window khi mất track."""
        self.kp_buffers.pop(person_id, None)
```

**Lý do quan trọng**: CPose hiện tại `Phase3Recognizer` và HAVEN `adl.py` đều làm cùng việc nhưng ở hai nơi khác nhau. Gộp vào `PoseADLEngine` eliminates code duplication và đảm bảo kết quả ADL nhất quán giữa offline demo và realtime.

---

## 6. Sơ Đồ Luồng Dữ Liệu Đầy Đủ

### 6.1 Offline Mode (Sequential Demo)

```
[User: upload video / ghi RTSP]
          │
          ▼
  ┌───────────────┐
  │ Phase 1       │  YOLOv8n detect người → ghi clip MP4
  │ RecorderMgr   │  emit: clip_saved {filename, duration, size}
  └───────┬───────┘
          │ raw_videos/*.mp4
          ▼
  ┌───────────────┐
  │ Phase 2       │  YOLOv11n offline → PNG frames + labels.txt
  │ Analyzer      │  emit: analysis_progress {pct, frames_saved}
  └───────┬───────┘
          │ output_labels/<stem>/
          ▼
  ┌───────────────────────────┐
  │ Phase 3                   │  yolo11n-pose → keypoints 17 COCO
  │ PoseADLRecognizer         │  rule_based_adl() sliding window W=30
  │ ─ calls PoseADLEngine ──► │  → ADL label + confidence
  └───────┬───────────────────┘
          │ output_pose/<stem>/
          ▼
  keypoints.txt + adl.txt + overlay_NNNN.png
          │
          ▼
  [Frontend Tab 5: Pose & ADL]
    - ADL distribution bar chart
    - Bảng keypoints/events
    - Preview overlay frame
```

### 6.2 Realtime Mode (RTSP Live)

```
resources.txt (N RTSP URLs)
          │
          ▼
  ┌───────────────────────────────────────────────────────┐
  │ CameraManager                                          │
  │ Spawn N CameraThread (daemon, auto-reconnect)          │
  └──┬────────────────────────────────────────────────────┘
     │
     ├─── CAM-01 (MASTER) ──────────────────────────────────►
     │         frame
     │           ↓
     │      FaceDetector [YuNet] → bbox + 5 landmarks
     │           ↓
     │      SFace embedding (128-dim)
     │           ↓
     │      IdentityManager.register_face()
     │           ↓
     │      GlobalID = "GID-001"
     │           ↓
     │      SQLite: INSERT face_embeddings
     │
     ├─── CAM-02/03 (SLAVE) ────────────────────────────────►
     │         frame
     │           ↓
     │      YOLOv8n detect person → N bboxes
     │           ↓
     │      DeepSort track → track_id stable
     │           ↓
     │      ReIDEngine.match_body(features)
     │           ├── Match found → GlobalID "GID-001"
     │           └── Not found  → UNKNOWN (đỏ)
     │           ↓
     │      PoseADLEngine.process(frame, bbox, track_id)
     │           ↓
     │      ADL: "walking" 0.87
     │           ↓
     │      draw_skeleton() + annotate(GlobalID, ADL)
     │           ↓
     │      [nếu có subscriber] encode JPEG → emit stream_frame
     │
     └─── CAM-04 (STRICT) ──────────────────────────────────►
               frame → [như SLAVE]
                 ↓
               UNKNOWN detected?
                 ├── YES → emit("security_event", {cam_id, timestamp, bbox})
                 │          → Overlay nhấp nháy đỏ
                 │          → Log intruder event
                 └── NO  → normal display
```

---

## 7. API Endpoints Mới (Bổ Sung)

### 7.1 Live Realtime Endpoints (thêm vào `routes.py`)

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/live/start` | Khởi động tất cả CameraThread theo `resources.txt` |
| `POST` | `/api/live/stop` | Dừng toàn bộ cameras |
| `GET` | `/api/live/status` | Trạng thái từng camera + FPS live |
| `POST` | `/api/persons/register` | `{name, age}` → tạo person, chờ CAM-01 thu face |
| `POST` | `/api/persons/register/stop` | Lưu embeddings, kết thúc registration |
| `GET` | `/api/persons` | Danh sách persons đã đăng ký |
| `DELETE` | `/api/persons/<id>` | Xoá person |
| `GET` | `/api/security/events` | Danh sách intruder events có phân trang |
| `POST` | `/api/cameras/roles` | Set role (MASTER/SLAVE/STRICT) cho từng cam |

### 7.2 SocketIO Events Bổ Sung (Server → Client)

| Event | Payload | Trigger |
|---|---|---|
| `live_detection` | `{cam_id, global_id, adl_label, conf, bbox, timestamp}` | Mỗi frame có người |
| `security_event` | `{cam_id, timestamp, bbox, threat_level}` | UNKNOWN ở CAM-04 |
| `identity_registered` | `{person_id, name, face_count, body_count}` | Đủ embeddings |
| `reid_match` | `{cam_id, global_id, name, similarity}` | Khi match thành công |

---

## 8. SPA Frontend — Tab 6 Mới: Live Monitor (Realtime)

CPose hiện tại đã có Tab 2 "Live Monitor" nhưng chỉ xem preview đơn giản. Tab mới tập trung vào **realtime Re-ID + ADL**:

```
Tab 6: Live Re-ID & ADL
  ┌─────────────────────────────────────────────────────┐
  │ Camera Grid (responsive)                             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
  │  │  CAM-01  │  │  CAM-02  │  │  CAM-04  │           │
  │  │  MASTER  │  │  SLAVE   │  │  STRICT  │           │
  │  │ [stream] │  │ [stream] │  │ [stream] │           │
  │  └──────────┘  └──────────┘  └──────────┘           │
  │  Click camera → lazy load stream (không load tất cả) │
  ├─────────────────────────────────────────────────────┤
  │ Persons Panel (live)                                 │
  │  GID-001: Nguyễn A  | WALKING (0.89) | CAM-02       │
  │  GID-002: Trần B    | SITTING (0.92) | CAM-03       │
  │  UNKNOWN            | N/A            | CAM-04 ⚠️     │
  ├─────────────────────────────────────────────────────┤
  │ Security Log                                         │
  │  [14:02:15] INTRUDER DETECTED — CAM-04              │
  │  [13:58:01] Re-ID: GID-001 matched (sim: 0.91)      │
  └─────────────────────────────────────────────────────┘
```

---

## 9. Unified Config (`unified_config.yaml` mở rộng)

```yaml
# ═══ GLOBAL ═══
project: havenpose
version: "1.0"
mode: dual    # "offline" | "realtime" | "dual"

# ═══ CAMERAS ═══
cameras:
  config_file: data/config/resources.txt
  roles:
    cam01: MASTER      # Face registration
    cam02: SLAVE       # Monitor
    cam03: SLAVE
    cam04: STRICT      # Security zone
  reconnect_timeout: 5    # seconds
  buffer_size: 1          # cv2 CAP_PROP_BUFFERSIZE

# ═══ MODELS ═══
models:
  phase1_detect:  models/yolov8n.pt
  phase2_detect:  models/yolo11n.pt
  phase3_pose:    models/yolo11n-pose.pt
  face_detect:    models/face_detection_yunet.onnx
  face_recog:     models/face_recognition_sface.onnx

# ═══ PHASE 1 ═══
phase1:
  pre_buffer_sec:    3
  post_buffer_sec:   3
  inference_every:   5
  min_clip_duration: 2.0
  conf_threshold:    0.35
  storage_limit_gb:  10.0

# ═══ PHASE 2 ═══
phase2:
  conf_threshold:  0.50
  progress_every:  10

# ═══ PHASE 3 (OFFLINE) ═══
phase3:
  conf_threshold:  0.45
  kp_conf_min:     0.3
  window_size:     30
  save_overlay:    true
  progress_every:  10

# ═══ REALTIME ADL ═══
realtime_adl:
  conf_threshold:  0.45
  kp_conf_min:     0.3
  window_size:     15    # nhỏ hơn offline để responsive hơn

# ═══ RE-ID ═══
reid:
  face_threshold:    0.363   # SFace cosine
  body_threshold:    0.85    # Color histogram
  min_face_samples:  5
  global_id_timeout: 30      # seconds không thấy → xoá

# ═══ ADL THRESHOLDS ═══
adl_thresholds:
  knee_bend_angle: 150
  hip_angle_lying: 160
  shoulder_raise:  45
  velocity_walk:   8.0

# ═══ OUTPUT ═══
output:
  raw_videos:    data/raw_videos
  output_labels: data/output_labels
  output_pose:   data/output_pose
  embeddings_db: data/embeddings.db
```

---

## 10. Dependency Bổ Sung (Requirements)

Giữ nguyên `requirements.txt` của CPose và thêm:

```
# Thêm từ HavenNet
deep-sort-realtime>=1.3,<2    # DeepSort tracker (thay ByteTrack)

# Đã có sẵn trong cả hai (không thêm)
# ultralytics, opencv-python, numpy, flask, flask-socketio
```

> **Không thêm** SQLAlchemy, FastAPI, Redis, mediapipe, mmpose — tuân thủ constraint của CPose CLAUDE.md.

---

## 11. Migration Path (Từ Code Hiện Tại)

### Bước 1 — Copy không thay đổi (an toàn)
```
CPose/app/               → HavenPose/app/               (giữ nguyên)
CPose/app/utils/         → HavenPose/app/utils/         (giữ nguyên)
CPose/static/            → HavenPose/static/            (giữ nguyên + thêm tab)
CPose/configs/           → HavenPose/configs/           (giữ + mở rộng)
```

### Bước 2 — Refactor từ HAVEN
```
HAVEN/backend/src/reid.py            → HavenPose/app/services/reid_engine.py
HAVEN/backend/src/core/global_id_manager.py → HavenPose/app/services/identity_manager.py
HavenNet/src/camera.py               → HavenPose/app/services/camera_manager.py
HavenNet/src/detectors/face.py       → HavenPose/core/face_detector.py
HavenNet/src/detectors/body.py       → HavenPose/core/body_detector.py
```

### Bước 3 — Tạo mới (Shared Core)
```
HavenPose/core/pose_adl_engine.py    # Gộp Phase3 + HAVEN adl.py
```

### Bước 4 — Mở rộng routes.py
```
Thêm endpoints: /api/live/*, /api/persons/*, /api/security/*
Thêm SocketIO events: live_detection, security_event, identity_registered
```

### Bước 5 — Frontend: Thêm Tab 6
```
Thêm "Live Re-ID & ADL" tab vào index.html
Lazy load camera stream (từ Bug Fix #6 trong CPose CLAUDE.md)
```

---

## 12. Điểm Khác Biệt Kiến Trúc Giữa Hai Mode

| Tiêu chí | Offline (CPose 3 Phase) | Realtime (HAVEN hierarchy) |
|---|---|---|
| **Input** | Video file hoặc RTSP → clip | RTSP live stream trực tiếp |
| **Latency** | Không yêu cầu | < 200ms/frame |
| **Threading** | 1 thread per clip (sequential) | N threads (daemon, per-camera) |
| **Identity** | Không có (chỉ person index) | GlobalID xuyên camera |
| **ADL window** | 30 frames (chính xác hơn) | 15 frames (responsive hơn) |
| **Output** | Files: .txt, .png (persistent) | MJPEG stream + SocketIO events |
| **Storage** | Có giới hạn & cleanup | Buffer in-memory (không ghi) |
| **Re-ID** | Không có | Color histogram + SFace |
| **Security** | Không có | Zone intrusion + alert |
| **Demo use case** | Báo cáo đồ án, dataset label | Phòng lab, giám sát live |

---

## 13. Rủi Ro Kỹ Thuật Và Cách Xử Lý

### Rủi ro 1: eventlet monkey-patch xung đột với threading của HAVEN
**Vấn đề**: CPose dùng `eventlet` async mode cho SocketIO. HAVEN dùng `ThreadPoolExecutor`. Có thể xung đột.

**Giải pháp**:
```python
# app/__init__.py — import eventlet TRƯỚC mọi thứ
import eventlet
eventlet.monkey_patch()

# CameraThread dùng eventlet.spawn() thay vì threading.Thread
# để compatible với eventlet event loop
from eventlet.green import threading as green_threading
```

### Rủi ro 2: PoseADLEngine dùng chung state giữa offline và realtime
**Vấn đề**: `kp_buffers` dict là state per-person. Nếu offline và realtime dùng chung instance sẽ lẫn state.

**Giải pháp**: Hai instance riêng biệt — Singleton pattern tách biệt:
```python
# Offline instance
pose_engine_offline = PoseADLEngine.get_instance("offline")

# Realtime instance (window_size nhỏ hơn)
pose_engine_live = PoseADLEngine.get_instance("realtime")
```

### Rủi ro 3: Model load quá chậm khi start
**Vấn đề**: Load 5 model (yolov8n, yolo11n, yolo11n-pose, yunet, sface) cùng lúc tốn nhiều thời gian và RAM.

**Giải pháp**: Lazy load theo mode được kích hoạt:
```python
# Chỉ load model khi phase/mode đó được gọi lần đầu
@cached_property
def pose_model(self):
    return YOLO(self.config["models"]["phase3_pose"])
```

### Rủi ro 4: RTSP stream drop giữa chừng làm mất tracking
**Vấn đề**: Khi RTSP reconnect, DeepSort và GlobalIDManager mất track cũ.

**Giải pháp**:
- `CameraThread` tự động clear `kp_buffers` khi reconnect
- `IdentityManager.reset_cam(cam_id)` xoá mapping local→global của camera đó
- Không xoá GlobalID gốc trong SQLite (người vẫn còn trong DB)

---

## 14. Tóm Tắt Nguyên Tắc Thiết Kế

1. **Single Codebase, Dual Mode**: Không tạo 2 project riêng biệt. Mode được chọn qua config hoặc UI.
2. **Shared Core không phụ thuộc Flask**: `core/` không import Flask, có thể test độc lập.
3. **Sequential Phase không bị phá vỡ**: CPose 3 phase giữ nguyên interface, chỉ bổ sung thêm pipeline mới.
4. **No new dependencies ngoài `deep-sort-realtime`**: Tuân thủ constraint CPose CLAUDE.md.
5. **Lazy stream subscription**: Camera stream chỉ encode/emit khi có subscriber (tối ưu CPU).
6. **ADL Window khác nhau theo mode**: Offline = 30 frames (chính xác), Realtime = 15 frames (responsive).
7. **State isolation**: Offline và Realtime PoseADLEngine dùng instance riêng, không share `kp_buffers`.
