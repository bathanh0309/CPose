
> **ĐỌC FILE NÀY TRƯỚC KHI SINH BẤT KỲ CODE NÀO.**
> Đây là tài liệu ràng buộc bắt buộc cho mọi AI assistant (Claude, Copilot, Codex, GPT-4, v.v.)
> khi làm việc với dự án CPose. Mọi quyết định thiết kế đã được cố định tại đây.
> Không suy luận, không tự ý thêm dependency, không thay đổi cấu trúc.

---

## 1. Bối cảnh & Mục tiêu Dự án

**CPose** là hệ thống thu thập, phân tích và nhận dạng hành vi người tự động từ nhiều camera IP (RTSP),
phục vụ nghiên cứu pose estimation và ADL (Activities of Daily Living).

**Loại đồ án:** Tốt nghiệp kỹ sư Điện tử Viễn thông & Kỹ thuật Máy tính chất lượng cao,
Trường Đại học Bách Khoa — Đại học Đà Nẵng.

**Mục tiêu kỹ thuật:**
- Phase 1: Đọc RTSP từ N camera IP → YOLOv8n real-time detect người → ghi clip MP4 ngắn
- Phase 2: Chạy YOLOv11n offline trên clip đã ghi → xuất frame PNG + file bounding box label
- Phase 3: Chạy YOLOv11-pose trên clip → xuất keypoints + ADL label (hành vi người)

**Môi trường triển khai:**
- Windows 10/11, Python 3.10+
- Phòng lab, camera IP (RTSP)
- Khởi động bằng `run.bat`, mở trình duyệt `http://localhost:5000`
- CUDA optional (auto-fallback sang CPU nếu không có GPU)

**KHÔNG PHẢI:** web service cloud, multi-user, database-driven, container-based.

---

## 2. Cấu trúc Thư mục (Canonical — KHÔNG THÊM, KHÔNG XÓA)

```
Capstone_Project/        ← thư mục gốc (root), KHÔNG đặt lại tên
├── app/
│   ├── __init__.py                 # Flask app factory: create_app() duy nhất
│   ├── api/
│   │   ├── __init__.py             # Đăng ký blueprint + SocketIO handler
│   │   └── routes.py               # TẤT CẢ REST endpoint + SocketIO events
│   ├── services/
│   │   ├── __init__.py
│   │   ├── phase1_recorder.py      # Class RecorderManager (singleton)
│   │   ├── phase2_analyzer.py      # Class Analyzer (singleton)
│   │   └── phase3_recognizer.py    # Class PoseADLRecognizer (singleton) ← PHASE 3
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py         # StorageManager: enforce_storage_limit(), list_videos()
│       ├── stream_probe.py         # probe_rtsp(url) → {width, height, fps}
│       └── pose_utils.py           # calc_angle(), calc_velocity(), rule_based_adl() ← PHASE 3
├── configs/
│   ├── unified_config.yaml         # Tham số chung toàn dự án
│   ├── pose_adl.yaml               # Config Phase 3: model, ADL classes, thresholds
│   └── runtime.env.example
├── data/
│   |
│   │ 
│   ├── raw_videos/                 # [git-ignored] Clip MP4 Phase 1
│   ├── output_labels/              # [git-ignored] PNG + TXT Phase 2
│   └── output_pose/                # [git-ignored] Keypoints + ADL Phase 3 ← PHASE 3
│       └── <clip_stem>/
│           ├── <clip_stem>_keypoints.txt
│           ├── <clip_stem>_adl.txt
│           └── <clip_stem>_overlay_NNNN.png   (optional: frame skeleton)
├── models/                         # [git-ignored nếu lớn] .pt weights YOLO
│   ├── yolov8n.pt                  # Phase 1
│   ├── yolov11n.pt                  # Phase 2
│   └── yolo11n-pose.pt             # Phase 3 ← PHASE 3
├── static/
│   ├── index.html                  # SPA duy nhất — KHÔNG tạo thêm html khác
│   ├── css/
│   │   └── style.css               # Toàn bộ style — KHÔNG import Tailwind/Bootstrap
│   └── js/
│       └── app.js                  # Toàn bộ frontend logic — Vanilla JS + Socket.IO CDN
├── main.py                         # Entry point: from app import create_app
├── run.bat                         # Windows launcher
├── requirements.txt
├── .env
├── .gitignore
├── CLAUDE.md                       # File này
└── README.md
```

**Quy tắc cứng:**
- Không tạo file Python nào ngoài cấu trúc trên
- Không tách routes thành nhiều file — tất cả trong `routes.py`
- Không dùng Blueprint lồng nhau — 1 blueprint duy nhất `api_bp`
- Không tạo folder `feat/`, `modules/`, `controllers/` hay bất kỳ cấu trúc khác
- Thư mục gốc là `Capstone_Project/` — KHÔNG đổi tên, KHÔNG tạo subfolder `CPose/`

---

## 3. Công nghệ & Dependency (CỨNG — KHÔNG THÊM)

### Backend
```
flask>=3.0,<4
flask-socketio>=5.3,<6
flask-cors>=4.0,<5
eventlet>=0.36,<1          # async mode cho SocketIO
ultralytics>=8.3,<9        # YOLOv8n + YOLOv11 + YOLOv11n-pose
opencv-python>=4.9,<5      # RTSP capture + VideoWriter
Pillow>=10.0,<11           # Lưu PNG frame
psutil>=5.9,<7             # Disk usage
PyYAML>=6,<7               # Đọc config
python-dotenv>=1,<2
numpy>=1.24,<3             # Tính toán keypoints, góc khớp
```

### Frontend (CDN — KHÔNG npm/webpack/vite)
```html
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<!-- Không dùng React, Vue, jQuery, Bootstrap, Tailwind -->
```

### KHÔNG DÙNG:
- FastAPI, Django, SQLAlchemy, SQLite
- React, Vue, Angular, jQuery
- Tailwind CSS, Bootstrap, Material UI
- Docker, celery, redis
- mediapipe, mmpose, ViTPose (dùng ultralytics pose thay thế)

---

## 4. Quy tắc Đặt tên File Output

| Loại | Pattern | Ví dụ |
|------|---------|-------|
| Clip Phase 1 | `YYYYMMDD_HHMMSS_camXX.mp4` | `20240315_143022_cam01.mp4` |
| Frame Phase 2 | `<clip_stem>_frame_<NNNN>.png` | `20240315_143022_cam01_frame_0042.png` |
| Label Phase 2 | `<clip_stem>_labels.txt` | `20240315_143022_cam01_labels.txt` |
| Keypoints Phase 3 | `<clip_stem>_keypoints.txt` | `20240315_143022_cam01_keypoints.txt` |
| ADL Label Phase 3 | `<clip_stem>_adl.txt` | `20240315_143022_cam01_adl.txt` |
| Overlay Phase 3 | `<clip_stem>_overlay_<NNNN>.png` | `20240315_143022_cam01_overlay_0042.png` |

- `camXX`: thứ tự camera trong `resources.txt`, đếm từ `01`, zero-padded 2 chữ số
- `NNNN`: frame index trong clip, zero-padded 4 chữ số, bắt đầu từ `0000`
- Output Phase 2: `data/output_labels/<clip_stem>/`
- Output Phase 3: `data/output_pose/<clip_stem>/`

---

## 5. Format File resources.txt

```
# Dòng bắt đầu bằng # là comment, bị bỏ qua
# Dòng trống cũng bị bỏ qua
rtsp://admin:password@192.168.1.101:554/stream1
rtsp://admin:password@192.168.1.102:554/stream1
```

- Chỉ RTSP URL, 1 URL/dòng
- Tên cam được tạo tự động: `cam01`, `cam02`, ...

---

## 6. Format File Label Phase 2 (labels.txt)

```
# frame_id x_min y_min x_max y_max
0 112 54 340 490
0 512 102 680 495
1 115 58 338 492
```

- Separator: dấu cách (space)
- Tọa độ pixel tuyệt đối (KHÔNG normalize về [0,1])
- Class luôn là `person` — KHÔNG ghi class vào file

---

## 7. Format File Keypoints Phase 3 (keypoints.txt)

```
# frame_id person_id kp0_x kp0_y kp0_conf kp1_x kp1_y kp1_conf ... kp16_x kp16_y kp16_conf
0 0 340.2 120.5 0.91 330.1 115.3 0.88 ... 280.1 410.3 0.85
0 1 510.4 200.1 0.79 ...
1 0 341.0 122.3 0.93 ...
```

- 17 keypoints theo chuẩn COCO (nose, left_eye, right_eye, left_ear, right_ear,
  left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
  left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)
- Tọa độ pixel tuyệt đối, confidence [0,1]
- `person_id` = index trong frame đó, bắt đầu từ 0

---

## 8. Format File ADL Label Phase 3 (adl.txt)

```
# frame_id person_id adl_label confidence
0  0  standing   0.92
30 0  walking    0.87
60 0  bending    0.74
90 0  falling    0.81
```

**ADL Classes được hỗ trợ:**
```
standing, sitting, walking, lying_down, falling, reaching, bending, unknown
```

---

## 9. Phase 1 — Recording Logic Chi tiết

### Hằng số (trong phase1_recorder.py — KHÔNG đổi tên)

```python
PRE_BUFFER_SEC     = 3
POST_BUFFER_SEC    = 3
INFERENCE_EVERY    = 5
MIN_CLIP_DURATION  = 2.0
PERSON_CLASS_ID    = 0
CONF_THRESHOLD_P1  = 0.35
DEFAULT_STORAGE_GB = 10.0
```

### Luồng xử lý
```
resources.txt → parse N URLs
      ↓
Với mỗi URL → tạo 1 CameraThread (daemon thread)
      ↓
CameraThread:
  cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
  Frame loop:
    ret, frame = cap.read()
    Append vào deque (PRE_BUFFER_SEC×fps)
    Mỗi INFERENCE_EVERY frame:
      results = yolo_n(frame, classes=[0])
      persons = [r for conf>CONF_P1]
    Nếu persons detected:
      Nếu chưa ghi → flush deque vào file
      Tiếp tục ghi frame
      no_person_counter = 0
    Nếu không có persons:
      no_person_counter += 1
      Nếu đang ghi: vẫn ghi (POST_BUFFER)
      Nếu no_person_counter>POST_FRAMES:
        Đóng VideoWriter
        Nếu duration >= MIN_CLIP → giữ
        Nếu duration < MIN_CLIP → xóa
        emit("clip_saved", {...})
        enforce_storage_limit()
```

---

## 10. Phase 2 — Analysis Logic Chi tiết

### Hằng số (trong phase2_analyzer.py)

```python
CONF_THRESHOLD_P2  = 0.50
PROGRESS_EVERY     = 10
PERSON_CLASS_ID    = 0
```

### Luồng xử lý
```
User chọn folder → POST /api/analysis/start
      ↓
Analyzer.run(folder_path) chạy trong Thread riêng
      ↓
Với mỗi .mp4 (sort alphabetically):
  clip_stem = Path(mp4).stem
  output_dir = data/output_labels/<clip_stem>/

  Với mỗi frame:
    results = yolo_l(frame, classes=[0], conf=CONF_P2)
    boxes = results[0].boxes.xyxy

    Nếu len(boxes) > 0:
      cv2.imwrite(out_png, frame)
      label_file.write(f"{frame_id} {x1} {y1} {x2} {y2}\\n")

    Emit progress mỗi PROGRESS_EVERY frame
      ↓
emit("analysis_complete", {...})
```

---

## 11. Phase 3 — Pose & ADL Logic Chi tiết

### Hằng số (trong phase3_recognizer.py)

```python
POSE_MODEL_NAME     = "yolo11n-pose.pt"
CONF_THRESHOLD_P3   = 0.45          # Confidence threshold detect người (pose model)
KP_CONF_MIN         = 0.3           # Minimum keypoint confidence để tính ADL
WINDOW_SIZE         = 30            # Số frame cho sliding window ADL classification
PROGRESS_EVERY      = 10            # Emit progress mỗi N frame
SAVE_OVERLAY        = True          # Có lưu frame skeleton PNG hay không
PERSON_CLASS_ID     = 0
```

### ADL Classes (pose_adl.yaml)
```yaml
adl_classes:
  - standing
  - sitting
  - walking
  - lying_down
  - falling
  - reaching
  - bending
  - unknown

# Thresholds góc khớp (độ) cho rule-based classifier
thresholds:
  knee_bend_angle: 150      # góc đầu gối < 150° → đang ngồi/ngồi xổm
  hip_angle_lying: 160      # góc hông > 160° khi nằm
  shoulder_raise: 45        # góc vai > 45° so với trục ngang → reaching/bending
  velocity_walk: 8.0        # pixel/frame trung bình để phân biệt đứng/đi
```

### Luồng xử lý
```
User chọn folder → POST /api/pose/start
      ↓
PoseADLRecognizer.run(folder_path) chạy trong Thread riêng
      ↓
Với mỗi .mp4 (sort alphabetically):
  clip_stem = Path(mp4).stem
  output_dir = data/output_pose/<clip_stem>/

  kp_buffer = {}   # person_id → deque(maxlen=WINDOW_SIZE)
  kp_file    = output_dir / f"{clip_stem}_keypoints.txt"
  adl_file   = output_dir / f"{clip_stem}_adl.txt"

  Với mỗi frame:
    results = pose_model(frame, classes=[0], conf=CONF_P3)
    keypoints_xy   = results[0].keypoints.xy    # [N, 17, 2]
    keypoints_conf = results[0].keypoints.conf  # [N, 17]

    Với mỗi person i:
      Lọc keypoints có conf >= KP_CONF_MIN
      Ghi vào kp_file

      kp_buffer[i].append(keypoints_xy[i])
      Nếu len(kp_buffer[i]) == WINDOW_SIZE:
        adl_label, conf = rule_based_adl(kp_buffer[i], config)
        Ghi vào adl_file: f"{frame_id} {i} {adl_label} {conf:.2f}\\n"

    Nếu SAVE_OVERLAY:
      overlay = draw_skeleton(frame, keypoints_xy, keypoints_conf)
      cv2.imwrite(overlay_path, overlay)

    Emit progress mỗi PROGRESS_EVERY frame
      ↓
emit("pose_complete", {clips_done, keypoints_written, adl_events})
```

### Rule-based ADL Classifier (pose_utils.py)

Hàm `rule_based_adl(window: deque, config: dict) -> (str, float)`:

```python
# Sử dụng frame cuối window để phân loại
# 1. Tính góc khớp: calc_angle(p1, vertex, p2) → degrees
# 2. Tính velocity trung bình qua window (ankle displacement)
# 3. Áp dụng rule theo thứ tự ưu tiên:

# Priority 1: falling (hip y thấp + thân nghiêng mạnh)
# Priority 2: lying_down (hip_angle > threshold khi hip.y gần frame bottom)
# Priority 3: sitting (knee_angle < threshold)
# Priority 4: bending (shoulder_angle > threshold, người không di chuyển)
# Priority 5: reaching (wrist.y < shoulder.y)
# Priority 6: walking (ankle velocity > threshold)
# Priority 7: standing (default)
# Fallback: unknown (quá ít keypoints hợp lệ)
```

Hàm tiện ích trong `pose_utils.py`:
```python
def calc_angle(p1, vertex, p2) -> float:
    """Tính góc (độ) tại vertex giữa vector p1-vertex và p2-vertex."""

def calc_velocity(positions: list) -> float:
    """Tính tốc độ trung bình (pixel/frame) từ chuỗi vị trí."""

def draw_skeleton(frame, keypoints_xy, keypoints_conf, min_conf=0.3) -> np.ndarray:
    """Vẽ skeleton COCO 17 keypoints + limbs lên frame, trả về frame mới."""

def rule_based_adl(window, config) -> tuple[str, float]:
    """Rule-based ADL classifier từ sliding window keypoints."""
```

---

## 12. API Endpoints (routes.py — ĐẦY ĐỦ)

### Phase 1 & 2 (giữ nguyên)

| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/` | Serve `static/index.html` |
| POST | `/api/config/upload` | Upload `resources.txt` |
| GET | `/api/config/cameras` | Danh sách camera |
| POST | `/api/cameras/probe` | Probe RTSP |
| POST | `/api/recording/start` | Bắt đầu Phase 1 |
| POST | `/api/recording/stop` | Dừng Phase 1 |
| GET | `/api/recording/status` | Trạng thái Phase 1 |
| GET | `/api/videos` | Danh sách clip MP4 |
| DELETE | `/api/videos/<filename>` | Xóa clip |
| POST | `/api/analysis/start` | Bắt đầu Phase 2 |
| POST | `/api/analysis/stop` | Dừng Phase 2 |
| GET | `/api/analysis/status` | Tiến độ Phase 2 |
| GET | `/api/analysis/results` | Kết quả Phase 2 |
| GET | `/api/storage/info` | Dung lượng raw_videos |
| POST | `/api/storage/limit` | Đặt giới hạn storage |

### Phase 3 (thêm mới)

| Method | Path | Body / Params | Response | Mô tả |
|--------|------|---------------|----------|-------|
| POST | `/api/pose/start` | `{folder: "data/raw_videos", save_overlay: true}` | `{status, total_clips}` | Bắt đầu Phase 3 |
| POST | `/api/pose/stop` | — | `{status}` | Dừng Phase 3 |
| GET | `/api/pose/status` | — | `{running, current_clip, progress_pct}` | Tiến độ Phase 3 |
| GET | `/api/pose/results` | `?folder=data/output_pose` | `{results: [{clip_stem, keypoints_count, adl_events, adl_summary}]}` | Kết quả Phase 3 |
| GET | `/api/pose/adl_summary` | `?clip=<clip_stem>` | `{clip, adl_distribution: {label: pct, ...}}` | Thống kê ADL theo clip |

### SocketIO Events (Server → Client)

| Event | Payload | Phase |
|-------|---------|-------|
| `camera_status` | `{cam_id, status, fps, resolution, timestamp}` | 1 |
| `detection_event` | `{cam_id, timestamp, person_count, confidence_max}` | 1 |
| `clip_saved` | `{filename, size_mb, duration_s, cam_id}` | 1 |
| `analysis_progress` | `{clip, frame_id, total_frames, pct, frames_saved}` | 2 |
| `analysis_complete` | `{clips_done, frames_saved, labels_written, elapsed_s}` | 2 |
| `pose_progress` | `{clip, frame_id, total_frames, pct, persons_detected}` | 3 |
| `pose_complete` | `{clips_done, keypoints_written, adl_events, elapsed_s}` | 3 |
| `storage_warning` | `{used_gb, limit_gb, pct}` | 1–3 |
| `error` | `{source, message, traceback}` | 1–3 |

---

## 13. Frontend Architecture (static/index.html + js/app.js)

### Cấu trúc SPA — 5 Tab

```
index.html
  ├── Tab 1: Configuration
  ├── Tab 2: Live Monitor
  ├── Tab 3: Analysis (Phase 2)
  ├── Tab 4: Results (Phase 2)
  └── Tab 5: Pose & ADL (Phase 3)   ← THÊM MỚI
```

### Tab 5: Pose & ADL
- Input đường dẫn folder clip (default: `data/raw_videos`)
- Nút **Load** → hiển thị danh sách clip
- Checkbox **Save overlay** (lưu frame skeleton PNG)
- Nút **Start Pose Analysis** → POST `/api/pose/start`
- Progress bar tổng thể + tên clip đang xử lý
- Nút **Stop**
- Sau khi hoàn tất: accordion từng clip hiển thị:
  - Tổng số keypoints được ghi
  - Biểu đồ ADL distribution (bar đơn giản CSS, không cần Chart.js)
  - Bảng `adl.txt`: frame_id | person_id | label | confidence
  - Preview overlay frame đầu tiên nếu có

---

## 14. CSS Rules (style.css)

- **Dark theme bắt buộc**: `background: #0f1117`, text `#e2e8f0`
- **Accent color**: `#3b82f6` (blue-500 equivalent)
- **Font**: `font-family: 'Inter', system-ui, sans-serif`

```css
:root {
  --bg-primary:   #0f1117;
  --bg-secondary: #1a1f2e;
  --bg-card:      #242938;
  --border:       #2d3748;
  --text-primary: #e2e8f0;
  --text-muted:   #718096;
  --accent:       #3b82f6;
  --success:      #48bb78;
  --warning:      #ed8936;
  --error:        #fc8181;
  --pose:         #a78bfa;   /* màu riêng cho Phase 3 */
}
```

---

## 15. Python Code Rules

### App factory (app/__init__.py)

```python
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

socketio = SocketIO()

def create_app():
    app = Flask(__name__, static_folder="../static", static_url_path="/static")
    CORS(app)
    socketio.init_app(app, async_mode="eventlet", cors_allowed_origins="*")
    from app.api import api_bp
    app.register_blueprint(api_bp)
    return app
```

### Path constants (BASE_DIR)

```python
from pathlib import Path

BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
RAW_VIDEOS      = DATA_DIR / "raw_videos"
OUTPUT_LABELS   = DATA_DIR / "output_labels"
OUTPUT_POSE     = DATA_DIR / "output_pose"       # Phase 3
CONFIG_DIR      = DATA_DIR / "config"
MODELS_DIR      = BASE_DIR / "models"
CONFIGS_DIR     = BASE_DIR / "configs"
RESOURCES       = CONFIG_DIR / "resources.txt"
POSE_ADL_CFG    = CONFIGS_DIR / "pose_adl.yaml"
```

### Singleton pattern

```python
class PoseADLRecognizer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### Logging convention

```python
logger.info("[Phase3] Processing clip %s", clip_name)
logger.info("[Phase3] ADL detected: %s (%.2f) frame %d", label, conf, frame_id)
logger.warning("[Phase3] Low keypoint confidence frame %d", frame_id)
logger.error("[Phase3] Error processing clip %s: %s", clip_name, err)
```

---

## 16. run.bat (Windows Launcher)

```bat
@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"

if not exist "%VENV%\\Scripts\\activate.bat" (
    echo [CPose] Tao virtual environment...
    python -m venv "%VENV%"
)

call "%VENV%\\Scripts\\activate.bat"

pip show flask >nul 2>&1
if errorlevel 1 (
    echo [CPose] Cai dat dependency...
    pip install -r "%ROOT%requirements.txt" --quiet
)

mkdir "%ROOT%data\\raw_videos"    2>nul
mkdir "%ROOT%data\\output_labels" 2>nul
mkdir "%ROOT%data\\output_pose"   2>nul
mkdir "%ROOT%data\\config"        2>nul
mkdir "%ROOT%models"              2>nul

echo [CPose] Khoi dong server tai http://localhost:5000
start "" timeout /t 2 /nobreak >nul && start http://localhost:5000
python "%ROOT%main.py"
pause
```

---

## 17. .gitignore

```
__pycache__/
*.py[cod]
venv/
.env

data/raw_videos/
data/output_labels/
data/output_pose/
models/

!data/raw_videos/.gitkeep
!data/output_labels/.gitkeep
!data/output_pose/.gitkeep
!data/config/.gitkeep
!models/.gitkeep

.DS_Store
Thumbbs.db
.vscode/
.idea/
```

---

## 18. Requirements (requirements.txt)

```
flask>=3.0,<4
flask-socketio>=5.3,<6
flask-cors>=4.0,<5
eventlet>=0.36,<1
ultralytics>=8.3,<9
opencv-python>=4.9,<5
Pillow>=10.0,<11
psutil>=5.9,<7
PyYAML>=6,<7
python-dotenv>=1,<2
numpy>=1.24,<3
```

---

## 19. Những điều TUYỆT ĐỐI không làm

| Không làm | Thay bằng |
|-----------|-----------|
| Thêm SQLite / database | Filesystem + JSON nếu cần persist |
| Thêm FastAPI, Django | Chỉ Flask |
| Dùng React/Vue | Vanilla JS + Socket.IO CDN |
| Tạo file route mới ngoài routes.py | Thêm endpoint vào routes.py |
| Hard-code path dạng string | pathlib.Path từ BASE_DIR |
| Dùng global variable | Singleton pattern (get_instance) |
| Commit .pt, .mp4, .png vào git | .gitignore |
| import os.path.join | from pathlib import Path |
| Tạo thread không phải daemon | thread.daemon = True |
| Blocking call trong route handler | Background thread + emit SocketIO |
| Dùng mediapipe/mmpose cho pose | ultralytics pose (yolo11n-pose.pt) |
| Tạo folder `feat-*` hay `src/` | Cấu trúc cố định mục 2 |

---

## 20. Workflow Người Dùng (End-to-End)

```
1. Chạy run.bat → Browser tự mở http://localhost:5000

2. Tab "Configuration"
   ├─ Upload resources.txt
   ├─ Probe Cameras → xem resolution/FPS
   ├─ Chọn resolution ghi hình
   └─ Set storage limit

3. Tab "Live Monitor"
   ├─ Start Recording → YOLOv8n detect người → ghi clip MP4
   ├─ Xem live log + storage gauge
   └─ Stop Recording khi xong

4. Tab "Analysis" (Phase 2)
   ├─ Load clip từ data/raw_videos
   ├─ Start Analysis → YOLOv11n offline
   └─ Theo dõi progress bar

5. Tab "Results" (Phase 2)
   ├─ Xem accordion từng clip
   ├─ Preview thumbnail PNG
   └─ Download label.txt

6. Tab "Pose & ADL" (Phase 3)
   ├─ Load clip (cùng folder hoặc thư mục khác)
   ├─ Bật/tắt Save overlay
   ├─ Start Pose Analysis → yolo11n-pose → keypoints + ADL
   ├─ Theo dõi progress
   └─ Xem ADL distribution + download keypoints/adl.txt
```

---

*Phiên bản: 3.0 — Bổ sung Phase 3 Pose Estimation & ADL Recognition.*
*Tài liệu này là nguồn sự thật duy nhất (single source of truth) cho mọi AI assistant.*
'''

readme_md = '''# CPose — Multi-Camera Person Detection, Auto-Labeling & ADL Recognition

> **Đồ án Tốt nghiệp Kỹ sư** — Ngành Điện tử Viễn thông & Kỹ thuật Máy tính Chất lượng cao
> Trường Đại học Bách Khoa — Đại học Đà Nẵng · 2024–2025

---

## Tổng quan

**CPose** là hệ thống thu thập, phân tích và nhận dạng hành vi người tự động từ nhiều camera IP (RTSP),
phục vụ nghiên cứu pose estimation và ADL (Activities of Daily Living).

Pipeline hoạt động hoàn toàn tự động qua **3 giai đoạn**:

| Giai đoạn | Mô hình | Mục tiêu | Chế độ |
|-----------|---------|----------|--------|
| **Phase 1** | YOLOv8n | Phát hiện người, kích hoạt ghi clip MP4 | Realtime — chạy ngầm liên tục |
| **Phase 2** | YOLOv11 | Trích xuất frame PNG + bounding box label | Offline — chạy theo yêu cầu |
| **Phase 3** | YOLOv11n-pose | Keypoints 17 điểm + ADL classification | Offline — chạy theo yêu cầu |

---

## Tính năng chính

### Phase 1 — Ghi hình thông minh
- Đọc đồng thời N luồng RTSP từ `resources.txt`
- Chỉ ghi clip khi phát hiện người (event-based) → tiết kiệm bộ nhớ
- Pre-buffer 3 giây và post-buffer 3 giây quanh sự kiện
- YOLO inference mỗi 5 frame để tối ưu tải CPU/GPU
- Tự động reconnect khi mất kết nối RTSP
- Quản lý bộ nhớ: xóa clip cũ nhất khi vượt giới hạn đặt trước

### Phase 2 — Auto-labeling chất lượng cao
- Chạy YOLOv11 offline trên toàn bộ clip hoặc thư mục chọn
- Chỉ lưu frame khi có người (confidence ≥ 0.50)
- Output chuẩn: frame PNG gốc + file label tọa độ pixel tuyệt đối
- Tracking tiến độ realtime qua SocketIO

### Phase 3 — Pose Estimation & ADL Recognition *(Mới)*
- Chạy YOLOv11n-pose trên clip đã ghi → 17 keypoints COCO format/người/frame
- Sliding window 30 frame → rule-based ADL classifier
- Nhận dạng 8 hoạt động: `standing`, `sitting`, `walking`, `lying_down`, `falling`, `reaching`, `bending`, `unknown`
- Tùy chọn lưu frame overlay skeleton PNG
- Thống kê phân bố ADL theo từng clip

### Dashboard Web (SPA)
- Giao diện dark theme duy nhất, 5 tab, không cần cài thêm phần mềm
- Probe tự động resolution và FPS của từng camera
- Live log: detection events, clip saved, storage usage
- Hoàn toàn Vanilla JS — không phụ thuộc framework

---

## Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| OS | Windows 10 / Linux | Windows 11 |
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA CUDA (tăng tốc YOLO) |
| Mạng | Truy cập RTSP camera | LAN tốc độ cao |

---

## Cấu trúc dự án

```
Capstone_Project/
├── app/
│   ├── __init__.py              # Flask app factory + path constants
│   ├── api/
│   │   └── routes.py            # Tất cả REST endpoint + SocketIO events
│   ├── services/
│   │   ├── phase1_recorder.py   # RecorderManager: RTSP + YOLOv8n + clip writer
│   │   ├── phase2_analyzer.py   # Analyzer: YOLOv11 + frame/label extractor
│   │   └── phase3_recognizer.py # PoseADLRecognizer: YOLOv11n-pose + ADL
│   └── utils/
│       ├── file_handler.py      # StorageManager: storage limit, file listing
│       ├── stream_probe.py      # probe_rtsp(): resolution + FPS từ RTSP
│       └── pose_utils.py        # calc_angle(), rule_based_adl(), draw_skeleton()
├── configs/
│   ├── unified_config.yaml      # Tham số chung toàn dự án
│   ├── pose_adl.yaml            # Config Phase 3: ADL classes, thresholds
│   └── runtime.env.example
├── data/
│   ├── config/
│   │   └── resources.txt        # Danh sách RTSP URL (1 URL/dòng)
│   ├── raw_videos/              # [git-ignored] Clip MP4 Phase 1
│   ├── output_labels/           # [git-ignored] PNG + TXT Phase 2
│   └── output_pose/             # [git-ignored] Keypoints + ADL Phase 3
│       └── <clip_stem>/
│           ├── <clip_stem>_keypoints.txt
│           ├── <clip_stem>_adl.txt
│           └── <clip_stem>_overlay_NNNN.png
├── models/                      # [git-ignored] YOLOv8/11 .pt weights
│   ├── yolov8n.pt               # Phase 1
│   ├── yolo11n.pt               # Phase 2
│   └── yolo11n-pose.pt          # Phase 3
├── static/
│   ├── index.html               # Dashboard SPA (5 tab)
│   ├── css/style.css
│   └── js/app.js
├── main.py
├── run.bat                      # Windows one-click launcher
├── requirements.txt
├── CLAUDE.md                    # AI assistant codex (đọc trước khi code)
└── .gitignore
```

---

## Cài đặt & Chạy

### Windows — One-click

```bat
run.bat
```

Script tự động:
1. Tạo virtual environment `.venv` nếu chưa có
2. Cài đặt tất cả dependency từ `requirements.txt`
3. Tạo các thư mục dữ liệu cần thiết (`raw_videos`, `output_labels`, `output_pose`)
4. Khởi động server Flask tại `http://localhost:5000`
5. Mở trình duyệt tự động

### Manual (Windows / Linux)

```bash
pip install -r requirements.txt
python main.py
```

Mở trình duyệt: `http://localhost:5000`

---

## Hướng dẫn sử dụng

### Bước 1 — Cấu hình camera (Tab: Configuration)

Tạo file `resources.txt`:

```
# Mỗi dòng là 1 RTSP URL. Dòng bắt đầu bằng # là comment.
rtsp://admin:password@192.168.1.100:554/stream1
rtsp://admin:password@192.168.1.101:554/stream1
```

1. Tải `resources.txt` lên (kéo thả hoặc click)
2. Nhấn **Probe Cameras** → lấy resolution và FPS của từng camera
3. Chọn độ phân giải ghi hình (chỉ hiện option ≤ max resolution của cam)
4. Thiết lập giới hạn storage (mặc định 10 GB)

### Bước 2 — Ghi hình (Tab: Live Monitor)

1. Nhấn **Start Recording**
2. YOLOv8n chạy ngầm mỗi 5 frame/camera
3. Clip được đặt tên: `YYYYMMDD_HHMMSS_camXX.mp4`
4. Nhấn **Stop Recording** khi hoàn tất

### Bước 3 — Auto-labeling (Tab: Analysis)

1. Load clip từ `data/raw_videos`
2. Nhấn **Start Analysis** — YOLOv11 offline
3. Output tại `data/output_labels/<clip_stem>/`:

```
data/output_labels/
└── 20240315_143022_cam01/
    ├── 20240315_143022_cam01_frame_0000.png
    ├── 20240315_143022_cam01_frame_0003.png
    └── 20240315_143022_cam01_labels.txt
```

### Bước 4 — Xem kết quả Phase 2 (Tab: Results)

- Accordion theo từng clip: số frame PNG, số bounding box
- Preview thumbnail, tải về `labels.txt`

### Bước 5 — Pose & ADL (Tab: Pose & ADL) *(Mới)*

1. Load clip (cùng folder `data/raw_videos` hoặc thư mục khác)
2. Bật/tắt **Save overlay** (lưu frame skeleton PNG)
3. Nhấn **Start Pose Analysis**
4. Output tại `data/output_pose/<clip_stem>/`:

```
data/output_pose/
└── 20240315_143022_cam01/
    ├── 20240315_143022_cam01_keypoints.txt
    ├── 20240315_143022_cam01_adl.txt
    └── 20240315_143022_cam01_overlay_0000.png
```

5. Xem biểu đồ phân bố ADL + tải về file kết quả

---

## Định dạng file output

### Phase 2 — labels.txt

```
# frame_id  x_min  y_min  x_max  y_max
0           112    54     340    490
0           512    102    680    495
1           115    58     338    492
```

| Trường | Mô tả |
|--------|-------|
| `frame_id` | Index frame trong clip, từ 0 |
| `x_min, y_min` | Góc trên bên trái bounding box (pixel tuyệt đối) |
| `x_max, y_max` | Góc dưới bên phải bounding box (pixel tuyệt đối) |

> Class luôn là `person` (COCO class 0) — không ghi vào file. Tọa độ **không normalize**.

### Phase 3 — keypoints.txt

```
# frame_id person_id kp0_x kp0_y kp0_conf ... kp16_x kp16_y kp16_conf
0 0 340.2 120.5 0.91 330.1 115.3 0.88 ... 280.1 410.3 0.85
```

- 17 keypoints COCO (nose → right_ankle)
- Tọa độ pixel tuyệt đối, confidence [0,1]

### Phase 3 — adl.txt

```
# frame_id person_id adl_label confidence
0  0  standing  0.92
30 0  walking   0.87
60 0  bending   0.74
```

**ADL classes:** `standing` · `sitting` · `walking` · `lying_down` · `falling` · `reaching` · `bending` · `unknown`

---

## API Reference

### Phase 1 & 2

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Dashboard |
| POST | `/api/config/upload` | Upload `resources.txt` |
| GET | `/api/config/cameras` | Danh sách camera |
| POST | `/api/cameras/probe` | Probe RTSP |
| POST | `/api/recording/start` | Bắt đầu Phase 1 |
| POST | `/api/recording/stop` | Dừng Phase 1 |
| GET | `/api/recording/status` | Trạng thái Phase 1 |
| GET | `/api/videos` | Danh sách clip MP4 |
| DELETE | `/api/videos/<filename>` | Xóa clip |
| POST | `/api/analysis/start` | Bắt đầu Phase 2 |
| POST | `/api/analysis/stop` | Dừng Phase 2 |
| GET | `/api/analysis/status` | Tiến độ Phase 2 |
| GET | `/api/analysis/results` | Kết quả Phase 2 |
| GET | `/api/storage/info` | Dung lượng raw_videos |
| POST | `/api/storage/limit` | Đặt giới hạn storage |

### Phase 3 *(Mới)*

| Method | Endpoint | Body / Params | Mô tả |
|--------|----------|---------------|-------|
| POST | `/api/pose/start` | `{folder, save_overlay}` | Bắt đầu Phase 3 |
| POST | `/api/pose/stop` | — | Dừng Phase 3 |
| GET | `/api/pose/status` | — | Tiến độ Phase 3 |
| GET | `/api/pose/results` | `?folder=data/output_pose` | Kết quả Phase 3 |
| GET | `/api/pose/adl_summary` | `?clip=<clip_stem>` | ADL distribution theo clip |

### SocketIO Events (Server → Client)

| Event | Mô tả | Phase |
|-------|-------|-------|
| `camera_status` | Trạng thái kết nối, FPS, resolution | 1 |
| `detection_event` | Phát hiện người: cam, số người | 1 |
| `clip_saved` | Clip được lưu: tên, size, duration | 1 |
| `analysis_progress` | Tiến độ Phase 2: frame / tổng | 2 |
| `analysis_complete` | Phase 2 hoàn tất | 2 |
| `pose_progress` | Tiến độ Phase 3: frame / tổng, số người | 3 |
| `pose_complete` | Phase 3 hoàn tất: keypoints, ADL events | 3 |
| `storage_warning` | Dung lượng > 90% | 1–3 |
| `error` | Lỗi backend: source, message, traceback | 1–3 |

---

## YOLO Models

| Model | File | Phase | Confidence | Tải tự động |
|-------|------|-------|------------|-------------|
| YOLOv8n | `models/yolov8n.pt` | 1 | 0.35 | ✅ Lần đầu Phase 1 |
| YOLOv11 | `models/yolo11n.pt` | 2 | 0.50 | ✅ Lần đầu Phase 2 |
| YOLOv11n-pose | `models/yolo11n-pose.pt` | 3 | 0.45 | ✅ Lần đầu Phase 3 |

> Weights cache vào `models/` sau lần tải đầu. Yêu cầu Internet lần đầu.

---

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python 3.10+, Flask 3, Flask-SocketIO 5 (eventlet) |
| AI Engine | Ultralytics YOLOv8/11 (detect + pose) |
| Video | OpenCV (RTSP capture, VideoWriter, frame extraction) |
| Math | NumPy (tính góc khớp, velocity, sliding window) |
| Frontend | HTML5, CSS3, Vanilla JavaScript, Socket.IO CDN |
| Concurrency | `threading.Thread` (daemon) + `threading.Lock` |
| Storage | Filesystem (`pathlib.Path`) — không dùng database |

---

## Quản lý bộ nhớ

- Giới hạn mặc định: **10 GB** (có thể thay qua giao diện)
- Khi `raw_videos/` vượt giới hạn → tự động xóa clip cũ nhất cho đến < 80%
- Cảnh báo SocketIO khi dung lượng > 90%
- `output_labels/` và `output_pose/` không bị ảnh hưởng bởi cơ chế này

---

## Ghi chú kỹ thuật

- Mỗi camera chạy trên **daemon thread độc lập** — thread-safe với `threading.Lock`
- Pre-buffer 3 giây (deque) + post-buffer 3 giây sau khi mất detection
- YOLOv8n inference Phase 1: mỗi 5 frame (giảm tải CPU, giữ real-time)
- Clip < 2 giây tự động bị xóa (lọc false detection thoáng qua)
- Tự động reconnect RTSP sau 5 giây khi mất kết nối
- Phase 3: sliding window 30 frame → ADL rule-based classification theo góc khớp + velocity
- ADL rule priority: `falling` > `lying_down` > `sitting` > `bending` > `reaching` > `walking` > `standing`

---

*Phát triển phục vụ nghiên cứu đồ án tốt nghiệp kỹ sư — Điện tử Viễn thông & Kỹ thuật Máy tính Chất lượng cao, Đại học Bách Khoa Đà Nẵng.*
'''