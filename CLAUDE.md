# CLAUDE.md — Codex Contract for CPose

> **ĐỌC FILE NÀY TRƯỚC KHI SINH BẤT KỲ CODE NÀO.**
> Đây là tài liệu ràng buộc bắt buộc cho mọi AI assistant (Claude, Copilot, Codex, GPT-4, v.v.)
> khi làm việc với dự án CPose. Mọi quyết định thiết kế đã được cố định tại đây.
> Không suy luận, không tự ý thêm dependency, không thay đổi cấu trúc.

---

## 1. Bối cảnh & Mục tiêu Dự án

**CPose** là hệ thống thu thập dữ liệu giám sát người theo hai giai đoạn phục vụ nghiên cứu
pose estimation và ADL (Activities of Daily Living).

**Loại đồ án:** Tốt nghiệp kỹ sư Điện tử Viễn thông & Kỹ thuật Máy tính chất lượng cao,
Trường Đại học Bách Khoa Đà Nẵng.

**Mục tiêu kỹ thuật:**
- Phase 1: Đọc RTSP từ N camera IP → YOLOv8n real-time detect người → ghi clip MP4 ngắn
- Phase 2: Chạy YOLOv8l offline trên clip đã ghi → xuất frame PNG + file bounding box label

**Môi trường triển khai:**
- Windows 10/11, Python 3.10+
- Phòng lab, camera IP (RTSP)
- Khởi động bằng `run.bat`, mở trình duyệt `http://localhost:5000`
- CUDA optional (auto-fallback sang CPU nếu không có GPU)

**KHÔNG PHẢI:** web service cloud, multi-user, database-driven, container-based.

---

## 2. Cấu trúc Thư mục (Canonical — KHÔNG THÊM, KHÔNG XÓA)

```
CPose/
├── app/
│   ├── __init__.py                 # Flask app factory: create_app() duy nhất
│   ├── api/
│   │   ├── __init__.py             # Đăng ký blueprint + SocketIO handler
│   │   └── routes.py               # TẤT CẢ REST endpoint + SocketIO events
│   ├── services/
│   │   ├── __init__.py
│   │   ├── phase1_recorder.py      # Class RecorderManager (singleton)
│   │   └── phase2_analyzer.py      # Class Analyzer (singleton)
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py         # StorageManager: enforce_storage_limit(), list_videos()
│       └── stream_probe.py         # probe_rtsp(url) → {width, height, fps}
├── data/
│   ├── config/
│   │   └── resources.txt           # RTSP URLs, 1 dòng/URL, # = comment, dòng trống bỏ qua
│   ├── raw_videos/                 # [git-ignored] Clip MP4 Phase 1
│   └── output_labels/              # [git-ignored] PNG + TXT Phase 2
├── models/                         # [git-ignored nếu lớn] .pt weights YOLO
├── static/
│   ├── index.html                  # SPA duy nhất — KHÔNG tạo thêm html khác
│   ├── css/
│   │   └── style.css               # Toàn bộ style — KHÔNG import Tailwind/Bootstrap
│   └── js/
│       └── app.js                  # Toàn bộ frontend logic — Vanilla JS + Socket.IO CDN
├── main.py                         # Entry point: from app import create_app
├── run.bat                         # Windows launcher
├── requirements.txt
├── .gitignore
├── CLAUDE.md                       # File này
└── README.md
```

**Quy tắc cứng:**
- Không tạo file Python nào ngoài cấu trúc trên
- Không tách routes thành nhiều file — tất cả trong `routes.py`
- Không dùng Blueprint lồng nhau — 1 blueprint duy nhất `api_bp`
- Không tạo folder `feat/`, `modules/`, `controllers/` hay bất kỳ cấu trúc khác

---

## 3. Công nghệ & Dependency (CỨNG — KHÔNG THÊM)

### Backend
```
flask>=3.0,<4
flask-socketio>=5.3,<6
flask-cors>=4.0,<5
eventlet>=0.36,<1          # async mode cho SocketIO
ultralytics>=8.3,<9        # YOLOv8n + YOLOv8l
opencv-python>=4.9,<5      # RTSP capture + VideoWriter
Pillow>=10.0,<11           # Lưu PNG frame
psutil>=5.9,<7             # Disk usage
PyYAML>=6,<7               # Đọc config nếu cần
python-dotenv>=1,<2
```

### Frontend (CDN — KHÔNG npm/webpack/vite)
```html
<!-- Socket.IO client -->
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<!-- Không dùng React, Vue, jQuery, Bootstrap, Tailwind -->
```

### KHÔNG DÙNG:
- FastAPI, Django, SQLAlchemy, SQLite
- React, Vue, Angular, jQuery
- Tailwind CSS, Bootstrap, Material UI
- Docker, celery, redis

---

## 4. Quy tắc Đặt tên File Output

| Loại | Pattern | Ví dụ |
|------|---------|-------|
| Clip Phase 1 | `YYYYMMDD_HHMMSS_camXX.mp4` | `20240315_143022_cam01.mp4` |
| Frame Phase 2 | `<clip_stem>_frame_<NNNN>.png` | `20240315_143022_cam01_frame_0042.png` |
| Label Phase 2 | `<clip_stem>_labels.txt` | `20240315_143022_cam01_labels.txt` |

- `camXX`: thứ tự camera trong `resources.txt`, đếm từ `01`, zero-padded 2 chữ số
- `NNNN`: frame index trong clip, zero-padded 4 chữ số, bắt đầu từ `0000`
- Output Phase 2 lưu trong sub-folder: `data/output_labels/<clip_stem>/`

---

## 5. Format File resources.txt

```
# Dòng bắt đầu bằng # là comment, bị bỏ qua
# Dòng trống cũng bị bỏ qua
rtsp://admin:password@192.168.1.101:554/stream1
rtsp://admin:password@192.168.1.102:554/stream1
rtsp://user:pass@10.0.0.5:554/live
```

- Chỉ RTSP URL, 1 URL/dòng
- Không có tên cam trong file — tên cam được tạo tự động: `cam01`, `cam02`, ...
- File được upload qua giao diện web → lưu vào `data/config/resources.txt`

---

## 6. Format File Label (label.txt)

```
# frame_id x_min y_min x_max y_max
0 112 54 340 490
0 512 102 680 495
1 115 58 338 492
3 220 80 410 500
```

- Separator: dấu cách (space), không phải tab, không phải CSV
- Tọa độ pixel tuyệt đối (KHÔNG normalize về [0,1])
- `frame_id` = index frame trong clip gốc, bắt đầu từ 0
- Class luôn là `person` (class 0 COCO) — KHÔNG ghi class vào file
- Dòng đầu là comment header
- Mỗi dòng sau = 1 bounding box (nhiều box trên cùng frame_id là hợp lệ)

---

## 7. Phase 1 — Recording Logic Chi tiết

### Luồng xử lý

```
resources.txt → parse N URLs
      ↓
Với mỗi URL → tạo 1 CameraThread (daemon thread)
      ↓
CameraThread:
  cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
  ┌─────────────────────────────────────────┐
  │ Frame loop:                              │
  │   ret, frame = cap.read()               │
  │   Append vào deque (PRE_BUFFER_SEC×fps) │
  │   Mỗi INFERENCE_EVERY frame:            │
  │     results = yolo_n(frame, classes=[0])│
  │     persons = [r for conf>CONF_P1]      │
  │   Nếu persons detected:                 │
  │     Nếu chưa ghi → flush deque vào file │
  │     Tiếp tục ghi frame hiện tại         │
  │     no_person_counter = 0               │
  │   Nếu không có persons:                 │
  │     no_person_counter += 1              │
  │     Nếu đang ghi: vẫn ghi (POST_BUFFER) │
  │     Nếu no_person_counter>POST_FRAMES → │
  │       Đóng VideoWriter                  │
  │       Nếu duration >= MIN_CLIP → giữ   │
  │       Nếu duration < MIN_CLIP → xóa    │
  │       emit("clip_saved", {...})         │
  │       enforce_storage_limit()           │
  └─────────────────────────────────────────┘
```

### Hằng số (trong phase1_recorder.py — KHÔNG đổi tên)

```python
PRE_BUFFER_SEC     = 3      # Giây buffer trước detection (deque size = fps × 3)
POST_BUFFER_SEC    = 3      # Giây buffer sau mất detection
INFERENCE_EVERY    = 5      # Chạy YOLO mỗi N frame để giảm tải CPU
MIN_CLIP_DURATION  = 2.0    # Giây tối thiểu để giữ clip (ngắn hơn → xóa)
PERSON_CLASS_ID    = 0      # Class "person" trong COCO
CONF_THRESHOLD_P1  = 0.35   # Confidence threshold Phase 1 (thấp hơn để không miss)
DEFAULT_STORAGE_GB = 10.0   # GB mặc định
```

### Resolution selection
- Khi user upload resources.txt → tự động probe từng URL
- Probe trả về: `{width, height, fps, status}` qua `stream_probe.probe_rtsp()`
- Frontend hiển thị resolution tối đa của cam và cho chọn trong dropdown:
  `[Tối đa: WxH, 720p, 480p, 360p]` (chỉ các option <= resolution tối đa của cam)
- Resolution được gửi qua `/api/recording/start` body: `{"resolution": "1280x720"}`
- RecorderManager set `cv2.CAP_PROP_FRAME_WIDTH`, `cv2.CAP_PROP_FRAME_HEIGHT` trước khi record

### Storage enforcement
- `enforce_storage_limit(raw_videos_dir, limit_gb)` trong `file_handler.py`
- Tính tổng size tất cả `.mp4` trong `raw_videos/`
- Nếu total > limit → xóa file cũ nhất (sort by `Path.stat().st_mtime` tăng dần)
- Xóa cho đến khi total < 80% limit (headroom buffer)
- Emit `"storage_warning"` khi total > 90% limit

---

## 8. Phase 2 — Analysis Logic Chi tiết

### Luồng xử lý

```
User chọn folder (path trên server) → POST /api/analysis/start
      ↓
Analyzer.run(folder_path) chạy trong Thread riêng
      ↓
Với mỗi .mp4 trong folder (sort alphabetically):
  clip_stem = Path(mp4).stem
  output_dir = data/output_labels/<clip_stem>/
  output_dir.mkdir(parents=True, exist_ok=True)
  label_file = output_dir / f"{clip_stem}_labels.txt"

  cap = cv2.VideoCapture(mp4_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_id = 0

  Với mỗi frame:
    ret, frame = cap.read()
    results = yolo_l(frame, classes=[0], conf=CONF_P2)
    boxes = results[0].boxes.xyxy  # [x_min, y_min, x_max, y_max]

    Nếu len(boxes) > 0:
      # Lưu frame PNG
      out_png = output_dir / f"{clip_stem}_frame_{frame_id:04d}.png"
      cv2.imwrite(str(out_png), frame)

      # Append label
      for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        label_file.write(f"{frame_id} {x1} {y1} {x2} {y2}\n")

    frame_id += 1
    # Emit progress mỗi 10 frame
    if frame_id % 10 == 0:
      emit("analysis_progress", {clip, frame_id, total_frames, pct})
      ↓
emit("analysis_complete", {clips_done, frames_saved, labels_written})
```

### Hằng số (trong phase2_analyzer.py)

```python
CONF_THRESHOLD_P2  = 0.50   # Confidence threshold Phase 2 (cao hơn để chính xác)
PROGRESS_EVERY     = 10     # Emit progress mỗi N frame
PERSON_CLASS_ID    = 0
```

---

## 9. API Endpoints (routes.py — ĐẦY ĐỦ, KHÔNG THÊM endpoint khác)

| Method | Path | Body / Params | Response | Mô tả |
|--------|------|---------------|----------|-------|
| GET | `/` | — | HTML | Serve `static/index.html` |
| POST | `/api/config/upload` | `multipart: file=resources.txt` | `{cameras: [{id,url,name}]}` | Upload + parse config |
| GET | `/api/config/cameras` | — | `{cameras: [{id,url,name}]}` | Danh sách camera hiện tại |
| POST | `/api/cameras/probe` | `{camera_ids: [1,2,...]}` | `{results: [{cam_id,width,height,fps,ok}]}` | Probe RTSP |
| POST | `/api/recording/start` | `{resolution: "1280x720", storage_limit_gb: 10}` | `{status}` | Bắt đầu Phase 1 |
| POST | `/api/recording/stop` | — | `{status}` | Dừng Phase 1 |
| GET | `/api/recording/status` | — | `{running, cameras: [{cam_id,status,fps}]}` | Trạng thái Phase 1 |
| GET | `/api/videos` | `?folder=data/raw_videos` | `{videos: [{filename,size_mb,duration_s,mtime}]}` | Danh sách clip |
| DELETE | `/api/videos/<filename>` | — | `{deleted}` | Xóa clip cụ thể |
| POST | `/api/analysis/start` | `{folder: "data/raw_videos"}` | `{status, total_clips}` | Bắt đầu Phase 2 |
| POST | `/api/analysis/stop` | — | `{status}` | Dừng Phase 2 giữa chừng |
| GET | `/api/analysis/status` | — | `{running, current_clip, progress_pct}` | Trạng thái Phase 2 |
| GET | `/api/analysis/results` | `?folder=data/output_labels` | `{results: [{clip_stem,frames,labels_count}]}` | Kết quả Phase 2 |
| GET | `/api/storage/info` | — | `{used_gb, limit_gb, pct, file_count}` | Dung lượng raw_videos |
| POST | `/api/storage/limit` | `{limit_gb: 10.0}` | `{limit_gb}` | Đặt giới hạn storage |

### SocketIO Events (Server → Client emit)

| Event | Payload | Khi nào |
|-------|---------|---------|
| `camera_status` | `{cam_id, status, fps, resolution, timestamp}` | Mỗi 2 giây khi đang record |
| `detection_event` | `{cam_id, timestamp, person_count, confidence_max}` | Mỗi khi YOLO detect |
| `clip_saved` | `{filename, size_mb, duration_s, cam_id}` | Khi clip được đóng và lưu |
| `analysis_progress` | `{clip, frame_id, total_frames, pct, frames_saved}` | Mỗi 10 frame Phase 2 |
| `analysis_complete` | `{clips_done, frames_saved, labels_written, elapsed_s}` | Phase 2 hoàn tất |
| `storage_warning` | `{used_gb, limit_gb, pct}` | Khi dung lượng > 90% |
| `error` | `{source, message, traceback}` | Mọi exception không xử lý được |

---

## 10. Frontend Architecture (static/index.html + js/app.js)

### Cấu trúc SPA

```
index.html
  ├── <head>: CSS link, Socket.IO CDN script
  ├── <nav>: 4 tab links (Config, Monitor, Analysis, Results)
  └── <main>:
       ├── #tab-config     (mặc định active)
       ├── #tab-monitor
       ├── #tab-analysis
       └── #tab-results
```

### Tab 1: Configuration
- Upload `resources.txt` (drag & drop + click)
- Sau upload: hiển thị bảng cameras với `[cam_id | URL | Resolution | FPS | Status]`
- Nút "Probe All Cameras" → gọi `/api/cameras/probe` → fill resolution/fps vào bảng
- Dropdown chọn resolution cho từng cam (chỉ option <= max resolution của cam)
- Input set storage limit (GB), nút Apply
- Hiển thị storage bar: used / limit

### Tab 2: Live Monitor
- Nút "Start Recording" / "Stop Recording"
- Grid cards cho mỗi camera: tên cam, trạng thái (●online/●offline), FPS thực, resolution đang dùng
- Live log: hiển thị 50 dòng cuối từ `detection_event` và `clip_saved` events
- Storage gauge realtime (update qua SocketIO)

### Tab 3: Analysis
- Input đường dẫn folder chứa clip (default: `data/raw_videos`)
- Nút "Browse / Load" → gọi `/api/videos` → hiển thị danh sách clip
- Bảng clip: `[filename | size_mb | duration | checkbox]`
- Nút "Start Analysis" → gọi `/api/analysis/start` với folder đã chọn
- Progress bar tổng thể + tên clip đang xử lý
- Nút "Stop" để dừng giữa chừng

### Tab 4: Results
- Gọi `/api/analysis/results` khi tab được mở
- Accordion per clip: hiển thị tên clip, số frame, số label
- Click → show grid preview thumbnail PNG frame đầu tiên
- Nút download label.txt

### JS Rules (app.js)
- Không dùng class, chỉ module pattern với IIFE hoặc ES6 modules
- Socket.IO: `const socket = io()` — tự kết nối localhost
- Mọi fetch đều có error handling và toast notification
- Toast: green = success, orange = warning, red = error, tự dismiss sau 4 giây
- KHÔNG dùng alert(), confirm() của browser
- Tab switching: toggle `hidden` class, không reload

---

## 11. CSS Rules (style.css)

- **Dark theme bắt buộc**: `background: #0f1117`, text `#e2e8f0`
- **Accent color**: `#3b82f6` (blue-500 equivalent)
- **Font**: `font-family: 'Inter', system-ui, sans-serif` (CDN Google Fonts)
- Không dùng `!important` trừ khi absolutely necessary
- Responsive: mobile-first, breakpoint duy nhất ở 768px
- CSS custom properties cho color scheme:
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
  }
  ```

---

## 12. Python Code Rules

### App factory (app/__init__.py)

```python
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

socketio = SocketIO()  # module-level singleton

def create_app():
    app = Flask(__name__, static_folder="../static", static_url_path="/static")
    CORS(app)
    socketio.init_app(app, async_mode="eventlet", cors_allowed_origins="*")

    from app.api import api_bp
    app.register_blueprint(api_bp)

    return app
```

### main.py

```python
from app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
```

### Singleton pattern (KHÔNG dùng global)

```python
# app/services/phase1_recorder.py
class RecorderManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### Logging convention
```python
import logging
logger = logging.getLogger(__name__)

# Prefix bắt buộc:
logger.info("[Phase1] Starting camera %s", cam_id)
logger.info("[Phase2] Processing clip %s", clip_name)
logger.info("[API] POST /api/recording/start")
logger.warning("[Storage] Usage at %.1f%%", pct)
logger.error("[Phase1] RTSP disconnect cam%s: %s", cam_id, err)
```

### Path handling
```python
from pathlib import Path

# LUÔN dùng pathlib, KHÔNG dùng os.path.join
BASE_DIR     = Path(__file__).parent.parent  # root CPose/
DATA_DIR     = BASE_DIR / "data"
RAW_VIDEOS   = DATA_DIR / "raw_videos"
OUTPUT_DIR   = DATA_DIR / "output_labels"
CONFIG_DIR   = DATA_DIR / "config"
MODELS_DIR   = BASE_DIR / "models"
RESOURCES    = CONFIG_DIR / "resources.txt"
```

### Thread safety
```python
import threading

class RecorderManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: dict[str, CameraThread] = {}
        self._storage_limit_gb: float = 10.0

    def start_recording(self, ...):
        with self._lock:
            # thao tác với self._cameras
```

### Exception handling
```python
from app import socketio

try:
    # risky operation
except Exception as e:
    logger.error("[Phase1] Unexpected error: %s", e, exc_info=True)
    socketio.emit("error", {
        "source": "phase1",
        "message": str(e),
        "traceback": traceback.format_exc()
    })
```

---

## 13. run.bat (Windows Launcher)

```bat
@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"

:: Tạo venv nếu chưa có
if not exist "%VENV%\Scripts\activate.bat" (
    echo [CPose] Tao virtual environment...
    python -m venv "%VENV%"
)

call "%VENV%\Scripts\activate.bat"

:: Cài dependency nếu thiếu
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [CPose] Cai dat dependency...
    pip install -r "%ROOT%requirements.txt" --quiet
)

:: Tạo thư mục cần thiết
mkdir "%ROOT%data\raw_videos" 2>nul
mkdir "%ROOT%data\output_labels" 2>nul
mkdir "%ROOT%data\config" 2>nul
mkdir "%ROOT%models" 2>nul

echo [CPose] Khoi dong server tai http://localhost:5000
echo [CPose] Nhan Ctrl+C de dung
echo.

:: Mở browser sau 2 giây
start "" timeout /t 2 /nobreak >nul && start http://localhost:5000

python "%ROOT%main.py"
pause
```

---

## 14. .gitignore

```
# Python
__pycache__/
*.py[cod]
*.pyo
venv/
.env

# Data (lớn, không commit)
data/raw_videos/
data/output_labels/
models/

# Giữ lại cấu trúc folder
!data/raw_videos/.gitkeep
!data/output_labels/.gitkeep
!data/config/.gitkeep
!models/.gitkeep

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## 15. Requirements (requirements.txt)

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
```

---

## 16. Những điều TUYỆT ĐỐI không làm

| Không làm | Thay bằng |
|-----------|-----------|
| Thêm SQLite / database | Dùng filesystem + JSON file nếu cần persist |
| Thêm FastAPI, Django | Chỉ Flask |
| Dùng React/Vue | Vanilla JS + Socket.IO CDN |
| Tạo file route mới ngoài routes.py | Thêm endpoint vào routes.py |
| Hard-code path dạng string | pathlib.Path từ BASE_DIR |
| Dùng global variable | Singleton pattern (get_instance) |
| Commit .pt, .mp4, .png vào git | .gitignore |
| import os.path.join | from pathlib import Path |
| Tạo thread không phải daemon | thread.daemon = True |
| Blocking call trong route handler | Chạy trong background thread, emit SocketIO |

---

## 17. Workflow Người Dùng (End-to-End)

```
1. Chạy run.bat
   └─ Browser tự mở http://localhost:5000

2. Tab "Configuration"
   ├─ Upload resources.txt (danh sách RTSP URL)
   ├─ Nhấn "Probe Cameras" → xem resolution/FPS từng cam
   ├─ Chọn resolution ghi hình cho từng cam
   └─ Set storage limit (mặc định 10 GB)

3. Tab "Live Monitor"
   ├─ Nhấn "Start Recording"
   │   └─ Backend: RecorderManager mở N RTSP stream
   │              YOLOv8n chạy ngầm mỗi 5 frame
   │              Phát hiện người → ghi clip MP4
   │              Clip tên: YYYYMMDD_HHMMSS_camXX.mp4
   ├─ Xem live log: detection events + clip saved
   ├─ Xem storage gauge tăng dần
   └─ Nhấn "Stop Recording" khi xong

4. Tab "Analysis"
   ├─ Folder mặc định: data/raw_videos
   ├─ Load → hiện danh sách clip .mp4
   ├─ Chọn clip hoặc "Select All"
   ├─ Nhấn "Start Analysis"
   │   └─ Backend: Analyzer chạy YOLOv8l trên từng clip
   │              Lưu frame PNG + label.txt
   └─ Theo dõi progress bar

5. Tab "Results"
   ├─ Xem accordion từng clip
   ├─ Preview frame thumbnail
   └─ Download label.txt
```

---

*Phiên bản: 2.0 — Cập nhật cho cấu trúc refactor CPose.*
*Tài liệu này là nguồn sự thật duy nhất (single source of truth) cho mọi AI assistant.*
