# CPose — Multi-Camera Person Detection & Auto-Labeling System

> **Đồ án Tốt nghiệp Kỹ sư** — Ngành Điện tử Viễn thông & Kỹ thuật Máy tính Chất lượng cao  
> Trường Đại học Bách Khoa — Đại học Đà Nẵng · 2024–2025

---

## Tổng quan

**CPose** là hệ thống thu thập dữ liệu giám sát người tự động từ nhiều camera IP (RTSP) theo kiến trúc hai giai đoạn — phục vụ nghiên cứu pose estimation và ADL (Activities of Daily Living).

Pipeline hoạt động hoàn toàn tự động:

- **Phase 1** — Real-time: đọc N luồng RTSP song song, dùng YOLOv8n (nhẹ) phát hiện người theo sự kiện, ghi clip MP4 tự động với pre/post buffer
- **Phase 2** — Offline: chạy YOLOv8l (chính xác cao) trên clip đã ghi, xuất frame PNG và file bounding box label phục vụ training dataset

| Giai đoạn | Mô hình | Mục tiêu | Chế độ |
|-----------|---------|----------|--------|
| Phase 1 | YOLOv8n | Phát hiện người, kích hoạt ghi hình | Realtime — chạy ngầm liên tục |
| Phase 2 | YOLOv11 | Trích xuất frame + bounding box | Offline — chạy theo yêu cầu |

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
- Chạy YOLOv11 trên toàn bộ clip hoặc thư mục chọn
- Chỉ lưu frame khi có người (confidence ≥ 0.50)
- Output chuẩn: frame PNG gốc + file label tọa độ tuyệt đối
- Tracking tiến độ realtime qua SocketIO

### Dashboard Web (SPA)
- Giao diện dark theme duy nhất, không cần cài thêm phần mềm
- Probe tự động resolution và FPS của từng camera
- Cho phép chọn độ phân giải ghi hình (≤ max của cam)
- Live log: detection events, clip saved, storage usage
- Hoàn toàn Vanilla JS — không phụ thuộc framework

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| OS | Windows 10 / Linux | Windows 11 |
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA CUDA (tăng tốc YOLO) |
| Mạng | Truy cập RTSP camera | LAN tốc độ cao |

---

## Cấu trúc dự án

```
CPose/
├── app/
│   ├── __init__.py              # Flask app factory + path constants
│   ├── api/
│   │   └── routes.py            # Tất cả REST endpoint + SocketIO events
│   ├── services/
│   │   ├── phase1_recorder.py   # RecorderManager: RTSP + YOLOv8n + clip writer
│   │   └── phase2_analyzer.py   # Analyzer: YOLOv8l + frame/label extractor
│   └── utils/
│       ├── file_handler.py      # StorageManager: storage limit, file listing
│       └── stream_probe.py      # probe_rtsp(): resolution + FPS từ RTSP
├── data/
│   ├── config/
│   │   └── resources.txt        # Danh sách RTSP URL (1 URL/dòng)
│   ├── raw_videos/              # [git-ignored] Clip MP4 Phase 1
│   └── output_labels/           # [git-ignored] PNG + TXT Phase 2
├── models/                      # [git-ignored] YOLOv8 .pt weights
├── static/
│   ├── index.html               # Dashboard duy nhất (SPA)
│   ├── css/style.css
│   └── js/app.js
├── main.py                      # Entry point: python main.py
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

Script tự động thực hiện:
1. Tạo virtual environment `.venv` nếu chưa có
2. Cài đặt tất cả dependency từ `requirements.txt`
3. Tạo các thư mục dữ liệu cần thiết
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

Tạo file `resources.txt` theo định dạng:

```
# Mỗi dòng là 1 RTSP URL. Dòng bắt đầu bằng # là comment.
rtsp://admin:password@192.168.1.100:554/stream1
rtsp://admin:password@192.168.1.101:554/stream1
rtsp://user:pass@10.0.0.5:554/live
```

Thao tác trên giao diện:
1. Tải `resources.txt` lên (kéo thả hoặc click)
2. Nhấn **Probe Cameras** → hệ thống tự lấy resolution và FPS tối đa của từng camera
3. Chọn độ phân giải ghi hình (chỉ hiện option ≤ max resolution của cam)
4. Thiết lập giới hạn storage (mặc định 10 GB)

### Bước 2 — Ghi hình (Tab: Live Monitor)

1. Nhấn **Start Recording**
2. Hệ thống kết nối RTSP, chạy YOLOv8n ngầm trên từng camera
3. Phát hiện người → tự động ghi clip MP4:
   ```
   YYYYMMDD_HHMMSS_camXX.mp4
   Ví dụ: 20240315_143022_cam01.mp4
   ```
4. Theo dõi live log: trạng thái kết nối, sự kiện detect, clip được lưu
5. Nhấn **Stop Recording** khi hoàn tất

### Bước 3 — Phân tích (Tab: Analysis)

1. Nhập đường dẫn thư mục clip (mặc định: `data/raw_videos`)
2. Nhấn **Load** → hiển thị danh sách clip với kích thước và thời lượng
3. Chọn clip cần phân tích (hoặc chọn tất cả)
4. Nhấn **Start Analysis** — YOLOv8l chạy offline trên từng clip
5. Theo dõi progress bar và tên clip đang xử lý

Output lưu tại `data/output_labels/<clip_stem>/`:
```
data/output_labels/
└── 20240315_143022_cam01/
    ├── 20240315_143022_cam01_frame_0000.png
    ├── 20240315_143022_cam01_frame_0003.png
    ├── 20240315_143022_cam01_frame_0007.png
    └── 20240315_143022_cam01_labels.txt
```

### Bước 4 — Xem kết quả (Tab: Results)

- Accordion theo từng clip: số frame PNG, số bounding box
- Preview thumbnail frame đầu tiên
- Tải về `label.txt` của từng clip

---

## Định dạng file label

```
# frame_id  x_min  y_min  x_max  y_max
0           112    54     340    490
0           512    102    680    495
1           115    58     338    492
3           220    80     410    500
```

| Trường | Mô tả |
|--------|-------|
| `frame_id` | Index frame trong clip, bắt đầu từ 0 |
| `x_min, y_min` | Góc trên bên trái bounding box (pixel tuyệt đối) |
| `x_max, y_max` | Góc dưới bên phải bounding box (pixel tuyệt đối) |

> Class luôn là `person` (COCO class 0) — không ghi vào file.  
> Separator: dấu cách (space). Tọa độ **không normalize**.

---

## API Reference

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Serve dashboard `index.html` |
| POST | `/api/config/upload` | Upload `resources.txt` |
| GET | `/api/config/cameras` | Danh sách camera đã load |
| POST | `/api/cameras/probe` | Probe RTSP → resolution + FPS |
| POST | `/api/recording/start` | Bắt đầu Phase 1 |
| POST | `/api/recording/stop` | Dừng Phase 1 |
| GET | `/api/recording/status` | Trạng thái recording |
| GET | `/api/videos` | Danh sách clip MP4 |
| DELETE | `/api/videos/<filename>` | Xóa clip |
| POST | `/api/analysis/start` | Bắt đầu Phase 2 |
| POST | `/api/analysis/stop` | Dừng Phase 2 |
| GET | `/api/analysis/status` | Tiến độ Phase 2 |
| GET | `/api/analysis/results` | Danh sách kết quả |
| GET | `/api/storage/info` | Dung lượng đang dùng |
| POST | `/api/storage/limit` | Đặt giới hạn storage |

### SocketIO Events (Server → Client)

| Event | Mô tả |
|-------|-------|
| `camera_status` | Trạng thái kết nối, FPS, resolution theo thời gian thực |
| `detection_event` | Phát hiện người: cam, timestamp, số người |
| `clip_saved` | Clip vừa được lưu: tên file, kích thước, thời lượng |
| `analysis_progress` | Tiến độ Phase 2: frame hiện tại / tổng frame |
| `analysis_complete` | Phase 2 hoàn tất: số clip, frame, label |
| `storage_warning` | Cảnh báo dung lượng > 90% |
| `error` | Lỗi từ backend: nguồn, message, traceback |

---

## Quản lý bộ nhớ

- Giới hạn mặc định: **10 GB** (có thể thay đổi qua giao diện)
- Khi `raw_videos/` vượt giới hạn → **tự động xóa clip cũ nhất** cho đến khi còn < 80% giới hạn
- Cảnh báo SocketIO khi dung lượng > 90%
- Phase 2 output (`output_labels/`) **không bị ảnh hưởng** bởi cơ chế này

---

## YOLO Models

| Model | File | Phase | Confidence | Tải tự động |
|-------|------|-------|------------|------------|
| YOLOv8n | `models/yolov8n.pt` | 1 | 0.35 | ✅ Lần đầu khởi động Phase 1 |
| YOLOv1l | `models/yolov11.pt` | 2 | 0.50 | ✅ Lần đầu khởi động Phase 2 |

> Weights được cache vào `models/` sau lần tải đầu tiên. Yêu cầu Internet lần đầu.

---

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python 3.10+, Flask 3, Flask-SocketIO 5 (eventlet) |
| AI Engine | Ultralytics YOLOv8 (n + l) |
| Video | OpenCV (RTSP capture, VideoWriter, frame extraction) |
| Frontend | HTML5, CSS3, Vanilla JavaScript, Socket.IO CDN |
| Concurrency | `threading.Thread` (daemon) + `threading.Lock` |
| Storage | Filesystem (`pathlib.Path`) — không dùng database |

---

## Ghi chú kỹ thuật

- Mỗi camera chạy trên **daemon thread độc lập** — thread-safe với `threading.Lock`
- Pre-buffer 3 giây trước detection (deque) + post-buffer 3 giây sau khi mất detection
- YOLOv8n inference Phase 1: mỗi 5 frame (giảm tải CPU, giữ real-time ổn định)
- Clip ngắn hơn 2 giây sẽ bị xóa tự động (lọc false detection thoáng qua)
- Tự động reconnect RTSP sau 5 giây khi mất kết nối

---

*Phát triển phục vụ nghiên cứu đồ án tốt nghiệp kỹ sư, chuyên ngành Điện tử Viễn thông & Kỹ thuật Máy tính Chất lượng cao — Trường Đại học Bách Khoa, Đại học Đà Nẵng.*
