# CPose: "Cross-Camera Pose & Action Recognition"

Hệ thống AI realtime trên CPU kết hợp

```
RTMPose-t (Pose Estimation)
OSNet-0.25 (ReID)
ByteTrack (Multi-Object Tracking)
EfficientGCN (Action Recognition)
```

---

## [INSTRUCTIONS — ĐỌC TRƯỚC KHI CHỈNH CODE]

Dự án đã chốt cấu trúc nghiêm ngặt. **BẮT BUỘC** tuân thủ các ràng buộc sau:

### 1. Giao diện (UI)

- **TUYỆT ĐỐI KHÔNG** refactor HTML/CSS/JS. Giữ nguyên Light Theme, layout ngang nút điều khiển, ô nhập dữ liệu `<input type="text">` (không dùng `<select>`).
- Khung Log của camera nào thì nằm ngay dưới camera đó. Cấm gom log chung.

### 2. Logging

- **KHÔNG** log rác mỗi frame (cấm: `"Đang xử lý frame 1, 2, 3..."`).
- Log AI **CHỈ** được sinh ra khi có sự kiện thay đổi trạng thái: phát hiện `global_id` mới, hành động nghi ngờ từ EfficientGCN, hoặc mất track.
- Format chuẩn: `[Time] [Level] Message`.

### 3. Chuẩn đầu vào/đầu ra các Module

| Module | Input | Output |
|---|---|---|
| **RTMPose-t** | Frame | `Bboxes` + `17 Keypoints` |
| **ByteTrack** | `Bboxes` | `local_track_id` (không mất/nhảy ID khi bị che ngắn hạn) |
| **OSNet-0.25** | Cropped person image | Feature embedding → Cosine Similarity → `global_id` xuyên camera (chỉ tính mỗi `N` frame hoặc khi có ID mới) |
| **EfficientGCN** | Buffer chuỗi Keypoints (vd: 30 frames) | Action label (không chạy blocking luồng đọc video chính) |

---

## Cấu trúc thư mục

```
CPose/
│
├── main.py                        # Entry point: FastAPI app + WebSocket handler
├── requirements.txt               # Python dependencies
├── packages.txt                   # System-level packages (apt)
├── run-push.bat                   # Script tiện ích cho Windows (git push nhanh)
│
├── configs/                       # Toàn bộ config YAML của hệ thống
│   └── system/
│       ├── pipeline.yaml          # Config chính: path model, threshold, interval ReID, seq_len GCN
│       └── benchmark.yaml        # Target metric từng module (FPS, mAP, MOTA, ...)
│
├── data/                          # KHÔNG THAY ĐỔI CẤU TRÚC THƯ MỤC NÀY
│   ├── input/                     # Video .mp4 dùng để test và inference
│   │   └── *.mp4
│   └── embeddings/                # Embedding face/body đã trích xuất trước (.npy)
│       └── *.npy
│
├── models/                        # Trọng số (weights) của tất cả module AI
│   ├── rtmpose/
│   │   └── rtmpose-t.onnx                             # RTMPose-t (ONNX)
│   ├── reid/
│   │   └── osnet0.25.pth                              # OSNet-0.25 (PyTorch)
│   ├── gcn/
│   │   └── efficientgcn.pth                           # EfficientGCN (PyTorch)
│   └── tracking/
│       └── bytetrack.yaml                             # ByteTrack config (Ultralytics)
│
├── src/                           # Source code core logic
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pose_estimator.py      # RTMPose-t wrapper (ONNX Runtime)
│   │   ├── tracker.py             # ByteTrack wrapper
│   │   ├── reid_extractor.py      # OSNet-0.25: extract embedding + cosine matching
│   │   └── action_recognizer.py   # EfficientGCN: buffer keypoints + infer action
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── draw.py                # Vẽ keypoints, bbox, ID, action label lên frame
│   │   ├── gallery.py             # Quản lý gallery embedding cho ReID
│   │   └── logger.py              # Logger chuẩn format [Time] [Level] Message
│   │
│   └── __init__.py
│
├── apps/                          # Các app/mode chạy riêng lẻ (CLI test, batch, ...)
│   └── run_single.py              # Chạy pipeline đơn camera qua CLI (không cần UI)
│
├── scripts/                       # Script tiện ích: chuẩn bị data, export model, benchmark
│   ├── export_onnx.py             # Convert model sang ONNX
│   ├── build_gallery.py           # Trích xuất embedding và lưu vào data/embeddings/
│   └── eval_pipeline.py           # Đánh giá FPS, MOTA, ReID accuracy
│
├── static/                        # Frontend Web UI (chỉ HTML/CSS/JS tĩnh)
│   ├── index.html                 # Giao diện chính: 2 camera stream + control panel
│   ├── style.css
│   └── app.js
│
└── docs/                          # Tài liệu bổ sung
    ├── architecture.md            # Sơ đồ kiến trúc pipeline
    └── benchmark_results.md       # Kết quả benchmark thực đo trên CPU
```

---

## Pipeline AI (Realtime trên CPU)

```
Video Frame
    │
    ▼
[RTMPose-t]  ── ONNX Runtime ──►  Bboxes + 17 Keypoints
    │
    ▼
[ByteTrack]  ──────────────────►  local_track_id (per camera)
    │
    ├──►  [OSNet-0.25]  ──────►  Feature Embedding
    │         │                       │
    │         └── Cosine Match ───►  global_id (cross-camera ReID)
    │
    └──►  [EfficientGCN]  ────►  Keypoint Buffer (30 frames)
                │                       │
                └── Async Thread ──►  Action Label (ADL)
```

**Lý do chọn stack này để chạy realtime trên CPU:**

- **RTMPose-t**: lightweight ONNX model (~5MB), latency ~25-40ms/frame trên CPU
- **OSNet-0.25**: backbone siêu nhỏ (0.25×), inference ~8-15ms/crop, phù hợp ReID interval
- **ByteTrack**: không cần model học (pure Kalman + IoU), overhead ~1-3ms/frame
- **EfficientGCN**: chạy async trên thread riêng, không blocking luồng đọc video chính

---

## Tải Model Weights

Tải về và đặt đúng vị trí theo bảng sau:

| Module | File | Download |
|---|---|---|
| **Pose — RTMPose-t** | `models/rtmpose/rtmpose-t_8xb256-420e_coco-256x192.onnx` | [MMPose Model Zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) |
| **ReID — OSNet-0.25** | `models/reid/osnet_x0_25_market.pth` | [torchreid Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) |
| **Action — EfficientGCN** | `models/gcn/efficientgcn_b0_ntu60.pth` | [EfficientGCN repo](https://github.com/yfsong0709/EfficientGCNv1) |
| **Tracking — ByteTrack** | `models/tracking/bytetrack.yaml` | Built-in Ultralytics config |

Tất cả hyperparameter (threshold, ReID interval, GCN seq_len, ...) được quản lý tập trung tại `configs/system/pipeline.yaml`.

---

## Benchmark Target (CPU)

| Module | Metric | Target |
|---|---|---|
| **RTMPose-t** | FPS (CPU) | 15–25 FPS |
| **RTMPose-t** | Latency | 25–60 ms/frame |
| **RTMPose-t** | OKS AP | ≥ 0.55 |
| **ByteTrack** | MOTA | ≥ 0.60 |
| **ByteTrack** | ID Switches | < 5 lần / 1.000 frame |
| **OSNet-0.25** | CMC Rank-1 | ≥ 0.70 |
| **OSNet-0.25** | Cosine threshold | Sweep 0.30–0.90 (default 0.55) |
| **EfficientGCN** | Latency/clip | 50–150 ms/clip (async) |
| **EfficientGCN** | Top-1 Accuracy | ≥ 0.65 |
| **Full Pipeline** | FPS (CPU) | 10–18 FPS |
| **Full Pipeline** | End-to-end latency | < 200 ms/frame |
| **Full Pipeline** | ReID match rate | ≥ 70% track có `global_id` ổn định |

Chi tiết target từng metric xem tại `configs/system/benchmark.yaml`.

---

## Cài đặt & Chạy

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động Backend (FastAPI + WebSocket)

```bash
uvicorn main:app --reload --port 8000
```

### 3. Mở Web UI

Mở file `static/index.html` bằng trình duyệt, sau đó:

1. Nhập đường dẫn video `.mp4` vào ô Camera 1 hoặc Camera 2 (ví dụ: `data/input/cam2_2026-01-29_16-26-40.mp4`).
2. Chọn các module cần bật: **Pose**, **Track**, **ReID**, **Action**.
3. Nhấn **▶ ON** để bắt đầu.

### 4. Chạy CLI đơn camera (không cần UI)

```bash
python apps/run_pipeline.py --source data/test_video.mp4
```

### 5. Build gallery embedding

```bash
python data/build-body-gallery.py --input data/body --output data/embeddings/
```

---

## Thư mục `data/` — Quy ước lưu trữ

```
data/
├── input/          # Video .mp4 gốc (camera feed hoặc file test)
└── embeddings/     # File .npy chứa face/body embedding đã trích xuất trước
                    # Tên file gợi ý: {person_id}_{camera_id}.npy
```

> **Lưu ý:** Thư mục `data/` không được commit video lên Git (đã có trong `.gitignore`). Chỉ commit file `.npy` nhỏ nếu cần seed gallery ban đầu.

---

## Tài liệu liên quan

- [RTMPose paper](https://arxiv.org/abs/2303.07399)
- [ByteTrack paper](https://arxiv.org/abs/2110.06864)
- [OSNet paper](https://arxiv.org/abs/1905.00953)
- [EfficientGCN paper](https://arxiv.org/abs/2106.15125)
