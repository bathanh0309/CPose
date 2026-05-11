# CLAUDE.md — CPose Project AI Coding Guidelines

> **Mục đích**: File này ràng buộc mọi AI assistant (Claude, Copilot, Cursor, GPT-4...)
> khi làm việc trong repo này. Đọc toàn bộ file trước khi viết bất kỳ dòng code nào.

---

## 1. Tổng quan dự án

**Tên**: CPose — Hệ thống phân tích hành vi toàn cảnh theo thời gian thực  
**Stack chính**: Python 3.11 · PyTorch · ONNX · InsightFace · Ultralytics YOLO · FAISS · Flask · FastAPI  
**Nền tảng**: Windows 10/11 (dev) · Linux x64 (server) · Jetson Orin (edge)  
**Kiến trúc**: Module A (Face Gate) + Module B (Multi-Camera ReID + ADL + Fall Detection)

```
CPose/
├── src/
│   ├── core/          # AI inference: detector, tracker, face_recognizer, body_embedder,
│   │                  #               pose_estimator, adl_classifier, fall_detector
│   ├── pipeline/      # Camera pipelines: cam1, cam2, slave, live
│   ├── reid/          # Identity management: gallery, matcher, global_id_manager
│   ├── backend/       # Storage, schemas, event record, logging
│   ├── utils/         # video_io, visualization, timer, topology, validate_io
│   ├── eval/          # Metrics, benchmark, paper tables — KHÔNG chạy trong runtime
│   ├── config.py      # Single config entry point
│   ├── orchestrator.py
│   └── main.py
├── configs/
│   ├── base/          # models.yaml, thresholds.yaml, pipeline.yaml, adl.yaml, logging.yaml
│   ├── camera/        # topology.yaml, multicam_manifest.json
│   └── profiles/      # dev.yaml, edge.yaml, benchmark.yaml
├── models/            # ONNX / PT weights (không commit file lớn)
├── data/face/         # Face gallery: {person}/embeddings.npy + meta.json
├── dataset/           # Evaluation datasets (không commit)
├── tests/             # Pytest tests
└── app/               # Flask/FastAPI app factory
```

---

## 2. Nguyên tắc bắt buộc (MUST)

### 2.1 Cấu trúc file

- **MỌI file AI inference** phải nằm trong `src/core/` — một file = một class = một nhiệm vụ
- **Pipeline** chỉ gọi các class từ `src/core/`, không chứa raw inference logic
- **Config** chỉ đọc từ `src/config.py` — không hardcode path, threshold trong pipeline code
- **Test** phải nằm trong `tests/` — không để `test.py` trong `src/`
- **Eval scripts** (`benchmark_all.py`, `make_paper_tables.py`...) chỉ trong `src/eval/`

### 2.2 Import

```python
# ĐÚNG — import tuyệt đối từ src
from src.core.face_recognizer import FaceRecognizer
from src.config import load_config

# SAI — relative import mơ hồ
from ..core.face import *
import face_recognizer
```

- Không dùng `import *` — bao giờ cũng explicit import
- Không import module nặng (torch, cv2) ở top-level `__init__.py` của package

### 2.3 Config

```python
# ĐÚNG
cfg = load_config(profile="dev")
threshold = cfg["face"]["cosine_threshold"]

# SAI — hardcode
threshold = 0.45
model_path = "models/buffalo_s/w600k_mbf.onnx"
```

- Mọi threshold, path, hyperparameter phải đọc từ `configs/base/*.yaml`
- Credentials (RTSP URL, passwords) chỉ trong `.env` — KHÔNG commit lên git

### 2.4 Model loading

```python
# ĐÚNG — lazy load, load một lần duy nhất khi khởi tạo class
class FaceRecognizer:
    def __init__(self, cfg: dict):
        self.model = self._load_model(cfg["models"]["arcface"])
    
    def _load_model(self, path: str):
        ...

# SAI — load model trong mỗi lần inference
def recognize(img):
    model = onnxruntime.InferenceSession("models/...")  # load lại mỗi call!
    ...
```

### 2.5 Threading & real-time pipeline

- Mỗi camera chạy trong **thread riêng** — dùng `threading.Thread`, không dùng `multiprocessing` trừ khi có lý do cụ thể
- Frame queue giữa decode thread và inference thread: dùng `queue.Queue(maxsize=2)` — không để queue không giới hạn
- Không block inference thread bằng I/O (file write, DB insert) — log/save phải async

```python
# ĐÚNG — non-blocking queue put
try:
    self.frame_queue.put_nowait(frame)
except queue.Full:
    pass  # drop frame, không block

# SAI — block pipeline
self.frame_queue.put(frame)  # chờ vô hạn nếu queue đầy
```

---

## 3. Quy ước code

### 3.1 Naming

| Loại | Convention | Ví dụ |
|------|-----------|-------|
| Class | PascalCase | `FaceRecognizer`, `GlobalIDManager` |
| Function / method | snake_case | `extract_embedding()`, `match_identity()` |
| Constant | UPPER_SNAKE | `COSINE_THRESHOLD`, `MAX_GALLERY_SIZE` |
| Config key | snake_case | `cosine_threshold`, `model_path` |
| File | snake_case | `face_recognizer.py`, `body_embedder.py` |

### 3.2 Type hints — BẮT BUỘC

```python
# ĐÚNG
def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
    ...

def match(self, query: np.ndarray, gallery: dict[str, np.ndarray]) -> tuple[str, float]:
    ...

# SAI — không có type hint
def extract_embedding(self, face_img):
    ...
```

### 3.3 Docstring — BẮT BUỘC cho public method

```python
def recognize(self, frame: np.ndarray) -> list[dict]:
    """
    Nhận diện khuôn mặt trong frame.

    Args:
        frame: BGR image array shape (H, W, 3)

    Returns:
        List of dicts: [{"person_id": str, "score": float, "bbox": list[int]}]
    
    Raises:
        ValueError: Nếu frame rỗng hoặc sai shape
    """
```

### 3.4 Logging — dùng loguru, KHÔNG dùng print

```python
# ĐÚNG
from loguru import logger
logger.info(f"[FaceRecognizer] Loaded model: {model_path}")
logger.warning(f"[Matcher] Low confidence: {score:.3f} < {threshold}")
logger.error(f"[Pipeline] Frame decode failed: {e}")

# SAI
print(f"Loaded model {model_path}")
```

Format log prefix: `[ClassName]` — ví dụ `[FaceRecognizer]`, `[GlobalIDManager]`

---

## 4. AI Models — quy định sử dụng

### Module A (Face Gate)

| Nhiệm vụ | Model | File |
|---------|-------|------|
| Face detect | RetinaFace | `models/buffalo_s/det_500m.onnx` |
| Face align | 2D106 landmark | `models/buffalo_s/2d106det.onnx` |
| Face embedding | ArcFace | `models/buffalo_s/w600k_mbf.onnx` |
| Anti-spoof | MiniFASNet | `models/face_antispoof/best_model_quantized.onnx` |

### Module B (Indoor ReID + ADL)

| Nhiệm vụ | Model | File |
|---------|-------|------|
| Person detect | YOLOv8n | `models/human_detect/yolov8n.pt` |
| Pose keypoints | YOLO Pose | `models/pose_estimation/yolov8n-pose.pt` |
| Body embedding | OSNet | `models/global_reid/osnet_x0_25_msmt17.onnx` |
| ADL classify | Rule-based + custom | `src/core/adl_classifier.py` |
| Fall detect | Multi-frame logic | `src/core/fall_detector.py` |

**Quy tắc model:**
- Tất cả inference chạy qua ONNX Runtime khi deploy — không dùng PyTorch `.pt` trực tiếp trên edge
- Export ONNX trước khi deploy: `python src/utils/export_onnx.py --model face_recognizer`
- Không download model tự động trong runtime code — model phải có sẵn, nếu thiếu thì raise lỗi rõ ràng

---

## 5. Embedding & Identity Management

### Face gallery format

```
data/face/{person_name}/
├── embeddings.npy    # shape: (N, 512), dtype: float32
└── meta.json         # {"name": str, "n_samples": int, "enrolled_at": str}
```

### Cosine similarity

```python
# ĐÚNG — normalize trước rồi dot product (nhanh hơn)
def cosine_sim(q: np.ndarray, g: np.ndarray) -> float:
    q_norm = q / (np.linalg.norm(q) + 1e-8)
    g_norm = g / (np.linalg.norm(g) + 1e-8)
    return float(np.dot(q_norm, g_norm))

# SAI — tính thủ công từng lần
score = np.dot(q, g) / (np.linalg.norm(q) * np.linalg.norm(g))
```

### Dual-threshold matching

```python
# Không tự đặt threshold — đọc từ config
ACCEPT_THRESHOLD = cfg["reid"]["accept_threshold"]   # ví dụ: 0.55
REJECT_THRESHOLD = cfg["reid"]["reject_threshold"]   # ví dụ: 0.35

if score >= ACCEPT_THRESHOLD:
    return person_id, "CONFIRMED"
elif score >= REJECT_THRESHOLD:
    return person_id, "CANDIDATE"
else:
    return "UNK", "UNKNOWN"
```

---

## 6. Anti-patterns — KHÔNG làm

```python
# ❌ Hardcode model path
model = onnxruntime.InferenceSession("D:/Capstone_Project/models/...")

# ❌ Load model nhiều lần
for frame in stream:
    model = load_model(...)  # thảm họa performance

# ❌ Global mutable state không kiểm soát
GLOBAL_ID_COUNTER = 0  # race condition khi multi-thread

# ❌ Bắt exception trống
try:
    result = model.run(...)
except:
    pass  # nuốt lỗi, không biết gì xảy ra

# ❌ Magic number không giải thích
if score > 0.45 and ratio < 2.1:
    ...

# ❌ Blocking sleep trong pipeline thread
time.sleep(0.1)  # dùng threading.Event.wait() thay thế

# ❌ Print thay vì logger
print("Processing frame", frame_id)

# ❌ Commit file nhạy cảm
# configs/_private.yaml chứa RTSP credentials — phải ở .gitignore

# ❌ Để test.py trong src/
# src/test.py → chuyển vào tests/

# ❌ Một function làm nhiều việc
def detect_track_reid_adl_log(frame):  # quá nhiều trách nhiệm
    ...
```

---

## 7. Performance constraints

| Metric | Target | Hard limit |
|--------|--------|------------|
| End-to-end latency / frame | ≤ 66ms | 100ms |
| FPS per camera | ≥ 15 FPS | 10 FPS |
| RAM usage (toàn hệ thống) | ≤ 4GB | 6GB |
| FAISS query time | ≤ 5ms | 20ms |
| ID Switch rate | ≤ 5% | 10% |

**Nếu thêm feature mới làm FPS giảm xuống dưới 10, phải tối ưu trước khi merge.**

---

## 8. Git workflow

### Commit message format

```
[module] action: mô tả ngắn

# Ví dụ:
[core] feat: add OSNet body embedder with ONNX runtime
[pipeline] fix: resolve frame queue blocking on cam2
[reid] refactor: merge gallery.py cosine matching with FAISS index
[eval] add: Market-1501 Rank-1 benchmark script
[config] fix: remove _private.yaml from git tracking
```

### Branch naming

```
feature/module-a-antispoof
fix/cam2-pipeline-queue-block
refactor/src-restructure-core-modules
eval/market1501-benchmark
```

### Không commit lên main trực tiếp

- Mọi thay đổi qua Pull Request
- Cần pass `pytest tests/` trước khi merge
- File model (`.onnx`, `.pt`) không commit — dùng `.gitignore` hoặc Git LFS

---

## 9. Testing

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chạy test theo module
pytest tests/test_face_recognizer.py -v
pytest tests/test_matcher.py -v

# Chạy với coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Mỗi class trong `src/core/` phải có test tương ứng trong `tests/`:**

```
src/core/face_recognizer.py  →  tests/test_face_recognizer.py
src/core/body_embedder.py    →  tests/test_body_embedder.py
src/reid/matcher.py          →  tests/test_matcher.py
```

Test tối thiểu cho mỗi class:
- `test_init()` — khởi tạo với config hợp lệ
- `test_inference_shape()` — output đúng shape/type
- `test_edge_case()` — input rỗng, ảnh đen, không có face

---

## 10. Checklist trước khi AI viết code

- [ ] File mới có đúng vị trí trong cấu trúc thư mục không?
- [ ] Config được đọc từ `load_config()` thay vì hardcode?
- [ ] Model chỉ load một lần trong `__init__`?
- [ ] Có type hints đầy đủ?
- [ ] Có docstring cho public methods?
- [ ] Dùng `logger` thay vì `print`?
- [ ] Không có magic number — dùng constant hoặc config?
- [ ] Exception được xử lý và log cụ thể?
- [ ] Thread-safe nếu code chạy trong multi-thread context?
- [ ] Có test tương ứng trong `tests/`?

---

*File này được maintain bởi Nguyễn Bá Thành & Nguyễn Văn Huy — Capstone Project 2025-2026*  
*Cập nhật lần cuối: tháng 5/2026*
