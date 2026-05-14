# CLAUDE.md — CPose Project Intelligence (v2)

> **Dành cho AI coding agents: Claude, Codex, Cursor, Copilot.**
> Đọc toàn bộ file này trước khi sửa bất kỳ file nào trong repo.
> Nếu cần phá một rule, ghi rõ lý do trong commit message hoặc response.

---

## 0. Tóm Tắt Nhanh Cho Agent (Token-Efficient)

```
Pipeline:  Frame → YoloPoseTracker → ByteTrackWrapper → FastReIDExtractor
                 → ReIDGallery → GlobalIDManager → PoseSequenceBuffer
                 → PoseC3DRunner → EventBus → JSONL

Identity:  track_id  = local int, unique trong 1 camera session (từ Ultralytics)
           global_id = string, cross-camera ("APhu", "gid_00001", ...)
           cache key = (camera_id: str, local_track_id: int)

Config:    MỌI path/threshold/hyperparam → configs/system/pipeline.yaml
           Validate bằng src/utils/config.py::validate_cfg() trước khi build model
           Resolve path bằng resolve_cfg_paths(cfg, ROOT)

Logging:   get_logger(__name__) — KHÔNG dùng print() trong pipeline

Error:     Frame-level: log + continue. Init-level: raise sớm. Resource: finally cleanup.
```

---

## 1. Kiến Trúc Pipeline (Chi Tiết)

```
Video / Camera frame (BGR numpy)
    │
    ▼
YoloPoseTracker.infer(frame, persist=True)          src/detectors/yolo_pose.py
    └─ Ultralytics YOLO.track() — ByteTrack nội bộ qua tracker_yaml
    └─ Output: list[DetectionDict], raw_result
    │
    ▼
ByteTrackWrapper.update(frame)                       src/trackers/bytetrack.py
    └─ Thin wrapper, chỉ delegate sang detector.infer()
    └─ KHÔNG có tracking logic riêng — ByteTrack do Ultralytics xử lý
    │
    ▼
[Per-detection loop]
    │
    ├─► crop = frame[y1:y2, x1:x2]
    │
    ▼
GlobalIDManager.assign(camera_id, track_id, crop, frame_idx)
    │                                                src/core/global_id.py
    ├─ Cache hit + reid_interval chưa hết → return (cached_gid, 1.0)
    ├─ Cache miss hoặc interval elapsed →
    │       FastReIDExtractor.extract(crop)          src/reid/fast_reid.py
    │           └─ L2-normalized float32 [D]
    │       ReIDGallery.query(feat, threshold)       src/reid/gallery.py
    │           └─ cosine similarity O(N) scan
    │           └─ ("person_id", score) hoặc ("unknown", score)
    └─ Return (global_id: str, reid_score: float)
    │
    ▼
PoseSequenceBuffer.update(...)                       src/action/pose_buffer.py
    └─ Accumulate keypoints, sliding window seq_len/stride
    └─ Khi đủ window → export .pkl (MMAction2 format)
    └─ _gc() dọn dead tracks mỗi frame
    │
    ▼
PoseC3DRunner.run_test(pkl_path)   [chỉ khi auto_infer=true]
    └─ subprocess gọi MMAction2 tools/test.py       src/action/posec3d.py
    └─ ⚠️ HIỆN TẠI: kết quả label KHÔNG được đọc lại vào pipeline
    │
    ▼
EventBus.emit(event_type, payload)                   src/core/event.py
    └─ Ghi JSONL, payload phải JSON-serializable
```

---

## 2. Cấu Trúc Thư Mục Chuẩn

```
CPose/
├── apps/               Entry points ONLY — không chứa business logic
│   ├── run_pipeline.py
│   ├── run_pose.py
│   ├── run_reid.py
│   └── run_adl.py
├── configs/
│   ├── system/pipeline.yaml    ← MỌI config tập trung ở đây
│   ├── fast-reid/R50.yml
│   └── posec3d/posec3d.py
├── src/
│   ├── action/         PoseSequenceBuffer, PoseC3DRunner
│   ├── core/           EventBus, GlobalIDManager
│   ├── detectors/      YoloPoseTracker
│   ├── reid/           FastReIDExtractor, ReIDGallery
│   ├── trackers/       ByteTrackWrapper
│   └── utils/          logger, config, io, vis
├── data/
│   ├── gallery/        Ảnh reference ReID (không commit)
│   ├── face/           Face embeddings theo person_id/ (không commit .npy)
│   └── output/         Tất cả output runtime (không commit)
├── models/             Weights (không commit — xem README)
├── static/             Web assets (hiện rỗng)
└── docs/               Papers, diagrams (không commit .pdf trực tiếp)
```

**Rule:** Logic mới → `src/<package>/`. Không nhét vào `apps/`.

---

## 3. Detection Dict Contract (KHÔNG THAY ĐỔI)

Mọi detection dict trong toàn bộ pipeline phải theo schema này:

```python
detection = {
    "bbox":             [x1, y1, x2, y2],    # float, pixel coords
    "score":            float,                # confidence [0.0, 1.0]
    "class_id":         int,                  # 0 = person
    "track_id":         int,                  # -1 nếu chưa tracked
    "keypoints":        [[x, y], ...] | None, # list of [x, y], length=V
    "keypoint_scores":  [float, ...] | None,  # length=V, cùng V với keypoints
}
```

- Đọc optional keys bằng `det.get("key")`, **không** dùng `det["key"]` cho optional fields.
- `V` = số keypoints (mặc định 17 COCO). Nếu model khác dùng số V khác, phải cập nhật `PoseSequenceBuffer` và MMAction2 config đồng bộ.

---

## 4. PoseSequenceBuffer Output Contract (KHÔNG THAY ĐỔI)

`.pkl` export phải giữ đúng format MMAction2 pose annotation:

```python
{
    "split": {"test": [sample_id: str]},
    "annotations": [{
        "frame_dir":       str,
        "total_frames":    int,
        "img_shape":       (H: int, W: int),
        "original_shape":  (H: int, W: int),
        "label":           int,
        "keypoint":        np.float32,   # shape [M=1, T, V, C=2]
        "keypoint_score":  np.float32,   # shape [M=1, T, V]
    }]
}
```

**Không đổi format này** nếu không sửa đồng bộ PoseC3D config và MMAction2 dataloader.

---

## 5. Rules Không Được Vi Phạm

### 5.1 Path Rules

```python
# ✅ ĐÚNG — resolve từ __file__
ROOT = Path(__file__).resolve().parents[1]  # apps/ → repo root

# ❌ SAI — hardcode absolute path
ROOT = Path(r"D:\Capstone_Project")
ROOT = Path("/home/user/CPose")
```

- `configs/system/pipeline.yaml` dùng **relative path**. Code resolve bằng `resolve_cfg_paths(cfg, ROOT)`.
- Không hardcode path trong `configs/posec3d/posec3d.py` hoặc `configs/fast-reid/R50.yml`. Xem Bug #6.

### 5.2 Config Rules

- Path, threshold, hyperparameter → `configs/system/pipeline.yaml`.
- Config mới → thêm vào `validate_cfg()` trong `src/utils/config.py`.
- Validate **trước** khi build bất kỳ model/module nào.
- Không hardcode magic value trong business logic nếu cần tune.

### 5.3 Error Handling Rules

```python
# Frame-level: log + skip
try:
    detections, _ = tracker.update(frame)
except Exception as exc:
    logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
    continue

# Resource: phải có finally
try:
    cap = cv2.VideoCapture(source)
    ...
finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

# Init-level: raise sớm với message rõ ràng
if not self.mmaction_root.exists():
    raise FileNotFoundError(f"mmaction_root not found: {self.mmaction_root}")
```

- **Không** dùng bare `except:`. Luôn dùng `except Exception as exc:` hoặc exception cụ thể hơn.

### 5.4 Logging Rules

```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```

- **Không** dùng `print()` trong pipeline. Chỉ chấp nhận trong script debug độc lập.
- `logger.debug` cho per-frame verbose, `logger.info` cho milestones, `logger.warning` cho recoverable errors, `logger.error` cho fatal.

### 5.5 Identity Rules

- `global_id` là string: tên người từ gallery hoặc `gid_NNNNN`.
- `local_track_id` chỉ unique trong 1 camera session. KHÔNG dùng làm cross-camera key.
- Cache key = `(camera_id: str, local_track_id: int)`.
- Gọi `forget_track(camera_id, track_id)` khi tracker báo lost. Hiện tại pipeline chưa làm — xem Bug #2.
- **Không** thêm embedding của unknown track vào gallery với key `"unknown"` — xem Bug #1.

### 5.6 Security Rules

- Credential (RTSP URL với user/pass) phải ở `.env`, KHÔNG commit vào repo.
- `.env` phải có trong `.gitignore`.
- `data/config/resources.txt` hiện vi phạm rule này — xem Bug #9.

### 5.7 ReID Rules

- Không gọi ReID CNN mỗi frame. Dùng `reid_interval` để re-check định kỳ.
- Khi gallery scale lớn (>50 người), phải dùng FAISS — xem Bug #8.
- EMA update prototype sau mỗi `add_embedding`. Công thức: `proto = alpha * proto + (1-alpha) * feat`.

### 5.8 ADL Rules

- ADL label kết quả **phải** được đọc từ stdout/file của subprocess và emit qua EventBus.
- `PoseC3DRunner.run_test()` hiện không trả label về pipeline — xem Bug #4.
- Keypoint count V phải match với MMAction2 model (NTU60 = 17 COCO).

---

## 6. Danh Sách Bug Đã Xác Định (Phải Fix, Không Workaround)

### Bug #1 — `gallery.add_embedding("unknown", feat)` [CRITICAL]

**File:** `src/core/global_id.py` dòng ~37
**Vấn đề:** Khi `matched_id == "unknown"`, code gọi `self.gallery.add_embedding("unknown", feat)`. Điều này:
- Tạo key `"unknown"` trong `gallery.prototypes` và `gallery.memory`
- Mọi track sau khi gán `unknown` sẽ tích lũy embedding dưới cùng 1 prototype
- Sau nhiều frame, prototype "unknown" converge thành mean của mọi người lạ → gây false match
- Memory leak vô hạn

**Fix:**
```python
# global_id.py :: assign()
if matched_id == "unknown":
    matched_id = self.track_to_global.get(key) or self._new_global_id()
    # Chỉ add embedding cho global_id mới, KHÔNG add cho "unknown"
    self.gallery.add_embedding(matched_id, feat)
    logger.info(f"New global_id={matched_id} camera={camera_id} track={local_track_id}")
```

### Bug #2 — `GlobalIDManager` memory leak [HIGH]

**File:** `src/core/global_id.py`
**Vấn đề:** `track_to_global` và `track_frame_count` tăng vô hạn. `forget_track()` tồn tại nhưng không bao giờ được gọi trong `run_pipeline.py`. Sau hàng giờ chạy với nhiều người đi qua frame, dict này sẽ rất lớn.

**Fix:** Thêm `_gc(current_frame_idx)` vào `GlobalIDManager` tương tự `PoseSequenceBuffer`, hoặc gọi `forget_track()` trong pipeline khi `track_id` không còn xuất hiện trong detections của frame hiện tại.

```python
# Ở cuối mỗi frame trong run_pipeline.py
active_tracks = {det["track_id"] for det in detections if det.get("track_id", -1) >= 0}
for tid in list(global_id_manager.track_to_global.keys()):
    cam, ltid = tid
    if cam == args.camera_id and ltid not in active_tracks:
        global_id_manager.forget_track(cam, ltid)
```

### Bug #3 — `run_pose.py` hardcode model path [MEDIUM]

**File:** `apps/run_pose.py` dòng 11
**Vấn đề:** `MODEL_PATH = ROOT / "models" / "yolo11n-pose.pt"` hardcode, vi phạm Config Rule 5.2.

**Fix:**
```python
# Xóa hardcode, load từ config
cfg = load_pipeline_cfg(ROOT / "configs" / "system" / "pipeline.yaml", ROOT)
model = YOLO(cfg["pose"]["weights"])
```

### Bug #4 — ADL label không được đọc lại vào pipeline [HIGH]

**File:** `src/action/posec3d.py`, `apps/run_pipeline.py`
**Vấn đề:** `PoseC3DRunner.run_test()` chạy subprocess và trả về `subprocess.CompletedProcess`, không parse label. Pipeline không đọc output, không emit `adl_result` event. Toàn bộ action recognition loop vô nghĩa.

**Fix:** `run_test()` phải:
1. Parse kết quả từ stdout hoặc một output JSON/txt được viết ra `work_dir`
2. Trả về `(label: int, label_name: str, confidence: float)` hoặc `None` nếu fail
3. Pipeline phải emit `event_bus.emit("adl_result", {...})` với kết quả

### Bug #5 — Relative `_BASE_` trong `configs/fast-reid/R50.yml` [MEDIUM]

**File:** `configs/fast-reid/R50.yml`
**Vấn đề:** `_BASE_: ../../.github/fast-reid/configs/Market1501/bagtricks_R50.yml`
YACS resolve path này relative so với file yml. Nếu thư mục `.github/fast-reid` di chuyển, hoặc CWD thay đổi, sẽ fail silently.

**Fix:** Dùng absolute path được inject bởi `FastReIDExtractor` lúc runtime, hoặc luôn chạy với `cwd=ROOT`.

### Bug #6 — Hardcode relative path trong `configs/posec3d/posec3d.py` [MEDIUM]

**File:** `configs/posec3d/posec3d.py`
**Vấn đề:**
```python
load_from = r'../../models/posec3d_r50_ntu60.pth'  # Chỉ đúng nếu CWD là configs/posec3d/
```
`PoseC3DRunner.build_temp_test_config()` đã dùng absolute path trong temp config, nhưng base config này vẫn có relative path.

**Fix:** `PoseC3DRunner` phải resolve `self.base_config` thành absolute path trước khi dùng làm `_base_` trong temp config. Hoặc xóa `load_from` khỏi `posec3d.py` và chỉ set trong temp config.

### Bug #7 — `bytetrack.yaml` bare filename [LOW]

**File:** `configs/system/pipeline.yaml` → `tracker.tracker_yaml`
**Vấn đề:** `"bytetrack.yaml"` không có path prefix. Ultralytics tìm file này trong package của nó (`ultralytics/cfg/trackers/`), hoạt động với built-in name. Nhưng nếu muốn dùng custom bytetrack config, phải là absolute/relative path rõ ràng.

**Fix:** Thêm note vào config rằng đây là Ultralytics built-in name. Nếu cần custom, dùng `str(ROOT / "configs" / "bytetrack_custom.yaml")`.

### Bug #8 — FAISS trong requirements nhưng không dùng [MEDIUM]

**File:** `src/reid/gallery.py`, `requirements.txt`
**Vấn đề:** `ReIDGallery.query()` dùng O(N) cosine scan. `faiss-cpu` được cài nhưng không dùng. Với gallery 100+ người × nhiều embeddings, mỗi frame ReID sẽ chậm.

**Fix:** Implement FAISS index trong `ReIDGallery`:
```python
import faiss

class ReIDGallery:
    def __init__(self, ...):
        self.index = None   # faiss.IndexFlatIP
        self.id_map = []    # person_id tương ứng với mỗi vector trong index

    def _rebuild_index(self):
        if not self.prototypes:
            self.index = None
            return
        vecs = np.stack(list(self.prototypes.values()), axis=0).astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.id_map = list(self.prototypes.keys())

    def query(self, feat, threshold=0.55):
        if self.index is None or self.index.ntotal == 0:
            return "unknown", -1.0
        q = feat[None, :].astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, 1)
        score = float(D[0, 0])
        if score < threshold:
            return "unknown", score
        return self.id_map[I[0, 0]], score
```

Gọi `_rebuild_index()` sau mỗi `add_embedding()` hoặc dùng `IndexIDMap` để incremental add.

### Bug #9 — RTSP credentials plaintext trong repo [CRITICAL - Security]

**File:** `data/config/resources.txt`
**Vấn đề:**
```
Cam Nha __rtsp://bathanh0309:bathanh0309@192.168.100.160:554/stream2
```
Username/password camera bị expose nếu repo là public hoặc có thêm collaborator.

**Fix:**
1. Xóa file này khỏi git history: `git filter-branch` hoặc `git-filter-repo`
2. Đổi password camera ngay
3. Tạo `.env`:
   ```
   CAM_NHA_URL=rtsp://user:pass@192.168.100.160:554/stream2
   CAM_SAN_URL=rtsp://user:pass@192.168.100.242:554/stream2
   ```
4. Thêm `.env` vào `.gitignore`
5. Load trong pipeline: `from dotenv import load_dotenv; load_dotenv()`

### Bug #10 — `FaceGallery` class thiếu [HIGH]

**File:** `data/face/README.md` reference, nhưng class không tồn tại trong `src/`
**Vấn đề:** `data/face/` có cấu trúc embeddings (`.npy`, `meta.json`) cho face recognition, nhưng không có `FaceGallery` class nào implement logic load/query. Đây là một subsystem chưa được implement.

**Fix:** Cần tạo `src/reid/face_gallery.py` với:
- `load(face_dir)` → stack embeddings từ `embeddings.npy` hoặc `emb_NN.npy`
- `query(feat, threshold)` → cosine hoặc FAISS
- Phân biệt rõ với `ReIDGallery` (body appearance) vs `FaceGallery` (face)

### Bug #11 — `GlobalIDManager` trả `1.0` khi cache hit [LOW]

**File:** `src/core/global_id.py` dòng ~22
```python
if key in self.track_to_global and not should_reid:
    return self.track_to_global[key], 1.0   # ← confidence giả
```
Dashboard hoặc event log sẽ thấy `reid_score=1.0` cho mọi cached frame, misleading.

**Fix:** Cache cả score cùng với global_id:
```python
self.track_to_global[key] = (matched_id, float(score))
# khi return cache:
gid, cached_score = self.track_to_global[key]
return gid, cached_score
```

### Bug #12 — Keypoint count không được validate [MEDIUM]

**File:** `src/action/pose_buffer.py`
**Vấn đề:** `PoseSequenceBuffer.update()` không kiểm tra `keypoints_xy.shape[0] == 17`. Nếu model YOLO được đổi sang variant khác (ví dụ 133 keypoints wholebody), MMAction2 PoseC3D sẽ nhận tensor sai shape và crash hoặc cho kết quả sai.

**Fix:**
```python
EXPECTED_KEYPOINTS = 17  # COCO

if keypoints_xy.shape[0] != EXPECTED_KEYPOINTS:
    logger.warning(
        f"Unexpected keypoint count {keypoints_xy.shape[0]} != {EXPECTED_KEYPOINTS}, skipping"
    )
    return None
```
Hoặc đưa `expected_keypoints` vào config để flexible.

### Bug #13 — `ByteTrackWrapper` misleading name [LOW]

**File:** `src/trackers/bytetrack.py`, `README.md`
**Vấn đề:** README nói "download `bytetrack_s_mot17.pth.tar`" nhưng file này không được load ở đâu cả. ByteTrack được Ultralytics xử lý nội bộ qua `tracker_yaml`. `bytetrack_s_mot17.pth.tar` trong `models/` là thừa.

**Fix:** Xóa `bytetrack_s_mot17.pth.tar` khỏi bảng model trong README, hoặc thêm note rõ "ByteTrack được Ultralytics quản lý, không cần download riêng".

---

## 7. Module Contracts

### `YoloPoseTracker` — `src/detectors/yolo_pose.py`

```
Input:   BGR frame (numpy.ndarray H×W×3)
Output:  (list[DetectionDict], ultralytics.Results)
Method:  infer(frame, persist=True)
Note:    persist=True bắt buộc để ByteTrack trong Ultralytics giữ track memory
         classes=[0] — chỉ detect người
```

### `ByteTrackWrapper` — `src/trackers/bytetrack.py`

```
Input:   BGR frame
Output:  (list[DetectionDict], raw_result)
Method:  update(frame)
Note:    Thin wrapper. Không thêm tracking logic riêng.
         ByteTrack thực sự nằm trong Ultralytics, không phải đây.
```

### `FastReIDExtractor` — `src/reid/fast_reid.py`

```
Input:   BGR crop (numpy.ndarray)
Output:  L2-normalized float32 vector shape [D] (D=2048 cho R50)
Method:  extract(image_bgr)
Note:    sys.path.insert chỉ làm 1 lần, có check duplicate.
         Nếu fastreid_root không tồn tại → raise FileNotFoundError
```

### `ReIDGallery` — `src/reid/gallery.py`

```
build()              Load ảnh từ gallery_dir/{person_name}/*.{jpg,png,...}
query(feat, thr)     → (person_id: str, score: float) | ("unknown", score)
add_embedding(id, feat) → update prototype (phải gọi _rebuild_index sau đó)
```

### `GlobalIDManager` — `src/core/global_id.py`

```
assign(camera_id, local_track_id, crop_bgr, frame_idx) → (global_id: str, score: float)
forget_track(camera_id, local_track_id)                → cleanup cache
```

### `PoseSequenceBuffer` — `src/action/pose_buffer.py`

```
update(...) → Path | None   (None nếu chưa đủ seq_len hoặc chưa đến stride)
reset_track(camera_id, local_track_id) → manual cleanup
_gc(current_frame_idx) → auto cleanup idle tracks
```

### `PoseC3DRunner` — `src/action/posec3d.py`

```
run_test(ann_file: str) → subprocess.CompletedProcess
⚠️ HIỆN TẠI chưa parse label. Phải fix Bug #4 trước khi dùng trong production.
```

### `EventBus` — `src/core/event.py`

```
emit(event_type: str, payload: dict) → ghi JSONL
payload phải JSON-serializable (không chứa numpy array, Path object)
Events chuẩn: "track_update", "pose_clip_exported", "adl_result" (chưa implement)
```

---

## 8. Config Schema Đầy Đủ (`configs/system/pipeline.yaml`)

```yaml
system:
  device: "cuda"          # hoặc "cpu"
  event_log: "data/output/events/pipeline.jsonl"
  vis_dir: "data/output/vis"
  save_video: true

pose:
  weights: "models/yolo11n-pose.pt"
  conf: 0.45              # float [0.0, 1.0]
  iou: 0.5                # float [0.0, 1.0]

tracker:
  tracker_yaml: "bytetrack.yaml"  # Ultralytics built-in name

reid:
  fastreid_root: ".github/fast-reid"
  config: "configs/fast-reid/R50.yml"
  weights: "models/fastreid_market_R50.pth"
  gallery_dir: "data/gallery"
  threshold: 0.55         # float, cosine similarity cutoff
  reid_interval: 10       # int, frames giữa 2 lần query gallery

adl:
  mmaction_root: ".github/pose-c3d"
  base_config: "configs/posec3d/posec3d.py"
  weights: "models/posec3d_r50_ntu60.pth"
  seq_len: 48             # int, frames per clip
  stride: 12              # int, export stride
  max_idle_frames: 150    # int, gc threshold
  export_dir: "data/output/clips_pkl"
  default_label: 0        # int, nhãn mặc định khi chưa infer
  work_dir: "data/output/posec3d"
  auto_infer: false       # bool, bật PoseC3D subprocess
```

Mọi key mới phải được thêm vào `validate_cfg()` trong `src/utils/config.py`.

---

## 9. Quy Trình Thêm Tính Năng

1. **Đặt logic** vào đúng package trong `src/`, không nhét vào `apps/`.
2. **Config** → `configs/system/pipeline.yaml`.
3. **Validate** → update `validate_cfg()`.
4. **Logger** → `get_logger(__name__)`.
5. **Không phá** DetectionDict contract và PoseSequenceBuffer `.pkl` format.
6. **Compile check**: `python -m compileall apps src`
7. **Security**: Không commit credential, weight, PDF.

---

## 10. Code Style

- Import order: stdlib → third-party → local (isort).
- Type hints cho public API của mỗi class/function mới.
- **Không** bare `except:`.
- **Không** refactor rộng nếu task chỉ cần bugfix hẹp.
- **Không** revert thay đổi không liên quan trong dirty worktree.

---

## 11. Bảng "Không Làm"

| Không làm | Lý do |
|-----------|-------|
| Hardcode absolute path | Chỉ chạy trên 1 máy |
| `print()` trong pipeline | Thiếu timestamp, không filter được |
| `gallery.add_embedding("unknown", feat)` | Pollute gallery, false match |
| Gọi ReID CNN mỗi frame | Không realtime với nhiều người |
| Để `track_to_global` tăng vô hạn | Memory leak |
| Bỏ `try/finally` cho OpenCV resource | Camera/writer/window leak |
| Commit `.pt`, `.pth`, `.pkl`, `.pdf`, `.npy` | Làm repo nặng |
| Commit credential vào repo | Security risk |
| Đổi `.pkl` output format tùy tiện | Break MMAction2 |
| Lặp `sys.path.insert` không kiểm soát | Side effect global process |
| Dùng FAISS index không rebuild sau add | Stale index, wrong results |
| Parse PoseC3D kết quả bằng regex trên stdout | Fragile; dùng output file |

---

## 12. Dependency Ngoài Pip

| Library | Config key | Ghi chú |
|---------|-----------|---------|
| FastReID | `reid.fastreid_root` | Clone vào `.github/fast-reid` |
| MMAction2 / PoseC3D | `adl.mmaction_root` | Clone vào `.github/pose-c3d` nếu bật `auto_infer` |

`FastReIDExtractor` tự inject path. Không thêm `sys.path.insert` cho FastReID ở module khác.

---

## 13. Checklist Trước Khi Commit

- [ ] Không còn hardcoded `D:\...`, `/home/...` trong `apps/`, `src/`, `configs/`.
- [ ] Không có credential (IP cam, password) trong code hoặc config file.
- [ ] Resource OpenCV/file/subprocess được cleanup bằng `finally`.
- [ ] Exception được bắt đúng cấp (frame-level vs init-level).
- [ ] Config mới đã được validate trong `validate_cfg()`.
- [ ] `gallery.add_embedding()` không được gọi với `person_id="unknown"`.
- [ ] `forget_track()` được gọi khi track lost.
- [ ] `.pt`, `.pth`, `.pkl`, `.pdf`, `.npy`, `.env` không bị staged.
- [ ] `python -m compileall apps src` pass không có error.
- [ ] DetectionDict contract và PoseSequenceBuffer `.pkl` contract không bị phá.
- [ ] Mọi event payload JSON-serializable (không có numpy, Path).

---

## 14. Files Cần Cẩn Thận Khi Sửa

| File | Rủi ro khi sửa |
|------|----------------|
| `configs/system/pipeline.yaml` | Ảnh hưởng toàn pipeline |
| `src/action/pose_buffer.py::_export_current_window()` | Format `.pkl` cố định |
| `src/reid/gallery.py::add_embedding()` | Dễ gây Bug #1 quay lại |
| `src/core/global_id.py::assign()` | Logic identity dễ race condition |
| `src/utils/config.py::validate_cfg()` | Thêm key mới phải thêm vào đây |
| `requirements.txt` | Kiểm tra compatibility trước khi đổi version |

---

