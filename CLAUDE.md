# CLAUDE.md - CPose Project Intelligence

> Dành cho AI coding agents như Claude, Codex, Cursor, Copilot.
> Đọc file này trước khi sửa repo. Nếu cần phá một rule, phải ghi rõ lý do trong thay đổi hoặc phản hồi.

---

## 1. Tổng Quan Hệ Thống

CPose là pipeline real-time cho multi-person pose tracking, ReID và action recognition dựa trên pose.

Luồng chính:

```text
Video/Camera frame
    |
    v
YoloPoseTracker       src/detectors/yolo_pose.py
    -> bbox, keypoints, local track_id
    |
    v
ByteTrackWrapper      src/trackers/bytetrack.py
    -> detection dict list
    |
    v
FastReIDExtractor     src/reid/fast_reid.py
    -> normalized float32 embedding
    |
    v
ReIDGallery           src/reid/gallery.py
    -> matched person_id or "unknown"
    |
    v
GlobalIDManager       src/core/global_id.py
    -> stable global_id
    |
    v
PoseSequenceBuffer    src/action/pose_buffer.py
    -> MMAction2 .pkl clip when seq_len is reached
    |
    v
PoseC3DRunner         src/action/posec3d.py
    -> ADL/action inference, optional
    |
    v
EventBus              src/core/event.py
    -> JSONL event log
```

Entry points:

| File | Purpose |
| --- | --- |
| `apps/run_pipeline.py` | Main full pipeline |
| `apps/run_pose.py` | Debug YOLO pose only |
| `apps/run_reid.py` | Debug ReID only |
| `apps/run_adl.py` | Debug PoseC3D only |

---

## 2. Cấu Trúc Thư Mục

```text
CPose/
|-- apps/                  entry points only
|-- configs/
|   |-- system/pipeline.yaml
|   |-- fast-reid/R50.yml
|   `-- posec3d/posec3d.py
|-- src/
|   |-- action/
|   |   |-- pose_buffer.py
|   |   `-- posec3d.py
|   |-- core/
|   |   |-- event.py
|   |   `-- global_id.py
|   |-- detectors/yolo_pose.py
|   |-- reid/
|   |   |-- fast_reid.py
|   |   `-- gallery.py
|   |-- trackers/bytetrack.py
|   `-- utils/
|       |-- config.py
|       |-- io.py
|       |-- logger.py
|       `-- vis.py
|-- data/
|   |-- gallery/
|   `-- output/
|-- models/                local weights, do not commit
|-- static/
`-- docs/
```

---

## 3. Rules Không Được Vi Phạm

### 3.1 Path Rules

Luôn resolve path từ repo root hoặc từ `Path(__file__)`.

```python
ROOT = Path(__file__).resolve().parents[1]  # apps/ -> repo root
```

Không hardcode absolute path:

```python
ROOT = Path(r"D:\Capstone_Project")  # sai
ROOT = Path("/home/user/CPose")      # sai
```

`configs/system/pipeline.yaml` nên dùng relative path. Code sẽ resolve bằng `src/utils/config.py::resolve_cfg_paths()`.

### 3.2 Config Rules

- Path, threshold và hyperparameter phải nằm trong `configs/system/pipeline.yaml`.
- Không hardcode magic values trong business logic nếu giá trị đó cần tune.
- Mọi config mới phải được validate trong `src/utils/config.py::validate_cfg()`.
- Config phải được validate trước khi build model/pipeline.

### 3.3 Error Handling Rules

Lỗi ở cấp frame, detection hoặc track: log rồi skip, không kill toàn pipeline.

```python
try:
    detections, _ = tracker.update(frame)
except Exception as exc:
    logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
    continue
```

Resource như `VideoCapture`, `VideoWriter`, window OpenCV phải cleanup bằng `finally`.

```python
try:
    ...
finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
```

Lỗi ở cấp khởi tạo model: raise sớm với message rõ ràng.

### 3.4 Logging Rules

Mỗi module có logic runtime nên dùng logger:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

Không dùng `print()` cho business logic lâu dài. `print()` chỉ chấp nhận trong script debug nhỏ nếu không ảnh hưởng pipeline.

### 3.5 Detection Dict Contract

Mọi detection dict phải theo schema:

```python
detection = {
    "bbox": [x1, y1, x2, y2],       # float pixel coords
    "score": float,                 # confidence [0, 1]
    "class_id": int,                # 0 = person
    "track_id": int,                # -1 if untracked
    "keypoints": [[x, y], ...] | None,
    "keypoint_scores": [float, ...] | None,
}
```

Đọc optional keys bằng `det.get("key")`, không dùng `det["key"]` nếu key có thể thiếu.

### 3.6 Global ID Rules

- `global_id` là string: person name từ gallery hoặc `gid_NNNNN`.
- `local_track_id` chỉ unique trong một camera/session.
- Cache key là `(camera_id, local_track_id)`.
- `GlobalIDManager.assign(camera_id, local_track_id, crop_bgr, frame_idx=None)` trả `(global_id, reid_score)`.
- Không gọi ReID CNN mỗi frame. Dùng cache và `reid_interval` để re-check định kỳ.
- Gọi `forget_track(camera_id, local_track_id)` khi track lost nếu có tín hiệu lost từ tracker.

### 3.7 PoseSequenceBuffer Output Contract

`.pkl` export phải giữ format MMAction2 pose annotation:

```python
{
    "split": {"test": [sample_id]},
    "annotations": [{
        "frame_dir": str,
        "total_frames": int,
        "img_shape": (H, W),
        "original_shape": (H, W),
        "label": int,
        "keypoint": np.float32,        # [M=1, T, V=17, C=2]
        "keypoint_score": np.float32,  # [M=1, T, V=17]
    }]
}
```

Không đổi format này nếu không sửa tương ứng PoseC3D/MMAction2 config.

---

## 4. Module Contracts

### `YoloPoseTracker` - `src/detectors/yolo_pose.py`

- Input: BGR frame as numpy array.
- Public method: `infer(frame, persist=True) -> (list[dict], result)`.
- `persist=True` cần thiết để ByteTrack giữ track memory.

### `ByteTrackWrapper` - `src/trackers/bytetrack.py`

- Thin wrapper, chỉ delegate sang `YoloPoseTracker.infer()`.
- Public method: `update(frame) -> (detections, raw_result)`.
- Không thêm tracking logic riêng ở đây nếu không có lý do rõ.

### `FastReIDExtractor` - `src/reid/fast_reid.py`

- Input: BGR crop.
- Output: L2-normalized `np.float32` vector shape `[D]`.
- `sys.path.insert` chỉ làm một lần, có check duplicate.
- Nếu `fastreid_root` không tồn tại, raise `FileNotFoundError`.

### `ReIDGallery` - `src/reid/gallery.py`

- `build()` load ảnh từ `gallery_dir/{person_name}/*`.
- `query(feat, threshold)` trả `(person_id, score)` hoặc `("unknown", score)`.
- `add_embedding(person_id, feat)` update online prototype.

### `GlobalIDManager` - `src/core/global_id.py`

- Public methods:
  - `assign(camera_id, local_track_id, crop_bgr, frame_idx=None)`.
  - `forget_track(camera_id, local_track_id)`.
- Dùng `reid_interval` để giới hạn số lần query gallery/CNN trên một track.

### `PoseSequenceBuffer` - `src/action/pose_buffer.py`

- Accumulate keypoints theo sliding window `seq_len`, stride `stride`.
- `_gc(current_frame_idx)` dọn state idle quá `max_idle_frames`.
- `reset_track(camera_id, local_track_id)` dọn state thủ công.

### `PoseC3DRunner` - `src/action/posec3d.py`

- Gọi MMAction2 `tools/test.py` qua subprocess.
- Build temp config theo clip.
- `run_test(ann_file)` hiện là synchronous subprocess.

### `EventBus` - `src/core/event.py`

- `emit(event_type, payload)` ghi JSONL.
- Payload phải JSON-serializable, không chứa numpy array.
- Events chuẩn hiện có: `track_update`, `pose_clip_exported`.

---

## 5. Quy Trình Thêm Tính Năng

1. Đặt logic vào đúng package trong `src/`, không nhét business logic vào `apps/`.
2. Thêm config vào `configs/system/pipeline.yaml` nếu giá trị cần tune.
3. Update `validate_cfg()` nếu thêm required config.
4. Dùng `get_logger(__name__)`.
5. Giữ detection dict và `.pkl` output contract.
6. Chạy ít nhất `python -m compileall apps src`.

---

## 6. Code Style

- Import order: stdlib, third-party, local.
- Ưu tiên type hints cho public functions/classes khi sửa code liên quan.
- Không dùng bare `except`.
- Không revert thay đổi không liên quan trong dirty worktree.
- Tránh refactor rộng nếu task chỉ cần bugfix hẹp.

---

## 7. Không Làm

| Không làm | Lý do |
| --- | --- |
| Hardcode absolute path | Chỉ chạy được trên một máy |
| Dùng `print()` trong pipeline | Thiếu timestamp/filter |
| Gọi ReID CNN mỗi frame | Không realtime khi nhiều người |
| Để state buffer tăng vô hạn | Memory leak khi chạy lâu |
| Bỏ `try/finally` cho OpenCV resources | Dễ leak camera/writer/window |
| Commit `.pt`, `.pth`, `.pkl`, `.pdf` | Làm repo nặng |
| Đổi `.pkl` output format tùy tiện | Break PoseC3D/MMAction2 |
| Lặp `sys.path.insert` không kiểm soát | Side effect global process |

---

## 8. File Cần Cẩn Thận Khi Sửa

- `configs/system/pipeline.yaml`: ảnh hưởng toàn pipeline.
- `src/action/pose_buffer.py::_export_current_window()`: format output cố định.
- `requirements.txt`: cần kiểm tra compatibility trước khi đổi version.
- Model files trong `models/`: không commit weight mới vào git.

---

## 9. Checklist Trước Khi Commit

- [ ] Không còn hardcoded `D:\...`, `/home/...` trong `apps`, `src`, `configs`.
- [ ] Resource OpenCV/file/subprocess được cleanup hợp lý.
- [ ] Exception được bắt ở đúng cấp.
- [ ] Config mới đã được validate.
- [ ] `.pt`, `.pth`, `.pkl`, `.pdf` không bị commit.
- [ ] `python -m compileall apps src` pass.
- [ ] Contract detection dict và PoseC3D `.pkl` không bị phá.

---

## 10. Dependency Ngoài Pip

| Library | Config key | Ghi chú |
| --- | --- | --- |
| FastReID | `reid.fastreid_root` | Clone/vendor local, hiện repo dùng `.github/fast-reid` |
| MMAction2 | `adl.mmaction_root` | Cần cài hoặc clone riêng nếu bật `adl.auto_infer` |

`FastReIDExtractor` tự xử lý path injection cho FastReID. Không thêm `sys.path.insert` cho FastReID ở module khác nếu không cần thiết.
