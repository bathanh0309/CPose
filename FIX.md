# CLAUDE.md — CPose Binding Rules

> **BẮT BUỘC ĐỌC TRƯỚC KHI SỬA BẤT KỲ FILE NÀO.**
> File này là nguồn sự thật duy nhất về kiến trúc, convention và ràng buộc của CPose.
> Mọi thay đổi code phải tuân thủ 100% các quy tắc dưới đây.

---

## 1. Cấu trúc thư mục

```
CPose/
├── apps/
│   ├── run_track.py        # ONLY: detect + track người (ByteTrack)
│   ├── run_pose.py         # ONLY: YOLO11-Pose skeleton
│   ├── run_reid.py         # ONLY: crop → ReID → global_id
│   ├── run_adl.py          # ONLY: skeleton buffer → ADL recognition
│   ├── run_object.py       # ONLY: custom object detector (pickleball, v.v.)
│   ├── run_pipeline.py     # ALL: track + pose + reid + adl + metrics + UI
│   └── run_web_cmd.py      # launcher uvicorn
├── configs/system/
│   ├── pipeline.yaml       # config gốc duy nhất
│   └── benchmark.yaml      # benchmark targets
├── models/                 # weights (không push git)
├── src/
│   ├── action/             # pose_buffer.py, posec3d.py
│   ├── core/               # event.py, global_id.py, metrics.py, ui_logger.py
│   ├── detectors/          # yolo_pose.py, pedestrian_yolo.py
│   ├── reid/               # fast_reid.py, gallery.py
│   ├── trackers/           # bytetrack.py
│   └── utils/              # config.py, filters.py, io.py, logger.py, naming.py, video.py, vis.py
├── static/                 # index.html, style.css, scripts.js  ← KHÔNG REFACTOR UI
├── main.py                 # FastAPI backend
└── CLAUDE.md
```

---

## 2. Trách nhiệm từng app (NGHIÊM CẤM vi phạm)

| App | Chạy | KHÔNG chạy |
|---|---|---|
| `run_track.py` | detect + ByteTrack người | ReID, ADL, pose buffer |
| `run_pose.py` | YOLO11-Pose + keypoints | ReID, ADL, tracking riêng |
| `run_reid.py` | detect tối thiểu → crop → ReID → global_id | ADL, pose buffer |
| `run_adl.py` | detect tối thiểu → pose → skeleton buffer → ADL | ReID, full pipeline |
| `run_object.py` | custom object detector (pickleball/paddle/ball) | người, ReID, ADL |
| `run_pipeline.py` | **TẤT CẢ**: track + pose + reid + adl + metrics + UI | — |

> **KHÔNG tạo `run_pedestrian_track.py`** — chức năng đã gom vào `run_track.py`.
> Nếu cần pedestrian tracking riêng, dùng `PedestrianYoloTracker` bên trong `run_track.py`.

---

## 3. CLI thống nhất — TẤT CẢ app phải có đủ các flag sau

```
--source          đường dẫn video / RTSP URL / webcam index
--camera-id       định danh camera (ví dụ: cam01, cam02)
--config          đường dẫn pipeline.yaml (default: configs/system/pipeline.yaml)
--show            hiện cửa sổ OpenCV
--save-video      ghi video kết quả ra data/output/vis/
--output          override đường dẫn output video
--max-frames      giới hạn số frame xử lý (0 = không giới hạn)
--ui-log          gửi metrics/log tới UILogger (optional)
```

> **NGHIÊM CẤM hard-code đường dẫn tuyệt đối** như `D:\Capstone_Project\...` vào source code.
> Đường dẫn đó chỉ dùng trong command test.

**Command test chuẩn:**

```bash
python apps/run_track.py   --source "D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4" --camera-id cam02 --show --save-video --max-frames 200
python apps/run_pose.py    --source "D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4" --camera-id cam02 --show --save-video --max-frames 200
python apps/run_reid.py    --source "D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4" --camera-id cam02 --show --save-video --max-frames 200
python apps/run_adl.py     --source "D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4" --camera-id cam02 --show --save-video --max-frames 250
python apps/run_pipeline.py --source "D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4" --camera-id cam02 --show --save-video --max-frames 250
```

---

## 4. Detection filtering — chống detect nhầm người

Tất cả app dùng `src/utils/filters.py::is_valid_person_detection()`.

**Config mặc định (pipeline.yaml):**

```yaml
tracking:
  person_conf: 0.60        # ngưỡng confidence detect người
  iou: 0.5
  min_box_area: 2500       # pixel² tối thiểu
  min_keypoints: 5         # số keypoint visible tối thiểu
  min_keypoint_score: 0.35 # avg keypoint score tối thiểu
  tracker_yaml: "bytetrack.yaml"
```

**Logic filter (theo thứ tự):**

1. `class_id == 0` → không phải người → loại
2. `score < person_conf` → confidence thấp → loại (`filtered_low_conf`)
3. `bbox_area < min_box_area` → bbox quá nhỏ → loại (`filtered_small_box`)
4. Nếu có keypoints: `visible_kpts < min_keypoints` hoặc `avg_kpt_score < min_keypoint_score` → loại (`filtered_bad_pose`)

**Khi bị loại nhiều track thật (người bị che khuất):**

```yaml
tracking:
  person_conf: 0.55        # giảm xuống 0.55
  min_keypoints: 5         # giữ nguyên
  min_keypoint_score: 0.35 # giữ nguyên
```

> Không tăng conf quá 0.65 — sẽ mất track người đang bị che khuất nhẹ.

**Overlay hiển thị khi bị filter:**

```
[12:33:30] Warning: filtered false person conf=0.34
```

---

## 5. Detection/Display conf tách biệt

| Config key | Dùng cho |
|---|---|
| `tracking.person_conf` | threshold lọc detection trước khi vào tracker |
| `pose.conf` | YOLO inference conf (có thể thấp hơn để không bỏ sót) |

Nếu muốn hiển thị nhãn conf khác với threshold lọc, dùng `detection_conf` và `display_conf` riêng.

---

## 6. YOLO11-Pose ≠ Object Detector

> **YOLO11-Pose chỉ detect NGƯỜI + 17 keypoints COCO.**
> **TUYỆT ĐỐI KHÔNG dùng YOLO11-Pose để detect pickleball, paddle, ball.**

| Cần detect | Dùng model |
|---|---|
| Người + skeleton | `models/yolo11n-pose.pt` → `run_pose.py` |
| Pickleball / paddle / ball | Custom YOLO train riêng → `run_object.py` |

`run_object.py` dùng `PedestrianYoloTracker` hoặc YOLO detect thuần với config:

```yaml
object:
  enabled: false
  weights: "models/yolo11n.pt"   # phải train riêng
  conf: 0.7
  iou: 0.5
  classes:
    - pickleball
    - paddle
    - ball
```

---

## 7. Identity contract

| Field | Nguồn | Ý nghĩa |
|---|---|---|
| `track_id` | ByteTrack (int, ≥ 0) | ID cục bộ trong 1 camera, 1 session |
| `global_id` | `GlobalIDManager` (str) | ID xuyên camera, ổn định theo gallery |

- **CẤMM** làm mất hoặc nhảy `track_id` khi đối tượng bị che khuất ngắn hạn.
- `global_id` chỉ được tính mỗi `reid_interval` frame hoặc khi xuất hiện ID mới — không tính mỗi frame.
- Khi track mất (`lost_track_ids`), gọi `gid_mgr.forget_track(camera_id, tid)` và xóa khỏi `track_status`.

---

## 8. ADL contract

- **Input:** buffer chuỗi keypoints `seq_len=48` frame, `stride=12`.
- **Output:** pkl file → PoseC3D → `action_label` + `action_score`.
- **CẤAM** chạy ADL blocking trên luồng đọc video chính — phải async hoặc subprocess.
- Engine duy nhất: `src/action/pose_buffer.py` → `src/action/posec3d.py`.
- **TUYỆT ĐỐI KHÔNG** tạo file ADL engine thứ hai ngoài `pose_buffer.py`.

**Status values của `PoseSequenceBuffer.update()`:**

```
collecting    → đang gom skeleton, chưa đủ seq_len
exported      → đã xuất pkl (khi --save-clips)
disabled      → clip_export_disabled (không dùng --save-clips)
inferred      → PoseC3D đã trả nhãn
failed        → PoseC3D subprocess lỗi
skipped       → keypoint shape sai / count sai
waiting       → chưa có keypoints
```

---

## 9. Metrics schema

Mỗi module sinh metrics JSON theo cấu trúc sau (qua `src/core/metrics.py::ModuleMetrics`):

```json
{
  "camera_id": "cam02",
  "module": "tracking",
  "frame_idx": 111,
  "fps": 9.3,
  "device": "cpu",
  "status": "running",
  "message": "OK"
}
```

**Tracking metrics thêm:**
`detections`, `tracked`, `filtered_low_conf`, `filtered_small_box`, `filtered_bad_pose`, `active_track_ids`

**Pose metrics thêm:**
`persons`, `avg_keypoint_score`, `valid_skeletons`, `invalid_skeletons`

**ReID metrics thêm:**
`local_track_id`, `global_id`, `reid_score`, `reid_status`, `gallery_size`

**ADL metrics thêm:**
`track_id`, `global_id`, `sequence_len`, `seq_len_required`, `adl_status`, `action_label`, `action_score`, `exported_pkl`

**Pipeline metrics thêm:**
`persons`, `tracked`, `reid_assigned`, `adl_collecting`, `adl_exported`, `events_per_second`

> Metrics chỉ emit khi có **sự kiện thay đổi trạng thái** (ID mới, action mới, mất track).
> CẤAM spam metrics mỗi frame vào log.

---

## 10. UILogger contract

File: `src/core/ui_logger.py`

```python
logger.log(camera_id, level, module, message, data=None)
# level: "INFO" | "WARNING" | "ERROR" | "METRIC"

logger.metric(camera_id, metrics_dict)
```

**Màu log trên UI:**

| Level | Màu |
|---|---|
| INFO | xanh lá |
| WARNING | vàng |
| ERROR | đỏ |
| METRIC | xanh dương |

**Format dòng log:**

```
[12:33:26] Tracking: persons=2 tracked=2 fps=9.3
[12:33:27] Pose: valid_skeletons=1 avg_kpt=0.72
[12:33:28] ReID: track=1 gid=APhu score=0.81
[12:33:29] ADL: collecting 25/48
[12:33:30] Warning: filtered false person conf=0.34
```

**Log Camera 1 nằm dưới Camera 1, Log Camera 2 nằm dưới Camera 2. CẤAM gom chung.**

---

## 11. FastAPI backend endpoints (main.py)

**Hiện có:**

```
GET  /                          → index.html
GET  /api/cameras               → danh sách camera
POST /api/upload                → upload video
POST /api/save-video/{sid}      → ghi buffer ra MP4
GET  /api/sessions              → danh sách session
WS   /ws/stream/{cam_id}        → stream + modules routing
```

**Cần thêm (nếu bật UILogger):**

```
GET  /api/logs/{camera_id}      → UILogger.get_logs(camera_id)
GET  /api/metrics/{camera_id}   → UILogger.get_metrics(camera_id)
GET  /api/status                → trạng thái tất cả camera
```

**WebSocket events (nếu bật SocketIO):**

```
camera_log      → dòng log theo camera_id
camera_metrics  → metrics JSON theo camera_id
module_status   → trạng thái từng module
```

---

## 12. UI — TUYỆT ĐỐI KHÔNG refactor

- Giữ nguyên Light Theme, layout ngang các nút điều khiển.
- Ô nhập nguồn là `<input type="text">` — không đổi thành `<select>`.
- Log Camera 1 nằm dưới Camera 1. Log Camera 2 nằm dưới Camera 2. CẤAM gom chung.
- Module buttons: khi bấm trên UI, backend chỉ bật module tương ứng:
  - Không chọn module → raw video stream.
  - Tracking → chỉ `run_track.py` logic.
  - Pose → chỉ `run_pose.py` logic.
  - ReID → detect tối thiểu + ReID.
  - ADL → pose tối thiểu + skeleton buffer + ADL.
  - Pipeline → bật tất cả.

---

## 13. Logging rules (chống spam)

- **CẤAM** log mỗi frame: `"Đang xử lý frame 1, 2, 3..."`.
- Log AI chỉ khi có **sự kiện**: ID mới, action mới, mất track, filter nhầm.
- Format bắt buộc: `[Time] [Level] Message`.
- Không log `track_update` event mỗi frame — chỉ log khi `status` thay đổi.

---

## 14. Known bugs (đang track)

| Bug | File | Ghi chú |
|---|---|---|
| Dual ADL classifiers | `run_adl.py` vs `run_pipeline.py` | Chỉ dùng `pose_buffer.py` làm engine |
| FAISS index không rebuild sau EMA update | `src/reid/gallery.py` | Cần rebuild index sau `add_embedding()` |
| `manifest.py` module missing | src/ | Chưa tạo file này |
| detect nhầm người (conf=0.34) | `yolo_pose.py` | Fix bằng `filters.py` + `person_conf=0.60` |

---

## 15. Config chuẩn (pipeline.yaml snippet)

```yaml
tracking:
  enabled: true
  model_type: "pose"
  person_conf: 0.60
  iou: 0.5
  min_box_area: 2500
  min_keypoints: 5
  min_keypoint_score: 0.35
  tracker_yaml: "bytetrack.yaml"

pose:
  enabled: true
  weights: "models/yolo11n-pose.pt"
  conf: 0.60
  iou: 0.5
  min_keypoints: 5
  min_keypoint_score: 0.35

object:
  enabled: false
  weights: "models/yolo11n-pickleball.pt"
  conf: 0.45
  iou: 0.5
  classes: [pickleball, paddle, ball]

ui:
  enabled: true
  log_metrics: true
  max_log_lines: 300
  metrics_interval_frames: 5

output:
  save_video: true
  save_json: false
  short_names: true

logging:
  save_events: false
  level: "INFO"
```

---

## 16. Overlay chuẩn cho mỗi module

**run_track.py:**
```
Module: Pedestrian Tracking
Camera: cam02  |  Frame: 111/213
Detections: 2  |  Tracked: 2  |  Filtered: 1
Device: cpu    |  FPS: 9.3
```

Window title: `"CPose - Pedestrian Tracking"`

**run_pose.py:**
```
Module: YOLO11-Pose
Camera: cam02  |  Frame: 111/213
Persons: 2     |  Valid skeletons: 2  |  Avg kpt: 0.72
Device: cpu    |  FPS: 9.3
```

Window title: `"CPose - Pose Estimation"`

**run_reid.py:**
```
Module: YOLO+ByteTrack+FastReID
Camera: cam02  |  Frame: 111/213
Persons: 2     |  Gallery: 3
Device: cpu    |  FPS: 9.3
```

Window title: `"CPose - ReID"`

**run_adl.py:**
```
Module: YOLO+Pose Buffer+PoseC3D
Camera: cam02  |  Frame: 111/213
Persons: 2     |  Device: cpu  |  FPS: 9.3
ADL: collecting 25/48
```

Window title: `"CPose - ADL"`

**run_pipeline.py:**
```
Module: Full CPose
Camera: cam02  |  Frame: 111/213
Persons: 2     |  Gallery: 3
Device: cpu    |  FPS: 9.3
ADL: collecting 25/48
```

Window title: `"CPose - Full Pipeline"`

---

## 17. Model weights (không push git — xem .gitignore)

| Module | File | Nguồn |
|---|---|---|
| Pose | `models/yolo11n-pose.pt` | Ultralytics HuggingFace |
| Tracking (custom ped) | `models/tracking.pt` | pedestrian-tracking repo |
| ReID | `models/fastreid_market_R50.pth` | JDAI-CV FastReID Model Zoo |
| ADL | `models/posec3d_r50_ntu60.pth` | MMAction2 PoseC3D |
| Object (optional) | `models/yolo11n-pickleball.pt` | **phải train/fine-tune riêng** |

---

*Last updated: theo codebase review + refactor request (2026-01)*