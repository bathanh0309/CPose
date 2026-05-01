# CPose Unified AI Modules Guide

**Dự án:** CPose — Time-First Cross-Camera Pose–ADL–ReID  
**Repo:** <https://github.com/bathanh0309/CPose>  
**Mục tiêu trước mắt:** demo ổn định trên **PC i5**, **4 camera**, **3 người**, dùng short video trong `data-test/`.  
**Mục tiêu sau:** mở rộng RTX/GPU, thay các baseline nhẹ bằng model mạnh hơn.

---

## 0. Quyết định kỹ thuật thống nhất

### 0.1 Chọn YOLOv8 hay YOLOv11?

Để tránh lan man và tránh mỗi module dùng một phiên bản khác nhau, tài liệu này **thống nhất dùng YOLOv8 cho bản demo trước**:

| Hạng mục | Lựa chọn chính cho PC i5 | Lý do |
|---|---|---|
| Human detection | `yolov8n.pt` | nhẹ, ổn định, dễ chạy CPU, tích hợp Ultralytics tốt |
| Pose estimation | `yolov8n-pose.pt` | có sẵn 17 COCO keypoints, dễ chạy, đủ cho demo 3 người |
| Tracking | ByteTrack trong Ultralytics | không cần train, hợp với video ngắn và terminal demo |
| ADL | Rule-based skeleton baseline | nhẹ, giải thích được, dễ debug trước khi train GCN |
| Face detect | SCRFD/RetinaFace ONNX qua InsightFace | chính xác hơn Haar, vẫn có thể chạy CPU nếu chỉ dùng theo track crop |
| Face recognition | ArcFace ONNX | chuẩn công nghiệp, embedding 512-D |
| Face anti-spoofing | MiniFASNet/CDCN ONNX, optional | chỉ bật khi có model; không có thì ghi `spoof_status: unchecked` |
| Cross-camera Global ReID | TFCS-PAR score fusion | đóng góp chính của CPose |

**Không trộn YOLOv8 và YOLOv11 trong bản demo.**  
Nếu sau này deploy RTX, có thể tạo nhánh benchmark riêng so sánh `YOLOv8n` vs `YOLO11n`, nhưng paper/demo chính nên giữ một pipeline nhất quán.

---

## 1. Kiến trúc tổng thể CPose

```text
data-test/
  cam01_YYYY-MM-DD_HH-MM-SS.mp4
  cam02_YYYY-MM-DD_HH-MM-SS.mp4
  cam03_YYYY-MM-DD_HH-MM-SS.mp4
  cam04_YYYY-MM-DD_HH-MM-SS.mp4
        │
        ▼
[1] Human Detection
        │ detections.json
        ▼
[2] Human Tracking
        │ tracks.json
        ▼
[3] Pose Estimation
        │ keypoints.json
        ▼
[4] Skeleton + ADL Recognition
        │ adl_events.json
        ▼
[5] Face Detect + Face Recognition + Anti-Spoofing
        │ face_events.json
        ▼
[6] Cross-Camera Global ReID / TFCS-PAR
        │ reid_tracks.json + global_person_table.json
        ▼
[7] Benchmark + Evaluation
        │ benchmark_summary.json / csv
        ▼
Paper-ready metrics
```

### 1.1 Nguyên tắc quan trọng

1. **Track ID không phải Global ID.**  
   `track_id` chỉ ổn định trong một camera/clip. `global_id` mới là định danh xuyên camera.

2. **Không dùng index detection làm person ID.**  
   Detection index thay đổi theo frame, dễ trộn người.

3. **Không báo accuracy nếu chưa có ground truth.**  
   Nếu không có nhãn chuẩn, ghi `metric_type: proxy`.

4. **Xử lý time-first trước, camera-order sau.**  
   Với 4 camera, clip phải được sort theo timestamp hoặc manifest.

5. **Demo i5 ưu tiên ổn định hơn SOTA.**  
   Dùng model nhẹ, giảm frame rate xử lý, ghi log rõ ràng.

---

## 2. Cấu trúc thư mục đề xuất

```text
Capstone-Project/
├── src/
│   ├── common/
│   │   ├── paths.py
│   │   ├── video_io.py
│   │   ├── schemas.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── manifest.py
│   │   ├── topology.py
│   │   ├── model_registry.py
│   │   └── errors.py
│   │
│   ├── human_detection/
│   │   ├── api.py
│   │   ├── detector_yolov8.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── human_tracking/
│   │   ├── api.py
│   │   ├── tracker_bytetrack.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── pose_estimation/
│   │   ├── api.py
│   │   ├── pose_yolov8.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── adl_recognition/
│   │   ├── api.py
│   │   ├── skeleton_features.py
│   │   ├── adl_rules.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── face/
│   │   ├── api.py
│   │   ├── face_detector.py
│   │   ├── face_recognizer.py
│   │   ├── anti_spoofing.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── global_reid/
│   │   ├── api.py
│   │   ├── body_features.py
│   │   ├── fusion_score.py
│   │   ├── state_machine.py
│   │   ├── global_id_manager.py
│   │   ├── metrics.py
│   │   └── main.py
│   │
│   ├── evaluation/
│   │   ├── detection_eval.py
│   │   ├── tracking_eval.py
│   │   ├── pose_eval.py
│   │   ├── adl_eval.py
│   │   ├── reid_eval.py
│   │   └── main.py
│   │
│   └── pipeline/
│       ├── run_all.py
│       └── benchmark_all.py
│
├── configs/
│   ├── model_registry.demo_i5.yaml
│   ├── model_registry.rtx.yaml
│   ├── multicam_manifest.example.json
│   └── camera_topology.example.yaml
├──data-test/
├── dataset/
│   ├── annotations/
│   └── outputs/
│
├── models/
│   ├── yolo/
│   ├── face/
│   └── reid/
│
├── scripts/
└── README.md
```

---

## 3. Cấu hình demo PC i5

### 3.1 `configs/model_registry.demo_i5.yaml`

```yaml
runtime:
  device: cpu
  demo_mode: true
  target_resolution: 640
  process_every_n_frames: 2
  max_people_per_frame: 3

human_detection:
  backend: ultralytics
  model: models/yolo/yolov8n.pt
  fallback: yolov8n.pt
  imgsz: 640
  conf: 0.50
  iou: 0.50
  class_id: 0

human_tracking:
  backend: ultralytics_bytetrack
  tracker_config: bytetrack.yaml
  min_hits: 3
  max_age: 30
  min_track_quality: 0.45

pose_estimation:
  backend: ultralytics
  model: models/yolo/yolov8n-pose.pt
  fallback: yolov8n-pose.pt
  imgsz: 640
  conf: 0.45
  keypoint_conf: 0.30
  run_on_confirmed_tracks_only: true

adl_recognition:
  method: rule_based
  window_size: 30
  min_visible_keypoints: 8
  smoothing_frames: 7
  labels:
    - standing
    - sitting
    - walking
    - lying_down
    - falling
    - reaching
    - bending
    - unknown

face:
  enabled: true
  run_every_n_frames: 10
  min_face_size: 40
  detector: scrfd_500m
  recognizer: arcface_r100
  anti_spoofing_enabled: false
  anti_spoofing_model: models/face/minifasnet.onnx
  spoof_threshold: 0.50

global_reid:
  method: tfcs_par
  strong_threshold: 0.65
  weak_threshold: 0.45
  confirm_frames: 3
  max_candidate_age_sec: 300
  weights:
    face: 0.30
    body: 0.20
    pose: 0.15
    height: 0.10
    time: 0.15
    topology: 0.10
```

### 3.2 Cấu hình RTX sau này

| Module | PC i5 demo | RTX/GPU sau này |
|---|---|---|
| Human detect | YOLOv8n | YOLOv8s/m hoặc YOLO11n/s ablation |
| Pose | YOLOv8n-pose | RTMPose-m, RTMO-l, YOLOv8s-pose |
| Tracking | ByteTrack | BoT-SORT / Deep OC-SORT |
| ADL | Rule-based | CTR-GCN, BlockGCN, SkateFormer |
| Face | ArcFace CPU ONNX | ArcFace + RetinaFace GPU |
| Anti-spoof | optional | MiniFASNet/CDCN enabled |
| Global ReID | TFCS-PAR baseline | KPR / Pose2ID-assisted fusion |

---

## 4. Chuẩn dữ liệu demo

### 4.1 Video input

```text
data-test/
  cam01_2026-01-29_16-26-25.mp4
  cam02_2026-01-29_16-26-40.mp4
  cam03_2026-01-29_16-26-55.mp4
  cam04_2026-01-29_16-27-20.mp4
```

### 4.2 Manifest input

Nên dùng manifest để không phụ thuộc tuyệt đối vào tên file:

```json
[
  {
    "video": "cam01_2026-01-29_16-26-25.mp4",
    "camera_id": "cam01",
    "start_time": "2026-01-29T16:26:25+07:00",
    "fps": 30,
    "location": "entrance",
    "timezone": "Asia/Ho_Chi_Minh"
  },
  {
    "video": "cam02_2026-01-29_16-26-40.mp4",
    "camera_id": "cam02",
    "start_time": "2026-01-29T16:26:40+07:00",
    "fps": 30,
    "location": "corridor",
    "timezone": "Asia/Ho_Chi_Minh"
  }
]
```

### 4.3 Camera topology

```yaml
transitions:
  - from: cam01
    to: cam02
    min_sec: 0
    max_sec: 60
    from_zone: exit_right
    to_zone: entry_left
    confidence: 1.0

  - from: cam02
    to: cam03
    min_sec: 0
    max_sec: 60
    from_zone: exit_right
    to_zone: entry_left
    confidence: 1.0

  - from: cam03
    to: cam04
    min_sec: 20
    max_sec: 180
    from_zone: elevator
    to_zone: elevator
    confidence: 0.9

  - from: cam04
    to: cam04
    min_sec: 5
    max_sec: 300
    from_zone: room_door
    to_zone: room_door
    confidence: 0.7
```

---

# PART A — MODULE SPECIFICATION

---

## 5. Module Human Detection

### 5.1 Mục tiêu

Phát hiện người trong từng frame video. Chỉ lấy class `person`.

### 5.2 Model thống nhất

| Mục | Giá trị |
|---|---|
| Model | `yolov8n.pt` |
| Class | COCO class `0 = person` |
| Input | video `.mp4` |
| Output | bbox người theo frame |
| Demo i5 | chạy mỗi 2 frame hoặc resize 640 |
| RTX sau | có thể nâng lên YOLOv8s/m |

### 5.3 Công thức

**Bounding box:**

```math
B_i = (x_1, y_1, x_2, y_2, c_i)
```

**IoU giữa hai box:**

```math
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}
```

**Precision / Recall / F1:**

```math
Precision = \frac{TP}{TP + FP}
```

```math
Recall = \frac{TP}{TP + FN}
```

```math
F1 = \frac{2PR}{P+R}
```

### 5.4 Output JSON

```json
[
  {
    "frame_id": 0,
    "timestamp_sec": 0.0,
    "camera_id": "cam01",
    "detections": [
      {
        "bbox": [320, 120, 520, 680],
        "confidence": 0.91,
        "class_id": 0,
        "class_name": "person",
        "failure_reason": "OK"
      }
    ]
  }
]
```

### 5.5 Metrics

| Metric | Có ground truth | Không ground truth |
|---|---:|---:|
| Precision | yes | no |
| Recall | yes | no |
| F1 | yes | no |
| mAP@50 | yes | no |
| total_detections | yes | yes |
| avg_confidence | yes | yes |
| detection_fps | yes | yes |
| metric_type | `ground_truth` | `proxy` |

---

## 6. Module Human Tracking

### 6.1 Mục tiêu

Gán `track_id` ổn định cho mỗi người trong từng camera/clip.

### 6.2 Model/thuật toán

| Mục | Giá trị |
|---|---|
| Tracker mặc định | ByteTrack |
| Input | `detections.json` hoặc video trực tiếp |
| Output | `tracks.json`, overlay video |
| Dùng cho | ADL window, Global ID candidate |

ByteTrack phù hợp demo vì liên kết cả detection confidence thấp để giảm fragmentation trong tracking-by-detection.

### 6.3 Công thức

**Cost IoU:**

```math
C_{iou}(i,j)=1-IoU(B_i^{track},B_j^{det})
```

**EMA cho feature track nếu có appearance:**

```math
e_t = \alpha e_{t-1} + (1-\alpha)f_t
```

**Track quality score đề xuất:**

```math
Q_{track}=0.5\bar{c}+0.3\min(\frac{age}{W},1)-0.2\min(\frac{misses}{max\_age},1)
```

Trong đó:

- \(\bar{c}\): confidence trung bình
- `age`: số frame track tồn tại
- `misses`: số frame mất detection
- \(W\): ADL window, ví dụ 30

### 6.4 Output JSON

```json
[
  {
    "frame_id": 30,
    "timestamp_sec": 1.0,
    "camera_id": "cam01",
    "tracks": [
      {
        "track_id": 1,
        "bbox": [320, 120, 520, 680],
        "confidence": 0.88,
        "class_name": "person",
        "age": 30,
        "hits": 28,
        "misses": 2,
        "is_confirmed": true,
        "fragment_count": 0,
        "quality_score": 0.84,
        "failure_reason": "OK"
      }
    ]
  }
]
```

### 6.5 Metrics

| Metric | Ý nghĩa |
|---|---|
| total_tracks | tổng track sinh ra |
| active_track_count_mean | số track trung bình/frame |
| track_fragmentation_proxy | số lần track bị đứt ước lượng |
| id_switch_count | cần GT |
| IDF1 | cần GT |
| tracking_fps | tốc độ module |

---

## 7. Module Pose Estimation

### 7.1 Mục tiêu

Trích xuất skeleton COCO-17 cho từng người đã tracking.

### 7.2 Model thống nhất

| Mục | Giá trị |
|---|---|
| Model | `yolov8n-pose.pt` |
| Keypoints | COCO-17 |
| Input | video + tracks |
| Output | `keypoints.json` |
| Demo i5 | chỉ chạy trên confirmed track, mỗi 2 frame nếu cần |

### 7.3 COCO-17 keypoints

| ID | Keypoint | ID | Keypoint |
|---:|---|---:|---|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow |  |  |

### 7.4 Công thức

**Keypoint:**

```math
K_i=(x_i,y_i,c_i)
```

**Visible keypoint:**

```math
visible(K_i)=
\begin{cases}
1, & c_i \geq \tau_{kp}\\
0, & c_i < \tau_{kp}
\end{cases}
```

**Visible ratio:**

```math
r_{visible}=\frac{1}{17}\sum_{i=0}^{16}visible(K_i)
```

**PCK nếu có ground truth:**

```math
PCK@a = \frac{1}{N}\sum_i \mathbb{1}\left(\frac{||\hat{K}_i-K_i||_2}{s} < a\right)
```

### 7.5 Output JSON

```json
[
  {
    "frame_id": 30,
    "timestamp_sec": 1.0,
    "camera_id": "cam01",
    "persons": [
      {
        "track_id": 1,
        "bbox": [320, 120, 520, 680],
        "visible_keypoint_ratio": 0.82,
        "keypoints": [
          {"id": 0, "name": "nose", "x": 421.0, "y": 150.0, "confidence": 0.91}
        ],
        "failure_reason": "OK"
      }
    ]
  }
]
```

---

## 8. Module Skeleton Feature Extraction

### 8.1 Mục tiêu

Từ COCO-17 keypoints, trích xuất đặc trưng hình học cho ADL và Global ReID.

### 8.2 Skeleton feature đề xuất

| Feature | Công thức/ý nghĩa | Dùng cho |
|---|---|---|
| torso_angle | độ nghiêng thân | ADL |
| knee_angle | góc gối | sitting/falling |
| ankle_velocity | vận tốc cổ chân | walking/falling |
| bbox_aspect_ratio | rộng/cao bbox | lying_down |
| height_ratio | chiều cao tương đối | Global ReID |
| shoulder_width_ratio | tỉ lệ vai | ReID phụ |
| visible_ratio | chất lượng skeleton | quality gate |
| gait_signature | thống kê chuyển động cổ chân/hông | Global ReID |

### 8.3 Công thức góc khớp

Với ba điểm \(p_1, v, p_2\):

```math
\theta = \cos^{-1}\left(
\frac{(p_1-v)\cdot(p_2-v)}
{||p_1-v||||p_2-v||}
\right)
```

### 8.4 Vận tốc cổ chân

```math
v_{ankle}=\frac{1}{W-1}\sum_{t=2}^{W}||A_t-A_{t-1}||_2
```

Trong đó \(W\) là sliding window, mặc định 30 frame.

### 8.5 Chiều cao tương đối

```math
h_{rel}=\frac{y_{ankle}-y_{head}}{H_{frame}}
```

### 8.6 Gait signature đơn giản

```math
g = [\mu(v_{ankle}), \sigma(v_{ankle}), \mu(v_{hip}), \sigma(v_{hip}), h_{rel}]
```

---

## 9. Module ADL Recognition

### 9.1 Mục tiêu

Nhận dạng hành vi sinh hoạt hằng ngày từ skeleton sequence.

### 9.2 Class demo

| Label | Ý nghĩa |
|---|---|
| standing | đứng |
| sitting | ngồi |
| walking | đi |
| lying_down | nằm |
| falling | té/ngã |
| reaching | với tay |
| bending | cúi người |
| unknown | không đủ dữ liệu |

### 9.3 Rule-based ADL baseline

Rule baseline dùng cho demo i5:

```text
if visible_keypoints < 8:
    unknown

elif torso_angle > 68 and ankle_velocity high:
    falling

elif torso_angle > 68 and bbox_aspect_ratio > 1.15:
    lying_down

elif knee_angle < 150 and ankle_velocity low:
    sitting

elif torso_angle > 45 and ankle_velocity low:
    bending

elif wrist_above_shoulder:
    reaching

elif ankle_velocity > threshold:
    walking

else:
    standing
```

### 9.4 Temporal smoothing

```math
\hat{y}_t = mode(y_{t-k},...,y_t)
```

Dùng majority vote 5–7 frame để giảm label flickering.

### 9.5 Output JSON

```json
[
  {
    "frame_id": 60,
    "timestamp_sec": 2.0,
    "camera_id": "cam01",
    "track_id": 1,
    "raw_label": "walking",
    "smoothed_label": "walking",
    "adl_label": "walking",
    "confidence": 0.79,
    "window_size": 30,
    "visible_keypoint_ratio": 0.82,
    "failure_reason": "OK"
  }
]
```

### 9.6 Metrics

| Metric | Ý nghĩa |
|---|---|
| total_adl_events | tổng event ADL |
| class_distribution | phân bố nhãn |
| unknown_rate | tỉ lệ unknown |
| avg_confidence | confidence trung bình |
| macro_f1 | cần ground truth |
| confusion_matrix | cần ground truth |

---

## 10. Module Face Detection

### 10.1 Mục tiêu

Phát hiện và căn chỉnh khuôn mặt trong crop người để hỗ trợ Global ID.

### 10.2 Lựa chọn demo

| Thành phần | Demo i5 |
|---|---|
| Detector | SCRFD/RetinaFace ONNX qua InsightFace |
| Chạy mỗi | 10 frame/track |
| Input | crop người từ bbox track |
| Output | face bbox + landmarks |

Không nên chạy face detect full frame mọi frame trên i5. Nên chạy theo track crop và giảm tần suất.

### 10.3 Face alignment

Dùng 5 landmarks: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải.

Similarity transform:

```math
x' = sRx + t
```

Sau alignment, resize về `112x112` cho ArcFace.

### 10.4 Output JSON

```json
{
  "frame_id": 120,
  "camera_id": "cam01",
  "track_id": 1,
  "face_detected": true,
  "face_bbox": [350, 130, 440, 230],
  "landmarks": [[370, 160], [415, 160], [392, 182], [375, 210], [410, 210]],
  "face_quality": 0.78,
  "failure_reason": "OK"
}
```

---

## 11. Module Face Recognition

### 11.1 Mục tiêu

Trích xuất embedding khuôn mặt và so khớp người.

### 11.2 Model

| Mục | Giá trị |
|---|---|
| Model | ArcFace ONNX |
| Embedding | 512-D |
| Similarity | cosine |
| Dùng trong CPose | evidence phụ cho Global ID |

### 11.3 ArcFace loss tham khảo

ArcFace dùng additive angular margin:

```math
L=-\frac{1}{N}\sum_{i=1}^{N}\log
\frac{e^{s\cos(\theta_{y_i}+m)}}
{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}}
```

### 11.4 Cosine similarity

```math
S_{face}(a,b)=\frac{f_a\cdot f_b}{||f_a||||f_b||}
```

Nếu embedding đã L2-normalized:

```math
S_{face}(a,b)=f_a\cdot f_b
```

### 11.5 Quy tắc dùng face

| Case | Quy tắc |
|---|---|
| mặt rõ | dùng face score mạnh |
| không thấy mặt | `score_face = null`, không fail |
| spoof risk | không update identity bằng face |
| quay lưng | dùng body + pose + time + topology |

---

## 12. Module Face Anti-Spoofing

### 12.1 Mục tiêu

Ngăn ảnh in/video replay được dùng để đăng ký hoặc cập nhật face embedding.

### 12.2 Demo i5

Face anti-spoofing nên để optional:

```yaml
anti_spoofing_enabled: false
```

Khi có model ONNX:

```yaml
anti_spoofing_enabled: true
spoof_threshold: 0.50
```

### 12.3 Output

```json
{
  "track_id": 1,
  "face_live_score": 0.83,
  "spoof_status": "live",
  "anti_spoofing_model": "MiniFASNet",
  "failure_reason": "OK"
}
```

Nếu chưa bật:

```json
{
  "spoof_status": "unchecked",
  "failure_reason": "ANTI_SPOOF_DISABLED"
}
```

### 12.4 Metric

| Metric | Cần dataset |
|---|---|
| APCER | Face anti-spoofing GT |
| BPCER | Face anti-spoofing GT |
| ACER | Face anti-spoofing GT |
| live/spoof count | không cần GT |

---

## 13. Module Body Appearance ReID

### 13.1 Mục tiêu

Tạo feature ngoại hình cơ thể để hỗ trợ Global ID khi face yếu.

### 13.2 Demo i5 feature nhẹ

Không cần model deep ReID ngay. Dùng:

| Feature | Kích thước |
|---|---:|
| HSV histogram head/body/legs | 168 |
| Hu moments | 7 |
| aspect ratio | 1 |
| height ratio | 1 |
| Tổng | ~177 |

### 13.3 HSV body split

```text
person crop
  head: 0%–20%
  body: 20%–70%
  legs: 70%–100%
```

### 13.4 Similarity

```math
S_{body}=\frac{h_a\cdot h_b}{||h_a||||h_b||}
```

### 13.5 Lưu ý

Body appearance dễ sai khi:

- thay áo;
- ánh sáng khác camera;
- crop bị che;
- người quay lưng.

Do đó body không được là tín hiệu duy nhất trong Global ID.

---

## 14. Module Cross-Camera Global ReID / TFCS-PAR

### 14.1 Mục tiêu

Gán Global ID ổn định xuyên 4 camera.

### 14.2 Input

| Input | Từ module |
|---|---|
| `tracks.json` | tracking |
| `keypoints.json` | pose |
| `adl_events.json` | ADL |
| `face_events.json` | face |
| `camera_topology.yaml` | config |
| `multicam_manifest.json` | config |

### 14.3 Công thức score tổng

```math
S_{total}=w_fS_{face}+w_bS_{body}+w_pS_{pose}+w_hS_{height}+w_tS_{time}+w_cS_{camera}
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| \(S_{face}\) | cosine similarity ArcFace |
| \(S_{body}\) | cosine similarity body feature |
| \(S_{pose}\) | similarity gait/pose signature |
| \(S_{height}\) | tương đồng chiều cao tương đối |
| \(S_{time}\) | hợp lệ theo transition window |
| \(S_{camera}\) | hợp lệ theo topology camera |

### 14.4 Time-window gating

```math
G_{time}(i,j)=
\begin{cases}
1, & \Delta t_{ij}\in[T_{min}^{i\rightarrow j},T_{max}^{i\rightarrow j}]\\
0, & otherwise
\end{cases}
```

### 14.5 Camera topology gating

```math
G_{camera}(c_i,c_j)=
\begin{cases}
1, & (c_i,c_j)\in E_{topology}\\
0, & otherwise
\end{cases}
```

### 14.6 Height similarity

```math
S_{height}=1-\min\left(\frac{|h_a-h_b|}{\tau_h},1\right)
```

### 14.7 Pose/gait similarity

```math
S_{pose}=\frac{g_a\cdot g_b}{||g_a||||g_b||}
```

### 14.8 Quyết định Global ID

```math
ID =
\begin{cases}
GID_{old}, & S_{total} \geq \tau_{strong}\\
GID_{old}^{soft}, & \tau_{weak}\leq S_{total}<\tau_{strong}\\
UNK, & S_{total}<\tau_{weak}
\end{cases}
```

Mặc định:

```yaml
strong_threshold: 0.65
weak_threshold: 0.45
confirm_frames: 3
```

### 14.9 State machine

| State | Ý nghĩa |
|---|---|
| ACTIVE | đang thấy người |
| PENDING_TRANSFER | vừa rời camera qua exit zone |
| IN_BLIND_ZONE | đang trong vùng mù như thang máy |
| IN_ROOM | vào phòng, có thể quay ra cùng camera |
| CLOTHING_CHANGE_SUSPECTED | body thay đổi mạnh sau khi vào phòng |
| DORMANT | mất lâu nhưng chưa đóng |
| CLOSED | kết thúc track |

### 14.10 Logic khi thay áo

Khi state là `IN_ROOM` hoặc `CLOTHING_CHANGE_SUSPECTED`:

```text
giảm weight body appearance
tăng weight time/topology/height/pose
face nếu có thì dùng, không bắt buộc
không tạo GID mới ngay nếu vẫn trong transition hợp lệ
```

### 14.11 Output JSON

```json
{
  "frame_id": 120,
  "timestamp_sec": 4.0,
  "camera_id": "cam03",
  "local_track_id": 2,
  "global_id": "GID-001",
  "state": "ACTIVE",
  "match_status": "strong_match",
  "score_total": 0.72,
  "score_face": null,
  "score_body": 0.51,
  "score_pose": 0.68,
  "score_height": 0.81,
  "score_time": 1.0,
  "score_topology": 1.0,
  "topology_allowed": true,
  "delta_time_sec": 42.0,
  "failure_reason": "OK"
}
```

---

# PART B — BENCHMARK & EVALUATION

---

## 15. Benchmark cần có cho demo

### 15.1 Benchmark runtime

| Module | Metric bắt buộc |
|---|---|
| Detection | FPS, latency/frame, total detections |
| Tracking | FPS, total tracks, fragmentation proxy |
| Pose | FPS, pose instances, visible ratio |
| ADL | events/sec, unknown rate, class distribution |
| Face | face detect rate, face embedding count |
| Global ReID | GID count, transfer count, conflict count |
| End-to-end | total runtime, end-to-end FPS |

### 15.2 Benchmark thật nếu có annotation

| Module | Metric thật |
|---|---|
| Detection | Precision, Recall, F1, mAP@50 |
| Tracking | IDF1, ID Switch, Fragmentation |
| Pose | PCK, missing keypoint rate |
| ADL | Accuracy, Macro-F1, confusion matrix |
| Global ReID | Global ID Accuracy, Cross-camera IDF1, False Split, False Merge |

### 15.3 Log terminal mẫu

```text
============================================================
CPose Pipeline Run
============================================================
Input folder : data-test/
Videos       : 4
People       : expected 3
Device       : CPU i5
Model        : YOLOv8n / YOLOv8n-pose

[1/4] cam01_2026-01-29_16-26-25.mp4
Detection FPS : 12.4
Tracking FPS  : 15.8
Pose FPS      : 7.2
ADL unknown   : 8.3%
Global IDs    : 3

Saved:
dataset/outputs/pipeline/2026-04-30_20-10-15/
============================================================
```

---

## 16. Paper-ready benchmark table template

### 16.1 Dataset summary

| Item | Value |
|---|---:|
| Cameras | 4 |
| Subjects | 3 |
| Total videos | TODO |
| Total duration | TODO |
| Resolution | TODO |
| FPS | TODO |
| ADL classes | 8 |
| Blind-zone scenarios | elevator, room, door |
| Clothing-change cases | TODO |
| No-face cases | TODO |

### 16.2 Module runtime results

| Module | Model/Method | Device | FPS | Latency ms/frame | Output |
|---|---|---|---:|---:|---|
| Detection | YOLOv8n | i5 CPU | TODO | TODO | detections.json |
| Tracking | ByteTrack | i5 CPU | TODO | TODO | tracks.json |
| Pose | YOLOv8n-pose | i5 CPU | TODO | TODO | keypoints.json |
| ADL | Rule-based | i5 CPU | TODO | TODO | adl_events.json |
| Global ReID | TFCS-PAR | i5 CPU | TODO | TODO | reid_tracks.json |

### 16.3 Global ID results

| Scenario | Metric | Value |
|---|---|---:|
| cam01 → cam02 | Transfer Success Rate | TODO |
| cam02 → cam03 | Transfer Success Rate | TODO |
| cam03 → elevator → cam04 | Blind-zone Recovery Rate | TODO |
| cam04 → room → cam04 | Room Re-entry Accuracy | TODO |
| clothing change | ID Preservation | TODO |
| all | ID Switch | TODO |
| all | False Split Rate | TODO |
| all | False Merge Rate | TODO |

---

# PART C — DATASETS TO DOWNLOAD

---

## 17. Dataset ưu tiên cho laptop

### 17.1 Dataset tự thu của CPose

Đây là quan trọng nhất cho demo và paper.

| Bộ dữ liệu | Mục tiêu | Số lượng tối thiểu |
|---|---|---:|
| CPose-4Cam-3Person | test Global ID | 4 cam × 3 người |
| CPose-ADL-Short | test ADL | 10–30 clip/class |
| CPose-ClothingChange | test thay áo | 5–10 sequence |
| CPose-NoFace | test quay lưng/mất mặt | 5–10 sequence |
| CPose-Occclusion | test che khuất | 5–10 sequence |

### 17.2 Dataset ngoài nên tải

| Nhóm | Dataset | Dùng cho | Link |
|---|---|---|---|
| Pose | COCO Keypoints | pose benchmark | <https://cocodataset.org> |
| Tracking | MOT17 | tracking benchmark | <https://motchallenge.net/data/MOT17> |
| ReID | Market-1501 | body ReID | <https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html> |
| ReID | DukeMTMC-reID | body ReID | thường qua Kaggle/Mirror |
| Occluded ReID | Occluded-Duke | occlusion ReID | paper/project page |
| Face recognition | LFW | face verification | <http://vis-www.cs.umass.edu/lfw> |
| Anti-spoof | CASIA-FASD | spoof/live | <http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.html> |
| Anti-spoof | Replay-Attack | spoof/live | <https://www.idiap.ch/en/dataset/replayattack> |
| ADL | Toyota Smarthome | ADL | <https://project.inria.fr/toyotasmarthome> |
| Fall/ADL | UR Fall | fall detection | <http://fenix.ur.edu.pl/~mkepski/ds/uf.html> |

### 17.3 Dataset không nên tải ngay cho demo i5

| Dataset | Lý do chưa ưu tiên |
|---|---|
| NTU RGB+D 120 full | rất lớn, không cần cho demo ngắn |
| Human3.6M full | nặng, 3D pose research, không cần runtime |
| CrowdHuman full | detection benchmark lớn, chưa cần |
| CelebA-Spoof full | lớn, chỉ cần nếu làm anti-spoof riêng |

---

# PART D — REFERENCE NUMBERS

---

## 18. Bảng số liệu từ nghiên cứu liên quan

### 18.1 Pose / ADL / Tracking

| Paper/Method | Dataset/Setup | Metric | Reported value |
|---|---|---|---:|
| RTMPose-m | COCO | AP | 75.8 |
| RTMPose-m | Intel i7 CPU | FPS | 90+ |
| RTMPose-m | GTX 1660Ti | FPS | 430+ |
| RTMPose-s | Snapdragon 865 | FPS | 70+ |
| RTMO-l | COCO val2017 | AP | 74.8 |
| RTMO-l | V100 GPU | FPS | 141 |
| RTMO-l | CrowdPose | AP | 73.2 |
| MotionBERT | NTU-60 XSub | Top-1 | 97.2% |
| π-ViT | Toyota Smarthome CS | mCA | 72.9% |
| π-ViT | NTU-60 | Top-1 | 94.0% |
| ByteTrack | MOT17 | HOTA | 63.1 |
| BoT-SORT | MOT17 | HOTA | 65.0 |
| Pose-assisted MCPT | AI City 2023 | IDF1 | 86.76% |

### 18.2 Human-centric unified models

| Paper | Scale/Result | Ý nghĩa cho CPose |
|---|---|---|
| HumanBench | 37 datasets, 11,019,187 images | chứng minh human-centric pretraining đa nhiệm hữu ích |
| UniHCP | 33 datasets, 5 tasks, 99.97% shared params | hướng dài hạn: unify pose/detect/ReID/attribute |
| UniHCP | Market1501 ReID mAP 90.3 | mốc tham khảo ReID mạnh |
| UniHCP | PA-100K mA 86.18 | mốc attribute recognition |
| UniHCP | CrowdHuman JI 85.8 | mốc pedestrian detection |

---

# PART E — GITHUB REPOS

---

## 19. Repo nên dùng

| Module | Repo | Link |
|---|---|---|
| YOLO detection/pose | Ultralytics | <https://github.com/ultralytics/ultralytics> |
| ByteTrack | FoundationVision | <https://github.com/FoundationVision/ByteTrack> |
| BoT-SORT | NirAharon | <https://github.com/NirAharon/BoT-SORT> |
| BoxMOT | mikel-brostrom | <https://github.com/mikel-brostrom/boxmot> |
| InsightFace | deepinsight | <https://github.com/deepinsight/insightface> |
| ArcFace | deepinsight | <https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch> |
| MiniFASNet | Silent-Face-Anti-Spoofing | <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing> |
| RTMPose/RTMO | MMPose | <https://github.com/open-mmlab/mmpose> |
| rtmlib | RTMPose inference | <https://github.com/Tau-J/rtmlib> |
| CTR-GCN | Uason-Chen | <https://github.com/Uason-Chen/CTR-GCN> |
| BlockGCN | ZhouYuxuanYX | <https://github.com/ZhouYuxuanYX/BlockGCN> |
| SkateFormer | KAIST-VICLab | <https://github.com/KAIST-VICLab/SkateFormer> |
| MotionBERT | Walter0807 | <https://github.com/Walter0807/MotionBERT> |
| KPR ReID | VlSomers | <https://github.com/VlSomers/keypoint_promptable_reidentification> |
| Pose2ID | yuanc3 | <https://github.com/yuanc3/Pose2ID> |

---

# PART F — CLI COMMANDS

---

## 20. Chạy từng module

```bash
python -m src.human_detection.main \
  --input data-test/ \
  --output data/outputs/1_detection \
  --config configs/model_registry.demo_i5.yaml
```

```bash
python -m src.human_tracking.main \
  --input data-test/ \
  --detections data/outputs/1_detection \
  --output data/outputs/2_tracking \
  --config configs/model_registry.demo_i5.yaml
```

```bash
python -m src.pose_estimation.main \
  --input data-test/ \
  --tracks data/outputs/2_tracking \
  --output data/outputs/3_pose \
  --config configs/model_registry.demo_i5.yaml
```

```bash
python -m src.adl_recognition.main \
  --pose-dir data/outputs/3_pose \
  --output data/outputs/4_adl \
  --config configs/model_registry.demo_i5.yaml
```

```bash
python -m src.face.main \
  --input data-test/ \
  --tracks data/outputs/2_tracking \
  --output data/outputs/5_face \
  --config configs/model_registry.demo_i5.yaml
```

```bash
python -m src.global_reid.main \
  --tracks data/outputs/2_tracking \
  --pose data/outputs/3_pose \
  --adl data/outputs/4_adl \
  --face data/outputs/5_face \
  --manifest configs/multicam_manifest.example.json \
  --topology configs/camera_topology.example.yaml \
  --output data/outputs/6_reid \
  --config configs/model_registry.demo_i5.yaml
```

## 21. Chạy full pipeline

```bash
python -m src.pipeline.run_all \
  --input data-test/ \
  --output data/outputs \
  --config configs/model_registry.demo_i5.yaml \
  --manifest configs/multicam_manifest.example.json \
  --topology configs/camera_topology.example.yaml
```

## 22. Chạy benchmark

```bash
python -m src.pipeline.benchmark_all \
  --run-dir data/outputs/pipeline/<timestamp>
```

## 23. Chạy evaluation nếu có ground truth

```bash
python -m src.evaluation.main \
  --outputs data/outputs/pipeline/<timestamp> \
  --gt data/annotations \
  --out data/outputs/pipeline/<timestamp>/evaluation
```

---

# PART G — ERROR TAXONOMY

---

## 24. Failure reasons chuẩn

| Code | Ý nghĩa |
|---|---|
| OK | không lỗi |
| NO_PERSON_DETECTED | không phát hiện người |
| LOW_DETECTION_CONFIDENCE | detection confidence thấp |
| TRACK_FRAGMENTED | track bị đứt |
| UNCONFIRMED_TRACK | track chưa đủ tin cậy |
| SHORT_TRACK_WINDOW | chưa đủ window cho ADL |
| LOW_KEYPOINT_VISIBILITY | thiếu keypoint |
| NO_FACE | không thấy mặt |
| FACE_SPOOF_RISK | nghi spoof |
| BODY_OCCLUDED | body bị che |
| TOPOLOGY_CONFLICT | chuyển camera không hợp topology |
| TIME_WINDOW_CONFLICT | lệch transition window |
| MULTI_CANDIDATE_CONFLICT | nhiều candidate giống nhau |
| MODEL_NOT_FOUND | không thấy model |
| INVALID_VIDEO | video lỗi |
| INVALID_MANIFEST | manifest lỗi |
| INVALID_TOPOLOGY | topology lỗi |
| ANTI_SPOOF_DISABLED | chưa bật anti-spoof |

---

# PART H — ROADMAP

---

## 25. Roadmap demo trước, research sau

### Phase 1 — Demo i5 ổn định

| Việc | Mục tiêu |
|---|---|
| YOLOv8n detection | chạy được 4 short videos |
| ByteTrack | track 3 người ổn định |
| YOLOv8n-pose | có skeleton overlay |
| Rule ADL | nhận standing/sitting/walking/unknown |
| TFCS-PAR | giữ GID qua 4 camera |
| Benchmark proxy | có FPS, latency, unknown rate |

### Phase 2 — Có số liệu paper

| Việc | Mục tiêu |
|---|---|
| Annotate 4 camera × 3 người | có GT Global ID |
| Annotate ADL events | tính Macro-F1 |
| Annotate bbox vài clip | tính detection F1 |
| Evaluate Global ReID | tính ID switch, false split, false merge |
| Ablation | bỏ face/body/pose/topology để so sánh |

### Phase 3 — RTX/GPU nâng cấp

| Việc | Mục tiêu |
|---|---|
| RTMPose/RTMO | tăng pose accuracy/FPS |
| BoT-SORT/Deep OC-SORT | tracking robust hơn |
| CTR-GCN/BlockGCN | ADL học sâu |
| KPR/Pose2ID | occluded ReID |
| MiniFASNet/CDCN | chống spoof thật |

---

# PART I — REFERENCES

---

## 26. Tài liệu tham khảo chính

[1] Ultralytics YOLOv8 Documentation and Repository. <https://github.com/ultralytics/ultralytics>  

[2] Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022. <https://github.com/FoundationVision/ByteTrack>  

[3] J. Kim et al., "Addressing the Occlusion Problem in Multi-Camera People Tracking with Human Pose Estimation", CVPR Workshops 2023. Reported IDF1: 86.76%.  

[4] J. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.  

[5] M. Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose", 2023.  

[6] P. Lu et al., "RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation", CVPR 2024.  

[7] W. Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations", ICCV 2023.  

[8] D. Reilly et al., "Just Add π! Pose Induced Video Transformers for Understanding Activities of Daily Living", CVPR 2024.  

[9] S. Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018.  

[10] Y. Chen et al., "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition", ICCV 2021.  

[11] Y. Zhou et al., "BlockGCN: Redefining Topology Awareness for Skeleton-Based Action Recognition", CVPR 2024.  

[12] V. Somers et al., "Keypoint Promptable Re-Identification", ECCV 2024. <https://github.com/VlSomers/keypoint_promptable_reidentification>  

[13] C. Yuan et al., "From Poses to Identity: Training-Free Person Re-Identification via Feature Centralization", CVPR 2025. <https://github.com/yuanc3/Pose2ID>  

[14] S. Tang et al., "HumanBench: Towards General Human-centric Perception with Projector Assisted Pretraining", CVPR 2023.  

[15] Y. Ci et al., "UniHCP: A Unified Model for Human-Centric Perceptions", CVPR 2023.  

[16] L. Zhou et al., "Human Pose-based Estimation, Tracking and Action Recognition with Deep Learning: A Survey", 2023.  

---

## 27. Checklist trước khi demo

- [ ] 4 video đặt trong `data-test/`
- [ ] Manifest đúng timestamp
- [ ] Topology đúng thứ tự camera
- [ ] `yolov8n.pt` và `yolov8n-pose.pt` tải sẵn
- [ ] Chạy được detection overlay
- [ ] Chạy được tracking overlay
- [ ] Chạy được pose overlay
- [ ] Chạy được ADL overlay
- [ ] Chạy được Global ID overlay
- [ ] Có `benchmark_summary.json`
- [ ] Không báo accuracy nếu chưa có ground truth
- [ ] Có video final cho từng camera
- [ ] Có log terminal rõ ràng

---

## 28. Kết luận thiết kế

Với cấu hình hiện tại, CPose nên được trình bày là:

> Một hệ thống giám sát đa camera chạy bằng terminal, dùng YOLOv8n/YOLOv8n-pose để phát hiện người và ước lượng tư thế, ByteTrack để duy trì local track, rule-based skeleton ADL để nhận diện hành vi nhẹ, ArcFace làm tín hiệu nhận dạng mặt phụ trợ, và TFCS-PAR để hợp nhất Global ID xuyên camera dựa trên face, body, pose, height, time và camera topology.

Điểm nghiên cứu không nằm ở việc thay YOLO bằng model mới hơn, mà nằm ở:

- xử lý time-first;
- giữ Global ID xuyên camera;
- xử lý blind-zone/room/clothing-change;
- có benchmark và failure reason rõ ràng;
- phân biệt proxy metrics và ground-truth metrics.
