# TOTAL.md — Tổng hợp đầy đủ thông số, benchmark, bảng so sánh và công thức cho paper CPose

> Tài liệu này là checklist tổng hợp để biến CPose từ một demo/app thành một đề tài nghiên cứu có cấu trúc rõ ràng. Nội dung tập trung vào những gì cần có khi viết bài báo về CPose: định vị nghiên cứu, thông số hệ thống, benchmark, bảng so sánh, công thức, ablation study, roadmap thực nghiệm và cách trả lời khi thầy/sếp/reviewer hỏi.

---

## 0. Định vị nghiên cứu CPose

### 0.1. Không nên định vị CPose là gì?

Không nên trình bày CPose đơn thuần là:

- một app dashboard nhận diện người;
- một demo dùng YOLO-Pose;
- một hệ thống vẽ skeleton lên video;
- một pipeline ghép detection + pose + ReID;
- một phần mềm giám sát camera thông thường.

Nếu trình bày như vậy, CPose dễ bị đánh giá là tích hợp model có sẵn, chưa có đóng góp nghiên cứu rõ ràng.

### 0.2. Nên định vị CPose là gì?

Nên định vị CPose là:

**CPose: A Time-First Cross-Camera Pose–ADL–ReID Framework for Sequential Multi-Camera Human Activity Monitoring**

Hoặc tiếng Việt:

**CPose: Khung suy luận không gian–thời gian theo thứ tự thời gian cho nhận diện tư thế, hoạt động và định danh xuyên camera trong hệ thống giám sát đa camera.**

Điểm nhấn không nằm ở YOLO-Pose, mà nằm ở:

1. xử lý nhiều camera theo trật tự thời gian;
2. giữ Global ID xuyên camera;
3. dùng topology camera và transition window;
4. dùng pose, ADL, face, body appearance, time và topology để giảm ID switch;
5. xử lý các vùng mù như thang máy, phòng kín, cửa ra/vào;
6. đánh giá bằng benchmark định lượng, không chỉ demo giao diện.

### 0.3. Câu thesis nên dùng cho paper

> CPose không chỉ phát hiện pose, mà giữ được danh tính và hoạt động của con người xuyên camera bằng suy luận không gian–thời gian, đặc biệt trong các tình huống vùng mù, mất mặt và thay đổi ngoại hình.

---

## 1. Các thông số hệ thống bắt buộc phải báo cáo

### 1.1. Dataset Summary

Bảng này bắt buộc phải có trong paper.

| Nhóm thông số | Nội dung cần báo cáo | Giá trị/TODO |
|---|---|---|
| Số camera | Tổng số camera sử dụng | TODO |
| Camera ID | cam01, cam02, cam03, cam04 | TODO |
| Topology camera | cam01 → cam02 → cam03 → elevator → cam04 | TODO |
| Số người thật | Tổng số người tham gia | TODO |
| Số Global ID thật | Ground-truth identity count | TODO |
| Số video/clip | Tổng số clip | TODO |
| Số clip/camera | Số clip theo từng camera | TODO |
| Tổng số frame | Tổng frame toàn dataset | TODO |
| FPS gốc | Ví dụ 25/30 FPS | TODO |
| Độ phân giải | Ví dụ 1280×720, 1920×1080 | TODO |
| Tổng thời lượng | Tổng số phút/giây video | TODO |
| Tổng số local track | Tổng số track nội bộ từng camera | TODO |
| Số ADL class | standing, sitting, walking, lying_down, falling, reaching, bending, unknown | 8 |
| Số kịch bản vùng mù | elevator, room, door blind zone | TODO |
| Số kịch bản khó | thay áo, che mặt, quay lưng, thiếu sáng, occlusion | TODO |

### 1.2. Test Scenario Summary

| Mã test | Kịch bản | Camera liên quan | Mục tiêu đánh giá | Metric chính |
|---|---|---|---|---|
| T1 | Đi bình thường từ cam01 sang cam02 | cam01, cam02 | Kiểm tra giữ ID qua camera gần | Transfer Success Rate, IDF1 |
| T2 | Đi từ cam02 sang cam03 | cam02, cam03 | Kiểm tra topology tuần tự | ID Switch, Fragmentation |
| T3 | Từ cam03 vào thang máy lên cam04 | cam03, cam04 | Kiểm tra vùng mù elevator | Blind-zone Recovery Rate |
| T4 | Vào thang máy rồi quay lại cam03 | cam03 | Kiểm tra return path | False Split Rate |
| T5 | Từ cam04 vào phòng rồi ra lại | cam04 | Kiểm tra room hold | Room Re-entry Accuracy |
| T6 | Vào phòng, thay áo, đi ra | cam04 | Kiểm tra clothing-change ReID | Clothing-change ID Preservation |
| T7 | Không thấy mặt/quay lưng | nhiều camera | Kiểm tra khi face cue yếu | No-face ID Accuracy |
| T8 | Occlusion một phần | nhiều camera | Kiểm tra pose mất keypoint | Missing Keypoint Rate, ADL F1 |
| T9 | Nhiều người xuất hiện cùng lúc | nhiều camera | Kiểm tra conflict identity | False Merge Rate |
| T10 | Người đi ngược topology | cam04→cam03 hoặc cam03→cam02 | Kiểm tra transition ngược | Transfer Accuracy |

---

## 2. Thông số model và cấu hình hiện tại

### 2.1. Bảng model chính

| Module | Model/Phương pháp | Thông số cần báo cáo |
|---|---|---|
| Person detector | YOLOv8n / YOLO11n | input size, confidence threshold, class ID |
| Pose estimator | YOLOv8n-pose / YOLO11n-pose / RTMPose | COCO-17 keypoints, keypoint confidence |
| Local tracking | IoU tracker / DeepSORT / ByteTrack | max age, IoU threshold, ID switch |
| ADL classifier | rule-based / ST-GCN / CTR-GCN | window size, number of classes |
| Face recognition | InsightFace / ArcFace | embedding dimension, similarity threshold |
| Body ReID | HSV + Hu / simple body ReID / deep ReID | similarity threshold, crop size |
| Vector DB | FAISS | top-k, embedding dim, index type |
| Global ID | CPose spatio-temporal gating | strong/weak threshold, confirm frames |
| Dashboard | Flask + Socket.IO | latency, update rate |

### 2.2. Phase 1 — Recorder / cắt clip tự động

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `conf_threshold` | 0.35 | Ngưỡng confidence detector |
| `person_conf_threshold` | 0.65 | Ngưỡng confidence riêng cho class person |
| `trigger_min_consecutive` | 3 | Số frame liên tiếp để kích hoạt ghi clip |
| `pre_roll_seconds` | 3 | Số giây lưu trước khi phát hiện |
| `post_roll_seconds` | 5 | Số giây lưu sau khi phát hiện |
| `rearm_cooldown_seconds` | 5 | Thời gian chờ trước khi kích hoạt lại |
| `min_clip_seconds` | 3.0 | Thời lượng clip tối thiểu |
| `max_clip_seconds` | 300.0 | Thời lượng clip tối đa |
| `min_box_area_ratio` | 0.0015 | Lọc bbox quá nhỏ |
| `snapshot_fps` | 10.0 | Tần suất snapshot |
| `jpeg_quality` | 75 | Chất lượng ảnh JPEG |
| `person_class_id` | 0 | Class person trong COCO |

### 2.3. Phase 2 — Offline analysis / label generation

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `model` | models/product/yolo11n.pt | Model detector |
| `person_class_id` | 0 | Person class |
| `conf_threshold` | 0.50 | Ngưỡng detect |
| `progress_every` | 3 | Tần suất cập nhật tiến độ |

### 2.4. Phase 3 — Pose + ADL

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `model` | models/product/yolov8n-pose.pt | Pose model |
| `person_class_id` | 0 | Person class |
| `conf_threshold` | 0.45 | Ngưỡng confidence pose/detection |
| `keypoint_conf_min` | 0.30 | Ngưỡng keypoint visible |
| `window_size` | 30 | Sliding window ADL |
| `progress_every` | 10 | Cập nhật tiến độ |
| `save_overlay` | true | Lưu video/ảnh overlay |

### 2.5. ADL classes hiện tại

| STT | Class |
|---:|---|
| 1 | standing |
| 2 | sitting |
| 3 | walking |
| 4 | lying_down |
| 5 | falling |
| 6 | reaching |
| 7 | bending |
| 8 | unknown |

### 2.6. Rule-based ADL thresholds

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `knee_bend_angle` | 150 | Góc gối để nhận diện sitting |
| `shoulder_raise` | 45 | Ngưỡng tay/shoulder để nhận reaching/bending |
| `velocity_walk` | 8.0 | Ngưỡng vận tốc để nhận walking |
| `min_visible_keypoints` | 8 | Số keypoint tối thiểu để không bị unknown |
| `falling_torso_angle` | 68 | Góc thân dùng cho falling |
| `falling_velocity_multiplier` | 1.1 | Hệ số vận tốc khi falling |
| `lying_aspect_ratio` | 1.15 | Tỉ lệ bbox cho lying_down |
| `bending_velocity_multiplier` | 0.6 | Hệ số vận tốc cho bending |
| `confidence_unknown` | 0.20 | Confidence mặc định unknown |
| `confidence_falling` | 0.88 | Confidence falling |
| `confidence_lying_down` | 0.84 | Confidence lying_down |
| `confidence_sitting` | 0.82 | Confidence sitting |
| `confidence_bending` | 0.78 | Confidence bending |
| `confidence_reaching` | 0.76 | Confidence reaching |
| `confidence_walking` | 0.79 | Confidence walking |
| `confidence_standing` | 0.75 | Confidence standing |

### 2.7. Global ID / Cross-camera ReID parameters

| Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---:|---|
| `strong_threshold` | 0.65 | Ngưỡng match mạnh |
| `weak_threshold` | 0.45 | Ngưỡng match yếu |
| `confirm_frames` | 3 | Số frame xác nhận ID |
| `top_k_candidates` | 20 | Số candidate lấy từ vector DB |
| `use_hungarian` | true | Có dùng Hungarian matching hay không |
| `max_unk_per_video` | 10 | Số UNK tối đa/video |
| `iou_resurrection_threshold` | 0.30 | Ngưỡng IoU để phục hồi UNK |
| `quality_update_threshold` | 0.70 | Ngưỡng chất lượng để update embedding |

### 2.8. Transition windows hiện tại

| Chuyển camera | Window | Ý nghĩa |
|---|---:|---|
| cam01 → cam02 | 0–60 s | Đi tuyến bình thường |
| cam02 → cam03 | 0–60 s | Đi tuyến bình thường |
| cam03 → cam02 | 10–120 s | Quay lại tầng/camera trước |
| cam03 → cam04 | 20–180 s | Đi qua thang máy/vùng mù |
| cam04 → cam03 | 20–180 s | Đi xuống từ cam04 về cam03 |
| cam04 → cam04 | 5–300 s | Vào phòng rồi ra lại |

### 2.9. ReID / Tracker / Persistence / VectorDB

| Nhóm | Tham số | Giá trị hiện tại | Ý nghĩa |
|---|---|---:|---|
| ReID | `threshold` | 0.65 | Ngưỡng ReID |
| ReID | `max_features` | 100 | Số feature tối đa lưu |
| ReID | `confirm_frames` | 3 | Số frame xác nhận |
| ReID | `min_crop_height` | 30 | Crop person tối thiểu |
| ReID | `min_crop_width` | 15 | Crop person tối thiểu |
| ReID | `top_k_similarity` | 5 | Top-k similarity |
| ReID | `pending_track_ttl_seconds` | 10.0 | TTL track chờ |
| ReID | `confirmed_track_ttl_seconds` | 60.0 | TTL track xác nhận |
| Tracker | `max_age` | 30 | Số frame giữ track mất dấu |
| Tracker | `n_init` | 3 | Số frame khởi tạo track |
| Tracker | `max_iou_distance` | 0.7 | Ngưỡng IoU |
| Tracker | `max_cosine_distance` | 0.4 | Ngưỡng cosine |
| Tracker | `half` | true | FP16 nếu hỗ trợ |
| Persistence | `embedding_dim` | 512 | Kích thước embedding |
| Persistence | `initial_memmap_size` | 10000 | Kích thước memmap ban đầu |
| Persistence | `expand_step` | 1000 | Bước mở rộng |
| Persistence | `ema_alpha` | 0.3 | EMA update embedding |
| VectorDB | `search_top_k` | 20 | Top-k search |
| VectorDB | `medium_dataset_threshold` | 1000 | Ngưỡng dataset vừa |
| VectorDB | `large_dataset_threshold` | 10000 | Ngưỡng dataset lớn |
| VectorDB | `hnsw_m` | 32 | HNSW M |
| VectorDB | `hnsw_ef_construction` | 200 | HNSW construction |
| VectorDB | `hnsw_ef_search` | 64 | HNSW search |
| VectorDB | `ivf_nlist` | 100 | IVF nlist |
| VectorDB | `ivf_nprobe` | 10 | IVF nprobe |

---

## 3. Benchmark bắt buộc phải có

### 3.1. Person detection benchmark

| Metric | Công dụng | Mức độ |
|---|---|---|
| Precision | Dự đoán người đúng trên tổng dự đoán người | Bắt buộc |
| Recall | Phát hiện được bao nhiêu người thật | Bắt buộc |
| F1-score | Cân bằng Precision và Recall | Bắt buộc |
| mAP@50 | Độ chính xác detection tại IoU 0.5 | Bắt buộc nếu có annotation |
| mAP@50–95 | Đánh giá nghiêm ngặt hơn | Nên có |
| FPS detector | Tốc độ riêng detector | Bắt buộc |
| Confusion matrix | Nếu có nhiều class | Nên có |

### 3.2. Pose estimation benchmark

| Metric | Công dụng | Mức độ |
|---|---|---|
| PCK@0.05 | Keypoint đúng ở ngưỡng chặt | Nên có nếu có label keypoint |
| PCK@0.1 | Keypoint đúng ở ngưỡng rộng hơn | Nên có |
| OKS-mAP | Chuẩn COCO pose | Nên có nếu có annotation |
| Mean keypoint confidence | Độ tin cậy keypoint trung bình | Bắt buộc nếu không có label |
| Visible keypoint ratio | Tỉ lệ frame có đủ keypoint | Bắt buộc |
| Missing keypoint rate | Tỉ lệ keypoint bị mất | Bắt buộc |
| Pose FPS | Tốc độ pose model | Bắt buộc |
| Pose failure rate | Tỉ lệ pose bị fail/unknown | Bắt buộc |

### 3.3. ADL recognition benchmark

| Metric | Công dụng | Mức độ |
|---|---|---|
| Accuracy | Tỉ lệ phân loại ADL đúng | Bắt buộc |
| Macro-F1 | Quan trọng nếu class mất cân bằng | Bắt buộc |
| Per-class Precision | Độ chính xác từng hành động | Bắt buộc |
| Per-class Recall | Độ phủ từng hành động | Bắt buộc |
| Per-class F1 | F1 từng hành động | Bắt buộc |
| Confusion matrix | Xem class nào bị nhầm | Bắt buộc |
| Window-size ablation | 10/15/30/45 frame | Nên có |
| ADL latency | ms/window | Bắt buộc |

### 3.4. Local tracking benchmark

| Metric | Công dụng | Mức độ |
|---|---|---|
| MOTA | Tracking accuracy tổng hợp | Nên có |
| IDF1 | Giữ đúng ID trong camera | Bắt buộc |
| ID Switch | Số lần đổi ID sai | Bắt buộc |
| Fragmentation | Số lần track bị đứt | Bắt buộc |
| HOTA | Cân bằng detection + association | Nên có |
| Tracker FPS | Tốc độ tracker | Bắt buộc |

### 3.5. Cross-camera Global ID benchmark

Đây là benchmark quan trọng nhất của CPose.

| Metric | Ý nghĩa | Mức độ |
|---|---|---|
| Global ID Accuracy | Tỉ lệ gán đúng Global ID | Bắt buộc |
| Cross-camera IDF1 | IDF1 trên toàn bộ camera | Bắt buộc |
| ID Switch across cameras | Số lần đổi ID khi qua camera khác | Bắt buộc |
| ID Fragmentation Rate | Một người bị tách thành nhiều GID | Bắt buộc |
| False Merge Rate | Hai người khác nhau bị gộp thành một GID | Bắt buộc |
| False Split Rate | Một người bị tách thành nhiều GID | Bắt buộc |
| Transfer Success Rate | Tỉ lệ chuyển camera giữ đúng ID | Bắt buộc |
| Unknown Rate | Tỉ lệ bị gán UNK | Bắt buộc |
| Soft-match Accuracy | Độ đúng của soft match | Nên có |
| Clothing-change ID Preservation | Giữ đúng ID khi thay áo | Bắt buộc nếu có test thay áo |
| Blind-zone Recovery Rate | Giữ ID đúng sau vùng mù | Bắt buộc |
| Room Re-entry Accuracy | Giữ ID khi vào/ra phòng | Nên có |

### 3.6. Full-system benchmark

| Metric | Ý nghĩa | Mức độ |
|---|---|---|
| Full pipeline FPS | Tốc độ toàn hệ thống | Bắt buộc |
| End-to-end latency | Thời gian từ input đến output | Bắt buộc |
| Detection latency | ms/frame | Bắt buộc |
| Pose latency | ms/frame | Bắt buộc |
| ADL latency | ms/window | Bắt buộc |
| ReID latency | ms/person | Bắt buộc |
| Render/dashboard latency | ms/update | Nên có |
| CPU usage | % | Nên có |
| GPU usage | % | Nên có |
| RAM usage | MB/GB | Nên có |
| VRAM usage | MB/GB | Nên có |
| Output storage size | MB/clip | Nên có |

---

## 4. Các bảng so sánh bắt buộc phải có trong paper

### 4.1. Table 1 — Dataset Summary

| Scenario | Cameras | Subjects | Clips | Frames | ADL labels | Special case |
|---|---:|---:|---:|---:|---:|---|
| Normal transfer | TODO | TODO | TODO | TODO | TODO | cam01→cam04 |
| Elevator transfer | TODO | TODO | TODO | TODO | TODO | cam03→cam04 |
| Room re-entry | TODO | TODO | TODO | TODO | TODO | cam04→cam04 |
| Clothing change | TODO | TODO | TODO | TODO | TODO | room@cam04 |
| Occlusion | TODO | TODO | TODO | TODO | TODO | partial body |
| No face | TODO | TODO | TODO | TODO | TODO | back view |

### 4.2. Table 2 — Hyperparameter Configuration

| Module | Parameter | Value |
|---|---|---:|
| Detector | `conf_threshold` | 0.35 / 0.45 |
| Pose | `keypoint_conf_min` | 0.30 |
| ADL | `window_size` | 30 |
| ADL | `min_visible_keypoints` | 8 |
| Global ID | `strong_threshold` | 0.65 |
| Global ID | `weak_threshold` | 0.45 |
| Global ID | `confirm_frames` | 3 |
| ReID | `threshold` | 0.65 |
| VectorDB | `search_top_k` | 20 |

### 4.3. Table 3 — Detection Benchmark

| Model | Precision | Recall | F1 | mAP@50 | mAP@50–95 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n pretrained | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLOv8n fine-tuned | TODO | TODO | TODO | TODO | TODO | TODO |
| YOLO11n | TODO | TODO | TODO | TODO | TODO | TODO |

### 4.4. Table 4 — Pose Benchmark

| Pose model | PCK@0.05 | PCK@0.1 | OKS-mAP | Missing keypoint rate | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-pose | TODO | TODO | TODO | TODO | TODO |
| YOLO11n-pose | TODO | TODO | TODO | TODO | TODO |
| RTMPose | TODO | TODO | TODO | TODO | TODO |

### 4.5. Table 5 — ADL Benchmark

| Method | Accuracy | Macro-F1 | Standing F1 | Sitting F1 | Walking F1 | Falling F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based ADL | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Rule-based + smoothing | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| ST-GCN / CTR-GCN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 4.6. Table 6 — Cross-camera Global ID Benchmark

| Method | Global ID Acc ↑ | IDF1 ↑ | IDSW ↓ | Fragmentation ↓ | False Merge ↓ | False Split ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Local tracker only | TODO | TODO | TODO | TODO | TODO | TODO |
| Face ReID only | TODO | TODO | TODO | TODO | TODO | TODO |
| Body ReID only | TODO | TODO | TODO | TODO | TODO | TODO |
| Pose/height/time only | TODO | TODO | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO |

### 4.7. Table 7 — Ablation Study

| Variant | Global ID Acc | IDF1 | IDSW | Fragmentation | Clothing-change success | Blind-zone success |
|---|---:|---:|---:|---:|---:|---:|
| CPose full | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o temporal gating | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o camera topology | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o face | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o body appearance | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o pose/gait | TODO | TODO | TODO | TODO | TODO | TODO |
| w/o ADL continuity | TODO | TODO | TODO | TODO | TODO | TODO |

### 4.8. Table 8 — Transition Window Ablation

| Window setting | Global ID Acc | IDSW | False Merge | False Split |
|---|---:|---:|---:|---:|
| No window | TODO | TODO | TODO | TODO |
| Strict window | TODO | TODO | TODO | TODO |
| Default CPose window | TODO | TODO | TODO | TODO |
| Loose window | TODO | TODO | TODO | TODO |

### 4.9. Table 9 — Clothing Change Benchmark

| Method | Same clothes | Changed clothes | No face | Changed clothes + no face |
|---|---:|---:|---:|---:|
| Body ReID only | TODO | TODO | TODO | TODO |
| Face ReID only | TODO | TODO | TODO | TODO |
| Time-topology only | TODO | TODO | TODO | TODO |
| CPose full | TODO | TODO | TODO | TODO |

### 4.10. Table 10 — Runtime Comparison

| System | Detector FPS | Pose FPS | ADL FPS | ReID FPS | Full FPS | Latency |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | TODO | TODO | TODO | TODO | TODO | TODO |
| GPU RTX/T4 | TODO | TODO | TODO | TODO | TODO | TODO |
| Jetson Nano / edge | TODO | TODO | TODO | TODO | TODO | TODO |

---

## 5. Công thức bắt buộc phải có

### 5.1. Intersection over Union — IoU

Dùng cho detection và tracking.

```math
IoU = \frac{Area(B_{pred} \cap B_{gt})}{Area(B_{pred} \cup B_{gt})}
```

### 5.2. Precision

```math
Precision = \frac{TP}{TP+FP}
```

### 5.3. Recall

```math
Recall = \frac{TP}{TP+FN}
```

### 5.4. F1-score

```math
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
```

### 5.5. Accuracy

```math
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
```

### 5.6. Average Precision — AP

```math
AP = \int_0^1 P(R)dR
```

### 5.7. Mean Average Precision — mAP

```math
mAP = \frac{1}{N}\sum_{i=1}^{N} AP_i
```

### 5.8. Percentage of Correct Keypoints — PCK

```math
PCK@\alpha = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}
\left(
\frac{||p_i-\hat{p}_i||_2}{s} < \alpha
\right)
```

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| `p_i` | keypoint ground truth |
| `p_hat_i` | keypoint dự đoán |
| `s` | scale, ví dụ chiều cao bbox người |
| `alpha` | ngưỡng, ví dụ 0.05 hoặc 0.1 |

### 5.9. Object Keypoint Similarity — OKS

```math
OKS = \frac{\sum_i exp\left(-\frac{d_i^2}{2s^2k_i^2}\right)\delta(v_i>0)}
{\sum_i \delta(v_i>0)}
```

### 5.10. Cosine Similarity

Dùng cho face embedding, body embedding và vector search.

```math
S_{cos}(a,b)=\frac{a \cdot b}{||a||||b||}
```

### 5.11. CPose Global ID Matching Score

Đây nên là công thức trung tâm của paper CPose.

```math
S_{total}=w_fS_{face}+w_bS_{body}+w_pS_{pose}+w_hS_{height}+w_tS_{time}+w_cS_{camera}
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `S_face` | Similarity khuôn mặt |
| `S_body` | Similarity ngoại hình/body |
| `S_pose` | Similarity pose/gait |
| `S_height` | Similarity chiều cao/tỉ lệ người |
| `S_time` | Hợp lệ theo thời gian chuyển camera |
| `S_camera` | Hợp lệ theo topology camera |

Quy tắc quyết định:

```math
ID =
\begin{cases}
GID_{old}, & S_{total} \geq \tau_{strong} \\
GID_{old}^{soft}, & \tau_{weak} \leq S_{total}<\tau_{strong} \\
UNK, & S_{total}<\tau_{weak}
\end{cases}
```

Theo cấu hình hiện tại:

```math
\tau_{strong}=0.65, \quad \tau_{weak}=0.45
```

### 5.12. Clothing-change Matching Score

Khi người thay áo, giảm trọng số body appearance:

```math
S_{change}=w_fS_{face}+w_b'S_{body}+w_p'S_{pose}+w_hS_{height}+w_t'S_{time}+w_c'S_{camera}
```

Điều kiện:

```math
w_b' < w_b
```

Ý nghĩa: không để màu áo quyết định Global ID.

### 5.13. Time-window Gating

```math
G_{time}(i,j)=
\begin{cases}
1, & \Delta t_{ij}\in[T_{min}^{c_i\rightarrow c_j},T_{max}^{c_i\rightarrow c_j}] \\
0, & otherwise
\end{cases}
```

Ví dụ:

```math
T^{cam03\rightarrow cam04}=[20,180]s
```

### 5.14. Camera Topology Gating

```math
G_{camera}(c_i,c_j)=
\begin{cases}
1, & (c_i,c_j)\in E_{topology} \\
0, & otherwise
\end{cases}
```

Trong đó `E_topology` là tập cạnh hợp lệ giữa các camera.

### 5.15. Temporal Voting / Confirm Frames

```math
V(GID_k)=\sum_{t=n-m+1}^{n}\mathbf{1}(ID_t=GID_k)
```

```math
ID_{final}=GID_k \quad \text{if} \quad V(GID_k)\geq N_{confirm}
```

Theo cấu hình hiện tại:

```math
N_{confirm}=3
```

### 5.16. MOTA

```math
MOTA = 1 - \frac{\sum_t(FN_t+FP_t+IDSW_t)}{\sum_t GT_t}
```

### 5.17. IDF1

```math
IDF1 = \frac{2IDTP}{2IDTP+IDFP+IDFN}
```

### 5.18. HOTA

```math
HOTA = \sqrt{DetA \cdot AssA}
```

Với:

```math
DetA = \frac{TP}{TP+FP+FN}
```

```math
AssA = \frac{1}{|TP|}\sum_{c \in TP} AssIoU(c)
```

### 5.19. ID Fragmentation Rate

```math
FragRate = \frac{N_{predicted\_GID}-N_{true\_ID}}{N_{true\_ID}}
```

Nếu 3 người thật nhưng hệ thống tạo 8 Global ID, hệ thống bị fragmentation nặng.

### 5.20. Transfer Success Rate

```math
TSR = \frac{N_{correct\_transfers}}{N_{total\_transfers}}
```

Dùng cho các chuyển tiếp như cam01→cam02, cam02→cam03, cam03→cam04, cam04→cam03.

### 5.21. False Merge Rate

```math
FMR = \frac{N_{false\_merge}}{N_{all\_merge}}
```

False merge xảy ra khi hai người khác nhau bị gộp thành một Global ID.

### 5.22. False Split Rate

```math
FSR = \frac{N_{false\_split}}{N_{true\_ID}}
```

False split xảy ra khi một người thật bị tách thành nhiều Global ID.

### 5.23. Unknown Rate

```math
UNKRate = \frac{N_{UNK}}{N_{all\_assignments}}
```

### 5.24. Blind-zone Recovery Rate

```math
BZRR = \frac{N_{correct\_blind\_zone\_recoveries}}{N_{blind\_zone\_events}}
```

### 5.25. Clothing-change ID Preservation Rate

```math
CCIPR = \frac{N_{correct\_ID\_after\_clothing\_change}}{N_{clothing\_change\_events}}
```

### 5.26. End-to-end Latency

```math
Latency_{total}=T_{detect}+T_{pose}+T_{ADL}+T_{ReID}+T_{render}
```

### 5.27. FPS

```math
FPS = \frac{N_{frames}}{T_{processing}}
```

---

## 6. Ablation study bắt buộc

Ablation là phần chứng minh CPose có đóng góp thật, không phải chỉ ghép module.

### 6.1. Các biến thể cần chạy

| Variant | Mô tả | Mục đích |
|---|---|---|
| CPose full | Tất cả module | Kết quả chính |
| w/o temporal gating | Bỏ transition time window | Chứng minh vai trò thời gian |
| w/o camera topology | Bỏ graph camera | Chứng minh vai trò topology |
| w/o face | Không dùng face embedding | Kiểm tra khi mất mặt |
| w/o body appearance | Không dùng màu/quần áo | Kiểm tra case thay áo |
| w/o pose/gait | Không dùng pose/gait signature | Chứng minh pose có ích cho ReID |
| w/o ADL continuity | Không dùng nhãn hành động để hỗ trợ | Chứng minh ADL giúp ổn định |
| w/o temporal smoothing | Bỏ smoothing pose/ADL | Kiểm tra flickering |
| local tracker only | Chỉ giữ ID trong từng camera | Baseline yếu nhất |
| appearance ReID only | Chỉ dùng body appearance | Baseline thường gặp |

### 6.2. Metric cho ablation

| Metric | Lý do |
|---|---|
| Global ID Accuracy | Chỉ số chính |
| Cross-camera IDF1 | Đo giữ ID xuyên camera |
| ID Switch | Đo lỗi đổi ID |
| Fragmentation Rate | Đo lỗi tách ID |
| False Merge Rate | Đo lỗi gộp người |
| False Split Rate | Đo lỗi tách người |
| Clothing-change ID Preservation | Đo khả năng xử lý thay áo |
| Blind-zone Recovery Rate | Đo khả năng xử lý vùng mù |
| Full FPS | Đảm bảo không quá chậm |

---

## 7. Benchmark tham khảo từ các bài liên quan trong nhóm

### 7.1. Bài pothole / road damage detection

| Thành phần trong bài pothole | Áp dụng vào CPose |
|---|---|
| Dataset nhiều nguồn | CPose nên có self-collected + public dataset nếu có |
| Train/val/test rõ ràng | CPose cần chia train/test hoặc scenario test rõ |
| mAP@50, mAP@50–95 | Dùng cho person detection/pose nếu fine-tune |
| FPS model và FPS toàn hệ thống | Bắt buộc báo cáo |
| OpenCV Python vs DeepStream C++ | CPose có thể so sánh CPU vs GPU hoặc unoptimized vs optimized |
| Edge deployment | Nếu chạy Jetson/edge thì paper mạnh hơn |

### 7.2. Bài lane detection

| Thành phần trong bài lane | Áp dụng vào CPose |
|---|---|
| Accuracy, F1, FP, FN | Dùng cho ADL và Global ID |
| Model size | Báo cáo size detector/pose/ADL model |
| FPS | Báo cáo tốc độ xử lý |
| Trade-off accuracy/speed | So sánh YOLOv8n-pose, YOLO11n-pose, RTMPose |
| Future work rõ | Nâng cấp rule-based ADL sang ST-GCN/CTR-GCN/Transformer |

### 7.3. Bài lane violation detection

Đây là bài nên học nhiều nhất.

| Bài lane violation | CPose tương ứng |
|---|---|
| Rear-wheel baseline | Pose/height/time/topology reasoning |
| Temporal filtering | Confirm frames / temporal voting |
| Bounding-box baseline | Local-tracking-only / appearance-only baseline |
| Accuracy 94% | CPose cần Global ID Accuracy |
| Ablation window size | CPose cần transition window ablation |
| False positive/false negative analysis | CPose cần false merge/false split analysis |

### 7.4. Bài traffic flow MOT

| Metric trong bài traffic MOT | CPose nên dùng |
|---|---|
| HOTA | Đánh giá tracking |
| MOTA | Đánh giá local tracking |
| IDF1 | Bắt buộc cho Global ID |
| FPS | Bắt buộc |
| So sánh ByteTrack, OCSORT, BoTSORT | CPose nên so với tracker/ReID baseline |
| Theo density/thời tiết/góc quay | CPose nên theo scenario: normal, occlusion, no face, clothing change |

---

## 8. Cấu trúc paper CPose đề xuất

### 8.1. Title đề xuất

**CPose: A Time-First Cross-Camera Pose–ADL–ReID Framework for Sequential Multi-Camera Human Activity Monitoring**

Hoặc:

**A Robust Cross-Camera Human Activity Monitoring Method from Multi-Camera Vision: A Spatio-Temporal Pose Reasoning Approach**

### 8.2. Abstract khung

Nội dung abstract nên có:

1. Bối cảnh: multi-camera human monitoring khó vì vùng mù, mất mặt, thay đổi ngoại hình, camera không đồng bộ.
2. Vấn đề: tracker/ReID thông thường dễ ID switch và fragmentation.
3. Phương pháp: CPose tích hợp detection, pose, ADL, face/body ReID và time-first camera topology reasoning.
4. Đóng góp: spatio-temporal Global ID reasoning, blind-zone buffer, clothing-change handling.
5. Kết quả: báo Global ID Accuracy, IDF1, Fragmentation, ADL F1, FPS.

### 8.3. Contribution nên viết

Nên viết 3 đóng góp chính:

1. **Framework contribution**: Đề xuất CPose, một framework end-to-end cho multi-camera pose, ADL và ReID.
2. **Algorithm contribution**: Đề xuất Time-First Cross-Camera Sequential Pose–ADL–ReID reasoning để giữ Global ID qua vùng mù và thay đổi ngoại hình.
3. **Evaluation contribution**: Xây dựng benchmark đa kịch bản và đánh giá bằng Global ID Accuracy, IDF1, ID Switch, Fragmentation, ADL Macro-F1 và FPS.

### 8.4. Outline paper

1. Introduction
2. Related Works
3. Proposed CPose Framework
4. Time-First Cross-Camera Pose–ADL–ReID Algorithm
5. Experimental Setup
6. Results and Ablation Study
7. Discussion
8. Conclusion

---

## 9. Các câu hỏi reviewer/thầy/sếp có thể hỏi

### 9.1. Điểm mới của CPose là gì?

Trả lời:

> Điểm mới không phải YOLO-Pose, mà là cơ chế suy luận Global ID xuyên camera theo thời gian trước, kết hợp camera topology, transition window, pose/ADL continuity, face/body similarity và blind-zone buffer để giảm ID switch và fragmentation.

### 9.2. Khác gì DeepSORT/ByteTrack/BoTSORT?

Trả lời:

> DeepSORT/ByteTrack/BoTSORT chủ yếu giữ ID trong một video/camera. CPose xử lý cross-camera identity association, đặc biệt khi có vùng mù như thang máy/phòng và khi ngoại hình thay đổi.

### 9.3. Nếu người thay áo thì sao?

Trả lời:

> CPose giảm trọng số body appearance, tăng vai trò face, pose/gait, height ratio, time window và camera topology. Nếu chỉ có một candidate hợp lệ trong RoomHoldBuffer, hệ thống ưu tiên giữ ID cũ ở dạng soft match.

### 9.4. Nếu không thấy mặt thì sao?

Trả lời:

> Hệ thống dùng body ratio, pose/gait signature, ADL continuity và topology/time gating. Confidence có thể thấp hơn nhưng không tạo ID mới ngay nếu candidate hợp lệ duy nhất.

### 9.5. Rule-based ADL có đủ để publish không?

Trả lời:

> Rule-based ADL là baseline ổn định. Để bài mạnh hơn, cần so sánh thêm với skeleton-based model như ST-GCN, CTR-GCN hoặc lightweight temporal model.

---

## 10. Roadmap thực nghiệm đề xuất

### 10.1. Giai đoạn 1 — Chuẩn hóa dataset

Việc cần làm:

- thu thập video đủ 4 camera;
- thống nhất tên file có timestamp;
- tạo ground truth Global ID;
- gán nhãn ADL theo frame hoặc theo đoạn;
- đánh dấu event: vào phòng, ra phòng, vào thang máy, ra thang máy, thay áo, mất mặt.

Output:

- `dataset_summary.csv`
- `global_id_gt.csv`
- `adl_gt.csv`
- `events_gt.csv`

### 10.2. Giai đoạn 2 — Chạy baseline

Baseline cần chạy:

1. local tracker only;
2. face ReID only;
3. body appearance ReID only;
4. time-topology only;
5. pose/height/time only;
6. CPose full.

### 10.3. Giai đoạn 3 — Chạy ablation

Ablation cần chạy:

1. CPose full;
2. w/o temporal gating;
3. w/o camera topology;
4. w/o face;
5. w/o body appearance;
6. w/o pose/gait;
7. w/o ADL continuity;
8. w/o temporal smoothing.

### 10.4. Giai đoạn 4 — Đo runtime

Cần đo:

- FPS detector;
- FPS pose;
- latency ADL;
- latency ReID;
- full pipeline FPS;
- CPU/GPU/RAM usage.

### 10.5. Giai đoạn 5 — Viết paper

Cấu trúc:

1. Introduction
2. Related Works
3. Proposed CPose Framework
4. Time-First Cross-Camera Pose–ADL–ReID Algorithm
5. Experimental Setup
6. Results and Ablation Study
7. Discussion
8. Conclusion

---

## 11. Checklist tối thiểu trước khi gửi thầy/sếp

### 11.1. Dataset

- [ ] Có bảng tổng quan dataset.
- [ ] Có topology camera.
- [ ] Có số clip/frame/người.
- [ ] Có nhãn Global ID ground truth.
- [ ] Có nhãn ADL ground truth.
- [ ] Có nhãn event vùng mù/thay áo.

### 11.2. Method

- [ ] Có sơ đồ pipeline.
- [ ] Có mô tả từng module.
- [ ] Có công thức Global ID matching score.
- [ ] Có time-window gating.
- [ ] Có camera topology gating.
- [ ] Có temporal voting.
- [ ] Có mô tả xử lý clothing-change.
- [ ] Có mô tả xử lý blind-zone.

### 11.3. Experiment

- [ ] Có detection benchmark.
- [ ] Có pose benchmark hoặc keypoint visibility benchmark.
- [ ] Có ADL benchmark.
- [ ] Có Global ID benchmark.
- [ ] Có ablation study.
- [ ] Có runtime benchmark.
- [ ] Có qualitative examples.
- [ ] Có failure case analysis.

### 11.4. Result

- [ ] Có bảng so sánh baseline.
- [ ] Có bảng ablation.
- [ ] Có confusion matrix ADL.
- [ ] Có biểu đồ ID switch/fragmentation.
- [ ] Có hình minh họa đúng/sai.
- [ ] Có phân tích tại sao sai.

---

## 12. Bộ benchmark tối thiểu nên chạy ngay

Ưu tiên theo thứ tự:

| Ưu tiên | Benchmark | Lý do |
|---:|---|---|
| 1 | Global ID Accuracy | Chỉ số chính của CPose |
| 2 | Cross-camera IDF1 | Đo khả năng giữ ID xuyên camera |
| 3 | ID Switch | Đo lỗi đổi ID |
| 4 | Fragmentation Rate | Đo lỗi tách một người thành nhiều ID |
| 5 | Transfer Success Rate | Đo khả năng qua camera |
| 6 | Blind-zone Recovery Rate | Đo khả năng xử lý thang máy/phòng |
| 7 | Clothing-change ID Preservation | Đo khả năng xử lý thay áo |
| 8 | ADL Accuracy + Macro-F1 | Đánh giá hoạt động |
| 9 | Full pipeline FPS | Đánh giá khả năng triển khai |
| 10 | Ablation Study | Chứng minh đóng góp từng thành phần |

---

## 13. Kết luận chiến lược

Các chỉ số quan trọng nhất cần ưu tiên:

1. Global ID Accuracy
2. Cross-camera IDF1
3. ID Switch
4. Fragmentation Rate
5. Transfer Success Rate
6. Blind-zone Recovery Rate
7. Clothing-change ID Preservation
8. ADL Macro-F1
9. Full pipeline FPS
10. Ablation Study

Nếu có đầy đủ các bảng, công thức và benchmark trong tài liệu này, CPose có thể được trình bày như một đề tài nghiên cứu nghiêm túc thay vì chỉ là một sản phẩm demo.
