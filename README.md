# CPose Terminal AI Pipeline — TFCS-PAR

Pipeline này chạy hoàn toàn bằng terminal, được thiết kế để phân tích video đa camera và gán định danh (Global ID) xuyên suốt các camera. Phương pháp phân tích là TFCS-PAR (Time-First Cross-Camera Sequential Pose-ADL-ReID).

Mọi kết quả chạy đều được tự động lưu theo timestamp trong thư mục `dataset/outputs/` giúp so sánh dễ dàng.

## 1. Yêu cầu cài đặt

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu máy chưa có PyTorch, vui lòng cài đặt `torch` và `torchvision` (hỗ trợ CUDA nếu có GPU) theo hướng dẫn trên trang chủ PyTorch trước khi chạy.

Để xem trực tiếp kết quả video MP4 trong VSCode, bạn nên cài đặt extension **Video Preview**:
- Tên extension: `Video Preview` (ID: `batchnepal.vscode-video-preview`)
- Hoặc tìm "mp4" trong tab Extensions của VSCode.

## 2. Chuẩn bị dữ liệu

Đặt video đầu vào (từ các camera) trong:
```text
data-test/
```

**Quy tắc đặt tên file cực kỳ quan trọng đối với thuật toán ReID:**
```text
cam<X>_YYYY-MM-DD_HH-MM-SS.mp4
Ví dụ: cam1_2026-01-29_16-26-25.mp4
```
Hệ thống sẽ đọc timestamp từ tên file để sắp xếp luồng xử lý theo đúng dòng thời gian thực tế.

## 3. Các Module

Hệ thống được chia thành 5 module xử lý nối tiếp. Bạn có thể chạy từng module bằng CLI hoặc chạy toàn bộ thông qua file python ở `src/pipeline/run_all.py`.

| Module | Input | Output |
| :--- | :--- | :--- |
| **1. Person Detection**| `.mp4` | *Video vẽ box không gán ID*<br/>`detections.json` |
| **2. Person Tracking** | <br/> `detections.json` | <br/> *Video vẽ box kèm ID cục bộ (`T1`, `T2`...)*<br/> `tracks.json` |
| **3. Pose Estimation**| <br/>`tracks.json` | `dataset/outputs/3_pose/<timestamp>/`<br/> *Video vẽ khung xương không ID*<br/> `keypoints.json` |
| **4. ADL Recognition** | <br/> `keypoints.json` | <br/>*Video vẽ nhãn hành vi (VD: `walking 0.85`)*<br/> `adl_events.json` |
| **5. Cross-Camera ReID** | <br/> `keypoints.json`<br/> `adl_events.json` | `dataset/outputs/5_reid/<timestamp>/`<br/> **Video vẽ Global ID (`GID-001`, `ACTIVE`...)**<br/> `reid_tracks.json`<br/> `global_person_table.json` |

## 4. Chạy toàn bộ Pipeline

Bạn có thể chạy tự động toàn bộ 5 bước trên theo thứ tự (từ Module 1 đến Module 5) bằng lệnh sau:

```bash
python -m src.pipeline.run_all \
  --input data-test/ \
  --output dataset/outputs \
  --config configs/model_registry.demo_i5.yaml \
  --manifest configs/multicam_manifest.example.json \
  --topology configs/camera_topology.example.yaml
```

Sau khi chạy xong, kết quả cuối cùng (kèm Global ID hoàn chỉnh) sẽ nằm trong `dataset/outputs/pipeline/<timestamp>/`. Cấu trúc chi tiết của các tham số và lệnh chạy từng phần được cung cấp trong file `CLAUDE.md`.

## 5. Cấu trúc Dataset & Tham số Đánh giá

Hệ thống hỗ trợ cơ chế tự động đánh giá (benchmark) thông qua thư mục `dataset/annotations/`. Để lấy các chỉ số (metrics) thực tế, bạn cần sắp xếp ground truth của các tập dữ liệu như sau:

| Module | Dataset / Nguồn | Vị trí thư mục Annotation | Tham số / Chỉ số chính được đo (Metrics) |
| :--- | :--- | :--- | :--- |
| **1. Detection** | COCO 2017 | `gt-person-detection/coco2017/` | Dùng `instances_val2017.json`.<br/>Đo *Precision, Recall, F1, mAP@50*. |
| **2. Tracking** | MOT17 | `gt-person-tracking/mot17/` | File cấu trúc MOT.<br/>Đo *IDF1, ID Switch, Fragmentation*. |
| **3. Pose** | COCO / MPII | `gt-pose-estimation/coco2017/` | Dùng `person_keypoints_val2017.json`.<br/>Đo *PCK, Missing keypoint rate*. |
| **4. ADL** | Tự thu / Toyota Smarthome | `gt-adl-recognition/` hoặc `gt-cpose-custom/adl_gt.csv` | File CSV chứa start_frame, end_frame, label.<br/>Đo *Accuracy, Macro-F1, Confusion Matrix*. |
| **5. Global ReID** | Market-1501 / CPose Custom | `gt-global-reid/` hoặc `gt-cpose-custom/global_id_gt.csv` | File chứa tọa độ bbox & Global ID chuẩn.<br/>Đo *Cross-camera IDF1, False Split, False Merge*. |

***Lưu ý:***
- Đối với **demo cục bộ**, dự án sử dụng các video kiểm thử đặt trong `data-test/`.
- File cấu hình `configs/model_registry.demo_i5.yaml` chứa toàn bộ ngưỡng cấu hình cho module (ví dụ: `conf: 0.5` cho YOLO, `strong_threshold: 0.65` cho ReID). Hệ thống sẽ tải thông số trực tiếp từ file yaml này trong quá trình xử lý.
