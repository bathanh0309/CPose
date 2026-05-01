# CPose Terminal AI Pipeline — TFCS-PAR

Pipeline này chạy hoàn toàn bằng terminal, được thiết kế để phân tích video đa camera và gán định danh (Global ID) xuyên suốt các camera. Phương pháp phân tích là TFCS-PAR (Time-First Cross-Camera Sequential Pose-ADL-ReID).

Mọi kết quả chạy đều được tự động lưu theo timestamp trong thư mục `dataset/outputs/` giúp so sánh dễ dàng.

## 1. Yêu cầu cài đặt

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
*(Ghi chú: `pytest` và các thư viện test đã được cấu hình sẵn bên trong `requirements.txt` tại mục `[12] DEV / TESTING`)*

Nếu máy chưa có PyTorch, vui lòng cài đặt `torch` và `torchvision` (hỗ trợ CUDA nếu có GPU) theo hướng dẫn trên trang chủ PyTorch trước khi chạy.

## 2. Cấu trúc mã nguồn (`src/`)

```text
src/
├── adl_recognition/   # Module 4: Nhận diện hành vi (Activities of Daily Living) dựa trên bộ khung xương.
├── common/            # Các tiện ích dùng chung: xử lý video I/O, vẽ hình ảnh (visualization), tính toán metrics.
├── evaluation/        # Scripts so sánh kết quả dự đoán với Ground Truth (tính mAP, IDF1, Accuracy...).
├── face/              # Module hỗ trợ trích xuất đặc trưng khuôn mặt (InsightFace).
├── global_reid/       # Module 5: Định danh chéo camera (Cross-Camera ReID) sử dụng TFCS-PAR.
├── human_detection/   # Module 1: Nhận diện người (Person Detection) dùng YOLO.
├── human_tracking/    # Module 2: Theo dõi người trong 1 camera (Local Tracking) dùng ByteTrack/IoU.
├── modules/           # Shim package (tương thích ngược) giúp import các module từ src.modules.
├── pipeline/          # Các script chạy toàn bộ quy trình từ đầu đến cuối (run_all.py) và tự động benchmark.
├── pose_estimation/   # Module 3: Ước lượng khung xương (Pose Estimation) dựa trên YOLO Pose.
└── reports/           # Scripts tạo bảng biểu, báo cáo markdown phục vụ viết bài báo nghiên cứu (Paper Tables).
```

## 3. Quy trình các Module xử lý (Pipeline)

Hệ thống được chia thành 5 module xử lý nối tiếp. Kết quả đầu ra của module trước làm đầu vào cho module sau.

### Module 1: Person Detection
```text
[Input]
- Video thô (VD: .mp4) từ thư mục `data-test/`

[Output]
- Video có vẽ bounding box (không gán ID)
- `detections.json` (chứa tọa độ khung hình)
```

### Module 2: Person Tracking
```text
[Input]
- Video thô
- `detections.json` (từ Module 1)

[Output]
- Video có vẽ bounding box và ID cục bộ (VD: T1, T2...)
- `tracks.json` (chứa tọa độ và Track ID)
```

### Module 3: Pose Estimation
```text
[Input]
- Video thô
- `tracks.json` (từ Module 2)

[Output]
- Video vẽ bộ khung xương người (Skeleton)
- `keypoints.json` (chứa tọa độ 17 điểm khớp trên cơ thể)
```

### Module 4: ADL Recognition
```text
[Input]
- Video thô
- `keypoints.json` (từ Module 3)

[Output]
- Video hiển thị nhãn hành vi bên dưới mỗi người (VD: walking, standing...)
- `adl_events.json` (chứa danh sách hành vi theo khung hình)
```

### Module 5: Cross-Camera Global ReID
```text
[Input]
- Video thô
- `keypoints.json` (từ Module 3)
- `adl_events.json` (từ Module 4)
- Các file cấu hình hệ thống camera (`multicam_manifest.json`, `camera_topology.yaml`)

[Output]
- Video vẽ nhãn GID hợp nhất xuyên camera (VD: GID-001) và trạng thái (ACTIVE, DORMANT...)
- `reid_tracks.json` (danh sách ID đã được hợp nhất)
- `global_person_table.json` (hồ sơ vòng đời của mỗi người trên toàn hệ thống)
```

## 4. Cách chạy toàn bộ Pipeline

Sử dụng script tiện ích `.bat` đã được cấu hình sẵn Python ảo (`.venv`):

```bat
run_06_pipeline.bat
```

Hoặc chạy thủ công qua lệnh CLI:

```bash
.venv\Scripts\python.exe -m src.pipeline.run_all \
  --input data-test/ \
  --output dataset/outputs \
  --config configs/model_registry.demo_i5.yaml \
  --manifest configs/multicam_manifest.json \
  --topology configs/camera_topology.yaml
```
