# CPose (Cross-camera Pose & Action Recognition)

Hệ thống AI xử lý thời gian thực kết hợp Nhận diện điểm ảnh (Pose), Theo dõi (Tracking), Định danh chéo camera (ReID) và Nhận diện hành vi (ADL).

---

## [INSTRUCTIONS MUST READ BEFORE MODIFYING CODE]

Dự án này đã được chốt cấu trúc nghiêm ngặt. Khi đọc hoặc chỉnh sửa code, **BẮT BUỘC** phải tuân thủ các quy tắc sau:

### 1. Ràng buộc về Giao diện (UI)
- **TUYỆT ĐỐI KHÔNG** refactor lại HTML/CSS/JS. Giữ nguyên Light Theme, layout ngang của các nút điều khiển, và ô nhập dữ liệu dạng `<input type="text">` (không dùng thẻ `<select>`).
- Khung Log của camera nào thì nằm ngay dưới camera đó. Cấm gom chung log.

### 2. Ràng buộc về Logging (Chống log bừa bãi)
- **KHÔNG ĐƯỢC** log rác (spam log) vào mỗi frame xử lý (ví dụ: cấm log "Đang xử lý frame 1, 2, 3...").
- Log AI **CHỈ ĐƯỢC PHÉP** sinh ra khi có sự kiện thay đổi trạng thái. Ví dụ: Phát hiện ID mới (`global_id`), phát hiện hành động nghi ngờ từ PoseC3D, hoặc khi mất track đối tượng.
- Chuẩn format: `[Time] [Level] Message`.

### 3. Ràng buộc chuẩn đầu vào/đầu ra các Modules
- **YOLO-Pose:** Đầu vào là Frame -> Đầu ra là `Bboxes` + `17 Keypoints`.
- **ByteTrack:** Đầu vào là `Bboxes` -> Đầu ra là `local_track_id` (Cấm làm mất/nhảy ID khi bị che khuất ngắn hạn).
- **FastReID:** Trích xuất Embedding -> So khớp Cosine Similarity -> Cấp `global_id` xuyên camera. Cấm tính toán ReID trên mọi frame để tránh giật lag (chỉ tính mỗi `N` frame hoặc khi có ID mới).
- **PoseC3D (ADL):** Nhận đầu vào là buffer chuỗi Keypoints (vd: 30 frames) -> Output nhãn hành động (Action Label). Cấm chạy ADL đồng bộ (blocking) luồng đọc video chính.

---

## 1. Tải và Cài đặt Mô hình (Model Weights)

Để hệ thống hoạt động, bạn cần tải các file trọng số (weights) và đặt đúng vào thư mục `models/` theo bảng sau:

| Module | Models | Link GitHub | Link Paper |
|---|---|---|---|
| **Pose (YOLO11-Pose)** | `models/yolo11n-pose.pt` | [Ultralytics YOLO11 Pose](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n-pose.pt) | [YOLO11 paper](https://arxiv.org/abs/2506.00915v1) |
| **Tracking (Pedestrian + ByteTrack)** | `models/tracking.pt` | [pedestrian-tracking](https://github.com/ErAkhileshSingh/pedestrian-tracking) + Ultralytics ByteTrack (`bytetrack.yaml`) | [ByteTrack paper](https://arxiv.org/abs/2110.06864) |
| **ReID (FastReID)** | `models/fastreid_market_R50.pth` | [JDAI-CV FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) | [FastReID paper](https://arxiv.org/abs/2006.02631) |
| **ADL (PoseC3D)** | `models/posec3d_r50_ntu60.pth` | [MMAction2 PoseC3D configs](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d) | [PoseC3D paper](https://arxiv.org/abs/2104.13586) |

*(Lưu ý: Đảm bảo thư mục `models/` được tạo ở thư mục gốc của project).*

Tất cả config của các module chính được gom trong `configs/system/pipeline.yaml`.

Benchmark/metric cho từng module được khai báo trong `configs/system/benchmark.yaml`.

| Module | Metric | Thông số ước lượng / target ban đầu |
|---|---|---|
| **Pose (YOLO11-Pose)** | FPS | 20-30 FPS trên GPU tầm trung, 5-12 FPS trên CPU. |
| **Pose (YOLO11-Pose)** | Latency | Khoảng 30-50 ms/frame trên GPU, 80-200 ms/frame trên CPU. |
| **Pose (YOLO11-Pose)** | Precision / Recall | Target >= 0.80 nếu video rõ, ít che khuất. |
| **Pose (YOLO11-Pose)** | mAP50 / mAP50-95 | Target mAP50 >= 0.75, mAP50-95 >= 0.45. |
| **Pose (YOLO11-Pose)** | OKS AP / OKS AR | Target OKS AP >= 0.60, OKS AR >= 0.65. |
| **Tracking (Pedestrian + ByteTrack)** | FPS | 18-30 FPS trên GPU, phụ thuộc số người trong khung hình. |
| **Tracking (Pedestrian + ByteTrack)** | Latency | Khoảng 35-60 ms/frame trên GPU. |
| **Tracking (Pedestrian + ByteTrack)** | MOTA / MOTP | Target MOTA >= 0.60, MOTP >= 0.70 khi có ground truth. |
| **Tracking (Pedestrian + ByteTrack)** | IDF1 | Target >= 0.65; thấp hơn nếu nhiều che khuất/chuyển camera. |
| **Tracking (Pedestrian + ByteTrack)** | ID Precision / ID Recall | Target >= 0.70 cho video camera cố định. |
| **Tracking (Pedestrian + ByteTrack)** | ID Switches | Target < 5 lần / 1.000 frame. |
| **Tracking (Pedestrian + ByteTrack)** | Mostly Tracked / Mostly Lost | Target Mostly Tracked >= 60%, Mostly Lost <= 20%. |
| **Tracking (Pedestrian + ByteTrack)** | False Positives / False Negatives | Target FP <= 10%, FN <= 20% tổng object ground truth. |
| **ReID (FastReID)** | Latency | Khoảng 5-20 ms/crop trên GPU, tùy số crop mỗi frame. |
| **ReID (FastReID)** | Top-1 / Top-5 Accuracy | Target Top-1 >= 0.75, Top-5 >= 0.90 với gallery đủ tốt. |
| **ReID (FastReID)** | mAP | Target >= 0.60 trên tập kiểm thử nội bộ. |
| **ReID (FastReID)** | CMC Rank-1 / Rank-5 / Rank-10 | Target Rank-1 >= 0.75, Rank-5 >= 0.90, Rank-10 >= 0.95. |
| **ReID (FastReID)** | Cosine threshold accuracy | Sweep 0.30-0.90, bước 0.05; threshold mặc định 0.55. |
| **ADL (PoseC3D)** | Latency | Khoảng 80-300 ms/clip skeleton, không nên chạy blocking trên luồng chính. |
| **ADL (PoseC3D)** | Top-1 / Top-5 Accuracy | Target Top-1 >= 0.65, Top-5 >= 0.85 khi label ADL đúng domain. |
| **ADL (PoseC3D)** | Precision / Recall / F1 Macro | Target F1 macro >= 0.60 cho tập ADL ban đầu. |
| **ADL (PoseC3D)** | Confusion Matrix | Xuất PNG/CSV; dùng để xem lớp hành động nào hay nhầm lẫn. |
| **ADL (PoseC3D)** | Per-class Accuracy | Target >= 0.60 cho mỗi lớp có đủ mẫu. |
| **Full CPose Pipeline** | FPS | Target 10-20 FPS khi bật pose + tracking + ReID interval. |
| **Full CPose Pipeline** | End-to-end latency | Target < 150 ms/frame khi chưa bật ADL realtime. |
| **Full CPose Pipeline** | Tracks / Persons per frame | Theo dõi trung bình số track/người mỗi frame để đánh giá tải hệ thống. |
| **Full CPose Pipeline** | ReID match rate | Target >= 70% track có `global_id` ổn định khi gallery đủ embedding. |
| **Full CPose Pipeline** | ADL clip export rate | Phụ thuộc `seq_len=48`, `stride=12`; ước lượng 1 clip mỗi 12 frame sau khi đủ buffer. |

---

## 2. Hướng dẫn Test Nhanh

Sử dụng file video đã được chỉ định sẵn trong thư mục `data/input/` để chạy kiểm thử luồng AI.

**Video dùng để Test:**
`D:\Capstone_Project\data\input\cam2_2026-01-29_16-26-40.mp4`

**Cách chạy Backend (FastAPI / Websocket):**
1. Mở Terminal / Command Prompt.
2. Di chuyển vào thư mục dự án:
   `cd D:\Capstone_Project\bathanh0309\cpose`
3. Khởi động Backend API (Cổng 8000):
   `uvicorn main:app --reload`

**Cách sử dụng trên UI:**
1. Mở file `static/index.html` bằng trình duyệt.
2. Dán đường dẫn test ở trên vào ô nhập dữ liệu của Camera 1 hoặc Camera 2.
3. Chọn các module cần chạy (Track, Pose, ReID, ADL).
4. Nhấn nút **▶ ON** để bắt đầu phân tích thời gian thực.




