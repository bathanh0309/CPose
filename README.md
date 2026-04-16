# CPose — Capstone Project (Advanced Human Activity Recognition)

CPose là một hệ thống giám sát thông minh tập trung vào nhận diện hành vi con người (Human Activity Recognition - HAR) thông qua mạng lưới Camera đa hướng (Multi-camera). Hệ thống sử dụng các mô hình Deep Learning tiên tiến để theo dõi, nhận diện tư thế và phân loại hành động của con người trong thời gian thực.

## Công nghệ & Công cụ (Tech Stack)

Hệ thống được xây dựng trên nền tảng kiến trúc linh hoạt, cho phép tích hợp cả ứng dụng thực tế (Product) và các nghiên cứu chuyên sâu (Research).

| Lớp | Công nghệ & Thư viện | Vai trò cốt lõi |
|:---|:---|:---|
| **Frontend UI** | HTML5, Vanilla JS, CSS3, Socket.IO Client | Dashboard điều khiển đa luồng, giám sát trạng thái & log thời gian thực. |
| **Backend API** | Python, Flask, Flask-SocketIO, Eventlet | Web Server hiệu suất cao, xử lý luồng (Concurrency) và Broker dữ liệu. |
| **AI / Model** | PyTorch, Ultralytics YOLOv8 / YOLO-pose | Nhận diện đối tượng (Object Detection) và ước lượng tư thế (Pose Estimation). |
| **Xử lý Ảnh/Video**| OpenCV (`cv2`), Pillow | Xử lý khung hình, giải mã/mã hóa luồng RTSP và vẽ Overlay kết quả. |
| **Dữ liệu & Vector**| Numpy, FAISS, Pandas | Xử lý mảng đa chiều, tính toán hình học (IoU) và tìm kiếm Vector Re-ID. |
| **Logging & UX**| Loguru, Rich, TQDM | Quản lý nhật ký hệ thống chuyên nghiệp và thanh tiến trình pipeline. |

---

## Phương pháp & Kỹ thuật cốt lõi

Dưới đây là các kỹ thuật và phương pháp chuyên sâu được áp dụng trong CPose để giải quyết các bài toán về độ trễ, nhiễu dữ liệu và bám vết xuyên camera.

| Kỹ thuật / Phương pháp | Mô tả chi tiết | Mục đích & Hiệu quả mang lại |
|:---|:---|:---|
| **TFCS-PAR Architecture** | *Time-First Cross-Camera Sequential Pose-ADL-ReID*. | Giải quyết bài toán mất ID khi người di chuyển qua các vùng mù giữa 4 Camera. |
| **BBox IoU Tracking** | Bám vết Bounding Box dựa trên chỉ số Intersection over Union và khoảng cách trọng tâm. | Tốc độ cực nhanh (Zero-lag), thay thế các giải pháp nặng như DeepSORT khi phần cứng hạn chế. |
| **Temporal Smoothing** | Sử dụng bộ đệm *Hysteresis* (5-8 frames) để trung hòa sai số dự đoán theo thời gian. | Loại bỏ hiện tượng "flickering" (nhấp nháy) của khung xương và nhãn hành động khi tín hiệu yếu. |
| **Re-ID Global Mapping** | Ánh xạ ID cục bộ (Local ID) sang ID toàn cục (Global ID) thông qua `CrossCameraIDMerger`. | Duy trì danh tính duy nhất của một người kể cả khi họ rời camera này và xuất hiện ở camera khác. |
| **FAISS Vector Search** | Sử dụng thư viện FAISS để tìm kiếm độ tương đồng vector đặc trưng giữa các người dùng. | Thực hiện Re-Identification (Re-ID) quy mô lớn với tốc độ tìm kiếm sub-millisecond. |
| **MJPEG Stream Polling** | Kỹ thuật trích xuất gói JPEG trực tiếp từ RAM Memory thay vì H.264 encoding. | Trình chiếu kết quả xử lý AI lên trình duyệt với độ trễ gần như bằng không (Zero-latency). |
| **Skeleton-based ADL** | Nhận diện hành động (ADL) dựa trên tọa độ 17 điểm keyword của cơ thể. | Giảm chi phí tính toán so với Video-based ADL, tăng độ chính xác nhờ tập trung vào chuyển động khớp. |
| **Job Queue & Worker** | Tách rời việc ghi hình (Producer) và xử lý AI (Consumer) qua hàng đợi clip. | Đảm bảo không mất khung hình khi GPU đạt đỉnh tải (Peak load) và tối ưu hóa I/O video. |

---

## Kiến trúc Pipeline Offline

Ngoài giao diện Real-time, hệ thống cung cấp pipeline xử lý offline mạnh mẽ:

1.  **Phase 1 (Recorder):** `scripts\run-phase1-recorder.bat`
    - Đọc `resources.txt`, ghi clip từ RTSP vào `data/raw_videos/`.
2.  **Phase 2 (Analyzer):** `scripts\run-phase2-analyzer.bat`
    - Trích xuất BBox và lưu labels vào `data/output_labels/`.
3.  **Phase 3 (ADL Recognition):** `scripts\run-phase3-adl.bat`
    - Nhận diện Pose & ADL, lưu kết quả ổn định vào `output_pose/`.

---

## Hướng dẫn cài đặt & Khởi chạy

### 1. Chuẩn bị môi trường (Windows)

Mở Terminal tại thư mục gốc và chuẩn bị môi trường Python:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Khởi động Dashboard

Để chạy phiên bản thương mại (Product), sử dụng:
```cmd
scripts\run-product.bat
```
Hoặc phiên bản dành cho nhà phát triển (Dev):
```cmd
scripts\run-dev.bat
```
Sau khi khởi động, truy cập giao diện điều khiển tại địa chỉ: `http://localhost:5000`

---

## 📈 Hướng phát triển (Research)

Hệ thống đã sẵn sàng các cổng kết nối (Interface) để tích hợp các mô hình nghiên cứu SOTA:
- **CTR-GCN / BlockGCN:** Tích hợp qua bộ thư viện `PYSKL` để nâng cao độ chính xác nhận diện hành động phức tạp.
- **Deep Re-ID:** Sử dụng `Pose2ID` để nhận dạng người dùng dựa trên khung xương khi khuôn mặt bị che khuất.
