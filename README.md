# CPose — CapstoneProject

## Công nghệ & Công cụ

| Lớp | Công nghệ & Thư viện | Vai trò cốt lõi |
|---|---|---|
| **Frontend UI** | HTML5, Vanilla JS, CSS3, Socket.IO Client | Dashboard giao diện đa luồng, giám sát & điều khiển thời gian thực (Real-time). |
| **Backend API** | Python, Flask, Flask-SocketIO | Web Server API, cầu nối giao tiếp hai chiều và bộ lên lịch Pipeline. |
| **AI / Model** | PyTorch, Ultralytics (YOLOv8) | Giải thuật Deep Learning nhận diện Person và trích xuất điểm ảnh (Pose Estimation). |
| **Xử lý Ảnh (CV)**| OpenCV (`cv2`) | Xử lý mảng ma trận pixel, nội suy ảnh và vẽ chồng biểu đồ khung xương (Overlay). |
| **Toán học & Data**| Numpy, FAISS, Dataclasses | Xử lý hình học đa hướng (Intersection over Union), khớp không gian và quản lý trạng thái. |

## Phương pháp thuật toán & Kiến trúc

| Kỹ thuật / Thuật toán | Mô tả giải pháp | Mục đích / Hiệu quả đem lại |
|---|---|---|
| **TFCS-PAR** | Không gian-Thời gian đa cam (*Time-First Cross-Camera Sequential Pose-ADL-ReID*). | Xử lý tình trạng mất ID người qua các vùng mù giữa nhiều Camera (Hành lang, thang máy). |
| **BBox IoU Tracking** | Bám vết Bounding Box bằng tỉ lệ độ phủ giao thoa và khoảng cách lệch tâm. | Cực kì nhẹ, tốc độ cao (0 độ trễ mạng), thay thế gọn gàng cho các thuật toán cực nặng như DeepSORT. |
| **Temporal Smoothing** | Bộ đệm *Hysteresis* lấy mẫu hành vi quá khứ nhằm tái tạo hoặc giữ nhịp tương lai (5-8 frames). | Triệt tiêu hoàn toàn sự "nháy giật" (flickering) khi khung xương/nhãn dự đoán bị mất trong vài nhịp của camera. |
| **Re-ID Global Map** | Từ điển ánh xạ liên thông `CrossCameraIDMerger` (Tra nối ID ảo - ID thật). | Duy trì xuyên suốt 1 mã người dùng (VD: Person_12) cho dù xuất hiện ngẫu nhiên ở 4 màn hình khác nhau. |
| **MJPEG Polling** | Client liên tục trích xuất gói JPEG sinh động từ Backend thông qua RAM Memory. | Trình diễn độ trễ bằng 0 (Zero-latency) luồng video xử lý khung xương trực tiếp, bỏ qua gánh nặng encode MP4/H264. |

## Cài đặt & Thiết lập

### 1. Cách tự động (Khuyên dùng)
Chạy tệp `run.bat` ở thư mục gốc. Script này sẽ tự động:
- Khởi tạo môi trường ảo Python (`venv`).
- Cài đặt thư viện từ `requirements.txt`.
- Tạo các thư mục dữ liệu cần thiết.

### 2. Cách thủ công (Dành cho Developer)

**Môi trường Python:**
```bash
# Tạo và kích hoạt venv
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/macOS

# Cài đặt thư viện backend
pip install -r requirements.txt
```

**Môi trường Frontend:**
```bash
npm install
# Hoặc:
pnpm install
```

## Khởi động 
Mở Terminal trực tiếp tại thư mục và chạy:
```cmd
run.bat
```
Truy cập Dashboard điều khiển hệ thống tại: `http://localhost:5000`
