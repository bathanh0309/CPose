```
cài build-tools do có insightface
https://visualstudio.microsoft.com/visual-cpp-build-tools/

cài python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install cython

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

```
Capstone_Project/
├── cpose/                          # THƯ VIỆN AI LÕI (Model-level wrappers, không dính tới Web)
│   ├── core/                       # Các module xử lý AI cấp thấp
│   │   ├── detection/              # Detector wrappers cho YOLO/RTDETR
│   │   │   ├── base.py             # Interface BaseDetector định nghĩa hàm detect()
│   │   │   ├── yolo_ultra.py       # Wrapper cho Ultralytics YOLO nhận diện người
│   │   │   └── factory.py          # Hàm build_detector() khởi tạo theo cấu hình
│   │   ├── pose_estimation/        # Module ước lượng khung xương
│   │   │   ├── base.py             # Interface BasePoseEstimator định nghĩa hàm estimate()
│   │   │   ├── yolo_pose.py        # Wrapper cho YOLO-Pose trích xuất 17 điểm khớp
│   │   │   ├── rtmpose.py          # Wrapper cho RTMPose (tùy chọn hiệu năng cao)
│   │   │   └── factory.py          # Hàm build_pose_estimator() khởi tạo bộ ước lượng
│   │   ├── face/                   # Module nhận diện khuôn mặt (ArcFace, SFace)
│   │   │   ├── base.py             # Interface BaseFaceRecognizer (encode/compare)
│   │   │   ├── insightface_arcface.py # Trích xuất 512-d embedding dùng ArcFace
│   │   │   └── factory.py          # Hàm build_face_recognizer() khởi tạo bộ nhận diện
│   │   ├── reid/                   # Module nhận dạng lại người (Body feature)
│   │   │   ├── base.py             # Interface BaseReIDModel định nghĩa hàm encode_body()
│   │   │   └── simple_body_reid.py # ReID dựa trên màu sắc và hình dáng cơ thể
│   │   └── vectordb/               # Tầng trừu tượng hóa cơ sở dữ liệu Vector
│   │       ├── base.py             # Interface BaseVectorDB (add/search/remove)
│   │       ├── faiss_db.py         # Backend FAISS hỗ trợ tìm kiếm láng giềng gần nhất
│   │       └── factory.py          # Hàm build_vectordb() khởi tạo backend (CPU/GPU)
│   ├── io/                         # Công cụ hỗ trợ Input/Output video chuyên dụng
│   │   ├── video_reader.py         # Đọc video hiệu năng cao dưới dạng generator
│   │   ├── camera_stream.py        # Quản lý luồng RTSP/Webcam với tự động kết nối lại
│   │   └── sink_writer.py          # Ghi frame kèm overlay (xương, nhãn) ra file video
│   ├── pipeline/                   # Các luồng tích hợp sẵn (Phase 1, 2, 3)
│   │   ├── multicam_recorder.py    # Phase 1: Nhận diện và cắt clip tự động
│   │   ├── multicam_analyzer.py    # Phase 2: Chạy Pose/Detection offline sinh file label
│   │   └── multicam_recognizer.py  # Phase 3: Tích hợp Pose+ADL+ReID cho multicam demo
│   └── utils/                      # Các tiện ích toán học và xử lý logic thuần túy
│       ├── pose_ops.py             # Tính góc khớp, tư thế và vector đặc trưng ADL
│       ├── bbox_ops.py             # Xử lý hộp bao: IOU, NMS, lọc kích thước
│       └── timing.py               # Đo FPS và phân tích độ trễ của các module
├── app/                            # ỨNG DỤNG DASHBOARD (Product Layer)
│   ├── api/                        # Tầng giao tiếp HTTP & WebSocket
│   │   ├── routes.py               # REST Endpoints: /config, /pose, /registration...
│   │   └── ws_handlers.py          # Sự kiện Socket.IO: camera_status, pose_status...
│   ├── core/                       # Logic nghiệp vụ của sản phẩm
│   │   ├── global_id.py            # GlobalPersonTable: Ánh xạ ID xuyên camera
│   │   ├── reid_logic.py           # Logic ReID thực tế dùng vectordb và cpose.core.reid
│   │   └── tracking.py             # Bộ theo dấu cục bộ (Local Tracker) trong từng clip
│   ├── services/                   # Lớp dịch vụ bao bọc các pipeline của cpose
│   │   ├── recorder_service.py     # Điều phối Phase 1 và thông báo trạng thái qua Socket
│   │   ├── analyzer_service.py     # Quản lý hàng đợi công việc gán nhãn Phase 2
│   │   ├── recognizer_service.py   # Điều phối Phase 3 và tổng hợp kết quả Pose+ADL
│   │   └── registration_service.py # Quản lý quy trình đăng ký danh tính khuôn mặt
│   ├── storage/                    # Tầng lưu trữ dữ liệu bền vững
│   │   ├── vector_db.py            # Quản lý index FAISS và hồ sơ người dùng
│   │   └── persistence.py          # Lưu trữ metadata, mapping ID và đường dẫn embedding
│   ├── utils/                      # Tiện ích phục vụ riêng cho ứng dụng web
│   │   ├── pose_utils.py           # Áp dụng các quy tắc ADL trả về nhãn hành động
│   │   ├── file_handler.py         # Quản lý file hệ thống và danh sách resources.txt
│   │   ├── stream_probe.py         # Kiểm tra thông tin luồng RTSP (FPS, Resolution)
│   │   └── runtime_config.py       # Tải và xác thực cấu hình config.yaml bằng Pydantic
│   └── bootstrap/                  # Khởi tạo ứng dụng Flask/SocketIO
│       ├── app_factory.py          # Hàm create_app() cấu hình server và đăng ký routes
│       └── config_loader.py        # Kiểm tra và tải cấu hình từ file YAML
├── research/                       # MÔI TRƯỜNG NGHIÊN CỨU (FastAPI Research Server)
│   ├── api/                        # Endpoint điều khiển thực nghiệm (Experiment Control)
│   │   ├── routes_experiments.py   # Quản lý trạng thái và kết quả các lần chạy thử
│   │   └── ws_experiments.py       # Stream tiến độ thực nghiệm thời gian thực
│   ├── services/                   # Logic xử lý nghiên cứu
│   │   ├── experiment_runner.py    # Thực thi các pipeline thử nghiệm trên tập dữ liệu
│   │   ├── model_registry.py       # Quản lý danh sách các mô hình đang thử nghiệm
│   │   └── benchmark_service.py    # Tính toán các chỉ số mAP, Accuracy, FPS...
│   └── schemas/                    # Định nghĩa cấu trúc dữ liệu Pydantic cho Research
├── shared/                         # THÀNH PHẦN CHIA SẺ (Dùng chung cho App & Research)
│   ├── io/                         # Quản lý đường dẫn và tệp tin tập trung
│   │   ├── paths.py                # Single source of truth cho mọi thư mục data/model
│   │   └── job_store.py            # Quản lý vòng đời trạng thái của các tác vụ AI
│   ├── contracts/                  # Định nghĩa các bản giao kèo dữ liệu (TypedDict)
│   └── adapters/                   # Lớp chuyển đổi giao tiếp giữa Flask và FastAPI
├── data/                           # KHO DỮ LIỆU (Video, Labels, Results)
│   ├── config/resources.txt        # Danh sách nguồn camera đầu vào
│   ├── multicam/                   # Dữ liệu video cho demo đa camera
│   ├── raw_videos/                 # Clip thô thu thập từ Phase 1
│   ├── output_labels/              # File nhãn JSON sinh ra từ Phase 2
│   ├── output_pose/                # Kết quả video và JSON từ Phase 3
│   └── research_runs/              # Nhật ký và kết quả các lần chạy nghiên cứu
├── models/                         # TRỌNG SỐ MÔ HÌNH (Product & Research)
├── static/                         # GIAO DIỆN NGƯỜI DÙNG (HTML/JS/CSS)
├── main.py                         # Điểm khởi đầu của ứng dụng Flask Dashboard
├── requirements.txt                # Danh sách thư viện cần cài đặt
├── AGENTS.md                       # Bản quy định bắt buộc cho AI trợ giúp dự án
├── PIPELINE.md                     # Tài liệu quy trình 3 giai đoạn xử lý
├── TFCS-PAR.md                     # Đặc tả thuật toán tracking xuyên camera
└── README.md                       # Tài liệu hướng dẫn sử dụng tổng quát

```

## Sơ đồ luồng dữ liệu (Data Flow)

```mermaid
    graph TD
    subgraph "1. Khởi tạo & nạp dữ liệu"
        A1[User tải resources.txt] --> A2[Backend: /api/config/upload<br/>Parse camera list]
        A3[User tải folder multicam] --> A4[Backend: đọc file .mp4<br/>Lọc đúng pattern camX_YYYY-MM-DD_HH-mm-ss.mp4]
        A4 --> A5[Parse timestamp & sắp xếp<br/>theo thời gian tăng dần]
        A5 --> A6[Tạo ClipQueue với trạng thái 'ready']
    end

    subgraph "2. Pipeline chính (START)"
        B1[Nhấn START] --> B2{Xác định mode}
        B2 -- resourcesLoaded --> B3[Mode RTSP: ghi short clip khi có người]
        B2 -- folderLoaded --> B4[Mode MULTICAM_FOLDER]
        B4 --> B5[Lấy clip tiếp theo từ Queue<br/>(đã sort time-first)]
    end

    subgraph "3. Xử lý từng clip (tuần tự)"
        C1[Bật đèn camera tương ứng] --> C2[Mở video & local tracking<br/>(DeepSORT)]
        C2 --> C3[Pose Estimation<br/>YOLO-pose → 17 keypoints]
        C3 --> C4[ADL Classification<br/>Rule-based (torso, knee, velocity...)]
        C4 --> C5[Cross-camera ReID<br/>+ spatiotemporal gating]
        C5 --> C6[FAISS Vector DB<br/>tra cứu/ghi embedding]
        C6 --> C7[Gán Global ID<br/>(giữ ID cũ nếu khớp, tạo mới nếu cần)]
        C7 --> C8[Ghi kết quả:<br/>- Video overlay<br/>- Keypoints, ADL, tracks, timeline JSON]
        C8 --> C9[Tắt đèn camera & phát socket event<br/>pose_progress, clip_saved]
        C9 --> C10{Còn clip trong Queue?}
        C10 -- Có --> B5
        C10 -- Không --> D[Kết thúc pipeline]
    end

    subgraph "4. Đăng ký người dùng (Registration)"
        E1[User mở webcam] --> E2[Backend tạo session]
        E2 --> E3[Thu thập ảnh mặt & embedding<br/>InsightFace ArcFace]
        E3 --> E4[Lưu embedding + metadata vào FAISS]
        E4 --> E5[Emit registration_done]
    end

    A2 --> B2
    A6 --> B2
    B3 --> C1
    C8 --> K[Web Dashboard / SocketIO]
    E5 --> K
    C6 --> I[(FAISS Vector DB)]
    I --> C6
```
