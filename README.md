# CPose (Cross camera Pose)

> Real-time pipeline: **YOLO11-Pose** -> **ByteTrack** -> **FastReID** -> **PoseC3D** -> ADL classification

---

## Pipeline Overview

```text
Video / Camera frame
    |
    v
[YoloPoseTracker]    -> bbox + keypoints + local track_id
    v
[ByteTrackWrapper]   -> stable tracking across frames
    v
[FastReIDExtractor]  -> appearance embedding (L2-norm)
    v
[ReIDGallery]        -> matched person_id
    v
[GlobalIDManager]    -> cross-camera global_id
    v
[PoseSequenceBuffer] -> .pkl clip (MMAction2 format, seq_len frames)
    v
[PoseC3DRunner]      -> ADL action label
    v
[EventBus]           -> JSONL event log
```

---

## Model Weights

Tải về và đặt đúng vị trí theo bảng dưới đây.

| Model | File name | Local path | Size | Download |
|---|---|---|---|---|
| YOLO11n-Pose | `yolo11n-pose.pt` | `models/yolo11n-pose.pt` | ~5 MB | [Ultralytics YOLO11 Pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) |
| ByteTrack (MOT17) | `bytetrack_s_mot17.pth.tar` | `models/bytetrack_s_mot17.pth.tar` | ~69 MB | [ByteTrack Releases](https://github.com/ifzhang/ByteTrack/releases/download/v0.1/bytetrack_s_mot17.pth.tar) |
| FastReID Market-1501 R50 | `fastreid_market_R50.pth` | `models/fastreid_market_R50.pth` | ~287 MB | [JDAI-CV FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) |
| PoseC3D NTU-60 R50 | `posec3d_r50_ntu60.pth` | `models/posec3d_r50_ntu60.pth` | ~200 MB | [MMAction2 Model Zoo](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md) |

### Cấu trúc sau khi tải

```text
D:/Capstone_Project/
└── models/
    ├── yolo11n-pose.pt
    ├── bytetrack_s_mot17.pth.tar
    ├── fastreid_market_R50.pth
    └── posec3d_r50_ntu60.pth
```

### Config tương ứng trong `configs/system/pipeline.yaml`

```yaml
pose:
  weights: models/yolo11n-pose.pt

tracker:
  tracker_yaml: bytetrack.yaml

reid:
  weights: models/fastreid_market_R50.pth

adl:
  weights: models/posec3d_r50_ntu60.pth
```

---

## Installation

### 1. Tạo virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài PyTorch

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### 3. Cài OpenMMLab stack

```bash
pip install --upgrade pip
pip install "setuptools<82" wheel
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
pip install mmaction2
```

### 4. Cài dependencies còn lại

```bash
pip install -r requirements.txt
```

---

## Project Structure

Ghi chú: `.git/`, `.venv/`, `__pycache__/` và các file cache sinh tự động không được liệt kê chi tiết. Các nhóm file lặp lại như video `.mp4`, embedding `.npy`, clip `.pkl`.

| Path | Loại | Chức năng |
|---|---|---|
| `.github/` | Folder | Chứa thư viện ngoài clone về để chạy pipeline: ByteTrack, FastReID, PoseC3D. |
| `.github/ByteTrack/` | Folder | Mã nguồn ByteTrack gốc, tài liệu, config và script thử nghiệm. |
| `.github/fast-reid/` | Folder | Mã nguồn FastReID gốc, dùng bởi `src/reid/fast_reid.py`. |
| `.github/pose-c3d/` | Folder | Mã nguồn PoseC3D/pyskl, dùng bởi `src/action/posec3d.py`. |
| `apps/` | Folder | Các entry point chạy từng module hoặc toàn bộ pipeline. |
| `apps/run_pose.py` | File | Chạy YOLO11-Pose + tracking để hiển thị bbox, skeleton và track id. |
| `apps/run_track.py` | File | Chạy YOLO11-Pose qua `ByteTrackWrapper` để kiểm tra tracking ổn định. |
| `apps/run_reid.py` | File | Chạy pose/tracking kèm FastReID và panel đối chiếu danh tính. |
| `apps/run_adl.py` | File | Thu chuỗi skeleton, export clip `.pkl`, tùy chọn chạy PoseC3D. |
| `apps/run_pipeline.py` | File | Chạy pipeline đầy đủ: pose, tracking, ReID, global id, ADL và event log. |
| `apps/run_web_cmd.py` | File | Khởi động FastAPI bằng uvicorn, mở trình duyệt và dừng process bằng Ctrl+Z. |
| `configs/` | Folder | Chứa cấu hình hệ thống và model. |
| `configs/system/` | Folder | Cấu hình chính của pipeline. |
| `configs/system/pipeline.yaml` | File | Khai báo device, source mặc định, output, YOLO, tracker, ReID và ADL. |
| `configs/fast-reid/` | Folder | Config riêng cho FastReID. |
| `configs/fast-reid/R50.yml` | File | Kế thừa config Market1501 BagTricks R50 và trỏ tới weight FastReID. |
| `configs/posec3d/` | Folder | Config riêng cho PoseC3D. |
| `configs/posec3d/posec3d.py` | File | Kế thừa config PoseC3D NTU60, đặt checkpoint, work dir và số class. |
| `data/` | Folder | Chứa nguồn video, gallery nhận diện, upload, output và config camera. |
| `data/config/` | Folder | Chứa cấu hình nguồn camera/RTSP cho web UI. |
| `data/config/resources.txt` | File | Danh sách camera dạng `Tên camera__rtsp://...` để backend đọc. |
| `data/face/` | Folder | Gallery embedding khuôn mặt/người cho ReID theo từng identity. |
| `data/face/APhu/` | Folder | Embedding đại diện cho identity `APhu`. |
| `data/face/APhu/meta.json` | File | Metadata định danh cho identity `APhu`. |
| `data/face/APhu/emb_00.npy` | File đại diện | Một embedding mẫu; các file `emb_01.npy`...`emb_11.npy` cùng chức năng. |
| `data/face/Huy/` | Folder | Embedding đại diện cho identity `Huy`. |
| `data/face/Huy/meta.json` | File | Metadata định danh cho identity `Huy`. |
| `data/face/Huy/emb_00.npy` | File đại diện | Một embedding mẫu; các file `emb_01.npy`...`emb_11.npy` cùng chức năng. |
| `data/face/Thanh/` | Folder | Embedding đại diện cho identity `Thanh`. |
| `data/face/Thanh/meta.json` | File | Metadata định danh cho identity `Thanh`. |
| `data/face/Thanh/emb_00.npy` | File đại diện | Một embedding mẫu; các file `emb_01.npy`...`emb_11.npy` cùng chức năng. |
| `data/face/Thanh/.gitkeep` | File | Giữ thư mục trong Git khi không có dữ liệu khác. |
| `data/gallery/` | Folder | Thư mục ảnh reference ReID nếu dùng gallery bằng ảnh. |
| `data/input/` | Folder | Video đầu vào mẫu cho pipeline command line. |
| `data/input/cam1_2026-01-29_16-26-25.mp4` | File đại diện | Video camera mẫu; các file `.mp4` khác trong `data/input` cùng chức năng. |
| `data/output/` | Folder | Kết quả sinh ra khi chạy pipeline. |
| `data/output/pose_results.json` | File | Kết quả pose đã xuất từ lần chạy trước. |
| `data/output/track_results.json` | File | Kết quả tracking đã xuất từ lần chạy trước. |
| `data/output/clips_pkl/` | Folder | Clip skeleton `.pkl` theo format MMAction2/PoseC3D. |
| `data/output/clips_pkl/c02_t001_g001_001.pkl` | File đại diện | Một clip skeleton mẫu; các file `.pkl` còn lại là các cửa sổ pose khác. |
| `data/output/clips_pkl/events/` | Folder | Event log JSONL của pipeline. |
| `data/output/clips_pkl/events/pipeline.jsonl` | File | Log sự kiện như track update, clip export và ADL result. |
| `data/output/vis/` | Folder | Video visualization đã render từ các module. |
| `data/output/vis/track_cam01.mp4` | File đại diện | Video output mẫu; các output `.mp4` khác nếu có cùng chức năng. |
| `data/skeleton/` | Folder | Vị trí dữ liệu skeleton trung gian nếu cần mở rộng dataset. |
| `data/uploads/` | Folder | Video do người dùng upload qua web UI. |
| `data/uploads/cam1_2026-01-29_16-26-25.mp4` | File đại diện | Video upload mẫu; các file `.mp4` khác trong `data/uploads` cùng chức năng. |
| `main.py` | File | FastAPI backend phục vụ web UI, upload video, đọc camera và stream frame qua WebSocket. |
| `models/` | Folder | Chứa model weights tải thủ công; không nên commit file weight lớn. |
| `models/yolo11n-pose.pt` | File model | Weight YOLO11n-Pose dùng để detect người và trích xuất 17 keypoint COCO. |
| `models/bytetrack_s_mot17.pth.tar` | File model | Checkpoint ByteTrack MOT17 tham khảo; pipeline hiện dùng `bytetrack.yaml` của Ultralytics. |
| `models/fastreid_market_R50.pth` | File model | Weight FastReID ResNet-50 Market1501 dùng để extract embedding nhận diện lại người. |
| `models/posec3d_r50_ntu60.pth` | File model | Weight PoseC3D R50 NTU60 dùng cho nhận diện hành động ADL từ skeleton clip. |
| `README.md` | File | Tài liệu tổng quan, cài đặt, cấu trúc file và hướng dẫn dự án. |
| `requirements.txt` | File | Danh sách package Python cần cài cho dự án. |
| `run-push-Me.bat` | File | Batch script tự động thêm/commit/push Git theo cấu hình cá nhân. |
| `run-push-MrPhu.bat` | File | Batch script tự động thêm/commit/push Git theo cấu hình MrPhu. |
| `run-web.bat` | File | Batch script chạy web backend bằng `apps/run_web_cmd.py`. |
| `src/` | Folder | Mã nguồn chính của CPose. |
| `src/action/` | Folder | Module xử lý ADL và PoseC3D. |
| `src/action/__init__.py` | File | Đánh dấu package `src.action`. |
| `src/action/pose_buffer.py` | File | Gom keypoint theo track thành cửa sổ `seq_len`, export `.pkl` cho PoseC3D. |
| `src/action/posec3d.py` | File | Wrapper gọi subprocess PoseC3D/MMAction2 để test clip skeleton. |
| `src/core/` | Folder | Thành phần lõi không phụ thuộc UI. |
| `src/core/event.py` | File | Ghi event pipeline ra JSONL hoặc bỏ qua bằng `NullEventBus`. |
| `src/core/global_id.py` | File | Quản lý map local track id sang global id bằng ReID và cache theo camera. |
| `src/detectors/` | Folder | Detector/pose estimator. |
| `src/detectors/yolo_pose.py` | File | Wrapper Ultralytics YOLO11-Pose, trả bbox, score, class, track id và keypoints. |
| `src/reid/` | Folder | Module nhận diện lại người. |
| `src/reid/fast_reid.py` | File | Load FastReID, extract embedding chuẩn hóa L2 từ crop người. |
| `src/reid/gallery.py` | File | Build gallery từ ảnh hoặc `.npy`, query cosine similarity và cập nhật prototype. |
| `src/trackers/` | Folder | Module tracking. |
| `src/trackers/bytetrack.py` | File | Wrapper mỏng quanh YOLO `.track(..., persist=True)` dùng ByteTrack của Ultralytics. |
| `src/utils/` | Folder | Tiện ích chung cho config, IO, device, video, logger, naming và visualization. |
| `src/utils/config.py` | File | Load/validate YAML config, resolve path tương đối và tự chọn CPU/CUDA. |
| `src/utils/device.py` | File | Resolve thiết bị Torch, fallback CPU nếu CUDA không khả dụng. |
| `src/utils/io.py` | File | Hàm tạo thư mục, đọc/ghi pickle, JSON, JSONL và timestamp ms. |
| `src/utils/logger.py` | File | Tạo logger chuẩn cho các module. |
| `src/utils/naming.py` | File | Chuẩn hóa tên camera/global id và sinh tên clip/video/json output. |
| `src/utils/video.py` | File | Tìm nguồn video mặc định, mở `cv2.VideoCapture`, lấy metadata và ghi video. |
| `src/utils/vis.py` | File | Vẽ bbox, skeleton COCO-17, panel info, trạng thái ADL, panel ReID và FPS. |
| `static/` | Folder | Frontend tĩnh cho control panel. |
| `static/index.html` | File | Giao diện 2 camera, chọn nguồn, upload video, bật/tắt module và log. |
| `static/scripts.js` | File | Gọi API camera/upload, quản lý WebSocket stream và cập nhật UI/log. |
| `static/style.css` | File | CSS layout dashboard, camera panel, nút, module toggle, stream và responsive. |

---

## Data Groups

| Nhóm | Số lượng hiện có | Đại diện | Chức năng |
|---|---:|---|---|
| Video input `.mp4` | 10 | `data/input/cam1_2026-01-29_16-26-25.mp4` | Dữ liệu camera mẫu để chạy pipeline. |
| Video upload `.mp4` | 2 | `data/uploads/cam1_2026-01-29_16-26-25.mp4` | Dữ liệu người dùng tải lên qua web UI. |
| Video output `.mp4` | 1 | `data/output/vis/track_cam01.mp4` | Video visualization sau khi chạy module. |
| Embedding `.npy` | 36 | `data/face/APhu/emb_00.npy` | Vector embedding theo identity cho ReID. |
| Clip skeleton `.pkl` | 29 | `data/output/clips_pkl/c02_t001_g001_001.pkl` | Cửa sổ keypoint đã export để PoseC3D đọc. |
