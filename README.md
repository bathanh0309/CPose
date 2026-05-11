# CPose — Multi-Camera AI Pipeline (TFCS-PAR)

## Cài đặt


```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Đặt file `.mp4` vào thư mục:

```
data-test/
├── cam1_2026-01-29_16-26-25.mp4
├── cam2_2026-01-28_15-57-54.mp4
└── ...   (đặt tên theo format: camX_YYYY-MM-DD_HH-MM-SS.mp4)
```

---

## Các file chạy (`.bat`)

Chạy tuần tự từ Module 1 đến Module 7, hoặc chọn module cần thiết.

| File | Module | Mô tả |
|---|---|---|
| `run_01_detection.bat` | Detection | Phát hiện người, lưu `detections.json` + crops |
| `run_02_tracking.bat` | Tracking | Theo dõi cục bộ, lưu `tracks.json` + `tracklets.json` + trajectory video |
| `run_03_pose.bat` | Pose | Ước lượng 17 điểm khớp, lưu `keypoints.json` |
| `run_04_adl.bat` | ADL | Nhận diện hành vi rule-based, lưu `adl_events.json` |
| `run_05_reid.bat` | Global ReID | Định danh xuyên camera TFCS-PAR, lưu `reid_tracks.json` |
| `run_06_pipeline.bat` | Full Pipeline | Chạy toàn bộ 5 module liên tiếp |
| `run_07_benchmark.bat` | Benchmark | Tổng hợp metrics tất cả module, xuất CSV + paper tables |

## Cấu hình

| File | Mục đích |
|---|---|
| `configs/profiles/dev.yaml` | Profile dev, override tren nen `configs/base/*.yaml` |
| `configs/camera/topology.yaml` | Ban do ket noi giua cac camera (dung cho ReID) |
| `configs/camera/multicam_manifest.json` | Metadata video: camera_id, start_time |
| `configs/unified_config.yaml` | Cau hinh app-level duy nhat thay cho `config.yaml` / `cpose_default.yaml` |

---

## Dataset benchmark

Tai COCO val2017 va Market-1501 theo huong dan trong [`docs/DATASETS.md`](docs/DATASETS.md). Hai script chinh:

```bat
bash scripts/download_coco.sh
bash scripts/download_market1501.sh
```

---

## Kết quả thực nghiệm

Bảng dưới đây tổng hợp dataset tham chiếu của từng module. Cột **Kết quả** gồm 3 dòng:

- **Ground truth (proxy):** Đo trực tiếp trên `data-test` (không có annotation).
- **Literature:** Số liệu tốt nhất từ paper gốc (nguồn: [`PAPERS.md`](PAPERS.md)).
- **CPose paper:** Kết quả sẽ điền sau khi có annotation + chạy benchmark.

| Dataset | Module | Kỹ thuật | Kết quả |
|---|---|---|---|
| **[MS COCO Keypoints](https://cocodataset.org)** (pretrain) <br> **CPose data-test** (inference) | **Detection** | YOLOv8n <br> conf=0.50, imgsz=640 |  **Proxy:** FPS ≈ 30–60 · Avg conf ≈ 0.82 · Avg quality ≈ 0.74 <br>  **Lit.:** YOLOv8n — COCO val mAP@50=52.3 <br>  **CPose:** *(chờ annotation — chạy `run_07_benchmark.bat`)* |
| **[MOT17](https://motchallenge.net/data/MOT17/)** (literature ref) <br> **CPose data-test** (inference) | **Tracking** | YOLOv8n + ByteTrack <br> ([ECCV 2022](https://arxiv.org/abs/2110.06864)) · min_hits=3 | **Proxy:** Confirmed ratio ≈ 0.80 · FragProxy < 0.15 <br> **Lit.:** ByteTrack — HOTA=63.1 · IDF1=77.3 · MOT17 <br>  **CPose:** *(chờ annotation)* |
| **[MS COCO Keypoints](https://cocodataset.org)** (pretrain) <br> **CPose data-test** (inference) | **Pose Estimation** | YOLOv8n-Pose <br> ([RTMPose](https://arxiv.org/abs/2303.07399)) · keypoint_conf=0.30 |  **Proxy:** Visible keypoint ratio ≈ 0.72 <br>  **Lit.:** RTMPose-m — 75.8 AP COCO · 90+ FPS CPU i7 <br>  **CPose:** *(chờ annotation)* |
| **[Toyota Smarthome](https://project.inria.fr/toyotasmarthome)** (literature ref) <br> **CPose data-test** (inference) | **ADL Recognition** | Rule-based TFCS <br> window=30 · smoothing=7 | **Proxy:** 5 nhãn: walking / standing / sitting / lying_down / falling <br>  **Lit.:** π-ViT ([CVPR 2024](https://arxiv.org/abs/2311.18840)) — 72.9% mCA Smarthome CS <br>  **CPose:** *(chờ annotation — so sánh với BlockGCN phase 3)* |
| **[Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501)** + **[DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID)** (literature ref) <br> **CPose data-test** (inference) | **Global ReID** | TFCS-PAR <br> strong_thresh=0.65 · topology-aware |  **Proxy:** Global IDs gán xuyên 4 camera <br>  **Lit.:** MCPT — 86.76% IDF1 AI City 2023 <br>  **CPose:** *(chờ annotation — global_person_table.json)* |

> **Chú thích:**
>
> - **Proxy** = metric tính không cần ground truth (tự đo trên `data-test`).
> - **Lit.** = số liệu từ paper gốc, **không phải** kết quả CPose — chỉ dùng để so sánh.
> - **CPose paper** = ô trống cho đến khi có annotation. Chạy `run_07_benchmark.bat` để điền.
