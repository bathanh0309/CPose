# CPose

> Real-time pipeline: **YOLO Pose** → **ByteTrack** → **FastReID** → **PoseC3D** → ADL recognition

---

## Pipeline Overview

```
Video / Camera frame
    │
    ▼
[YoloPoseTracker]   → bbox + keypoints + local track_id
    ▼
[ByteTrackWrapper]  → stable tracking
    ▼
[FastReIDExtractor] → appearance embedding
    ▼
[ReIDGallery]       → matched person_id
    ▼
[GlobalIDManager]   → cross-camera global_id
    ▼
[PoseSequenceBuffer]→ .pkl clip (MMAction2 format)
    ▼
[PoseC3DRunner]     → ADL action label
    ▼
[EventBus]          → JSONL event log
```

---

## Model

| Model | File name | Size | Download |
|-------|-----------|------|----------|
| YOLOv11-Pose (detection + keypoint) | `yolov11-pose.pt` | ~138 MB | [Ultralytics Releases](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n-pose.pt) |
| ByteTrack (MOT17 pretrained) | `bytetrack_s_mot17.pth.tar` | ~69 MB | [ByteTrack Releases](https://github.com/ifzhang/ByteTrack/releases/download/v0.1/bytetrack_s_mot17.pth.tar) |
| FastReID Market-1501 (ResNet-50) | `fastreid_market_R50.pth` | ~287 MB | [JDAI-CV FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) |
| PoseC3D ADL (finetuned) | `posec3d_r50_ntu60.pth` | ~200 MB |  [Pose C3D ](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d) |

**Sau khi tải về:**

```
CPose/
└── models/
    ├── yolov11n-pose.pt
    ├── bytetrack_s_mot17.pth.tar
    ├── fastreid_market_R50.pth
    └── posec3d_r50_ntu60.pth
```

Kiểm tra path trong `configs/system/pipeline.yaml`:

```yaml
detector:
  weights: models/yolov11n-pose.pt

tracker:
  weights: models/bytetrack_s_mot17.pth.tar

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
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Cài PyTorch (CUDA 11.8)

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

### 5. Clone external repos

```bash
# FastReID
git clone https://github.com/JDAI-CV/fast-reid.git third_party/fast-reid

# Cập nhật path trong configs/system/pipeline.yaml
# reid.fastreid_root: third_party/fast-reid
```

---

## Quick Start

```bash
# Chạy toàn bộ pipeline
python apps/run_pipeline.py --source data/sample.mp4 --camera-id cam01

# Debug từng module
python apps/run_pose.py --source 0              # webcam
python apps/run_reid.py --source data/test/
python apps/run_adl.py  --clip data/output/clips_pkl/sample.pkl
```

---

## Project Structure

```
CPose/
├── apps/                   # Entry points
├── configs/
│   ├── system/
│   │   └── pipeline.yaml   # ← MỌI config tập trung ở đây
│   ├── fast-reid/
│   └── posec3d/
├── src/
│   ├── action/             # PoseSequenceBuffer, PoseC3DRunner
│   ├── core/               # EventBus, GlobalIDManager
│   ├── detectors/          # YoloPoseTracker
│   ├── reid/               # FastReIDExtractor, ReIDGallery
│   ├── trackers/           # ByteTrackWrapper
│   └── utils/              # logger, config, io, vis
├── data/
│   ├── gallery/            # Ảnh reference ReID (không commit)
│   └── output/
├── models/                 # Weights (không commit — xem bảng trên)
├── third_party/            # fast-reid, mmaction2 (git clone)
└── CLAUDE.md               # AI agent project intelligence
```

---

## For AI Coding Agents

Đọc [`CLAUDE.md`](./CLAUDE.md) trước khi chạm vào bất kỳ file nào.

---
