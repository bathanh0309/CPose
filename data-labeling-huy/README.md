# Multi-Camera RTSP Data Collection & Auto-Labeling System

A high-performance, automated pipeline designed for real-time human detection from multiple RTSP streams, event-based recording, and high-precision offline labeling using **YOLOv11**.

---

## 🚀 Key Features

### 1. Phase 1: Real-time Data Collection (Background Service)
#### RTSP Ingestion ➔ YOLOv11-Nano Detection ➔ Motion-triggered MP4 Saving.
* **Automated Ingestion:** Dynamically reads RTSP stream configurations from `resources.txt`.
* **Edge Inference:** Integrated with a **Light-weight YOLOv11** model optimized for background execution.
* **Resource Efficiency:** Engineered for minimal CPU/GPU overhead to prevent frame drops and maintain stream integrity.
* **Event-based Recording:** Automatically triggers video capture upon human detection.
* **Standardized Output:** Saves clips in MP4 format using the convention: `yyyymmdd_CamXX.mp4`.

### 2. Phase 2: Offline Auto-Labeling (Post-processing)
#### MP4 Assets ➔ YOLOv11-Large Inference ➔ Frame Extraction & TXT Labeling.
* **Asynchronous Processing:** Decoupled from the live stream service to ensure system stability.
* **High-Precision Inference:** Utilizes the **YOLOv11 (Large/Extra-Large)** model for maximum detection accuracy.
* **Dataset Generation:** Automatically extracts frames and generates annotation files.

### 3. Web-based Management Dashboard (HTML5)
#### Real-time monitoring & Background task tracking.
* **Camera Status Monitor:** Real-time telemetry including Resolution, Bitrate, Online/Offline status, and actual FPS.
* **Selective Live View:** Single-channel preview mode to optimize UI bandwidth and client-side performance.
* **Non-disruptive Control:** UI controls (Play/Stop) only toggle the web preview; the backend collection engine (Phase 1) remains persistent.
* **Task Analytics:** Progress tracking for Phase 2 labeling queues.

---

## 🛠 Tech Stack
* **Core:** Python 3.x
* **AI Engine:** Ultralytics YOLOv11
* **Backend:** FastAPI / Starlette
* **Processing:** OpenCV, FFmpeg
* **Frontend:** HTML5, CSS3 (Dashboard)

---

## 📂 Project Structure
```text
├── app/
│   ├── api/             # API Route handlers
│   ├── services/        # Phase 1 & Phase 2 core logic
│   └── utils/           # File handling and helpers
├── data/
│   ├── config/          # resources.txt (RTSP links)
│   ├── raw_videos/      # Recorded MP4 clips (Git ignored)
│   └── output_labels/   # Generated frames and txt files
├── models/              # Models containers 
├── static/              # Web Dashboard assets
└── requirements.txt     # Project dependencies