# 🎯 HAVEN Multi-Camera Tracking System - Production Refactor

**Version:** 2.0 (Production-Ready)  
**Author:** Senior MLOps Engineer  
**Date:** 2026-02-02

---

## 📋 Executive Summary

This is a **complete refactor** of the HAVEN multi-camera person tracking and Re-Identification (ReID) system, transitioning from a **rule-based heuristic prototype** to a **production-grade deep learning system** that can:

✅ Track **1000+ people** without performance degradation (O(log N) search via FAISS)  
✅ **Persist state** across restarts (no data loss)  
✅ **Strict Master-Slave architecture**: Only cam2 creates GlobalIDs  
✅ **Robust cross-camera matching** using deep embeddings (OSNet)  
✅ **Spatiotemporal gating** to prevent physically impossible matches  
✅ **Dangerous Zone** and **Dangerous Object** detection modules  
✅ **Production observability**: structured logs, metrics, debug artifacts

---

## 🏗️ Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-CAMERA INPUT                       │
│  (cam1: display) (cam2: MASTER) (cam3: slave) (cam4: slave) │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Video Stream Manager   │
        │  (Multi-file segments)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   YOLO Detector +       │
        │   BoT-SORT Tracker      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   OSNet ReID Engine     │
        │   (512-dim embeddings)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  GlobalID Manager       │
        │  ┌──────────────────┐   │
        │  │ MASTER (cam2):   │   │
        │  │ Create G1, G2... │   │
        │  ├──────────────────┤   │
        │  │ SLAVE (cam3/4):  │   │
        │  │ Match or UNK     │   │
        │  └──────────────────┘   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  FAISS Vector Database  │
        │  (O(log N) ANN search)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Persistence Layer      │
        │  (SQLite + Memmap)      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Dangerous Zones +      │
        │  Objects Module         │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Visualization +        │
        │  Event Logging          │
        └─────────────────────────┘
```

---

## 🔑 Key Changes from Original System

| Aspect | Original (Heuristic) | Refactored (Production) |
|--------|----------------------|-------------------------|
| **ReID** | HSV histogram + Hu moments (176-dim) | OSNet deep embeddings (512-dim) |
| **Search** | O(N×K) linear scan | O(log N) FAISS HNSW |
| **Assignment** | Greedy first-come-first-serve | Hungarian algorithm (global optimization) |
| **Persistence** | Volatile (RAM only) | SQLite + memmap (durable) |
| **Scaling** | Fails at N>100 | Handles N=1000+ |
| **Domain Shift** | Poor (lighting sensitive) | Robust (learned features) |
| **Spatiotemporal** | None | Camera graph with travel time |
| **UNK Handling** | IoU resurrection only | IoU + temporal voting + quality gating |

---

## 📁 Directory Structure

```
haven_refactor/
├── core/
│   ├── global_id_manager.py      # Master-Slave logic ⭐
│   ├── reid_engine.py             # OSNet wrapper + feature bank
│   ├── spatiotemporal_gating.py   # Camera transition rules
│   └── matching_optimizer.py      # Hungarian assignment
│
├── models/
│   ├── osnet.py                   # OSNet ReID model
│   ├── detector.py                # YOLO wrapper
│   └── tracker.py                 # BoT-SORT integration
│
├── modules/
│   ├── dangerous_zone.py          # Polygon-based zone detection
│   ├── dangerous_object.py        # Weapon/fire detector
│   └── adl_detector.py            # (Optional) Pose-based ADL
│
├── pipeline/
│   ├── camera_stream.py           # Multi-file video loader
│   ├── processor.py               # Main inference loop
│   └── synchronizer.py            # Multi-cam sync
│
├── storage/
│   ├── persistence.py             # SQLite + memmap ⭐
│   └── vector_db.py               # FAISS wrapper ⭐
│
├── utils/
│   ├── metrics.py                 # IDS, MOTA, ReID accuracy
│   ├── logger.py                  # Structured JSON logging
│   └── visualizer.py              # Overlay rendering
│
├── config/
│   └── production.yaml            # Master configuration ⭐
│
├── tests/
│   ├── test_global_id_manager.py  # Unit tests
│   ├── test_reid_matching.py
│   └── test_persistence.py
│
└── scripts/
    ├── run_multi_camera.py        # Main runner
    ├── evaluate.py                # Offline evaluation
    └── tune_thresholds.py         # Threshold optimization
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/bathanh0309/HAVEN.git
cd HAVEN

# Create conda environment
conda create -n haven python=3.10
conda activate haven

# Install dependencies
pip install -r requirements_production.txt

# Install FAISS (CPU or GPU)
pip install faiss-cpu  # or faiss-gpu for CUDA
```

**requirements_production.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
torchreid>=1.4.0
faiss-cpu>=1.7.4  # or faiss-gpu
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
```

### 2. Prepare Data

```bash
# Data structure
data/
├── cam1/
│   ├── segment_001.mp4
│   ├── segment_002.mp4
│   └── ...
├── cam2/  # MASTER
│   ├── segment_001.mp4
│   └── ...
├── cam3/  # SLAVE
│   └── ...
└── cam4/  # SLAVE
    └── ...
```

### 3. Configure System

Edit `config/production.yaml`:

```yaml
system:
  data_root: "/path/to/data"
  master_camera: "cam2"
  persist_path: "/path/to/state"

reid:
  backbone: "osnet_x0_25"
  device: "cuda:0"  # or "cpu"

# ... (see full config)
```

### 4. Run

```bash
python scripts/run_multi_camera.py --config config/production.yaml
```

---

## 🧪 Testing & Validation

### Unit Tests

```bash
# Test GlobalIDManager
pytest tests/test_global_id_manager.py -v

# Test persistence
pytest tests/test_persistence.py -v

# Test ReID matching
pytest tests/test_reid_matching.py -v
```

### Key Test Cases

1. **Master-only ID creation:**
   - ✅ Cam2 creates G1, G2, G3...
   - ✅ Cam3/4 never create GlobalIDs

2. **No flicker ID switching:**
   - ✅ Temporal voting prevents jitter
   - ✅ Same person doesn't get multiple IDs

3. **Hungarian assignment:**
   - ✅ 2 people in frame don't steal each other's IDs
   - ✅ Best global match (not greedy)

4. **Persistence:**
   - ✅ Restart recovers GlobalIDs
   - ✅ No data loss on crash

### Offline Evaluation

```bash
python scripts/evaluate.py \
  --config config/production.yaml \
  --ground_truth annotations.json \
  --output results/
```

**Metrics:**
- **ID Switches (IDS):** Count of identity changes
- **MOTA/MOTP:** Multi-Object Tracking Accuracy/Precision
- **ReID Accuracy:** Top-1, Top-5 matching accuracy

---

## ⚙️ Tuning Guide

### Threshold Optimization

```bash
python scripts/tune_thresholds.py \
  --data_path /path/to/validation_set \
  --config config/production.yaml \
  --output tuned_config.yaml
```

This performs grid search over:
- `strong_threshold` (0.6 - 0.75)
- `weak_threshold` (0.4 - 0.5)
- `confirm_frames` (2 - 5)

### Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats scripts/run_multi_camera.py

# Visualize
snakeviz profile.stats
```

**Expected bottlenecks:**
1. YOLO inference (40-50% time)
2. OSNet embedding (20-30% time)
3. FAISS search (<5% time)

### Debug Mode

Enable debug crop saving in config:

```yaml
metrics:
  save_debug_crops: true
  debug_crop_limit: 50
  debug_crop_path: "/data/debug_crops"
```

This saves the top-50 worst matches with metadata for analysis.

---

## 📊 Monitoring & Observability

### Structured Logging

Logs are in JSON format:

```json
{
  "timestamp": "2026-02-02T10:30:45",
  "level": "INFO",
  "camera": "cam2",
  "event": "GLOBAL_ID_CREATED",
  "global_id": 42,
  "track_id": 5,
  "bbox": [100, 200, 150, 300],
  "match_score": null
}
```

### Metrics Dashboard (Optional)

Integrate with Prometheus:

```bash
pip install prometheus-client

# In code:
from prometheus_client import Counter, Histogram

id_switches_counter = Counter('haven_id_switches', 'ID switch events')
reid_latency = Histogram('haven_reid_latency_seconds', 'ReID inference time')
```

---

## 🐛 Troubleshooting

### Issue: High UNK Rate on Slave Cameras

**Symptoms:** Cam3/4 show mostly UNK instead of GlobalIDs.

**Possible Causes:**
1. **Lighting difference too severe**
   - Solution: Add per-camera color normalization in config
   
2. **Thresholds too strict**
   - Solution: Lower `strong_threshold` from 0.65 to 0.60
   
3. **OSNet model not loaded**
   - Solution: Check logs for "Model loaded successfully"

### Issue: ID Switches (Flickering)

**Symptoms:** Same person gets G5 → G3 → G5.

**Possible Causes:**
1. **Confirm frames too low**
   - Solution: Increase `confirm_frames` from 3 to 5
   
2. **Quality gating disabled**
   - Solution: Ensure `quality_threshold: 0.7` in config

### Issue: Slow FPS (<10 FPS)

**Symptoms:** System lags, frame drops.

**Possible Causes:**
1. **CPU-only OSNet**
   - Solution: Use GPU (`device: cuda:0`)
   
2. **Too many candidates in search**
   - Solution: Reduce `top_k_candidates` from 20 to 10
   
3. **FAISS not installed**
   - Solution: `pip install faiss-cpu`

---

## 🔒 Production Deployment Checklist

- [ ] **Hardware:**
  - [ ] GPU available for YOLO + OSNet
  - [ ] Sufficient disk for persistence (estimate: 100MB per 1000 IDs)
  
- [ ] **Configuration:**
  - [ ] `master_camera` correctly set to `cam2`
  - [ ] Slave cameras in `slave_cameras` list
  - [ ] Thresholds tuned on validation set
  
- [ ] **Persistence:**
  - [ ] `persist_path` writable and backed up
  - [ ] Auto-save interval reasonable (60s default)
  
- [ ] **Monitoring:**
  - [ ] Structured logging enabled
  - [ ] Metrics export configured (JSON/Prometheus)
  - [ ] Disk space alerts set up
  
- [ ] **Testing:**
  - [ ] All unit tests pass (`pytest tests/`)
  - [ ] Offline evaluation shows <5% ID switches
  - [ ] Restart recovery works (kill and restart)

---

## 📚 References

**ReID Models:**
- [OSNet Paper](https://arxiv.org/abs/1905.00953)
- [torchreid Library](https://github.com/KaiyangZhou/deep-person-reid)

**Tracking:**
- [BoT-SORT](https://arxiv.org/abs/2206.14651)
- [ByteTrack](https://arxiv.org/abs/2110.06864)

**Vector Databases:**
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 🤝 Contributing

For questions or issues:
1. Check this README and config comments
2. Review logs in `/var/log/haven`
3. Open GitHub issue with:
   - Config file
   - Sample video/image
   - Error logs

---

## 📄 License

[Specify license]

---

**Author:** Senior MLOps Engineer  
**Contact:** [Your contact info]
