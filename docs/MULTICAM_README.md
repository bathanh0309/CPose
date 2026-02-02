# HAVEN - Multi-Camera ReID System (Cam2 Master Edition)

## 🎯 Overview

Multi-camera person re-identification system với **Camera 2 làm Master duy nhất** cho việc tạo Global ID. System này được thiết kế để:

- ✅ **Cam2 (Parking)**: Tạo Global ID số (1, 2, 3...) cho mọi người xuất hiện lần đầu
- ✅ **Cam3 & Cam4**: Chỉ match với gallery từ Cam2 hoặc gán nhãn UNKNOWN
- ✅ **Anti-Flicker**: Chống cấp trùng ID do tracker mất vài frame
- ✅ **Open-Set Association**: Two-threshold với pending state
- ✅ **Spatiotemporal Gating**: Validate bằng camera graph
- ✅ **Security Module**: Danger zones + Dangerous objects detection
- ✅ **Folder-Based Input**: Auto-load videos từ thư mục, watch mode support

## 🏗️ Architecture

```
HAVEN/
├── backend/
│   ├── multi_camera_config.yaml    # Main configuration
│   ├── run_multi_camera.py         # Main entry point
│   ├── src/
│   │   ├── core/
│   │   │   ├── video_source.py     # Video source abstraction
│   │   │   └── security.py         # Security manager
│   │   ├── global_id/
│   │   │   └── manager.py          # Global ID manager (CAM2 MASTER)
│   │   ├── reid/
│   │   │   └── extractor.py        # ReID feature extractor
│   │   └── storage/
│   ├── models/                     # YOLO weights
│   ├── data/                       # Input videos & outputs
│   │   ├── multicam_videos/
│   │   │   ├── cam1/              # Video files for cam1
│   │   │   ├── cam2/              # Video files for cam2
│   │   │   ├── cam3/              # Video files for cam3
│   │   │   └── cam4/              # Video files for cam4
│   │   ├── global_id_state.db     # Persistent state
│   │   └── security_alerts.jsonl  # Alert logs
│   └── logs/
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repo
git clone https://github.com/bathanh0309/HAVEN.git
cd HAVEN

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (if not already present)
# The system will auto-download on first run
```

### 2. Prepare Data

Organize your videos in folder structure:

```
backend/data/multicam_videos/
├── cam1/
│   ├── cam1_20250131_100000.mp4
│   ├── cam1_20250131_100500.mp4
│   └── ...
├── cam2/  # MASTER CAMERA
│   ├── cam2_20250131_100000.mp4
│   ├── cam2_20250131_100500.mp4
│   └── ...
├── cam3/
│   └── ...
└── cam4/
    └── ...
```

**Filename format** (optional but recommended):
- `camX_YYYYMMDD_HHMMSS.mp4` → System extracts timestamp
- Otherwise uses file creation time

### 3. Configure

Edit `backend/multi_camera_config.yaml`:

```yaml
# Set your data path
data_source:
  mode: 'folder'
  data_root: 'D:/HAVEN/backend/data/multicam_videos'
  watch_mode: true  # Auto-detect new files

# Enable/disable cameras
cameras:
  - id: 'cam1'
    enabled: false  # Can disable cam1
  - id: 'cam2'
    enabled: true
    is_master: true  # IMPORTANT: Only cam2 creates IDs
  - id: 'cam3'
    enabled: true
  - id: 'cam4'
    enabled: true

# Master camera (CRITICAL)
master_camera: 'cam2'  # Only this camera creates new Global IDs

# Thresholds (tune these)
global_id:
  strong_threshold: 0.65   # >= this: match
  weak_threshold: 0.45     # <= this: unknown
  min_frames_stable: 15    # Anti-flicker
  cooldown_seconds: 10     # Re-attachment window

# Security zones (optional)
security:
  enabled: true
  danger_zones:
    cam2:
      - name: 'Restricted Area'
        polygon: [[100, 200], [300, 200], [300, 400], [100, 400]]
        alert_consecutive_frames: 10
```

### 4. Run

```bash
cd backend
python run_multi_camera.py --config multi_camera_config.yaml
```

**Keyboard Controls:**
- `q`: Quit
- `s`: Save checkpoint (manual)

## 📊 Expected Output

### Display Layout (2x2 Grid)

```
┌──────────────┬──────────────┐
│   Cam1       │   Cam2       │
│   (Optional) │   (MASTER)   │
│              │  ┌──────┐    │
│              │  │T1 G1 │    │
│              │  └──────┘    │
├──────────────┼──────────────┤
│   Cam3       │   Cam4       │
│  ┌──────┐    │  ┌────────┐  │
│  │T2 G1 │    │  │T3  UNK │  │
│  └──────┘    │  └────────┘  │
└──────────────┴──────────────┘
```

**Overlay Format:**
- `T{local_id}  G{global_id}` for known persons
- `T{local_id}  UNKNOWN` for unmatched persons (cam3/cam4 only)
- Color coding:
  - **Green**: Known/Matched
  - **Red**: Unknown/Security alert
  - **Yellow**: Pending decision

## 🔒 Key Guarantees

### 1. Cam2 Master Logic

✅ **Person enters Cam2 first time** → Assigned G1  
✅ **Next person enters Cam2** → Assigned G2  
✅ **G1 appears in Cam3/Cam4** → Correctly matched as G1  
❌ **New person in Cam3 (never in Cam2)** → Marked as UNKNOWN (not G3!)

### 2. Anti-Flicker Protection

Without anti-flicker:
```
Frame 1-10:  Person A → G1
Frame 11-15: Track lost (flicker)
Frame 16-30: Person A reappears → G2 ❌ WRONG!
```

With anti-flicker:
```
Frame 1-10:  Person A → G1
Frame 11-15: Track lost → Moved to cooldown
Frame 16-30: Person A reappears → Re-attached to G1 ✅ CORRECT!
```

### 3. Open-Set Association

```
Match Score >= 0.65: STRONG → Assign Global ID
Match Score 0.45-0.65: PENDING → Collect more frames
Match Score <= 0.45: WEAK → Unknown (cam3/4) or New ID (cam2)
```

### 4. Spatiotemporal Gating

Example:
```
G1 last seen in Cam2 at t=100s
G1 detected in Cam3 at t=110s → time_diff = 10s

Camera graph: cam2→cam3 min=5s, max=60s
→ 10s is VALID ✅

If detected at t=103s → time_diff = 3s
→ 3s < 5s → REJECT ❌ (too fast, probably wrong match)
```

## 🔧 Advanced Configuration

### Tuning Thresholds

Start with these values and adjust based on your data:

```yaml
global_id:
  # High precision (fewer false matches, more unknowns)
  strong_threshold: 0.70
  weak_threshold: 0.50
  
  # High recall (more matches, some false positives)
  strong_threshold: 0.60
  weak_threshold: 0.40
  
  # Balanced (recommended starting point)
  strong_threshold: 0.65
  weak_threshold: 0.45
```

### Camera Graph

Define realistic travel times:

```yaml
camera_graph:
  edges:
    - from: 'cam2'
      to: 'cam3'
      min_time: 5    # Minimum seconds
      max_time: 60   # Maximum seconds
```

### Gallery Update (Domain Shift)

Enable EMA update to adapt embeddings:

```yaml
global_id:
  enable_gallery_update: true
  gallery_update_alpha: 0.3    # 30% new, 70% old
  update_threshold: 0.70       # Only update on strong matches
```

## 🛡️ Security Features

### Danger Zones

Define polygons per camera:

```yaml
security:
  danger_zones:
    cam2:
      - name: 'Restricted Parking'
        polygon: [[100, 200], [300, 200], [300, 400], [100, 400]]
        alert_consecutive_frames: 10
```

**How it works:**
1. System checks if person's bbox bottom-center is inside polygon
2. Requires N consecutive frames before alerting
3. Logs to `backend/data/security_alerts.jsonl`

### Dangerous Objects

Detect weapons/dangerous items:

```yaml
security:
  dangerous_objects:
    enabled: true
    confidence_threshold: 0.5
    alert_consecutive_frames: 5
    objects:
      - 'knife'
      - 'scissors'
```

## 📁 Output Files

### Global ID State
**File**: `backend/data/global_id_state.db` (SQLite)

Stores:
- Next Global ID counter
- Gallery embeddings per ID
- Creation time/camera
- Total appearances

**Persistence**: System resumes from last state on restart

### Security Alerts
**File**: `backend/data/security_alerts.jsonl` (JSON Lines)

Format:
```json
{
  "alert_type": "zone_intrusion",
  "camera_id": "cam2",
  "timestamp": 1706745600.123,
  "datetime": "2025-01-31 14:30:00",
  "description": "Person entered Restricted Parking Area",
  "global_id": 1,
  "track_id": 5,
  "location": [100, 200, 150, 350],
  "metadata": {"zone_name": "Restricted Parking", "consecutive_frames": 12}
}
```

## 🐛 Troubleshooting

### Issue: Multiple IDs for same person in Cam2

**Cause**: Anti-flicker not working, tracker too unstable

**Solution**:
```yaml
global_id:
  min_frames_stable: 20      # Increase from 15
  cooldown_seconds: 15       # Increase from 10
```

### Issue: Cam3/Cam4 always shows UNKNOWN

**Cause**: Thresholds too high, or domain shift too severe

**Solution**:
```yaml
global_id:
  strong_threshold: 0.60     # Decrease from 0.65
  enable_gallery_update: true
  gallery_update_alpha: 0.4  # More aggressive adaptation
```

### Issue: False matches across cameras

**Cause**: Spatiotemporal gating too loose

**Solution**:
```yaml
camera_graph:
  edges:
    - from: 'cam2'
      to: 'cam3'
      min_time: 8     # Increase minimum
      max_time: 45    # Decrease maximum
```

### Issue: Videos not loading

**Check**:
1. Folder structure correct?
2. Video extensions in config?
3. Check logs for errors

```bash
# View logs
tail -f backend/logs/multicam_reid.log
```

## 📊 Performance Tips

### For Real-Time Processing

```yaml
performance:
  frame_skip: 2          # Process every 2nd frame
  num_workers: 4
  batch_size: 8

detection:
  model_path: 'yolo11n.pt'  # Use nano model (faster)
```

### For High Accuracy

```yaml
performance:
  frame_skip: 1          # Process all frames

detection:
  model_path: 'yolo11x.pt'  # Use extra-large model
  conf_threshold: 0.4

reid:
  model_type: 'osnet_ain_x1_0'  # Better ReID model
```

## 🔬 Testing

### Unit Tests

```bash
cd tests
python test_global_id_manager.py
```

### Integration Test

```bash
# Test with sample videos
python run_multi_camera.py --config test_config.yaml
```

**Expected**:
- Person in cam2 → G1
- Same person in cam3 → G1 (matched)
- New person in cam3 → UNKNOWN (not G2)

## 📝 Logging

### Log Levels

```yaml
logging:
  level: 'DEBUG'  # DEBUG, INFO, WARNING, ERROR
```

### View Real-Time Stats

Press any key during runtime to see:
- Global IDs created
- Active tracks per camera
- FPS per camera
- Pending matches

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Test thoroughly with multi-camera setup
4. Submit PR

## 📄 License

MIT License

## 🆘 Support

For issues:
1. Check logs: `backend/logs/multicam_reid.log`
2. Check config: `backend/multi_camera_config.yaml`
3. Open GitHub issue with:
   - Config file
   - Log excerpt
   - Expected vs actual behavior

## 🎓 Citation

```bibtex
@software{haven_multicam,
  title={HAVEN Multi-Camera ReID System},
  author={Your Name},
  year={2025},
  url={https://github.com/bathanh0309/HAVEN}
}
```

---

**Last Updated**: 2025-01-31  
**Version**: 2.0.0 (Cam2 Master Edition)
