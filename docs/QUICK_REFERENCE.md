# HAVEN Quick Reference - Cam2 Master Edition

## 🚀 Quick Start (1 Minute)

```bash
# 1. Extract files
tar -xzf HAVEN_REFACTORED.tar.gz
cd HAVEN_REFACTORED

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
mkdir -p backend/data/multicam_videos/{cam1,cam2,cam3,cam4}
# Copy your videos to cam2/, cam3/, cam4/ folders

# 4. Edit config
nano backend/multi_camera_config.yaml
# Set: data_root, enable cameras, thresholds

# 5. Run
./run.sh  # Linux/Mac
# or
run.bat   # Windows
```

## 🎯 Core Concept (30 Seconds)

```
┌─────────────────────────────────────────┐
│ CAM2 (Parking) = MASTER                 │
│ - Creates Global IDs: 1, 2, 3...       │
│ - Sequential, monotonic increasing      │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ CAM3 & CAM4 = SLAVES                    │
│ - Match existing IDs from cam2 gallery  │
│ - If no match → UNKNOWN (not new ID)    │
└─────────────────────────────────────────┘
```

## 📝 Key Config Settings

```yaml
# CRITICAL
master_camera: 'cam2'                    # Only cam2 creates IDs

# DATA
data_source:
  mode: 'folder'
  data_root: 'D:/HAVEN/backend/data/multicam_videos'

# THRESHOLDS (Tune these!)
global_id:
  strong_threshold: 0.65                 # >= match
  weak_threshold: 0.45                   # <= unknown
  min_frames_stable: 15                  # Anti-flicker
  cooldown_seconds: 10                   # Re-attachment window
```

## 🔧 Common Tasks

### Check if system works
```bash
cd tests
python test_global_id_manager.py
# Should see: ✅ ALL TESTS PASSED!
```

### Monitor in real-time
```bash
# In separate terminal
tail -f backend/logs/multicam_reid.log
```

### Save checkpoint manually
```bash
# During runtime, press 's' key
```

### Quit
```bash
# During runtime, press 'q' key
```

## 🐛 Troubleshooting (30 Seconds)

| Problem | Solution |
|---------|----------|
| Person gets multiple IDs | Increase `min_frames_stable` and `cooldown_seconds` |
| Cam3 always UNKNOWN | Lower `strong_threshold` or enable `gallery_update` |
| False matches | Increase `strong_threshold` or tune `camera_graph` |
| No videos loading | Check folder structure and file extensions |

## 📊 Expected Display Format

```
┌─────────┬─────────┐
│  CAM2   │  CAM3   │
│ ┌─────┐ │ ┌─────┐ │
│ │T1 G1│ │ │T2 G1│ │  ← Same person
│ └─────┘ │ └─────┘ │
│ ┌─────┐ │ ┌────────┐
│ │T2 G2│ │ │T3  UNK │ │  ← New person (not in cam2)
│ └─────┘ │ └────────┘ │
└─────────┴─────────┘
```

## 📂 File Locations

```
backend/
├── multi_camera_config.yaml        # Edit this!
├── run_multi_camera.py             # Run this!
├── data/
│   ├── multicam_videos/            # Put videos here
│   │   ├── cam2/  *.mp4
│   │   ├── cam3/  *.mp4
│   │   └── cam4/  *.mp4
│   ├── global_id_state.db          # State persists here
│   └── security_alerts.jsonl       # Alerts logged here
└── logs/
    └── multicam_reid.log            # Main log file
```

## 🎛️ Tuning Quick Guide

### More Precision (Fewer False Matches)
```yaml
strong_threshold: 0.70  # ↑
weak_threshold: 0.50    # ↑
```

### More Recall (Catch More People)
```yaml
strong_threshold: 0.60  # ↓
weak_threshold: 0.40    # ↓
enable_gallery_update: true
```

### Anti-Flicker (Noisy Tracker)
```yaml
min_frames_stable: 20   # ↑
cooldown_seconds: 15    # ↑
```

## 🔐 Security Features

### Add Danger Zone
```yaml
security:
  danger_zones:
    cam2:
      - name: 'Restricted Area'
        polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        alert_consecutive_frames: 10
```

### Enable Object Detection
```yaml
security:
  dangerous_objects:
    enabled: true
    objects: ['knife', 'scissors']
    confidence_threshold: 0.5
```

## 📞 Get Help

1. **README.md** - Full documentation
2. **MIGRATION_GUIDE.md** - If upgrading
3. **IMPLEMENTATION_SUMMARY.md** - Technical details
4. **tests/test_global_id_manager.py** - Code examples
5. **backend/logs/multicam_reid.log** - Error messages

## ⚡ Performance Tips

```yaml
# Real-time processing
performance:
  frame_skip: 2          # Process every 2nd frame
detection:
  model_path: 'yolo11n.pt'  # Use nano (faster)

# High accuracy
performance:
  frame_skip: 1          # Process all frames
detection:
  model_path: 'yolo11x.pt'  # Use extra-large
```

## 🎯 Acceptance Criteria Checklist

- [ ] Cam2 creates sequential IDs (1, 2, 3...)
- [ ] Cam3/4 never create new IDs
- [ ] No duplicate IDs for same person (anti-flicker works)
- [ ] Side-by-side display shows all cameras
- [ ] Overlay format: "T{local_id}  G{global_id}"
- [ ] State persists across restarts
- [ ] Security alerts work (if configured)
- [ ] No crashes when videos finish

## 🚦 Status Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green | Known person (matched) |
| 🔴 Red | Unknown person OR security alert |
| 🟡 Yellow | Pending decision (collecting frames) |

## 📈 Success Metrics

Monitor these to ensure system works:
- **Next Global ID**: Should increment steadily (only from cam2)
- **Gallery Size**: Number of unique persons registered
- **Unknown Rate**: % of unknowns in cam3/4 (should decrease as more people register in cam2)
- **Re-attachment Rate**: % of flickers successfully recovered

---

**Quick Reference Version**: 1.0  
**Last Updated**: 2025-01-31

For detailed documentation, see README.md
