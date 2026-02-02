# Migration Guide - From Old HAVEN to Cam2 Master Edition

## Overview

This guide helps you migrate from the old HAVEN system to the new Cam2 Master edition with enhanced features.

## Key Changes

### 1. Master Camera Concept

**Old System:**
```yaml
# Multiple cameras could create IDs
master_cameras: [1, 2]
registration_camera: 1
```

**New System:**
```yaml
# Single master camera (cam2 only)
master_camera: 'cam2'

cameras:
  - id: 'cam2'
    is_master: true  # Only this camera creates Global IDs
```

**Why?**
- Eliminates ID conflicts
- Predictable sequential IDs
- Easier debugging

### 2. Folder-Based Video Input

**Old System:**
```yaml
# Had to specify each video file
videos:
  cam1: 'path/to/cam1_video1.mp4'
  cam2: 'path/to/cam2_video1.mp4'
```

**New System:**
```yaml
# Just specify root folder
data_source:
  mode: 'folder'
  data_root: 'D:/HAVEN/backend/data/multicam_videos'
  watch_mode: true  # Auto-detect new files
```

**Benefits:**
- Automatic file discovery
- Continuous monitoring
- No manual path updates

### 3. Anti-Flicker Protection

**New Feature:** Prevents duplicate IDs when tracker temporarily loses track.

```yaml
global_id:
  min_frames_stable: 15    # Track must be stable
  cooldown_seconds: 10     # Re-attachment window
```

### 4. Open-Set Association

**New Feature:** Two-threshold matching with pending state.

```yaml
global_id:
  strong_threshold: 0.65   # >= this: confident match
  weak_threshold: 0.45     # <= this: unknown
  # Between: pending decision
```

### 5. Security Module

**New Features:**
- Danger zone detection
- Dangerous object detection

```yaml
security:
  enabled: true
  danger_zones:
    cam2:
      - name: 'Restricted Area'
        polygon: [[100, 200], [300, 200], [300, 400], [100, 400]]
```

## Migration Steps

### Step 1: Backup Old System

```bash
# Backup config
cp backend/configs/multicam.yaml backend/configs/multicam.yaml.old

# Backup data
cp -r backend/data backend/data.backup
```

### Step 2: Update Config Structure

Create new `multi_camera_config.yaml`:

```yaml
# OLD: multicam.yaml
master_cameras: [1, 2]
registration_camera: 1
videos:
  cam1: 'videos/cam1.mp4'
  cam2: 'videos/cam2.mp4'

# NEW: multi_camera_config.yaml
master_camera: 'cam2'  # Single master
data_source:
  mode: 'folder'
  data_root: 'backend/data/multicam_videos'
```

### Step 3: Reorganize Video Files

**Old Structure:**
```
backend/
└── data/
    ├── cam1_video1.mp4
    ├── cam2_video1.mp4
    └── ...
```

**New Structure:**
```
backend/
└── data/
    └── multicam_videos/
        ├── cam1/
        │   ├── cam1_20250131_100000.mp4
        │   └── cam1_20250131_100500.mp4
        ├── cam2/
        │   ├── cam2_20250131_100000.mp4
        │   └── cam2_20250131_100500.mp4
        ├── cam3/
        └── cam4/
```

**Migration Script:**
```bash
# Create new structure
mkdir -p backend/data/multicam_videos/{cam1,cam2,cam3,cam4}

# Move existing videos
mv backend/data/cam1*.mp4 backend/data/multicam_videos/cam1/
mv backend/data/cam2*.mp4 backend/data/multicam_videos/cam2/
mv backend/data/cam3*.mp4 backend/data/multicam_videos/cam3/
mv backend/data/cam4*.mp4 backend/data/multicam_videos/cam4/
```

### Step 4: Update Threshold Values

Recommended starting values:

```yaml
global_id:
  # Start conservative, then tune
  strong_threshold: 0.65
  weak_threshold: 0.45
  min_frames_stable: 15
  cooldown_seconds: 10
```

### Step 5: Add Camera Graph

Define realistic travel times between cameras:

```yaml
camera_graph:
  edges:
    - from: 'cam2'
      to: 'cam3'
      min_time: 5    # Minimum realistic time
      max_time: 60   # Maximum realistic time
```

**How to measure:**
- Walk between cameras
- Measure actual time
- Add safety margin (±20%)

### Step 6: Test Before Production

```bash
# Run tests
cd tests
python test_global_id_manager.py

# Test with sample data
cd backend
python run_multi_camera.py --config multi_camera_config.yaml
```

**Verify:**
- [ ] Cam2 creates IDs 1, 2, 3...
- [ ] Cam3/4 match or show UNKNOWN
- [ ] No duplicate IDs for same person
- [ ] Spatiotemporal gating works

## Code Changes

### GlobalIDManager API

**Old:**
```python
manager.register_person(camera_id, embedding)
manager.verify_person(camera_id, embedding)
```

**New:**
```python
global_id, status, confidence = manager.update_track(
    camera_id='cam2',
    track_id=1,
    embedding=embedding,
    bbox=[x1, y1, x2, y2],
    confidence=0.9,
    timestamp=timestamp
)
```

**Returns:**
- `global_id`: int or None
- `status`: 'known', 'unknown', 'pending', 'new'
- `confidence`: 0-1

### Video Source

**Old:**
```python
cap = cv2.VideoCapture('video.mp4')
```

**New:**
```python
from core.video_source import create_video_source

source = create_video_source(config, 'cam2')
ret, frame, timestamp = source.read()
```

**Benefits:**
- Automatic file switching
- Timestamp extraction
- Watch mode support

## Feature Comparison

| Feature | Old System | New System |
|---------|-----------|------------|
| Master Camera | Multiple | Single (Cam2) |
| Video Input | File paths | Folder-based |
| Anti-Flicker | No | Yes ✅ |
| Open-Set | No | Yes ✅ |
| Spatiotemporal | Basic | Enhanced ✅ |
| Security Zones | No | Yes ✅ |
| Dangerous Objects | No | Yes ✅ |
| State Persistence | JSON | SQLite ✅ |
| Watch Mode | No | Yes ✅ |

## Breaking Changes

### 1. Config File Format

**Action:** Must recreate config file with new schema.

**Old → New Mapping:**
```python
# OLD
master_cameras: [1, 2]
→ master_camera: 'cam2'

# OLD
registration_camera: 1
→ master_camera: 'cam2'

# OLD
videos: {cam1: 'path'}
→ data_source: {mode: 'folder', data_root: 'path'}
```

### 2. ID Assignment Logic

**Old:** Any camera could create IDs.

**New:** Only Cam2 creates IDs.

**Impact:**
- Person first seen in Cam3 → UNKNOWN (not new ID)
- Must appear in Cam2 first to get Global ID

**Workaround:** If person enters via Cam3:
1. Mark as UNKNOWN in Cam3
2. When appears in Cam2 → Gets Global ID
3. Next time in Cam3 → Matches to that ID

### 3. API Changes

**Old:**
```python
manager.register_person(cam_id, embedding)
```

**New:**
```python
manager.update_track(cam_id, track_id, embedding, bbox, conf, timestamp)
```

**Action:** Update all caller code.

## Troubleshooting Migration

### Issue: "Config file invalid"

**Cause:** Using old config format.

**Solution:** Use new template from `multi_camera_config.yaml`.

### Issue: "No videos found"

**Cause:** Videos not in correct folder structure.

**Solution:**
```bash
# Check structure
ls backend/data/multicam_videos/cam2/

# Should show .mp4 files
```

### Issue: "All persons show as UNKNOWN in Cam3"

**Cause:** Persons never appeared in Cam2 (master).

**Solution:** This is correct behavior! Only persons who appear in Cam2 get Global IDs.

### Issue: "Person gets multiple IDs (G1, G3, G5...)"

**Cause:** Anti-flicker not tuned properly.

**Solution:**
```yaml
global_id:
  min_frames_stable: 20     # Increase
  cooldown_seconds: 15      # Increase
```

## Rollback Plan

If migration fails:

```bash
# 1. Stop new system
# Press Ctrl+C or 'q'

# 2. Restore old config
cp backend/configs/multicam.yaml.old backend/configs/multicam.yaml

# 3. Restore old data
rm -rf backend/data
cp -r backend/data.backup backend/data

# 4. Run old system
python run_multicam_reid.py  # Old script
```

## Support

For migration issues:

1. **Check logs:**
   ```bash
   tail -f backend/logs/multicam_reid.log
   ```

2. **Verify config:**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('backend/multi_camera_config.yaml')))"
   ```

3. **Test components:**
   ```bash
   cd tests
   python test_global_id_manager.py
   ```

4. **Contact:** Open GitHub issue with:
   - Old config (sanitized)
   - New config
   - Error logs
   - Expected vs actual behavior

## Timeline Recommendation

### Week 1: Preparation
- [ ] Backup old system
- [ ] Read documentation
- [ ] Create new config
- [ ] Reorganize video files

### Week 2: Testing
- [ ] Run unit tests
- [ ] Test with sample data
- [ ] Tune thresholds
- [ ] Verify all cameras

### Week 3: Deployment
- [ ] Deploy to production
- [ ] Monitor for 24h
- [ ] Adjust thresholds
- [ ] Document issues

### Week 4: Optimization
- [ ] Fine-tune parameters
- [ ] Add security zones
- [ ] Enable watch mode
- [ ] Train team

---

**Last Updated:** 2025-01-31  
**Migration Guide Version:** 1.0
