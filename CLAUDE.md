# CLAUDE.md — CPose Project Intelligence

> For AI coding agents: Claude, Cursor, Codex, Copilot Workspace.
>
> Read this file before modifying any file in this repository.
>
> Project goal: build a real-time human perception pipeline:
>
> **YOLO11-Pose → ByteTrack → FastReID → PoseC3D → ADL visualization**
>
> Every application under `apps/` must be able to display processed video and optionally save output video.

---

# 0. Executive Summary

```text
Input:
  Video / Webcam / RTSP stream

Main pipeline:
  Frame
    → YOLO11-Pose
    → ByteTrack local tracking
    → Person crop
    → FastReID embedding
    → ReID gallery matching
    → Global ID assignment
    → Pose sequence buffer
    → PoseC3D ADL inference
    → Visualization + JSONL event logging

Core rule:
  configs/system/pipeline.yaml is the single source of truth.

Do not:
  - hard-code absolute paths
  - force CUDA on CPU-only machines
  - use default_label as real ADL prediction
  - store credentials in repo
  - change detection dict schema without updating all modules
```

# 1. Repository Structure

```text
CPose/
├── apps/
│   ├── run_pipeline.py      # Full pipeline
│   ├── run_pose.py          # YOLO11-Pose + ByteTrack visualizer
│   ├── run_reid.py          # ReID visualizer
│   └── run_adl.py           # ADL / skeleton action visualizer
│
├── configs/
│   ├── system/
│   │   └── pipeline.yaml    # Main config, single source of truth
│   ├── fast-reid/
│   │   └── R50.yml
│   └── posec3d/
│       └── posec3d.py
│
├── src/
│   ├── action/
│   │   ├── pose_buffer.py
│   │   └── posec3d.py
│   ├── core/
│   │   ├── event.py
│   │   └── global_id.py
│   ├── detectors/
│   │   └── yolo_pose.py
│   ├── reid/
│   │   ├── fast_reid.py
│   │   └── gallery.py
│   ├── trackers/
│   │   └── bytetrack.py
│   └── utils/
│       ├── config.py
│       ├── io.py
│       ├── logger.py
│       ├── video.py
│       └── vis.py
│
├── data/
│   ├── gallery/             # ReID reference images, not committed
│   └── output/              # Runtime outputs, not committed
│
├── models/                  # Model weights, not committed
├── README.md
└── CLAUDE.md

```

## Rules

apps/      = entry points only
src/       = business logic
configs/   = configuration only
models/    = local weights, not committed
data/      = local data/runtime output, not committed

# 2. Main Design Principles
## 2.1 Single Source of Truth

All paths, thresholds, model weights, output directories and hyperparameters must come from:

configs/system/pipeline.yaml

Do not hard-code values inside apps/ or src/.

Correct:

cfg = load_pipeline_cfg(Path(args.config), ROOT)
weights = cfg["pose"]["weights"]
device = cfg["system"]["device"]

Incorrect:

weights = "models/yolo11n-pose.pt"
device = "cuda"
ROOT = Path(r"D:\Capstone_Project")
## 2.2 Path Rule

Every relative path must be resolved from the project root, not from the current terminal directory.

Correct pattern inside apps/*.py:

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Then load config:

from src.utils.config import load_pipeline_cfg

cfg = load_pipeline_cfg(Path(args.config), ROOT)
## 2.3 Device Rule

The code must run on both CPU-only and CUDA machines.

pipeline.yaml should use:

system:
  device: "auto"

auto means:

if torch.cuda.is_available():
    use cuda
else:
    use cpu

Never force:

device = 0
device = "cuda"

unless the config loader already confirmed CUDA is available.

## 2.4 Visualization Rule

Every script under apps/ must support:

--source
--camera-id
--config
--show
--save-video
--output
--max-frames

Every script must:

- read frames from video/webcam/RTSP
- process frames
- draw results on frame
- call cv2.imshow() if --show is used
- write output .mp4 if --save-video is used
- release OpenCV resources safely
# 3. Standard CLI Contract for All Apps

Every apps/run_*.py should expose this argument structure:

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--camera-id", type=str, default="cam01")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)

    return parser.parse_args()

Expected commands:

python apps/run_pose.py --source data/sample.mp4 --show --max-frames 100

python apps/run_reid.py --source data/sample.mp4 --show --max-frames 100

python apps/run_adl.py --source data/sample.mp4 --show --max-frames 150

python apps/run_pipeline.py --source data/sample.mp4 --camera-id cam01 --show --max-frames 150
# 4. Detection Dict Contract

Every detector/tracker output must follow this schema exactly:

detection = {
    "bbox": [x1, y1, x2, y2],
    "score": float,
    "class_id": int,
    "track_id": int,
    "keypoints": [[x, y], ...] | None,
    "keypoint_scores": [float, ...] | None,
}

Meaning:

bbox:
  [x1, y1, x2, y2] in pixel coordinates

score:
  detection confidence

class_id:
  0 = person

track_id:
  local tracking ID from ByteTrack / Ultralytics tracker
  -1 means no valid track ID

keypoints:
  COCO-17 keypoints by default

keypoint_scores:
  confidence for each keypoint

Rules:

- Optional fields must be read with det.get(...)
- Do not assume keypoints always exist
- Do not assume track_id is always valid
- Do not use local track_id as global identity

Correct:

track_id = det.get("track_id", -1)
keypoints = det.get("keypoints")

Incorrect:

track_id = det["track_id"]
keypoints = det["keypoints"]
# 5. Identity Contract

There are two different identity concepts:

local_track_id:
  int
  generated by ByteTrack / Ultralytics tracker
  valid only inside one camera session

global_id:
  str
  generated by ReID / GlobalIDManager
  can be used across cameras or sessions if gallery is stable

Cache key must be:

key = (str(camera_id), int(local_track_id))

Do not use only track_id as key.

Correct:

key = (str(camera_id), int(local_track_id))

Incorrect:

key = int(local_track_id)
# 6. Module Responsibilities
## 6.1 YoloPoseTracker

File:

src/detectors/yolo_pose.py

Responsibility:

- load YOLO11-Pose model
- run YOLO.track()
- use Ultralytics ByteTrack through tracker YAML
- return DetectionDict list

Input:

frame: np.ndarray  # BGR image

Output:

detections, raw_result

Rules:

- validate weights path during initialization
- use normalized device from config
- do not hard-code CUDA
- classes should default to [0] for person only
## 6.2 ByteTrackWrapper

File:

src/trackers/bytetrack.py

Responsibility:

- thin wrapper around detector.infer()
- no independent ByteTrack checkpoint logic

Important clarification:

The current project uses Ultralytics built-in ByteTrack through:
tracker_yaml: "bytetrack.yaml"

It does not load bytetrack_s_mot17.pth.tar.
Do not claim the repo uses external ByteTrack checkpoint unless it is actually implemented.
## 6.3 FastReIDExtractor

File:

src/reid/fast_reid.py

Responsibility:

- load FastReID config
- load FastReID weights
- extract L2-normalized appearance embedding from person crop

Input:

image_bgr: np.ndarray

Output:

feature: np.ndarray  # float32, normalized

Rules:

- validate fastreid_root
- validate config_path
- validate weights_path
- insert fastreid_root into sys.path only once
- do not silently ignore missing files
## 6.4 ReIDGallery

File:

src/reid/gallery.py

Responsibility:

- build gallery from data/gallery/
- compute prototype embeddings for each person
- query nearest person by cosine similarity
- optionally support FAISS for large galleries

Expected gallery structure:

data/gallery/
├── person_001/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── person_002/
│   ├── 001.jpg
│   └── 002.jpg
└── person_003/
    └── 001.jpg

Query contract:

person_id, score = gallery.query(feat, threshold=0.55)

Return:

("person_001", 0.82)
or
("unknown", 0.31)

Rules:

- never add embedding under person_id="unknown"
- warn if gallery is empty
- pipeline must still run if gallery is empty
## 6.5 GlobalIDManager

File:

src/core/global_id.py

Responsibility:

- map local_track_id to global_id
- use ReID only every N frames
- cache identity results
- generate new gid_NNNNN when no gallery match is found

Expected return:

global_id, reid_score, status = manager.assign(...)

Recommended status values:

"cache_hit"
"gallery_match"
"new_global_id"
"reid_failed"

Rules:

- do not run ReID every frame
- keep global_id stable for the same track
- cache both global_id and last score
- do not return fake score 1.0 for cache hit
- provide cleanup for lost tracks
## 6.6 PoseSequenceBuffer

File:

src/action/pose_buffer.py

Responsibility:

- collect skeleton keypoints over time
- export fixed-length clip when seq_len is reached
- prepare MMAction2-compatible .pkl annotation

Required output .pkl format:

{
    "split": {
        "test": [sample_id]
    },
    "annotations": [
        {
            "frame_dir": sample_id,
            "total_frames": int,
            "img_shape": (H, W),
            "original_shape": (H, W),
            "label": int,
            "keypoint": np.ndarray,        # [M, T, V, C]
            "keypoint_score": np.ndarray,  # [M, T, V]
        }
    ]
}

For current YOLO11-Pose:

M = 1
T = seq_len
V = 17
C = 2

Rules:

- validate keypoint count = 17
- return status for visualization
- default_label is only for annotation export
- default_label is not real ADL prediction

Recommended return object:

{
    "status": "collecting" | "exported" | "skipped",
    "current_len": int,
    "seq_len": int,
    "pkl_path": str | None,
}
## 6.7 PoseC3DRunner

File:

src/action/posec3d.py

Responsibility:

- validate MMAction2 / PoseC3D root
- build temporary test config
- run action recognition
- return parsed ADL result

Recommended return:

{
    "ok": bool,
    "label": int | None,
    "label_name": str | None,
    "score": float | None,
    "stdout": str,
    "stderr": str,
}

Rules:

- do not crash full pipeline if auto_infer=false
- do not display default_label as prediction
- parse result from a stable file if possible, not fragile stdout regex
- handle subprocess failure clearly
## 6.8 EventBus

File:

src/core/event.py

Responsibility:

- write JSONL event logs

Standard event types:

track_update
pose_clip_exported
adl_result
reid_warning
pipeline_error

Rules:

- payload must be JSON-serializable
- convert Path to str
- convert numpy types to Python int/float/list
# 7. Required Config Schema

File:

configs/system/pipeline.yaml

Recommended schema:

system:
  device: "auto"
  event_log: "data/output/events/pipeline.jsonl"
  vis_dir: "data/output/vis"
  save_video: true

pose:
  weights: "models/yolo11n-pose.pt"
  conf: 0.45
  iou: 0.5

tracker:
  tracker_yaml: "bytetrack.yaml"

reid:
  fastreid_root: ".github/fast-reid"
  config: "configs/fast-reid/R50.yml"
  weights: "models/fastreid_market_R50.pth"
  gallery_dir: "data/gallery"
  threshold: 0.55
  reid_interval: 10

adl:
  mmaction_root: ".github/pose-c3d"
  base_config: "configs/posec3d/posec3d.py"
  weights: "models/posec3d_r50_ntu60.pth"
  seq_len: 48
  stride: 12
  max_idle_frames: 150
  export_dir: "data/output/clips_pkl"
  default_label: 0
  work_dir: "data/output/posec3d"
  auto_infer: false

visualization:
  draw_bbox: true
  draw_skeleton: true
  draw_track_id: true
  draw_global_id: true
  draw_reid_score: true
  draw_adl_status: true
  draw_fps: true

If new config keys are added, update:

src/utils/config.py::validate_cfg()
# 8. src/utils/config.py Standard

The config utility must provide:

load_pipeline_cfg(path: Path, root: Path) -> dict
validate_cfg(cfg: dict) -> None
resolve_cfg_paths(cfg: dict, root: Path) -> dict
normalize_device(value) -> str | None

Required behavior:

- load YAML
- validate required sections and keys
- resolve relative paths from project root
- keep Ultralytics built-in tracker names as bare names
- convert device "auto" into "cuda" or "cpu"
- fallback to CPU if CUDA is unavailable

Device normalization rule:

if device in {"auto", ""}:
    return "cuda" if torch.cuda.is_available() else "cpu"

if device in {"cuda", "cuda:0", "0"} and not torch.cuda.is_available():
    return "cpu"
# 9. src/utils/video.py Standard

Create this file if missing.

Required functions:

def parse_video_source(source: str):
    return int(source) if str(source).isdigit() else source


def open_video_source(source: str):
    ...


def get_video_meta(cap):
    ...


def create_video_writer(output_path, fps, width, height):
    ...


def safe_imshow(window_name, frame, delay=1):
    ...

Rules:

- support webcam index: "0"
- support video path
- support RTSP URL
- validate cap.isOpened()
- fallback FPS to 25.0 if invalid
- release resources in apps
# 10. Visualization Standard

File:

src/utils/vis.py

Required functions:

draw_detection(frame, det, label=None, color=None)
draw_skeleton_only(frame, keypoints, keypoint_scores=None, color=None)
draw_info_panel(frame, info: dict, pos=(10, 10))
draw_reid_panel(frame, query_crop, matches)
draw_adl_status(frame, status)

COCO-17 edge list:

COCO_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6),
]

Overlay examples:

track=12 conf=0.87
gid_00003 score=0.71
ADL: collecting 25/48
ADL: clip exported
ADL: walking score=0.83
FPS: 18.7
Device: cpu
# 11. Application Requirements
## 11.1 apps/run_pose.py

Must show:

- bbox
- skeleton
- local track_id
- detection confidence
- FPS
- device

Must not:

- hard-code model path
- hard-code output directory
- only save JSON

Expected command:

python apps/run_pose.py --source data/sample.mp4 --show --save-video
## 11.2 apps/run_reid.py

Must show:

- bbox
- skeleton if available
- local track_id
- global_id
- ReID score
- optional ReID side panel
- warning if gallery is empty

Must handle:

- missing gallery folder
- empty gallery
- missing FastReID dependency

Expected command:

python apps/run_reid.py --source data/sample.mp4 --show --save-video
## 11.3 apps/run_adl.py

Must show processed video, not only .pkl skeleton viewer.

Must show:

- bbox
- skeleton
- track_id
- ADL collection status
- ADL export status
- ADL prediction if auto_infer=true and inference succeeds

Must not:

- display default_label as real action
- require existing .pkl as only input mode

Expected command:

python apps/run_adl.py --source data/sample.mp4 --show --save-video
## 11.4 apps/run_pipeline.py

Must run full pipeline:

YOLO11-Pose
→ ByteTrack
→ FastReID
→ GlobalIDManager
→ PoseSequenceBuffer
→ PoseC3D optional inference
→ EventBus
→ Visualization

Must show:

- bbox
- skeleton
- local track_id
- global_id
- ReID score/status
- ADL status
- FPS
- device

Expected command:

python apps/run_pipeline.py --source data/sample.mp4 --camera-id cam01 --show --save-video
# 12. Error Handling Rules
Initialization-level errors

Raise early with clear message.

Examples:

if not weights_path.exists():
    raise FileNotFoundError(f"Model weights not found: {weights_path}")

if not config_path.exists():
    raise FileNotFoundError(f"Config not found: {config_path}")
Frame-level errors

Log and continue.

try:
    detections, _ = tracker.update(frame)
except Exception as exc:
    logger.warning(f"[frame {frame_idx}] tracker failed: {exc}", exc_info=True)
    continue
Resource cleanup

Always use finally.

try:
    ...
finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

Do not use bare except:

except:
    ...

Use:

except Exception as exc:
    ...
# 13. Logging Rules

Use:

from src.utils.logger import get_logger

logger = get_logger(__name__)

Do not use print() inside core pipeline modules.

Allowed logging levels:

logger.debug     per-frame verbose details
logger.info      module initialization / output saved
logger.warning   recoverable runtime error
logger.error     fatal error before exit
# 14. ReID Rules
Do not run ReID every frame

Use:

reid:
  reid_interval: 10
Do not pollute gallery with "unknown"

Incorrect:

gallery.add_embedding("unknown", feat)

Correct:

if matched_id == "unknown":
    matched_id = self.track_to_global.get(key) or self._new_global_id()
    self.gallery.add_embedding(matched_id, feat)
Cache identity and score

Incorrect:

return self.track_to_global[key], 1.0

Correct:

gid, cached_score = self.track_to_global[key]
return gid, cached_score, "cache_hit"
Empty gallery behavior

If gallery is empty:

- pipeline still runs
- new gid_NNNNN can be assigned
- overlay must show "ReID gallery empty"
- do not crash
# 15. ADL Rules
default_label is not prediction

default_label exists only for exported MMAction2 annotation.

Do not display:

Label: 0

as if it were a predicted action.

Display instead:

ADL: collecting 25/48
ADL: exported
ADL: inference disabled
ADL: walking score=0.83
Keypoint validation

Current expected keypoints:

COCO-17

Required check:

EXPECTED_KEYPOINTS = 17

if keypoints_xy.shape[0] != EXPECTED_KEYPOINTS:
    logger.warning(
        f"Unexpected keypoint count {keypoints_xy.shape[0]} != {EXPECTED_KEYPOINTS}"
    )
    return {
        "status": "skipped",
        "reason": "invalid_keypoint_count",
        "current_len": 0,
        "seq_len": self.seq_len,
        "pkl_path": None,
    }
# 16. Security Rules

Never commit:

.env
RTSP usernames/passwords
camera passwords
API keys
model weights
large datasets
runtime .pkl outputs
face embeddings

.gitignore must include:

.env
*.pt
*.pth
*.pth.tar
*.onnx
*.engine
*.pkl
*.npy
data/output/
data/gallery/
models/

If a credential was committed:

1. remove it from repo
2. rotate password immediately
3. purge from git history if public
4. move to .env
# 17. Model Files

Expected local model layout:

models/
├── yolo11n-pose.pt
├── fastreid_market_R50.pth
└── posec3d_r50_ntu60.pth

Important:

ByteTrack currently uses Ultralytics built-in tracker YAML:
tracker_yaml: "bytetrack.yaml"

The project does not need bytetrack_s_mot17.pth.tar unless a custom external ByteTrack implementation is added.
# 18. External Repositories

Expected external dependencies:

.github/
├── fast-reid/
└── pose-c3d/

FastReID:

git clone https://github.com/JDAI-CV/fast-reid.git .github/fast-reid

MMAction2 / PoseC3D:

git clone https://github.com/open-mmlab/mmaction2.git .github/pose-c3d

Rules:

- Do not import FastReID globally outside FastReIDExtractor
- Do not import MMAction2 globally unless needed
- Validate external repo root before use
# 19. Known High-Priority Fixes
Fix 1 — CPU/CUDA device bug

Problem:

torch.cuda.is_available() == False
but code still requests device=0 or cuda

Required fix:

- use device: "auto" in pipeline.yaml
- normalize device in src/utils/config.py
- never pass invalid CUDA device to Ultralytics
Fix 2 — App scripts do not consistently show processed video

Required fix:

Every apps/run_*.py must:
- open video source
- process every frame
- draw overlay
- call cv2.imshow if --show
- write .mp4 if --save-video
Fix 3 — run_pose.py hard-codes model path

Required fix:

Load YOLO weights from cfg["pose"]["weights"].
Fix 4 — ReID gallery empty or invalid

Required fix:

- Warn clearly
- Continue pipeline
- Show overlay warning
Fix 5 — ADL default label misused

Required fix:

Do not display default_label as predicted ADL result.
Show collecting/exported/inference-disabled/inferred status.
Fix 6 — PoseC3D inference result not returned

Required fix:

PoseC3DRunner.run_test() should return structured result:
{
  "ok": bool,
  "label": int | None,
  "label_name": str | None,
  "score": float | None,
  "stdout": str,
  "stderr": str,
}
Fix 7 — Memory leak in ID managers

Required fix:

- GlobalIDManager must clean old tracks
- PoseSequenceBuffer already has GC-like behavior; keep it
- Pipeline should identify active tracks per frame
# 20. Smoke Test Checklist

Before committing, run:

python -m compileall apps src

Then run:

python apps/run_pose.py --source data/sample.mp4 --show --max-frames 100

Expected:

- video window opens
- bbox visible
- skeleton visible
- track_id visible
- no CUDA error on CPU machine

Run:

python apps/run_pipeline.py --source data/sample.mp4 --camera-id cam01 --show --max-frames 150

Expected:

- video window opens
- bbox + skeleton visible
- global_id visible
- ReID score/status visible
- ADL collecting/export status visible
- output JSONL written
- no crash if gallery is empty

# 21. Code Style

Use standard import order:

```python
# standard library
import argparse
import sys
from pathlib import Path

# third-party
import cv2
import numpy as np

# local
from src.utils.config import load_pipeline_cfg
```

Rules:

- Public functions should use type hints where practical
- Avoid huge functions
- Keep apps thin
- Keep model logic inside src/
- Avoid broad refactor unless required
- Do not change file formats silently

# 22. JSONL Event Format

Every event row should look like:

{
  "ts_ms": 1710000000000,
  "type": "track_update",
  "payload": {
    "camera_id": "cam01",
    "frame_idx": 25,
    "local_track_id": 3,
    "global_id": "gid_00001",
    "reid_score": 0.72,
    "bbox": [100, 120, 250, 400]
  }
}

Rules:

- payload must not contain numpy.ndarray
- payload must not contain pathlib.Path
- convert all numpy int/float to Python int/float
# 23. Pull Request / Commit Response Format

When an AI agent modifies the repo, it must report:

Files changed:
- path/to/file.py
- path/to/config.yaml

What changed:
- short bullet per file

Why:
- explain bug or logic issue fixed

How to test:
- exact commands

Remaining limitations:
- honest notes

Example:

Files changed:
- src/utils/config.py
- apps/run_pose.py

What changed:
- Added safe device normalization
- Reworked run_pose.py to load config and show processed video

How to test:
python apps/run_pose.py --source data/sample.mp4 --show --max-frames 100
# 24. Do Not Do These
Do not hard-code absolute paths.
Do not force CUDA.
Do not use print in src modules.
Do not add unknown embeddings to gallery under "unknown".
Do not treat local track_id as global identity.
Do not display default_label as ADL prediction.
Do not commit model weights.
Do not commit credentials.
Do not change MMAction2 pkl format unless updating config too.
Do not let OpenCV windows/camera/writer leak.
Do not swallow exceptions silently.
# 25. Final Project Goal

A correct CPose demo should show, on live video:

Person bbox
COCO-17 skeleton
Local track_id
Global person ID
ReID similarity score
ADL status or action label
FPS
Device
Camera ID

Example overlay:

CPose Full Pipeline
camera: cam01
frame: 152
device: cpu
fps: 17.8

track=4
gid=gid_00003
reid=0.68
ADL: collecting 32/48
