# CLAUDE.md — CPose System Guide

> **Purpose:** This file is the primary reference for any AI assistant (Claude or otherwise) working on the CPose codebase. It defines architecture decisions, module contracts, CLI commands, metric policies, coding constraints, and review rules. Read this file completely before making any change.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Module Architecture & Contracts](#3-module-architecture--contracts)
4. [Pipeline & Orchestration](#4-pipeline--orchestration)
5. [Run Output Structure](#5-run-output-structure)
6. [CLI Commands](#6-cli-commands)
7. [Config Files](#7-config-files)
8. [JSON Schemas](#8-json-schemas)
9. [Metrics Policy](#9-metrics-policy)
10. [Terminal Logging Rules](#10-terminal-logging-rules)
11. [Video Comparison Rules](#11-video-comparison-rules)
12. [Error Taxonomy](#12-error-taxonomy)
13. [Paper Report Output](#13-paper-report-output)
14. [Coding Constraints](#14-coding-constraints)
15. [Review Constraints](#15-review-constraints)
16. [Roadmap](#16-roadmap)
17. [Reference Numbers](#17-reference-numbers)

---

## 1. Project Overview

**CPose** is a terminal-based, multi-camera human monitoring system designed for R&D Computer Vision and AI Edge deployment. It does **not** use a frontend, web server, Flask, or Socket.IO.

### System description

> CPose is a terminal-only, multi-camera human monitoring pipeline that uses YOLOv8n/YOLOv8n-pose for person detection and pose estimation, ByteTrack for local tracking, rule-based skeleton ADL for lightweight activity recognition, ArcFace as an auxiliary face signal, and TFCS-PAR for fusing Global IDs across cameras using face, body, pose, height, time, and camera topology.

### Research focus

The novelty of CPose is **not** about replacing YOLO with a newer model. It is about:

- Time-first, topology-gated cross-camera identity fusion (TFCS-PAR)
- Maintaining stable Global IDs through blind zones, room entries, and clothing changes
- Proxy vs. ground-truth metric distinction (no fake accuracy)
- Failure-reason taxonomy across all modules
- Paper-ready benchmark tables from every run

### Key constraints (non-negotiable)

| Constraint | Rule |
|---|---|
| No frontend | No Flask, no WebSocket, no HTML, no dashboard |
| No frame-level terminal log | Only summary per video, then overall summary |
| No fake metrics | Never compute accuracy without ground truth |
| No absolute paths | Use `pathlib` everywhere |
| No hardcoded paths | All paths from config or `paths.py` |
| UTF-8 JSON output | `json.dumps(..., ensure_ascii=False, indent=2)` |
| Proxy metric label | Any metric without GT must be `"metric_type": "proxy"` |
| GT metric label | Any metric with GT must be `"metric_type": "ground_truth"` |

---

## 2. Repository Structure

```
CPose/
├── README.md
├── CLAUDE.md                        ← this file
├── ACADEMIC.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── cpose_default.yaml           ← master runtime config
│   ├── model_registry.yaml          ← model paths + fallbacks
│   ├── model_registry.demo_i5.yaml  ← CPU-optimised variant
│   ├── camera_topology.yaml         ← cross-camera transition rules
│   ├── multicam_manifest.json
│   ├── logging.yaml
│   └── edge_profile.yaml
│
├── data-test/                       ← demo input videos (not committed)
│   └── README.md
│
├── dataset/
│   ├── annotations/
│   │   ├── detection_gt/
│   │   ├── tracking_gt/
│   │   ├── pose_gt/
│   │   ├── adl_gt/
│   │   └── global_id_gt/
│   ├── outputs/                     ← legacy compat only
│   └── runs/                        ← all research runs live here
│       └── <run_id>/
│
├── models/
│   ├── yolo/                     ← .pt / .onnx model files
│   ├── reid/
│   └── face/
│
├── src/
│   ├── __init__.py
│   │
│   ├── cli/                         ← thin CLI entrypoints
│   │   ├── run_detection.py
│   │   ├── run_tracking.py
│   │   ├── run_pose.py
│   │   ├── run_adl.py
│   │   ├── run_reid.py
│   │   ├── run_pipeline.py
│   │   └── run_benchmark.py
│   │
│   ├── common/                      ← shared utilities, no model logic
│   │   ├── paths.py
│   │   ├── config.py
│   │   ├── manifest.py
│   │   ├── topology.py
│   │   ├── schemas.py
│   │   ├── errors.py
│   │   ├── timer.py
│   │   ├── json_io.py
│   │   ├── video_io.py
│   │   ├── visualization.py
│   │   ├── logging_utils.py
│   │   └── paper_logger.py
│   │
│   ├── modules/
│   │   ├── detection/
│   │   │   ├── api.py               ← public API used by pipeline
│   │   │   ├── detector.py          ← model wrapper
│   │   │   ├── metrics.py
│   │   │   ├── schemas.py
│   │   │   └── main.py              ← CLI __main__ entrypoint
│   │   │
│   │   ├── tracking/
│   │   │   ├── api.py
│   │   │   ├── tracker.py
│   │   │   ├── metrics.py
│   │   │   ├── schemas.py
│   │   │   └── main.py
│   │   │
│   │   ├── pose_estimation/
│   │   │   ├── api.py
│   │   │   ├── pose_model.py
│   │   │   ├── keypoint_utils.py
│   │   │   ├── metrics.py
│   │   │   ├── schemas.py
│   │   │   └── main.py
│   │   │
│   │   ├── adl_recognition/
│   │   │   ├── api.py
│   │   │   ├── skeleton_features.py
│   │   │   ├── rule_based_adl.py
│   │   │   ├── smoothing.py
│   │   │   ├── metrics.py
│   │   │   ├── schemas.py
│   │   │   └── main.py
│   │   │
│   │   └── global_reid/
│   │       ├── api.py
│   │       ├── global_id_manager.py
│   │       ├── state_machine.py
│   │       ├── matching.py
│   │       ├── metrics.py
│   │       ├── schemas.py
│   │       └── main.py
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py          ← coordinates all modules
│   │   ├── run_all.py               ← full pipeline CLI entrypoint
│   │   ├── live_pipeline.py          ← live combined demo overlay
│   │   ├── stage_registry.py        ← module order + enable/disable
│   │   └── benchmark_all.py
│   │
│   ├── evaluation/
│   │   ├── detection_eval.py
│   │   ├── tracking_eval.py
│   │   ├── pose_eval.py
│   │   ├── adl_eval.py
│   │   ├── reid_eval.py
│   │   └── main.py
│   │
│   └── reports/
│       ├── make_paper_tables.py
│       ├── make_run_summary.py
│       └── export_metrics_csv.py
│
│ run_01_detection.bat
│ run_02_tracking.bat
│ run_03_pose.bat
│ run_04_adl.bat
│ run_05_reid.bat
│ run_06_pipeline.bat
│ run_07_benchmark.bat
│
├── docs/
│   ├── module_contracts.md
│   ├── metrics_policy.md
│   └── paper_logging.md
│
├── experiments/
│
└── tests/
    ├── test_manifest.py
    ├── test_topology.py
    ├── test_detection_schema.py
    ├── test_tracking_schema.py
    ├── test_pose_schema.py
    ├── test_adl_schema.py
    └── test_reid_schema.py
```

### Legacy compatibility

The legacy path `python -m src.pipeline.run_all` must still work via a compatibility wrapper. However, all new development targets:

```bash
python -m src.modules.<module>.main
python -m src.pipeline.run_all
```

---

## 3. Module Architecture & Contracts

Each module must have:
- `api.py` — callable from the pipeline orchestrator
- `main.py` — `__main__` entrypoint for standalone CLI use
- `metrics.py` — computes and saves metrics; never fakes numbers
- `schemas.py` — defines Pydantic or dataclass schemas for input/output

### 3.1 Module 1 — Detection

**Responsibility:** Detect persons in video frames only.

| Item | Value |
|---|---|
| Input | video file(s) in `data-test/` |
| Model | `yolo11n.pt` (fallback: `yolov8n.pt`) |
| Output | `detections.json`, `detection_overlay.mp4`, `detection_metrics.json` |
| Class filter | class_id = 0 (person) only |

**Forbidden:**
- Must NOT assign track IDs
- Must NOT run pose estimation
- Must NOT classify ADL labels

**Proxy metrics (no GT needed):**

| Metric | Description |
|---|---|
| `total_frames` | frames processed |
| `total_detections` | total person detections |
| `avg_confidence` | mean detection confidence |
| `avg_persons_per_frame` | average concurrent detections |
| `fps` | processing speed |
| `latency_ms` | ms per frame |

**GT metrics (requires `dataset/annotations/detection_gt/`):**
Precision, Recall, F1, mAP@50

---

### 3.2 Module 2 — Tracking

**Responsibility:** Assign stable local track IDs within a single camera/video.

| Item | Value |
|---|---|
| Input | `detections.json` + original video |
| Tracker | ByteTrack (default) |
| Output | `tracks.json`, `tracking_overlay.mp4`, `tracking_metrics.json` |

**Forbidden:**
- Must NOT create Global IDs
- Must NOT run pose estimation
- Must NOT classify ADL labels

**Track quality score formula:**

```
Q_track = 0.5 * mean_conf + 0.3 * min(age / W, 1) - 0.2 * min(misses / max_age, 1)
```

Where `W` = ADL window (default 30), `max_age` = 30.

**Proxy metrics:**

| Metric | Description |
|---|---|
| `total_tracks` | total distinct track IDs |
| `mean_track_age` | average track lifespan in frames |
| `confirmed_track_ratio` | ratio of confirmed vs tentative tracks |
| `fragment_proxy` | estimated track break count |
| `fps` | processing speed |
| `latency_ms` | ms per frame |

**GT metrics (requires `dataset/annotations/tracking_gt/`):**
IDF1, ID Switch Count, Fragmentation, HOTA

---

### 3.3 Module 3 — Pose Estimation

**Responsibility:** Extract COCO-17 keypoints for each tracked person.

| Item | Value |
|---|---|
| Input | video + `tracks.json` (or raw video if tracks unavailable) |
| Model | `yolo11n-pose.pt` (fallback: `yolov8n-pose.pt`) |
| Keypoints | COCO-17 (nose through right ankle) |
| Output | `keypoints.json`, `pose_overlay.mp4`, `pose_metrics.json` |

**Forbidden:**
- Must NOT infer ADL labels from keypoints
- Must NOT create Global IDs
- Must NOT re-run detection or tracking

**Visible keypoint formula:**

```
visible(K_i) = 1 if conf_i >= tau_kp else 0
r_visible = (1/17) * sum(visible(K_i) for i in 0..16)
```

Default `tau_kp = 0.30`.

**Proxy metrics:**

| Metric | Description |
|---|---|
| `total_pose_instances` | total persons with keypoints |
| `mean_keypoint_confidence` | average keypoint confidence |
| `visible_keypoint_ratio` | average `r_visible` across all instances |
| `missing_keypoint_rate` | 1 - visible_keypoint_ratio |
| `fps` | processing speed |
| `latency_ms` | ms per frame |

**GT metrics (requires `dataset/annotations/pose_gt/`):**
PCK@0.1, PCK@0.05, missing keypoint rate per joint

---

### 3.4 Module 4 — ADL Recognition

**Responsibility:** Classify Activities of Daily Living from skeleton sequences only.

| Item | Value |
|---|---|
| Input | `keypoints.json` + original video (for overlay only) |
| Method | Rule-based baseline (window size 30 frames) |
| Output | `adl_events.json`, `adl_overlay.mp4`, `adl_metrics.json` |

**Forbidden:**
- Must NOT create Global IDs
- Must NOT re-run detection, tracking, or pose

**ADL class labels:**

| Label | Description |
|---|---|
| `standing` | upright, low ankle velocity |
| `sitting` | low knee angle, low velocity |
| `walking` | high ankle velocity, upright torso |
| `lying_down` | high torso angle, wide bbox aspect ratio |
| `falling` | high torso angle + sudden velocity spike |
| `reaching` | wrist above shoulder |
| `bending` | torso angle 30–68°, low velocity |
| `unknown` | fewer than 8 visible keypoints |

**Rule-based decision tree (simplified):**

```
if visible_keypoints < 8:          → unknown
elif torso_angle > 68 and ankle_velocity high:  → falling
elif torso_angle > 68 and bbox_ar > 1.15:       → lying_down
elif knee_angle < 150 and ankle_velocity low:   → sitting
elif torso_angle > 45 and ankle_velocity low:   → bending
elif wrist_above_shoulder:                      → reaching
elif ankle_velocity > threshold:               → walking
else:                                          → standing
```

**Temporal smoothing:** majority vote over 5–7 frames to reduce label flickering.

**Proxy metrics:**

| Metric | Description |
|---|---|
| `total_adl_events` | total ADL event records |
| `class_distribution` | count and ratio per label |
| `unknown_rate` | ratio of `unknown` labels |
| `avg_confidence` | mean rule confidence |
| `fps_equivalent` | ADL events processed per second |

**GT metrics (requires `dataset/annotations/adl_gt/`):**
Accuracy, Macro-F1, per-class F1, Confusion Matrix

---

### 3.5 Module 5 — Global ReID (TFCS-PAR)

**Responsibility:** Assign stable Global IDs across cameras using time, topology, face, body, pose, and height evidence fusion.

| Item | Value |
|---|---|
| Input | `tracks.json`, `keypoints.json`, `adl_events.json`, manifest, topology |
| Method | TFCS-PAR score fusion |
| Output | `reid_tracks.json`, `global_person_table.json`, `reid_overlay.mp4`, `reid_metrics.json` |

**Forbidden:**
- Must NOT re-run detection, tracking, or pose if JSON inputs exist
- Must NOT use body appearance as the sole matching signal

**Score fusion formula:**

```
S_total = w_f*S_face + w_b*S_body + w_p*S_pose + w_h*S_height + w_t*S_time + w_c*S_camera
```

**Time-window gating:**

```
G_time(i,j) = 1 if delta_t in [T_min(i→j), T_max(i→j)] else 0
```

**Camera topology gating:**

```
G_camera(c_i, c_j) = 1 if (c_i, c_j) in topology_edges else 0
```

**Identity decision:**

```
if S_total >= strong_threshold (0.65):  → assign existing GID (strong_match)
elif S_total >= weak_threshold (0.45):  → assign tentatively (weak_match)
else:                                   → new GID or UNK
```

**Person state machine:**

| State | Meaning |
|---|---|
| `ACTIVE` | person currently visible in a camera |
| `PENDING_TRANSFER` | recently left via exit zone |
| `IN_BLIND_ZONE` | in elevator, stairwell, or unmapped area |
| `IN_ROOM` | entered a room, may exit same camera |
| `CLOTHING_CHANGE_SUSPECTED` | body appearance changed significantly post-room |
| `DORMANT` | not seen for a while but not yet closed |
| `CLOSED` | track ended |

**Clothing-change handling:** When state is `IN_ROOM` or `CLOTHING_CHANGE_SUSPECTED`, reduce body appearance weight, increase time/topology/height/pose weights. Do not create a new GID immediately if transition window is valid.

**Proxy metrics:**

| Metric | Description |
|---|---|
| `global_id_count` | total unique Global IDs assigned |
| `pending_count` | GIDs in PENDING_TRANSFER state |
| `conflict_count` | multi-candidate conflicts |
| `topology_conflict_count` | transitions rejected by topology |
| `unknown_match_count` | tracks that could not be matched |

**GT metrics (requires `dataset/annotations/global_id_gt/`):**
Global ID Accuracy, Cross-camera IDF1, False Split Rate, False Merge Rate, ID Switch Count

---

### 3.6 Module — Evaluation (standalone)

**Responsibility:** Compute ground-truth metrics from saved pipeline outputs. Does NOT modify pipeline outputs.

| Input | Pipeline run directory + `dataset/annotations/` |
| Output | Evaluation results in `07_evaluation/` within the run directory |

---

### 3.7 Module — Reports (standalone)

**Responsibility:** Generate paper-ready CSV and Markdown tables from saved metrics. Does NOT run any model.

| Input | Pipeline run directory |
| Output | CSV/Markdown files in `08_paper_report/` within the run directory |

---

## 4. Pipeline & Orchestration

### Pipeline execution order

```
Stage 0: Input validation + manifest loading
Stage 1: Detection        → 01_detection/
Stage 2: Tracking         → 02_tracking/
Stage 3: Pose Estimation  → 03_pose/
Stage 4: ADL Recognition  → 04_adl/
Stage 5: Global ReID      → 05_global_reid/
Stage 6: Comparison video → 06_comparison/
Stage 7: Evaluation       → 07_evaluation/  (only if --gt provided)
Stage 8: Paper Report     → 08_paper_report/
```

### Input video processing order

All modules that enumerate files from `data-test/` must process videos in chronological order, earliest first. The canonical order is parsed from filenames in this format:

```
<camera_id>_YYYY-MM-DD_HH-MM-SS.<ext>
```

Example: `cam2_2026-01-28_15-57-54.mp4` must be processed before `cam1_2026-01-29_16-26-25.mp4`, even though `cam1` sorts earlier alphabetically.

Rules:

- Never rely on filesystem enumeration order or alphabetical filename order for `data-test/`.
- Prefer manifest `start_time` when a manifest is provided; otherwise parse the timestamp embedded in the video filename.
- Sort ascending by timestamp, then by `camera_id`, then by filename as deterministic tie-breakers.
- The resolved chronological order must be reused consistently by detection, tracking, pose, ADL, Global ReID, comparison generation, metrics `input_videos`, and `run_manifest.json`.
- `--compare-count N` selects the first N videos after chronological sorting, not the first N alphabetically.
- If a filename has no parseable timestamp, log one warning and place it after timestamped videos using filesystem modified time as fallback.

### Error handling

- If a single video fails in any stage, log one error line and continue with the next video.
- If an entire stage fails, log the error and skip downstream stages that depend on it.
- Never crash the entire pipeline over one bad video.

### Stage registry

`pipeline/stage_registry.py` controls which stages are enabled. Each stage can be independently disabled via config or CLI flag.

---

## 5. Run Output Structure

Every pipeline run creates a unique directory:

```
dataset/runs/<run_id>/
├── run_config.yaml          ← copy of effective config for this run
├── run_manifest.json        ← resolved video list + camera IDs
├── console_summary.txt      ← full terminal output saved to file
├── 00_input/                ← symlinks or copies of input videos
├── 01_detection/
│   ├── <video_stem>_detections.json
│   ├── <video_stem>_detection_overlay.mp4
│   └── detection_metrics.json
├── 02_tracking/
│   ├── <video_stem>_tracks.json
│   ├── <video_stem>_tracking_overlay.mp4
│   └── tracking_metrics.json
├── 03_pose/
│   ├── <video_stem>_keypoints.json
│   ├── <video_stem>_pose_overlay.mp4
│   └── pose_metrics.json
├── 04_adl/
│   ├── <video_stem>_adl_events.json
│   ├── <video_stem>_adl_overlay.mp4
│   └── adl_metrics.json
├── 05_global_reid/
│   ├── <video_stem>_reid_tracks.json
│   ├── global_person_table.json
│   ├── <video_stem>_reid_overlay.mp4
│   └── reid_metrics.json
├── 06_comparison/
│   ├── <video_stem>_raw_vs_detection.mp4
│   ├── <video_stem>_raw_vs_tracking.mp4
│   ├── <video_stem>_raw_vs_pose.mp4
│   ├── <video_stem>_raw_vs_adl.mp4
│   ├── <video_stem>_raw_vs_reid.mp4
│   └── preview_index.json
├── 07_evaluation/
│   └── (populated only when --gt is provided)
└── 08_paper_report/
    ├── paper_metrics_summary.md
    ├── table_module_runtime.csv
    ├── table_detection_results.csv
    ├── table_tracking_results.csv
    ├── table_pose_results.csv
    ├── table_adl_results.csv
    ├── table_reid_results.csv
    └── figure_list.md
```

### Run ID format

```
<YYYY-MM-DD>_<HH-MM-SS>_<tag>

Examples:
  2026-05-01_15-30-22_baseline
  2026-05-01_20-10-15_ablation_no_face
```

The `<tag>` is user-supplied via `--run-id`; defaults to `run` if not provided.

### Legacy output path

`dataset/outputs/` is kept for backward compatibility only. All new research runs must use `dataset/runs/<run_id>/`.

---

## 6. CLI Commands

### Module-level commands (standalone)

```bash
# Detection
python -m src.modules.detection.main \
  --input data-test \
  --run-id test_detection \
  --make-comparison \
  --compare-count 2

# Tracking (reads from existing detection run)
python -m src.modules.tracking.main \
  --run-dir dataset/runs/test_detection \
  --make-comparison \
  --compare-count 2

# Pose estimation
python -m src.modules.pose_estimation.main \
  --run-dir dataset/runs/test_detection \
  --make-comparison \
  --compare-count 2

# ADL recognition
python -m src.modules.adl_recognition.main \
  --run-dir dataset/runs/test_detection \
  --make-comparison \
  --compare-count 2

# Global ReID
python -m src.modules.global_reid.main \
  --run-dir dataset/runs/test_detection \
  --make-comparison \
  --compare-count 2
```

### Full pipeline

`run_06_pipeline.bat` is the live demo path. It must show one combined overlay per frame with Detection + Tracking + Pose + ADL + ReID at the same time.

```bash
python -m src.pipeline.live_pipeline \
  --input data-test \
  --output dataset/runs \
  --models configs/model_registry.demo_i5.yaml \
  --topology configs/camera_topology.yaml \
  --run-id live_combined
```

The offline research pipeline remains available for saved stage-by-stage outputs:

```bash
python -m src.pipeline.run_all \
  --input data-test \
  --output dataset/runs \
  --config configs/cpose_default.yaml \
  --models configs/model_registry.yaml \
  --manifest configs/multicam_manifest.example.json \
  --topology configs/camera_topology.yaml \
  --run-id demo_baseline \
  --make-comparison \
  --compare-count 2
```

### Benchmark

```bash
python -m src.pipeline.benchmark_all \
  --run-dir dataset/runs/<run_id>
```

### Evaluation (with ground truth)

```bash
python -m src.evaluation.main \
  --run-dir dataset/runs/<run_id> \
  --gt dataset/annotations
```

### Paper report generation

```bash
python -m src.reports.make_paper_tables \
  --run-dir dataset/runs/<run_id>
```

### Legacy compatibility (still works, not recommended)

```bash
python -m src.pipeline.run_all   # legacy path, wraps new pipeline
```

### CLI flags reference

| Flag | Type | Description |
|---|---|---|
| `--input` | path | Input video directory |
| `--run-dir` | path | Existing run directory (for downstream modules) |
| `--run-id` | str | Tag appended to run timestamp |
| `--output` | path | Root output directory (`dataset/runs` by default) |
| `--config` | path | Path to `cpose_default.yaml` |
| `--models` | path | Path to `model_registry.yaml` |
| `--manifest` | path | Path to `multicam_manifest.json` |
| `--topology` | path | Path to `camera_topology.yaml` |
| `--gt` | path | Path to ground-truth annotations |
| `--make-comparison` | flag | Generate raw vs. processed comparison videos |
| `--compare-count` | int | Number of videos to create comparisons for (default: 2) |
| `--preview` | flag | Open OpenCV window for live preview (requires display) |
| `--debug` | flag | Print full stack traces and verbose logs |

---

## 7. Config Files

### `configs/cpose_default.yaml`

```yaml
project:
  name: CPose
  task: multi_camera_human_monitoring
  output_root: dataset/runs

input:
  default_video_dir: data-test
  video_exts: [".mp4", ".avi", ".mov", ".mkv"]

runtime:
  device: auto          # auto | cpu | cuda | mps
  save_overlay: true
  save_json: true
  save_metrics: true
  make_comparison: false
  compare_count: 2
  preview: false
  debug: false

logging:
  console_level: summary
  save_console_summary: true
  paper_ready_tables: true

metrics:
  require_ground_truth_for_accuracy: true
  default_metric_type: proxy
```

### `configs/model_registry.yaml`

```yaml
detector:
  name: yolo11n
  path: models/weights/yolo11n.pt
  fallback:
    - models/weights/yolov8n.pt
  conf: 0.5
  imgsz: 640
  class_id: 0

tracker:
  name: bytetrack
  min_hits: 3
  max_age: 30

pose:
  name: yolo11n-pose
  path: models/weights/yolo11n-pose.pt
  fallback:
    - models/weights/yolov8n-pose.pt
  conf: 0.45
  keypoint_conf: 0.30
  imgsz: 640

adl:
  method: rule_based
  window_size: 30
  min_visible_keypoints: 8
  smoothing_frames: 5

reid:
  method: tfcs_par
  strong_threshold: 0.65
  weak_threshold: 0.45
  top_k_candidates: 20
```

---

## 8. JSON Schemas

All JSON files must be UTF-8, indent=2, and include `failure_reason` on every record.

### `detections.json`

```json
[
  {
    "frame_id": 0,
    "timestamp_sec": 0.0,
    "camera_id": "cam1",
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.91,
        "class_id": 0,
        "class_name": "person",
        "failure_reason": "OK"
      }
    ]
  }
]
```

### `tracks.json`

```json
[
  {
    "frame_id": 0,
    "timestamp_sec": 0.0,
    "camera_id": "cam1",
    "tracks": [
      {
        "track_id": 1,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.88,
        "age": 45,
        "hits": 40,
        "misses": 2,
        "is_confirmed": true,
        "quality_score": 0.87,
        "failure_reason": "OK"
      }
    ]
  }
]
```

### `keypoints.json`

```json
[
  {
    "frame_id": 0,
    "timestamp_sec": 0.0,
    "camera_id": "cam1",
    "persons": [
      {
        "track_id": 1,
        "bbox": [x1, y1, x2, y2],
        "visible_keypoint_ratio": 0.82,
        "keypoints": [
          {"id": 0, "name": "nose", "x": 123.0, "y": 222.0, "confidence": 0.91},
          {"id": 1, "name": "left_eye", "x": 130.0, "y": 215.0, "confidence": 0.88}
        ],
        "failure_reason": "OK"
      }
    ]
  }
]
```

### `adl_events.json`

```json
[
  {
    "frame_id": 30,
    "timestamp_sec": 1.0,
    "camera_id": "cam1",
    "track_id": 1,
    "raw_label": "walking",
    "smoothed_label": "walking",
    "adl_label": "walking",
    "confidence": 0.79,
    "window_size": 30,
    "visible_keypoint_ratio": 0.82,
    "failure_reason": "OK"
  }
]
```

### `reid_tracks.json`

```json
[
  {
    "frame_id": 30,
    "timestamp_sec": 1.0,
    "camera_id": "cam2",
    "local_track_id": 1,
    "global_id": "GID-001",
    "state": "ACTIVE",
    "match_status": "strong_match",
    "score_total": 0.72,
    "score_time": 1.0,
    "score_topology": 1.0,
    "score_pose": 0.68,
    "score_body": 0.51,
    "score_face": null,
    "delta_time_sec": 42.0,
    "topology_allowed": true,
    "failure_reason": "OK"
  }
]
```

### Metrics JSON (all modules)

Every `*_metrics.json` file must include these top-level fields:

```json
{
  "metric_type": "proxy",
  "run_id": "2026-05-01_15-30-22_baseline",
  "module": "pose_estimation",
  "input_videos": ["cam1.mp4", "cam2.mp4"],
  "model_info": {"name": "yolo11n-pose", "path": "models/weights/yolo11n-pose.pt"},
  "output_paths": {"json": "...", "overlay": "...", "paper_table": "..."},
  "metrics": { ... }
}
```

---

## 9. Metrics Policy

### Core rule

> **Never report accuracy unless you have ground truth. Never call a proxy metric an accuracy.**

| Situation | `metric_type` field | Can use "accuracy"? |
|---|---|---|
| No ground truth available | `"proxy"` | NO |
| Ground truth loaded from `dataset/annotations/` | `"ground_truth"` | YES |

### Proxy metrics are valid research outputs

Proxy metrics measure system behaviour without GT labels. They are legitimate for:
- Benchmarking runtime (FPS, latency)
- Describing output volume (detections per frame, pose instances)
- Measuring signal quality (keypoint visibility, track confirmation rate)
- Comparing system variants (ablation studies)

### What to never do

- Do not compute `accuracy = correct / total` unless you have labelled GT.
- Do not call `avg_confidence` an accuracy metric — it is a proxy.
- Do not invent numbers. If a metric cannot be computed, omit it or set it to `null`.
- Do not pretend `unknown_rate` is a performance metric — it is a coverage proxy.

### Runtime metrics (always proxy)

| Metric | Unit |
|---|---|
| `module_fps` | frames / second |
| `end_to_end_fps` | frames / second |
| `total_runtime_sec` | seconds |
| `latency_ms_per_frame` | milliseconds |

---

## 10. Terminal Logging Rules

### What is allowed per module run

1. Header block with module name and run ID
2. Key config summary (input, output, model, video count, mode)
3. One line per video processed
4. One summary block at the end
5. Paths to all output files

### What is forbidden

- Per-frame log lines (absolutely forbidden)
- Stack traces unless `--debug` flag is set
- Verbose model loading output (suppress unless `--debug`)
- Repeated warnings for the same condition

### Reference terminal format

```
============================================================
CPose Pose Estimation | Run: 2026-05-01_15-30-22_baseline
============================================================
Input        : data-test/
Output       : dataset/runs/2026-05-01_15-30-22_baseline/03_pose
Model        : models/weights/yolo11n-pose.pt
Videos       : 2
Mode         : module-only
Comparison   : enabled, first 2 videos

[1/2] cam1_2026-01-29_16-26-25.mp4
Frames=1820 | PoseInst=1712 | KP-Visible=0.84 | FPS=28.6 | Latency=34.9ms

[2/2] cam2_2026-01-29_16-27-10.mp4
Frames=1765 | PoseInst=1630 | KP-Visible=0.81 | FPS=27.9 | Latency=35.8ms

-------------------- SUMMARY --------------------
Total frames        : 3585
Total pose instances: 3342
Mean KP visibility  : 0.825
Average FPS         : 28.2
Metric type         : proxy
Saved metrics       : dataset/runs/.../03_pose/pose_metrics.json
Paper table         : dataset/runs/.../08_paper_report/table_pose_results.csv
Comparison videos   : dataset/runs/.../06_comparison/
=================================================
```

### Error and warning line formats

```
[ERROR] cam3_footage.mp4 | INVALID_VIDEO | reason=codec not supported
[WARN]  model not found, fallback used: models/weights/yolov8n.pt
```

### `console_summary.txt`

The complete terminal output of every run must be saved verbatim to `console_summary.txt` in the run root directory.

---

## 11. Video Comparison Rules

### When to generate comparisons

Only when `--make-comparison` is passed. Default is off.

### Layout

Every comparison video is side-by-side:
- **Left half:** raw input frame
- **Right half:** processed output frame (with overlays)
- **Text overlay (bottom-left of each half, small font):**
  - Left: `LEFT: Raw Input`
  - Right: `RIGHT: CPose <Module Name>`

### Output naming

```
<video_stem>_raw_vs_detection.mp4
<video_stem>_raw_vs_tracking.mp4
<video_stem>_raw_vs_pose.mp4
<video_stem>_raw_vs_adl.mp4
<video_stem>_raw_vs_reid.mp4
```

All comparison videos go to `06_comparison/`.

### `--compare-count N`

Only generate comparisons for the first N videos in `data-test/`. Default N = 2.

### `--preview`

Open an OpenCV `imshow` window for live playback. If no display is available (headless server), issue one warning line and continue. **Never crash on missing display.**

### `preview_index.json`

```json
{
  "run_id": "2026-05-01_15-30-22_baseline",
  "comparisons": [
    {
      "video": "cam1_2026-01-29_16-26-25.mp4",
      "raw_vs_detection": "06_comparison/cam1_raw_vs_detection.mp4",
      "raw_vs_tracking": "06_comparison/cam1_raw_vs_tracking.mp4",
      "raw_vs_pose": "06_comparison/cam1_raw_vs_pose.mp4",
      "raw_vs_adl": "06_comparison/cam1_raw_vs_adl.mp4",
      "raw_vs_reid": "06_comparison/cam1_raw_vs_reid.mp4"
    }
  ]
}
```

---

## 12. Error Taxonomy

All modules must use the following error codes in `failure_reason` fields. Defined in `src/common/errors.py`.

| Code | Meaning |
|---|---|
| `OK` | No error |
| `NO_PERSON_DETECTED` | No person found in frame |
| `LOW_DETECTION_CONFIDENCE` | Detection below confidence threshold |
| `TRACK_FRAGMENTED` | Track was broken and restarted |
| `UNCONFIRMED_TRACK` | Track has not reached `min_hits` yet |
| `SHORT_TRACK_WINDOW` | Track too short for ADL window |
| `LOW_KEYPOINT_VISIBILITY` | Fewer than `min_visible_keypoints` keypoints visible |
| `NO_FACE` | No face detected in person crop |
| `BODY_OCCLUDED` | Person bounding box heavily occluded |
| `TOPOLOGY_CONFLICT` | Camera transition not permitted by topology |
| `TIME_WINDOW_CONFLICT` | Transition time outside allowed window |
| `MULTI_CANDIDATE_CONFLICT` | Multiple Global ID candidates above threshold |
| `MODEL_NOT_FOUND` | Model file missing; fallback attempted |
| `INVALID_VIDEO` | Video file cannot be opened or read |
| `INVALID_MANIFEST` | Manifest JSON is malformed |
| `INVALID_TOPOLOGY` | Topology YAML is malformed or missing edges |
| `MISSING_INPUT_JSON` | Required upstream JSON not found |
| `GROUND_TRUTH_NOT_FOUND` | GT annotation file missing for evaluation |

---

## 13. Paper Report Output

### Required files in `08_paper_report/`

| File | Contents |
|---|---|
| `paper_metrics_summary.md` | Narrative summary with all key metrics |
| `table_module_runtime.csv` | FPS and latency per module |
| `table_detection_results.csv` | Detection metrics |
| `table_tracking_results.csv` | Tracking metrics |
| `table_pose_results.csv` | Pose estimation metrics |
| `table_adl_results.csv` | ADL metrics |
| `table_reid_results.csv` | Global ReID metrics |
| `figure_list.md` | List of generated overlay and comparison videos |

### `table_module_runtime.csv` format

```csv
module,total_frames,total_instances,fps,latency_ms,metric_type
detection,3585,4201,31.2,32.1,proxy
tracking,3585,3988,29.4,34.0,proxy
pose,3585,3342,28.2,35.4,proxy
adl,3585,3200,120.5,8.3,proxy
global_reid,3585,14,300.0,3.2,proxy
```

### Rules for paper tables

- Only summary numbers — no per-frame data.
- Always include a `metric_type` column.
- If a metric could not be computed, write `N/A` — never a fabricated number.
- Tables must be directly paste-able into a LaTeX `tabular` or Markdown table without editing.

---

## 14. Coding Constraints

### Always

- Use `pathlib.Path` for all file and directory operations. Never use `os.path.join` with string concatenation.
- Save JSON with `json.dumps(data, ensure_ascii=False, indent=2)` and explicit UTF-8 encoding.
- Check if a module's input JSON exists before processing. Raise `MISSING_INPUT_JSON` if not.
- Use relative paths anchored from the project root. Never hardcode absolute paths.
- Load model paths from `model_registry.yaml`, not from function arguments or environment variables.
- Attempt fallback models in order when the primary model is not found. Log `[WARN]` for each fallback attempt.
- Wrap per-video processing in try/except. Log `[ERROR]` and continue — never crash the loop.

### Never

- Do not log per-frame data to the terminal.
- Do not open a GUI window unless `--preview` is explicitly passed.
- Do not import from `app/` in `src/`. They are separate packages.
- Do not run model inference inside `metrics.py`, `schemas.py`, or `main.py`. All inference belongs in the backend/model wrapper.
- Do not call `accuracy` any variable that is not `correct_predictions / total_predictions` against labelled ground truth.
- Do not write output files directly to `dataset/outputs/` from new code. All new code writes to `dataset/runs/<run_id>/`.

### Module isolation

- `detection` imports from: `src.common` only
- `tracking` imports from: `src.common`, reads `detections.json`
- `pose_estimation` imports from: `src.common`, reads `tracks.json`
- `adl_recognition` imports from: `src.common`, reads `keypoints.json`
- `global_reid` imports from: `src.common`, reads `tracks.json`, `keypoints.json`, `adl_events.json`
- `pipeline` imports from: all module `api.py` files
- No module imports directly from another module's internals (only via `api.py`)

---

## 15. Review Constraints

When reviewing or generating code for CPose, apply the following checks:

### Metric checks

- [ ] Is `metric_type` set on every metrics object?
- [ ] Does any code compute `accuracy` without checking for ground truth first?
- [ ] Are all `null` metrics omitted from paper tables rather than shown as zero?

### Output path checks

- [ ] Does any function hardcode an absolute path?
- [ ] Does any function write to `dataset/outputs/` directly (it should use the run directory)?
- [ ] Does any path construction use string concatenation instead of `pathlib`?

### Module boundary checks

- [ ] Does the detection module assign any `track_id`?
- [ ] Does the tracking module create any `global_id`?
- [ ] Does the pose module classify any ADL label?
- [ ] Does the ADL module create any Global ID?
- [ ] Does the global_reid module re-run detection/tracking/pose if JSON inputs exist?

### Log checks

- [ ] Does any module print per-frame output to the terminal without `--debug`?
- [ ] Is every video processing loop wrapped in try/except with a single error log line?
- [ ] Is the summary block printed at the end of each module run?

### JSON checks

- [ ] Is every JSON file written with `ensure_ascii=False, indent=2`?
- [ ] Does every record contain a `failure_reason` field?
- [ ] Does every metrics JSON contain `metric_type`, `run_id`, `module`, `input_videos`?

---

## 16. Roadmap

### Phase 1 — Demo on i5 CPU (current)

| Task | Target |
|---|---|
| YOLOv8n detection | Process 4 short videos |
| ByteTrack | Stable tracks for 3 persons |
| YOLOv8n-pose | Skeleton overlay working |
| Rule-based ADL | standing/sitting/walking/unknown working |
| TFCS-PAR | Global ID maintained across 4 cameras |
| Proxy benchmark | FPS, latency, unknown rate in paper table |

### Phase 2 — Ground-truth metrics for paper

| Task | Target |
|---|---|
| Annotate 4 cameras × 3 persons | GT Global IDs |
| Annotate ADL events | Macro-F1, Confusion Matrix |
| Annotate a few detection clips | Detection Precision/Recall |
| Evaluate Global ReID | ID Switch, False Split, False Merge |
| Ablation study | Remove face/body/pose/topology one at a time |

### Phase 3 — GPU-accelerated upgrade

| Task | Target |
|---|---|
| RTMPose / RTMO | Higher pose accuracy + FPS |
| BoT-SORT / Deep OC-SORT | More robust tracking |
| CTR-GCN / BlockGCN | Learned ADL recognition |
| KPR / Pose2ID | Occluded ReID |
| MiniFASNet / CDCN | Anti-spoofing |

---

## 17. Reference Numbers

### Pose/tracking benchmarks from literature

| Method | Dataset | Metric | Value |
|---|---|---|---:|
| RTMPose-m | COCO | AP | 75.8 |
| RTMPose-m | Intel i7 CPU | FPS | 90+ |
| RTMPose-m | GTX 1660Ti | FPS | 430+ |
| RTMO-l | COCO val2017 | AP | 74.8 |
| RTMO-l | V100 GPU | FPS | 141 |
| ByteTrack | MOT17 | HOTA | 63.1 |
| BoT-SORT | MOT17 | HOTA | 65.0 |
| Pose-assisted MCPT | AI City 2023 | IDF1 | 86.76% |
| MotionBERT | NTU-60 XSub | Top-1 | 97.2% |
| π-ViT | Toyota Smarthome CS | mCA | 72.9% |
| UniHCP | Market-1501 | ReID mAP | 90.3 |

These numbers are from published papers and are used as baselines for comparison. Do not present them as CPose results.

---

## 18. Pre-Demo Checklist

Before any live demonstration:

- [ ] At least 4 videos placed in `data-test/`
- [ ] `multicam_manifest.json` timestamps match video filenames
- [ ] `camera_topology.yaml` edges reflect actual camera layout
- [ ] `models/weights/yolov8n.pt` downloaded
- [ ] `models/weights/yolov8n-pose.pt` downloaded
- [ ] Detection overlay renders correctly
- [ ] Tracking overlay with stable IDs renders correctly
- [ ] Pose skeleton overlay renders correctly
- [ ] ADL labels appear on overlay
- [ ] Global ID labels appear on reid overlay
- [ ] `benchmark_summary.json` exists in run directory
- [ ] Terminal output shows no per-frame lines
- [ ] No metric is labelled "accuracy" without ground truth
- [ ] Comparison videos exist in `06_comparison/`
- [ ] `08_paper_report/table_module_runtime.csv` is populated

---
