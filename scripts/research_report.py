"""CPose research/evaluation report generator for paper experiments."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXTS = {".txt", ".json", ".csv"}
MODEL_EXTS = {".pt", ".onnx", ".engine", ".pth", ".bin"}
ADL_CLASSES = ["standing", "sitting", "walking", "lying_down", "falling", "reaching", "bending", "unknown"]
MISSING = "MISSING"
NA = "N/A"


def import_optional(module_name: str) -> Tuple[Optional[Any], str]:
    """Import a module without failing the run."""
    try:
        return importlib.import_module(module_name), "OK"
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"NOT_INSTALLED: {exc.__class__.__name__}"


@contextlib.contextmanager
def suppress_native_stderr() -> Iterable[None]:
    """Suppress noisy native-library stderr messages during metadata probing."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
    except Exception:
        yield


def package_version(distribution: str, import_name: str) -> str:
    """Get package version without importing noisy packages when possible."""
    try:
        return importlib_metadata.version(distribution)
    except Exception:
        module, status = import_optional(import_name)
        if module is None:
            return status
        return str(getattr(module, "__version__", "INSTALLED_VERSION_UNKNOWN"))


def setup_logging(run_dir: Path) -> None:
    """Configure console and file logging."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    """Write dictionaries to CSV using stdlib csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_cell(row.get(key, "")) for key in fieldnames})


def normalize_cell(value: Any) -> str:
    """Convert values to stable CSV/Markdown text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def run_command(args: Sequence[str], cwd: Path) -> str:
    """Run a command and return stdout, or N/A on failure."""
    try:
        result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            return result.stdout.strip() or NA
        return NA
    except Exception:
        return NA


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML with PyYAML if available."""
    yaml, status = import_optional("yaml")
    if yaml is None:
        logging.info(f"[WARN] Cannot parse YAML {path}: {status}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logging.info(f"[WARN] Cannot parse YAML {path}: {exc}")
        return {}


def deep_get(data: Dict[str, Any], path: str, default: Any = MISSING) -> Any:
    """Read dot-separated keys from nested dictionaries."""
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def collect_config(project_root: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Collect CPose config files."""
    config_paths = [
        project_root / ".env",
        project_root / "config.py",
        project_root / "configs" / "config.yaml",
        project_root / "configs" / "phase1.yaml",
        project_root / "app" / "config.yaml",
    ]
    for folder in [project_root / "configs", project_root / "app"]:
        if folder.exists():
            config_paths.extend(folder.glob("*.json"))
            config_paths.extend(folder.glob("*.toml"))
            config_paths.extend(folder.glob("*.yaml"))
            config_paths.extend(folder.glob("*.yml"))
    seen: set[Path] = set()
    merged: Dict[str, Any] = {}
    used: List[str] = []
    for path in config_paths:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        used.append(str(path.relative_to(project_root)))
        if path.suffix.lower() in {".yaml", ".yml"}:
            merged = merge_dicts(merged, load_yaml_file(path))
        elif path.suffix.lower() == ".json":
            try:
                merged = merge_dicts(merged, json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logging.info(f"[WARN] Cannot parse JSON {path}: {exc}")
    return merged, used


def collect_environment(project_root: Path) -> Dict[str, Any]:
    """Collect hardware, software, Python, Git, and package versions."""
    psutil, _ = import_optional("psutil")
    torch, torch_status = import_optional("torch")
    cv2, cv2_status = import_optional("cv2")
    env: Dict[str, Any] = {
        "project_root": str(project_root.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run_command(["git", "rev-parse", "HEAD"], project_root),
        "git_branch": run_command(["git", "branch", "--show-current"], project_root),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": NA,
        "gpu": NA,
        "vram_total_gb": NA,
        "cuda_available": NA,
        "cuda_version": NA,
        "cudnn_version": NA,
    }
    if psutil is not None:
        try:
            env["cpu"] = platform.processor() or f"{psutil.cpu_count(logical=True)} logical CPUs"
            env["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            env["cuda_available"] = cuda_available
            env["cuda_version"] = getattr(torch.version, "cuda", None) or NA
            env["cudnn_version"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else NA
            if cuda_available:
                env["gpu"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                env["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
        except Exception as exc:
            env["gpu"] = f"GPU_QUERY_FAILED: {exc.__class__.__name__}"
    else:
        env["cuda_available"] = torch_status
    if cv2 is not None:
        env["opencv_build"] = "cv2 import OK"
    else:
        env["opencv_build"] = cv2_status

    package_imports = {
        "torch": ("torch", "torch"),
        "torchvision": ("torchvision", "torchvision"),
        "ultralytics": ("ultralytics", "ultralytics"),
        "opencv-python/cv2": ("opencv-python", "cv2"),
        "numpy": ("numpy", "numpy"),
        "scipy": ("scipy", "scipy"),
        "pandas": ("pandas", "pandas"),
        "sklearn": ("scikit-learn", "sklearn"),
        "faiss": ("faiss-cpu", "faiss"),
        "flask": ("flask", "flask"),
        "socketio": ("python-socketio", "socketio"),
    }
    packages: Dict[str, str] = {}
    for display, (distribution, module_name) in package_imports.items():
        packages[display] = package_version(distribution, module_name)
    env["packages"] = packages
    return env


def guess_module(path: Path) -> str:
    """Guess model module from filename."""
    name = path.name.lower()
    guesses = []
    for token in ["yolo", "pose", "arcface", "dinov2", "clip", "reid", "faiss", "bytetrack", "deepsort"]:
        if token in name:
            guesses.append(token)
    return "+".join(guesses) if guesses else "unknown"


def scan_models(project_root: Path, skip_load: bool = False) -> List[Dict[str, Any]]:
    """Find model files and lightly inspect YOLO models when possible."""
    roots = [project_root / "models", project_root / "models" / "product", project_root / "weights", project_root / "checkpoints"]
    files: List[Path] = []
    for root in roots:
        if root.exists():
            files.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXTS])
    rows: List[Dict[str, Any]] = []
    YOLO = None
    if not skip_load:
        ultralytics, _ = import_optional("ultralytics")
        YOLO = getattr(ultralytics, "YOLO", None) if ultralytics is not None else None
    for path in sorted(set(files)):
        stat = path.stat()
        row = {
            "module": guess_module(path),
            "model_name": path.name,
            "path": str(path.relative_to(project_root)),
            "file_type": path.suffix.lower(),
            "size_mb": round(stat.st_size / (1024**2), 3),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "status": "FOUND",
            "notes": "",
        }
        if YOLO is not None and path.suffix.lower() == ".pt" and ("yolo" in path.name.lower() or "pose" in path.name.lower()):
            start = time.perf_counter()
            try:
                model = YOLO(str(path))
                elapsed = time.perf_counter() - start
                names = getattr(model, "names", None)
                task = getattr(model, "task", NA)
                row["status"] = "LOADABLE_ULTRALYTICS"
                row["notes"] = f"task={task}; classes={names}; nc={len(names) if isinstance(names, dict) else NA}; load_time_sec={elapsed:.3f}"
            except Exception as exc:
                row["status"] = "LOAD_FAILED"
                row["notes"] = f"{exc.__class__.__name__}: {exc}"
        elif skip_load:
            row["notes"] = "load test skipped"
        rows.append(row)
    if not rows:
        rows.append({"module": NA, "model_name": NA, "path": NA, "file_type": NA, "size_mb": NA, "modified_time": NA, "status": "NO_MODEL_FILES_FOUND", "notes": ""})
    return rows


def extract_pipeline_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract paper-relevant parameters from parsed config."""
    specs = {
        "Recorder": [
            "phase1.conf_threshold", "phase1.person_conf_threshold", "phase1.trigger_min_consecutive",
            "phase1.pre_roll_seconds", "phase1.post_roll_seconds", "phase1.rearm_cooldown_seconds",
            "phase1.min_clip_seconds", "phase1.max_clip_seconds", "phase1.min_box_area_ratio",
            "phase1.snapshot_fps", "phase1.jpeg_quality", "phase1.person_class_id", "phase1.inference_every",
        ],
        "Analyzer": ["phase2.model", "phase2.person_class_id", "phase2.conf_threshold", "phase2.progress_every"],
        "Pose/ADL": [
            "models.pose_model_path", "phase3.model", "phase3.conf_threshold", "phase3.keypoint_conf_min",
            "phase3.window_size", "pose_utils.min_visible_keypoints", "phase3.adl_classes",
            "pose_utils.knee_bend_angle", "pose_utils.shoulder_raise", "pose_utils.velocity_walk",
            "pose_utils.falling_torso_angle", "pose_utils.lying_aspect_ratio", "pose_utils.confidence_unknown",
            "pose_utils.confidence_falling", "pose_utils.confidence_lying_down", "pose_utils.confidence_sitting",
            "pose_utils.confidence_bending", "pose_utils.confidence_reaching", "pose_utils.confidence_walking",
            "pose_utils.confidence_standing",
        ],
        "Tracking": ["tracker.name", "tracker.max_age", "tracker.n_init", "tracker.max_iou_distance", "tracker.max_cosine_distance", "tracker.half", "tracker.embedder"],
        "ReID/Global ID": [
            "global_id.strong_threshold", "global_id.weak_threshold", "global_id.confirm_frames",
            "global_id.top_k_candidates", "vector_db.search_top_k", "reid.threshold", "reid.top_k_similarity",
            "reid.pending_track_ttl_seconds", "reid.confirmed_track_ttl_seconds",
            "global_id.iou_resurrection_threshold", "global_id.quality_update_threshold",
            "global_id.use_hungarian", "global_id.max_unk_per_video",
        ],
        "VectorDB/FAISS": [
            "persistence.embedding_dim", "vector_db.search_top_k", "vector_db.index_type",
            "vector_db.medium_dataset_threshold", "vector_db.large_dataset_threshold", "vector_db.hnsw_m",
            "vector_db.hnsw_ef_construction", "vector_db.hnsw_ef_search", "vector_db.ivf_nlist", "vector_db.ivf_nprobe",
        ],
    }
    rows: List[Dict[str, Any]] = []
    for module, paths in specs.items():
        for path in paths:
            rows.append({"module": module, "model_name": path, "path": "", "file_type": "pipeline_parameter", "size_mb": "", "status": "CONFIG", "notes": deep_get(config, path)})
    windows = deep_get(config, "global_id.transition_windows", {})
    wanted = ["cam01->cam02", "cam02->cam03", "cam03->cam02", "cam03->cam04", "cam04->cam03", "cam04->cam04"]
    for key in wanted:
        value = MISSING
        if isinstance(windows, dict):
            value = windows.get(key, windows.get(key.replace("cam0", "cam"), MISSING))
        rows.append({"module": "Transition windows", "model_name": key, "path": "", "file_type": "pipeline_parameter", "size_mb": "", "status": "CONFIG" if value != MISSING else "TODO", "notes": value})
    return rows


def parse_camera_id(path: Path) -> str:
    """Infer camera id from filename or parent folder."""
    text = " ".join([path.stem.lower(), *(p.lower() for p in path.parts[-4:])])
    match = re.search(r"cam(?:era)?[_-]?0?([1-9]\d*)", text)
    if match:
        return f"cam{int(match.group(1)):02d}"
    return "unknown"


def parse_timestamp(path: Path) -> str:
    """Infer timestamp from common CPose filenames."""
    text = path.stem
    patterns = [
        r"(20\d{2})[-_](\d{2})[-_](\d{2})[_-](\d{2})[-_](\d{2})[-_](\d{2})",
        r"(20\d{6})[_-](\d{6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and len(match.groups()) == 6:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
        if match and len(match.groups()) == 2:
            date, clock = match.groups()
            return f"{date[:4]}-{date[4:6]}-{date[6:]} {clock[:2]}:{clock[2:4]}:{clock[4:]}"
    return NA


def scan_dataset(project_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Path], List[Path]]:
    """Scan video/image/label data and compute metadata summaries."""
    roots = [project_root / name for name in ["data", "dataset", "datasets", "videos", "raw_videos", "clips", "output", "outputs", "output_labels", "processed", "research_data"]]
    files: List[Path] = []
    for root in roots:
        if root.exists():
            files.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS | LABEL_EXTS])
    cv2, _ = import_optional("cv2")
    video_rows: List[Dict[str, Any]] = []
    video_files = sorted({p for p in files if p.suffix.lower() in VIDEO_EXTS})
    label_files = sorted({p for p in files if p.suffix.lower() in LABEL_EXTS})
    image_files = sorted({p for p in files if p.suffix.lower() in IMAGE_EXTS})
    for path in video_files:
        row: Dict[str, Any] = {
            "path": str(path.relative_to(project_root)),
            "camera_id": parse_camera_id(path),
            "timestamp": parse_timestamp(path),
            "width": NA,
            "height": NA,
            "fps": NA,
            "frame_count": NA,
            "duration_sec": NA,
            "file_size_mb": round(path.stat().st_size / (1024**2), 3),
            "readable": False,
        }
        if cv2 is not None:
            try:
                with suppress_native_stderr():
                    cap = cv2.VideoCapture(str(path))
                    readable = bool(cap.isOpened())
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    cap.release()
                row.update({
                    "width": width or NA,
                    "height": height or NA,
                    "fps": round(fps, 3) if fps else NA,
                    "frame_count": frames or NA,
                    "duration_sec": round(frames / fps, 3) if fps and frames else NA,
                    "readable": readable,
                })
            except Exception as exc:
                row["readable"] = f"READ_FAILED: {exc.__class__.__name__}"
        video_rows.append(row)
    summary = summarize_dataset(video_rows, image_files, label_files)
    summary["annotation_summary"] = summarize_annotations(label_files)
    return video_rows, summary, video_files, label_files


def summarize_dataset(video_rows: Sequence[Dict[str, Any]], image_files: Sequence[Path], label_files: Sequence[Path]) -> Dict[str, Any]:
    """Aggregate video metadata."""
    total_frames = sum(int(r["frame_count"]) for r in video_rows if str(r.get("frame_count", "")).isdigit())
    total_duration = sum(float(r["duration_sec"]) for r in video_rows if is_number(r.get("duration_sec")))
    camera_counts: Counter[str] = Counter(str(r.get("camera_id", "unknown")) for r in video_rows)
    camera_frames: defaultdict[str, int] = defaultdict(int)
    camera_duration: defaultdict[str, float] = defaultdict(float)
    resolution: Counter[str] = Counter()
    fps_dist: Counter[str] = Counter()
    for row in video_rows:
        cam = str(row.get("camera_id", "unknown"))
        if str(row.get("frame_count", "")).isdigit():
            camera_frames[cam] += int(row["frame_count"])
        if is_number(row.get("duration_sec")):
            camera_duration[cam] += float(row["duration_sec"])
        resolution[f"{row.get('width')}x{row.get('height')}"] += 1
        fps_dist[str(row.get("fps"))] += 1
    camera_summary = [
        {
            "camera_id": cam,
            "clips": camera_counts[cam],
            "frames": camera_frames.get(cam, 0),
            "duration_sec": round(camera_duration.get(cam, 0.0), 3),
        }
        for cam in sorted(camera_counts)
    ]
    return {
        "total_videos": len(video_rows),
        "total_images": len(image_files),
        "total_label_files": len(label_files),
        "total_frames": total_frames,
        "total_duration_sec": round(total_duration, 3),
        "total_duration_min": round(total_duration / 60, 3),
        "resolution_distribution": dict(resolution),
        "fps_distribution": dict(fps_dist),
        "camera_count": len(camera_counts),
        "camera_summary": camera_summary,
    }


def is_number(value: Any) -> bool:
    """Return True if value can be parsed as a finite float."""
    try:
        float(value)
        return True
    except Exception:
        return False


def summarize_annotations(label_files: Sequence[Path]) -> Dict[str, Any]:
    """Count readable prediction/label files without assuming ground truth."""
    adl_files = [p for p in label_files if "_adl" in p.name.lower() or "adl" in p.name.lower()]
    track_files = [p for p in label_files if "track" in p.name.lower()]
    global_files = [p for p in label_files if "global" in p.name.lower() or "timeline" in p.name.lower()]
    counts = Counter()
    adl_rows = 0
    for path in adl_files:
        for item in read_adl_records(path):
            label = item.get("label")
            if label:
                counts[label] += 1
                adl_rows += 1
    return {
        "adl_files": len(adl_files),
        "adl_prediction_rows": adl_rows,
        "adl_prediction_distribution": dict(counts),
        "track_files": len(track_files),
        "global_or_timeline_files": len(global_files),
        "note": "file exists but parser not implemented for non-ADL labels" if track_files or global_files else "",
    }


def read_adl_records(path: Path) -> List[Dict[str, Any]]:
    """Read common ADL prediction/ground-truth formats."""
    records: List[Dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[\s,]+", line)
                if len(parts) >= 3:
                    records.append({"frame": parts[0], "track": parts[1], "label": parts[2], "confidence": parts[3] if len(parts) > 3 else NA})
        elif path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    label = row.get("adl") or row.get("adl_label") or row.get("label") or row.get("activity")
                    if label:
                        records.append({"frame": row.get("frame") or row.get("frame_id"), "track": row.get("track") or row.get("track_id"), "label": label, "confidence": row.get("confidence", NA)})
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            items = data if isinstance(data, list) else data.get("items", data.get("predictions", data.get("annotations", []))) if isinstance(data, dict) else []
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict):
                    label = item.get("adl") or item.get("adl_label") or item.get("label") or item.get("activity")
                    if label:
                        records.append({"frame": item.get("frame") or item.get("frame_id"), "track": item.get("track") or item.get("track_id"), "label": label, "confidence": item.get("confidence", NA)})
    except Exception:
        return records
    return records


def run_runtime_benchmark(project_root: Path, config: Dict[str, Any], videos: Sequence[Path], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run a small optional YOLO benchmark when model and video are available."""
    row = {
        "component": "full_pipeline",
        "detector_load_time": NA,
        "detector_inference_ms_per_frame": NA,
        "detector_fps": NA,
        "pose_load_time": NA,
        "pose_inference_ms_per_frame": NA,
        "pose_fps": NA,
        "tracking_ms_per_frame": NA,
        "adl_ms_per_window": NA,
        "reid_ms_per_person": NA,
        "global_id_ms_per_assignment": NA,
        "full_pipeline_ms_per_frame": NA,
        "full_pipeline_fps": NA,
        "status": "N/A",
        "notes": "",
    }
    if args.skip_inference:
        row["status"] = "SKIPPED"
        row["notes"] = "--skip-inference"
        return [row]
    if not videos:
        row["status"] = "N/A"
        row["notes"] = "No videos found"
        return [row]
    ultralytics, status = import_optional("ultralytics")
    cv2, cv2_status = import_optional("cv2")
    if ultralytics is None or cv2 is None:
        row["status"] = "MODULE_IMPORT_FAILED"
        row["notes"] = f"ultralytics={status}; cv2={cv2_status}"
        return [row]
    YOLO = getattr(ultralytics, "YOLO", None)
    model_path = deep_get(config, "phase2.model", deep_get(config, "models.detector_model_path", MISSING))
    if model_path == MISSING:
        row["status"] = "N/A"
        row["notes"] = "No detector model path in config"
        return [row]
    model_file = (project_root / str(model_path)).resolve()
    if not model_file.exists():
        row["status"] = "N/A"
        row["notes"] = f"Detector model missing: {model_path}"
        return [row]
    try:
        start = time.perf_counter()
        model = YOLO(str(model_file))
        row["detector_load_time"] = round(time.perf_counter() - start, 4)
        if args.device != "auto":
            try:
                model.to(args.device)
            except Exception as exc:
                row["notes"] += f" device_set_failed={exc.__class__.__name__};"
        infer_times: List[float] = []
        processed = 0
        for video in list(videos)[: args.max_videos]:
            cap = cv2.VideoCapture(str(video))
            local = 0
            while cap.isOpened() and local < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                start = time.perf_counter()
                model.predict(frame, conf=float(deep_get(config, "phase2.conf_threshold", 0.5)), verbose=False)
                elapsed = time.perf_counter() - start
                if processed >= args.warmup_frames:
                    infer_times.append(elapsed)
                processed += 1
                local += 1
            cap.release()
        if infer_times:
            ms = sum(infer_times) / len(infer_times) * 1000
            row["detector_inference_ms_per_frame"] = round(ms, 3)
            row["detector_fps"] = round(1000 / ms, 3) if ms else NA
            row["full_pipeline_ms_per_frame"] = row["detector_inference_ms_per_frame"]
            row["full_pipeline_fps"] = row["detector_fps"]
            row["status"] = "OK_METADATA_PLUS_DETECTOR"
            row["notes"] += f"frames_measured={len(infer_times)}"
        else:
            row["status"] = "N/A"
            row["notes"] += "No frames measured"
    except Exception as exc:
        row["status"] = "BENCHMARK_FAILED"
        row["notes"] += f"{exc.__class__.__name__}: {exc}"
    return [row]


def evaluate_adl(label_files: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate ADL when GT is available; otherwise report prediction distribution."""
    gt_files = [p for p in label_files if any(token in p.name.lower() for token in ["gt", "ground_truth", "annotation"]) and "adl" in p.name.lower()]
    pred_files = [p for p in label_files if "adl" in p.name.lower() and p not in gt_files]
    pred_records = [record for path in pred_files for record in read_adl_records(path)]
    gt_records = [record for path in gt_files for record in read_adl_records(path)]
    results: List[Dict[str, Any]] = []
    confusion: List[Dict[str, Any]] = []
    meta = {"gt_found": bool(gt_records), "pred_found": bool(pred_records)}
    if not gt_records:
        distribution = Counter(str(r.get("label", "unknown")) for r in pred_records)
        total = sum(distribution.values())
        if distribution:
            for label in ADL_CLASSES:
                count = distribution.get(label, 0)
                results.append({"metric": "prediction_distribution", "class": label, "value": count, "percentage": round(count / total * 100, 3) if total else 0, "status": "NO_GROUND_TRUTH"})
        else:
            results.append({"metric": "accuracy", "class": "all", "value": NA, "percentage": NA, "status": "MISSING_GROUND_TRUTH"})
        results.append({"metric": "macro_f1", "class": "all", "value": NA, "percentage": NA, "status": "MISSING_GROUND_TRUTH"})
        results.append({"metric": "note", "class": "all", "value": "ADL ground truth was not found; only prediction distribution was reported.", "percentage": NA, "status": "MISSING_GROUND_TRUTH"})
        return results, confusion, meta
    results.append({"metric": "accuracy", "class": "all", "value": "TODO", "percentage": NA, "status": "GT_FOUND_MATCHING_NOT_IMPLEMENTED"})
    results.append({"metric": "macro_f1", "class": "all", "value": "TODO", "percentage": NA, "status": "GT_FOUND_MATCHING_NOT_IMPLEMENTED"})
    return results, confusion, meta


def evaluate_global_id(label_files: Sequence[Path]) -> List[Dict[str, Any]]:
    """Create Global ID result rows, with placeholders when GT is missing."""
    gt_files = [p for p in label_files if any(token in p.name.lower() for token in ["global_id", "timeline", "ground_truth", "annotations", "gt", "tracks"])]
    gt_like = [p for p in gt_files if any(token in p.name.lower() for token in ["ground_truth", "annotations", "gt"])]
    status = "GT_FILE_FOUND_EVALUATOR_TEMPLATE" if gt_like else "MISSING_GROUND_TRUTH"
    metrics = [
        "global_id_accuracy", "cross_camera_idf1", "id_switch_across_cameras", "fragmentation_count",
        "fragmentation_rate", "false_merge", "false_merge_rate", "false_split", "false_split_rate",
        "transfer_success_rate", "unknown_rate", "blind_zone_recovery_rate",
        "clothing_change_id_preservation", "room_reentry_accuracy",
    ]
    return [{"metric": metric, "value": NA if not gt_like else "TODO", "status": status, "notes": f"candidate_files={len(gt_files)}"} for metric in metrics]


def make_ablation_template() -> List[Dict[str, Any]]:
    """Create ablation template rows."""
    methods = [
        "CPose full", "w/o temporal gating", "w/o camera topology", "w/o face cue",
        "w/o appearance cue", "w/o pose/gait cue", "w/o ADL continuity", "Local tracker only",
        "Face ReID only", "Appearance only", "Pose/height/time only",
    ]
    fields = ["global_id_acc", "idf1", "id_switch", "fragmentation", "false_merge", "false_split", "clothing_change_success", "blind_zone_success"]
    rows = []
    for method in methods:
        row = {"method": method, **{field: "TODO" for field in fields}, "notes": "template only"}
        rows.append(row)
    return rows


def markdown_table(rows: Sequence[Dict[str, Any]], max_rows: int = 20) -> str:
    """Render a compact Markdown table."""
    if not rows:
        return "N/A\n"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:max_rows]:
        out.append("| " + " | ".join(normalize_cell(row.get(h, "")).replace("\n", " ") for h in headers) + " |")
    if len(rows) > max_rows:
        out.append(f"\nShowing first {max_rows} of {len(rows)} rows.\n")
    return "\n".join(out) + "\n"


def latex_table(caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a copyable one-column LaTeX table."""
    safe_rows = rows or [["N/A" for _ in headers]]
    colspec = "l" * len(headers)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{escape_tex(caption)}}}",
        f"\\label{{tab:{label}}}",
        "\\resizebox{\\columnwidth}{!}{",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & ".join(escape_tex(str(h)) for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in safe_rows:
        lines.append(" & ".join(escape_tex(normalize_cell(v)) for v in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "}", "\\end{table}", ""])
    return "\n".join(lines)


def escape_tex(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    return "".join(replacements.get(ch, ch) for ch in text)


def write_paper_tables(run_dir: Path, summary: Dict[str, Any], model_rows: Sequence[Dict[str, Any]], runtime_rows: Sequence[Dict[str, Any]], adl_rows: Sequence[Dict[str, Any]], global_rows: Sequence[Dict[str, Any]], ablation_rows: Sequence[Dict[str, Any]]) -> None:
    """Write LaTeX paper tables."""
    dataset = summary["dataset"]
    env = summary["environment"]
    config_rows = [r for r in model_rows if r.get("file_type") == "pipeline_parameter"][:25]
    model_files = [r for r in model_rows if r.get("file_type") != "pipeline_parameter"]
    parts = [
        latex_table("Dataset Summary", "dataset_summary", ["Metric", "Value"], [
            ["Total videos", dataset["total_videos"]], ["Total frames", dataset["total_frames"]],
            ["Total duration (min)", dataset["total_duration_min"]], ["Camera count", dataset["camera_count"]],
        ]),
        latex_table("Hardware and Software Setup", "hardware_software", ["Item", "Value"], [
            ["OS", env.get("os", NA)], ["CPU", env.get("cpu", NA)], ["RAM (GB)", env.get("ram_total_gb", NA)],
            ["GPU", env.get("gpu", NA)], ["CUDA", env.get("cuda_version", NA)], ["Python", env.get("python_version", NA).split()[0]],
        ]),
        latex_table("Hyperparameter Configuration", "hyperparameters", ["Module", "Parameter", "Value"], [[r["module"], r["model_name"], r["notes"]] for r in config_rows]),
        latex_table("Detection and Tracking Results", "detection_tracking", ["Metric", "Value", "Status"], [["Detection mAP", NA, "MISSING_GROUND_TRUTH"], ["Tracking IDF1", NA, "MISSING_GROUND_TRUTH"], ["Model files", len(model_files), "metadata only"]]),
        latex_table("ADL Results", "adl_results", ["Metric", "Class", "Value", "Status"], [[r.get("metric"), r.get("class"), r.get("value"), r.get("status")] for r in adl_rows[:20]]),
        latex_table("Cross-camera Global ID Results", "global_id_results", ["Metric", "Value", "Status"], [[r.get("metric"), r.get("value"), r.get("status")] for r in global_rows]),
        latex_table("Ablation Study", "ablation", ["Method", "Global ID Acc", "IDF1", "Notes"], [[r["method"], r["global_id_acc"], r["idf1"], r["notes"]] for r in ablation_rows]),
        latex_table("Runtime Comparison", "runtime", ["Component", "ms/frame", "FPS", "Status"], [[r.get("component"), r.get("full_pipeline_ms_per_frame"), r.get("full_pipeline_fps"), r.get("status")] for r in runtime_rows]),
    ]
    (run_dir / "paper_tables.tex").write_text("\n".join(parts), encoding="utf-8")


def write_markdown(run_dir: Path, summary: Dict[str, Any], video_rows: Sequence[Dict[str, Any]], model_rows: Sequence[Dict[str, Any]], runtime_rows: Sequence[Dict[str, Any]], adl_rows: Sequence[Dict[str, Any]], global_rows: Sequence[Dict[str, Any]], ablation_rows: Sequence[Dict[str, Any]], missing_items: Sequence[str]) -> None:
    """Write research_summary.md."""
    env = summary["environment"]
    dataset = summary["dataset"]
    packages = env.get("packages", {})
    package_lines = "\n".join(f"- {name}: {version}" for name, version in packages.items())
    camera_rows = dataset.get("camera_summary", [])
    pipeline_rows = [r for r in model_rows if r.get("file_type") == "pipeline_parameter"]
    text = f"""# CPose Research Run Summary

## 1. Run Information
- timestamp: {env.get("timestamp")}
- git commit: {env.get("git_commit")}
- git branch: {env.get("git_branch")}
- project root: {env.get("project_root")}

## 2. Environment
- OS: {env.get("os")}
- CPU: {env.get("cpu")}
- RAM: {env.get("ram_total_gb")} GB
- GPU: {env.get("gpu")}
- CUDA: available={env.get("cuda_available")}, version={env.get("cuda_version")}, cuDNN={env.get("cudnn_version")}

{package_lines}

## 3. Dataset Summary
- total videos: {dataset.get("total_videos")}
- total frames: {dataset.get("total_frames")}
- total duration: {dataset.get("total_duration_sec")} sec ({dataset.get("total_duration_min")} min)
- camera count: {dataset.get("camera_count")}
- resolution distribution: {dataset.get("resolution_distribution")}
- fps distribution: {dataset.get("fps_distribution")}

Table: Dataset Summary

| Metric | Value |
| --- | --- |
| Total videos | {dataset.get("total_videos")} |
| Total frames | {dataset.get("total_frames")} |
| Total duration sec | {dataset.get("total_duration_sec")} |
| Total duration min | {dataset.get("total_duration_min")} |
| Label files | {dataset.get("total_label_files")} |

Table: Camera Summary

{markdown_table(camera_rows)}

Table: Video Metadata

{markdown_table(list(video_rows), max_rows=30)}

## 4. Model and Pipeline Configuration

{markdown_table(list(model_rows), max_rows=60)}

Pipeline parameters:

{markdown_table(pipeline_rows, max_rows=80)}

## 5. Runtime Benchmark

{markdown_table(list(runtime_rows))}

## 6. ADL Evaluation

{markdown_table(list(adl_rows), max_rows=40)}

## 7. Global ID Evaluation

{markdown_table(list(global_rows), max_rows=40)}

## 8. Ablation Study

{markdown_table(list(ablation_rows), max_rows=20)}

## 9. Missing Items for Paper
{chr(10).join(f"- {item}" for item in missing_items)}

## 10. Recommended Next Actions
- annotate global ID
- annotate ADL ground truth
- run ablation
- export confusion matrix
- measure runtime on target edge device
"""
    (run_dir / "research_summary.md").write_text(text, encoding="utf-8")


def compute_missing_items(summary: Dict[str, Any], adl_meta: Dict[str, Any], global_rows: Sequence[Dict[str, Any]], model_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Identify paper gaps from generated evidence."""
    missing: List[str] = []
    if not adl_meta.get("gt_found"):
        missing.append("missing ADL ground truth")
    if any(r.get("status") == "MISSING_GROUND_TRUTH" for r in global_rows):
        missing.append("missing Global ID ground truth")
    missing.extend([
        "missing clothing-change event annotations",
        "missing blind-zone event annotations",
    ])
    env = summary.get("environment", {})
    if env.get("gpu") in {NA, "", None} or str(env.get("gpu", "")).startswith("GPU_QUERY_FAILED"):
        missing.append("missing hardware GPU information")
    config_params = [r for r in model_rows if r.get("module") == "Transition windows"]
    if any(r.get("status") == "TODO" for r in config_params):
        missing.append("missing transition window config")
    if not config_params:
        missing.append("missing camera topology config")
    return missing


def value_from_config(config: Dict[str, Any], path: str) -> Any:
    """Return config value as paper-friendly text."""
    value = deep_get(config, path, "TODO")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return value


def concise_hyperparameters(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the short Table 2 hyperparameter list from TOTAL.md."""
    items = [
        ("Detector", "phase1.conf_threshold", "phase1.conf_threshold"),
        ("Detector", "phase2.conf_threshold", "phase2.conf_threshold"),
        ("Pose", "phase3.keypoint_conf_min", "phase3.keypoint_conf_min"),
        ("ADL", "phase3.window_size", "phase3.window_size"),
        ("ADL", "pose_utils.min_visible_keypoints", "pose_utils.min_visible_keypoints"),
        ("Global ID", "strong_threshold", "global_id.strong_threshold"),
        ("Global ID", "weak_threshold", "global_id.weak_threshold"),
        ("Global ID", "confirm_frames", "global_id.confirm_frames"),
        ("ReID", "threshold", "reid.threshold"),
        ("VectorDB", "search_top_k", "vector_db.search_top_k"),
    ]
    return [{"module": module, "parameter": parameter, "value": value_from_config(config, path)} for module, parameter, path in items]


def concise_model_table(config: Dict[str, Any], project_root: Path) -> List[Dict[str, Any]]:
    """Return only the main model/method rows needed by TOTAL.md."""
    rows = [
        ("Person detector", value_from_config(config, "phase2.model"), f"conf={value_from_config(config, 'phase2.conf_threshold')}, class_id={value_from_config(config, 'phase2.person_class_id')}"),
        ("Pose estimator", value_from_config(config, "phase3.model"), f"COCO-17, keypoint_conf_min={value_from_config(config, 'phase3.keypoint_conf_min')}"),
        ("Local tracking", "DeepSORT / IoU tracker", f"max_age={value_from_config(config, 'tracker.max_age')}, max_iou={value_from_config(config, 'tracker.max_iou_distance')}"),
        ("ADL classifier", "rule-based", f"window_size={value_from_config(config, 'phase3.window_size')}, classes=8"),
        ("Face recognition", "InsightFace / ArcFace", f"threshold={value_from_config(config, 'detectors.face_similarity_threshold')}"),
        ("Body ReID", "simple body ReID", f"threshold={value_from_config(config, 'reid.threshold')}"),
        ("Vector DB", "FAISS", f"top_k={value_from_config(config, 'vector_db.search_top_k')}, dim={value_from_config(config, 'persistence.embedding_dim')}"),
        ("Global ID", "CPose spatio-temporal gating", f"strong={value_from_config(config, 'global_id.strong_threshold')}, weak={value_from_config(config, 'global_id.weak_threshold')}"),
    ]
    out = []
    for module, method, notes in rows:
        model_path = project_root / str(method)
        size = round(model_path.stat().st_size / (1024**2), 3) if model_path.exists() else NA
        out.append({"module": module, "model_or_method": method, "model_size_mb": size, "notes": notes})
    return out


def concise_transition_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return transition windows from TOTAL.md."""
    windows = deep_get(config, "global_id.transition_windows", {})
    meanings = {
        "cam01->cam02": "normal transfer",
        "cam02->cam03": "sequential transfer",
        "cam03->cam02": "return path",
        "cam03->cam04": "elevator / blind zone",
        "cam04->cam03": "elevator return",
        "cam04->cam04": "room hold / re-entry",
    }
    rows = []
    for key, meaning in meanings.items():
        value = "TODO"
        if isinstance(windows, dict):
            value = windows.get(key, windows.get(key.replace("cam0", "cam"), "TODO"))
        rows.append({"transition": key, "window_sec": value, "meaning": meaning})
    return rows


def concise_dataset_rows(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return short dataset summary aligned with TOTAL.md."""
    annotation = dataset.get("annotation_summary", {})
    return [
        {"metric": "number_of_cameras", "value": dataset.get("camera_count", "TODO"), "status": "MEASURED"},
        {"metric": "camera_ids", "value": ", ".join(r["camera_id"] for r in dataset.get("camera_summary", [])), "status": "MEASURED"},
        {"metric": "camera_topology", "value": "cam01 -> cam02 -> cam03 -> elevator -> cam04", "status": "FROM_TOTAL_MD"},
        {"metric": "subjects", "value": "TODO", "status": "MISSING_GROUND_TRUTH"},
        {"metric": "true_global_ids", "value": "TODO", "status": "MISSING_GROUND_TRUTH"},
        {"metric": "clips", "value": dataset.get("total_videos", 0), "status": "MEASURED"},
        {"metric": "frames", "value": dataset.get("total_frames", 0), "status": "MEASURED"},
        {"metric": "duration_min", "value": dataset.get("total_duration_min", 0), "status": "MEASURED"},
        {"metric": "resolution_distribution", "value": dataset.get("resolution_distribution", {}), "status": "MEASURED"},
        {"metric": "fps_distribution", "value": dataset.get("fps_distribution", {}), "status": "MEASURED"},
        {"metric": "adl_classes", "value": ", ".join(ADL_CLASSES), "status": "CONFIG"},
        {"metric": "adl_prediction_rows", "value": annotation.get("adl_prediction_rows", 0), "status": "PREDICTION_ONLY"},
        {"metric": "local_tracks", "value": "TODO", "status": "MISSING_TRACK_GT"},
        {"metric": "blind_zone_scenarios", "value": "elevator, room, door blind zone", "status": "FROM_TOTAL_MD_TEMPLATE"},
        {"metric": "hard_cases", "value": "clothing change, no face, back view, low light, occlusion", "status": "FROM_TOTAL_MD_TEMPLATE"},
    ]


def concise_benchmark_rows(runtime_rows: Sequence[Dict[str, Any]], adl_rows: Sequence[Dict[str, Any]], global_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the priority benchmark list from TOTAL.md."""
    runtime = runtime_rows[0] if runtime_rows else {}
    global_map = {r.get("metric"): r for r in global_rows}
    adl_macro = next((r for r in adl_rows if r.get("metric") == "macro_f1"), {})
    priorities = [
        ("Global ID Accuracy", global_map.get("global_id_accuracy", {}).get("value", NA), "MISSING_GROUND_TRUTH"),
        ("Cross-camera IDF1", global_map.get("cross_camera_idf1", {}).get("value", NA), "MISSING_GROUND_TRUTH"),
        ("ID Switch", global_map.get("id_switch_across_cameras", {}).get("value", NA), "MISSING_GROUND_TRUTH"),
        ("Fragmentation Rate", global_map.get("fragmentation_rate", {}).get("value", NA), "MISSING_GROUND_TRUTH"),
        ("Transfer Success Rate", global_map.get("transfer_success_rate", {}).get("value", NA), "MISSING_GROUND_TRUTH"),
        ("Blind-zone Recovery Rate", global_map.get("blind_zone_recovery_rate", {}).get("value", NA), "MISSING_EVENT_GT"),
        ("Clothing-change ID Preservation", global_map.get("clothing_change_id_preservation", {}).get("value", NA), "MISSING_EVENT_GT"),
        ("ADL Macro-F1", adl_macro.get("value", NA), adl_macro.get("status", "MISSING_GROUND_TRUTH")),
        ("Full pipeline FPS", runtime.get("full_pipeline_fps", NA), runtime.get("status", NA)),
        ("Ablation Study", "TODO", "TEMPLATE_ONLY"),
    ]
    return [{"priority": idx + 1, "metric": metric, "value": value, "status": status} for idx, (metric, value, status) in enumerate(priorities)]


def concise_test_scenarios() -> List[Dict[str, Any]]:
    """Return the scenario checklist from TOTAL.md."""
    return [
        {"test_id": "T1", "scenario": "normal transfer cam01->cam02", "main_metric": "Transfer Success Rate, IDF1", "status": "TODO_NEEDS_GT"},
        {"test_id": "T2", "scenario": "cam02->cam03 sequential transfer", "main_metric": "ID Switch, Fragmentation", "status": "TODO_NEEDS_GT"},
        {"test_id": "T3", "scenario": "cam03->elevator->cam04", "main_metric": "Blind-zone Recovery Rate", "status": "TODO_NEEDS_EVENT_GT"},
        {"test_id": "T4", "scenario": "elevator return to cam03", "main_metric": "False Split Rate", "status": "TODO_NEEDS_GT"},
        {"test_id": "T5", "scenario": "cam04 room re-entry", "main_metric": "Room Re-entry Accuracy", "status": "TODO_NEEDS_EVENT_GT"},
        {"test_id": "T6", "scenario": "room clothing change", "main_metric": "Clothing-change ID Preservation", "status": "TODO_NEEDS_EVENT_GT"},
        {"test_id": "T7", "scenario": "no face / back view", "main_metric": "No-face ID Accuracy", "status": "TODO_NEEDS_EVENT_GT"},
        {"test_id": "T8", "scenario": "partial occlusion", "main_metric": "Missing Keypoint Rate, ADL F1", "status": "TODO_NEEDS_KEYPOINT_GT"},
        {"test_id": "T9", "scenario": "multiple people conflict", "main_metric": "False Merge Rate", "status": "TODO_NEEDS_GT"},
        {"test_id": "T10", "scenario": "reverse topology", "main_metric": "Transfer Accuracy", "status": "TODO_NEEDS_GT"},
    ]


def write_concise_markdown(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Write short paper-focused Markdown report."""
    env = summary["environment"]
    text = f"""# CPose Essential Research Summary

## Run
- timestamp: {env.get("timestamp")}
- project root: {env.get("project_root")}
- git: {env.get("git_branch")} / {env.get("git_commit")}
- hardware: CPU={env.get("cpu")}; RAM={env.get("ram_total_gb")} GB; GPU={env.get("gpu")}
- software: Python {env.get("python_version", "").split()[0]}, torch {env.get("packages", {}).get("torch")}, ultralytics {env.get("packages", {}).get("ultralytics")}, OpenCV {env.get("packages", {}).get("opencv-python/cv2")}

## 1. Dataset Summary
{markdown_table(summary["essential_dataset"], max_rows=20)}

## 2. Test Scenarios From TOTAL.md
{markdown_table(summary["test_scenarios"], max_rows=20)}

## 3. Model / Method
{markdown_table(summary["model_table"], max_rows=20)}

## 4. Hyperparameters
{markdown_table(summary["hyperparameters"], max_rows=20)}

## 5. Transition Windows
{markdown_table(summary["transition_windows"], max_rows=20)}

## 6. Priority Benchmarks
{markdown_table(summary["priority_benchmarks"], max_rows=20)}

## 7. ADL Prediction Distribution
Ground truth ADL was not found, so this is prediction distribution only.

{markdown_table(summary["adl_distribution"], max_rows=20)}

## 8. Ablation Template
{markdown_table(summary["ablation"], max_rows=20)}

## 9. Missing Items
{chr(10).join(f"- {item}" for item in summary["missing_items_for_paper"])}

## 10. Next Actions
- Create `global_id_gt.csv`, `adl_gt.csv`, and `events_gt.csv`.
- Run scenarios T1-T10 after ground truth is ready.
- Fill Global ID, ADL, runtime, and ablation tables with measured results only.
"""
    (run_dir / "research_summary.md").write_text(text, encoding="utf-8")


def write_concise_paper_tables(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Write only the paper tables requested by TOTAL.md."""
    parts = [
        latex_table("Dataset Summary", "dataset_summary", ["Metric", "Value", "Status"], [[r["metric"], r["value"], r["status"]] for r in summary["essential_dataset"]]),
        latex_table("Hyperparameter Configuration", "hyperparameters", ["Module", "Parameter", "Value"], [[r["module"], r["parameter"], r["value"]] for r in summary["hyperparameters"]]),
        latex_table("Priority Benchmark Results", "priority_benchmarks", ["Priority", "Metric", "Value", "Status"], [[r["priority"], r["metric"], r["value"], r["status"]] for r in summary["priority_benchmarks"]]),
        latex_table("Cross-camera Scenario Checklist", "test_scenarios", ["Test", "Scenario", "Metric", "Status"], [[r["test_id"], r["scenario"], r["main_metric"], r["status"]] for r in summary["test_scenarios"]]),
        latex_table("Ablation Study Template", "ablation", ["Variant", "Global ID Acc", "IDF1", "IDSW", "Fragmentation"], [[r["method"], r["global_id_acc"], r["idf1"], r["id_switch"], r["fragmentation"]] for r in summary["ablation"]]),
    ]
    (run_dir / "paper_tables.tex").write_text("\n".join(parts), encoding="utf-8")


def run_concise_report(project_root: Path, output_base: Path, run_dir: Path, args: argparse.Namespace) -> int:
    """Generate the concise TOTAL.md-aligned report."""
    logging.info("[CPose Research] Mode: concise, TOTAL.md focused")
    config, config_files = collect_config(project_root)
    env = collect_environment(project_root)
    video_rows, dataset_summary, video_files, label_files = scan_dataset(project_root)
    runtime_rows = run_runtime_benchmark(project_root, config, video_files, args)
    adl_rows, _, adl_meta = evaluate_adl(label_files)
    global_rows = evaluate_global_id(label_files)
    ablation_rows = make_ablation_template()
    model_rows = concise_model_table(config, project_root)
    summary: Dict[str, Any] = {
        "mode": "concise_TOTAL_md",
        "output_dir": str(run_dir),
        "config_files": config_files,
        "environment": env,
        "dataset": dataset_summary,
        "essential_dataset": concise_dataset_rows(dataset_summary),
        "test_scenarios": concise_test_scenarios(),
        "model_table": model_rows,
        "hyperparameters": concise_hyperparameters(config),
        "transition_windows": concise_transition_rows(config),
        "runtime": runtime_rows,
        "adl_distribution": [r for r in adl_rows if r.get("metric") == "prediction_distribution"],
        "global_id": global_rows,
        "priority_benchmarks": concise_benchmark_rows(runtime_rows, adl_rows, global_rows),
        "ablation": ablation_rows,
    }
    summary["missing_items_for_paper"] = compute_missing_items(summary, adl_meta, global_rows, [{"module": "Transition windows", "status": "CONFIG"}])

    write_csv(run_dir / "essential_results.csv", summary["priority_benchmarks"], ["priority", "metric", "value", "status"])
    write_concise_markdown(run_dir, summary)
    write_concise_paper_tables(run_dir, summary)
    (run_dir / "research_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"[OK] Dataset: cameras={dataset_summary.get('camera_count')}, clips={dataset_summary.get('total_videos')}, frames={dataset_summary.get('total_frames')}, duration_min={dataset_summary.get('total_duration_min')}")
    logging.info(f"[OK] ADL predictions: rows={dataset_summary.get('annotation_summary', {}).get('adl_prediction_rows', 0)}; ground_truth={adl_meta.get('gt_found')}")
    logging.info(f"[OK] Runtime: full_fps={runtime_rows[0].get('full_pipeline_fps') if runtime_rows else NA}; status={runtime_rows[0].get('status') if runtime_rows else NA}")
    logging.info("[OK] Files: research_summary.md, research_summary.json, essential_results.csv, paper_tables.tex, console.log")
    logging.info("[CPose Research] Missing:")
    for item in summary["missing_items_for_paper"]:
        logging.info(f"  - {item}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate CPose research report outputs.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--output-dir", default="research_outputs", help="Output base directory")
    parser.add_argument("--max-videos", type=int, default=3, help="Max videos for benchmark")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames per video for benchmark")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Warmup frames excluded from benchmark")
    parser.add_argument("--skip-inference", action="store_true", help="Skip model inference benchmark")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--run-ablation", action="store_true", help="Reserved for future real ablation support")
    parser.add_argument("--full-report", action="store_true", help="Export the older verbose multi-CSV report")
    return parser.parse_args()


def main() -> int:
    """Run the report pipeline."""
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_base = (project_root / args.output_dir).resolve()
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_base / run_id
    setup_logging(run_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    logging.info(f"[CPose Research] Project root: {project_root}")
    logging.info(f"[CPose Research] Python: {sys.executable}")
    logging.info(f"[CPose Research] Output: {run_dir}")

    if not args.full_report:
        rc = run_concise_report(project_root, output_base, run_dir, args)
        try:
            shutil.copyfile(run_dir / "console.log", output_base / "latest_console.log")
        except Exception:
            pass
        logging.info(f"[CPose Research] Completed: {run_dir}")
        return rc

    config, config_files = collect_config(project_root)
    logging.info(f"[OK] Config files scanned: {', '.join(config_files) if config_files else 'none'}")

    env = collect_environment(project_root)
    logging.info(f"[OK] Environment logged: OS={env.get('os')}, CPU={env.get('cpu')}, GPU={env.get('gpu')}")

    model_file_rows = scan_models(project_root, skip_load=args.skip_inference)
    pipeline_rows = extract_pipeline_config(config)
    model_rows = model_file_rows + pipeline_rows
    write_csv(run_dir / "model_config.csv", model_rows, ["module", "model_name", "path", "file_type", "size_mb", "modified_time", "status", "notes"])
    logging.info(f"[OK] Model config exported: {len(model_file_rows)} model rows, {len(pipeline_rows)} pipeline parameter rows")

    video_rows, dataset_summary, video_files, label_files = scan_dataset(project_root)
    write_csv(run_dir / "dataset_summary.csv", video_rows, ["path", "camera_id", "timestamp", "width", "height", "fps", "frame_count", "duration_sec", "file_size_mb", "readable"])
    logging.info(f"[OK] Dataset scanned: videos={dataset_summary['total_videos']}, frames={dataset_summary['total_frames']}, duration_min={dataset_summary['total_duration_min']}")

    runtime_rows = run_runtime_benchmark(project_root, config, video_files, args)
    write_csv(run_dir / "runtime_benchmark.csv", runtime_rows)
    logging.info(f"[OK] Runtime benchmark finished: {runtime_rows[0].get('status')}")

    adl_rows, confusion_rows, adl_meta = evaluate_adl(label_files)
    write_csv(run_dir / "adl_results.csv", adl_rows)
    if confusion_rows:
        write_csv(run_dir / "adl_confusion_matrix.csv", confusion_rows)
    logging.info(f"[OK] ADL evaluation exported: gt_found={adl_meta.get('gt_found')}, pred_found={adl_meta.get('pred_found')}")

    global_rows = evaluate_global_id(label_files)
    write_csv(run_dir / "global_id_results.csv", global_rows)
    logging.info(f"[OK] Global ID evaluation exported: {global_rows[0].get('status') if global_rows else NA}")

    ablation_rows = make_ablation_template()
    write_csv(run_dir / "ablation_template.csv", ablation_rows, ["method", "global_id_acc", "idf1", "id_switch", "fragmentation", "false_merge", "false_split", "clothing_change_success", "blind_zone_success", "notes"])
    if args.run_ablation:
        logging.info("[WARN] --run-ablation requested, but no config-driven ablation runner is exposed; template only.")

    summary = {
        "environment": env,
        "config_files": config_files,
        "dataset": dataset_summary,
        "models": model_file_rows,
        "pipeline_config": pipeline_rows,
        "runtime": runtime_rows,
        "adl": adl_rows,
        "global_id": global_rows,
        "ablation": ablation_rows,
        "output_dir": str(run_dir),
    }
    missing_items = compute_missing_items(summary, adl_meta, global_rows, model_rows)
    summary["missing_items_for_paper"] = missing_items

    (run_dir / "research_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(run_dir, summary, video_rows, model_rows, runtime_rows, adl_rows, global_rows, ablation_rows, missing_items)
    write_paper_tables(run_dir, summary, model_rows, runtime_rows, adl_rows, global_rows, ablation_rows)
    logging.info("[OK] Paper tables generated")

    latest_console = output_base / "latest_console.log"
    try:
        shutil.copyfile(run_dir / "console.log", latest_console)
    except Exception:
        pass

    logging.info("[CPose Research] Missing items for paper:")
    for item in missing_items:
        logging.info(f"  - {item}")
    logging.info(f"[CPose Research] Completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
