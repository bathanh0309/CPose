"""
src/core/web_runtime.py  — CPose realtime AI processor (patched)

FIXES applied in this version
──────────────────────────────
1. EfficientGCNADL import is kept inside _ensure_adl() (lazy).
   Missing or incompatible weights fall back to "unknown" without crashing.
2. PersonGateDetector properly imported from src.detectors.person_gate.
3. TrackCache.active() may return stale (cached) detections for display;
   ADL / ReID / gallery now receive only *fresh* detections from the
   current frame so stale crops don't pollute those pipelines.
4. module state: "No AI module selected" is logged only when modules are
   truly empty; the early-return path is guarded accordingly.
5. Pose model is NOT invoked when only Tracking is active.
6. ReID runs only when the ReID module is active AND the track passes
   min_track_age / min_crop_area guards.
7. Gallery events are deduplicated per track via gallery_interval_frames.
"""

from __future__ import annotations

import base64
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.core.model_registry import ModelRegistry
from src.core.track_cache import TrackCache
from src.detectors.person_gate import PersonGateDetector          # ← was missing
from src.detectors.yolo_ultralytics import YOLODetectUltralytics, YOLOPoseUltralytics
from src.trackers.bytetrack import ByteTrackNumpy
from src.utils.filters import keypoint_quality
from src.utils.one_euro_filter import KeypointSmoother
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel


# ── helpers ───────────────────────────────────────────────────────────────────

def clipped_crop(frame: np.ndarray, bbox: list[float]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def draw_action_label(frame: np.ndarray, bbox: list[float], action_label: str) -> None:
    if not action_label or action_label == "unknown":
        return
    h, w = frame.shape[:2]
    x1, y1, x2, _ = map(int, bbox)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    text = str(action_label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
    y0 = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, y0), (min(w - 1, x1 + tw + 8), y1), (0, 80, 160), -1)
    cv2.putText(frame, text, (x1 + 3, max(th + 2, y1 - 7)), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# ── main class ────────────────────────────────────────────────────────────────

class WebAIProcessor:
    def __init__(
        self,
        camera_id: str,
        modules: set[str],
        cfg: dict[str, Any],
        model_registry: ModelRegistry | None = None,
    ):
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self.modules: set[str] = set(modules)
        self.frame_idx = -1
        self.fps_counter = FPSCounter()
        self.mode = os.getenv("CPOSE_MODE", "realtime").strip().lower() or "realtime"
        self.use_gpu = os.getenv("CPOSE_USE_GPU", "false").strip().lower() == "true"
        self.mode_cfg = self._load_mode_config()

        self.last_detect_ms = 0.0
        self.last_pose_ms = 0.0
        self.last_reid_ms = 0.0
        self.last_adl_ms = 0.0
        self.last_total_ms = 0.0
        self._pose_ms_window: deque[float] = deque(maxlen=100)
        self._reid_ms_window: deque[float] = deque(maxlen=100)
        self._adl_ms_window: deque[float] = deque(maxlen=100)
        self._total_ms_window: deque[float] = deque(maxlen=100)
        self.keypoint_smoother = KeypointSmoother(freq=float(cfg.get("web", {}).get("target_fps", 25)))
        self.registry = model_registry or ModelRegistry(cfg)
        if model_registry is None:
            self.registry.preload({"detect"})

        self._print_device_allocation_header()

        tracking_cfg = cfg.get("tracking", {})
        self.detector = self.registry.get_detector()
        need_pose_at_start = bool({"pose", "adl"} & self.modules)
        self.pose_model: YOLOPoseUltralytics | None = (
            self.registry.get_pose_model() if need_pose_at_start else getattr(self.registry, "pose_model", None)
        )
        self._pose_model_error: str | None = self.registry.error_for("pose")

        self.tracker = ByteTrackNumpy(
            high_thresh=float(tracking_cfg.get("track_thresh", tracking_cfg.get("conf", 0.45))),
            low_thresh=float(tracking_cfg.get("low_thresh", 0.10)),
            match_thresh=float(tracking_cfg.get("match_thresh", 0.30)),
            max_age=int(tracking_cfg.get("track_buffer", tracking_cfg.get("track_ttl_frames", 25))),
        )
        print("[ByteTrack] device=CPU")

        # ── Legacy / gate ──────────────────────────────────────────────────
        self.person_gate: PersonGateDetector | None = None
        self.person_gate_error: str | None = None

        # ── ReID ───────────────────────────────────────────────────────────
        self.reid = self.registry.get_reid_model() if "reid" in self.modules else getattr(self.registry, "reid_model", None)
        self.reid_error: str | None = self.registry.error_for("reid")
        self.reid_gid_cache: dict[int, dict[str, Any]] = {}
        self.reid_last_frame: dict[int, int] = {}
        self._reid_warned = False
        self._reid_gid_log_frame: dict[int, int] = {}
        self.temp_gid_counter = 0
        self.temp_gid_by_track: dict[int, str] = {}

        # ── ADL ────────────────────────────────────────────────────────────
        self.adl = self.registry.get_adl_model() if "adl" in self.modules else None
        self.adl_enabled = self.adl is not None and not getattr(self.adl, "load_error", None)
        self.adl_error: str | None = self.registry.error_for("adl")
        if "adl" not in self.modules and self.adl is None:
            self.adl_error = "ADL model was not preloaded for this session"
        self._adl_warned = False
        self.adl_status_by_track: dict[int, dict[str, Any]] = {}
        self.adl_last_seen_frame: dict[int, int] = {}
        self.lost_track_threshold = int(cfg.get("adl", {}).get("lost_track_threshold", 30))

        # ── State ──────────────────────────────────────────────────────────
        self.person_active = False
        self.no_person_frames = 0
        self.last_wait_log_frame = -999999
        self.last_person_count = 0
        self.last_gate_detections: list[dict[str, Any]] = []
        self.last_gate_best_conf: float = 0.0
        self.last_gate_status = "disabled"

        # TrackCache
        track_ttl = int(cfg.get("tracking", {}).get("track_ttl_frames", 12))
        min_conf_cache = float(cfg.get("tracking", {}).get("min_conf_for_cache", 0.45))
        max_tracks_cache = int(cfg.get("tracking", {}).get("max_cached_tracks", 5))
        self.track_cache = TrackCache(
            ttl_frames=track_ttl,
            min_conf_for_cache=min_conf_cache,
            max_cached_tracks=max_tracks_cache,
        )

        # Gallery events
        self.gallery_events: list[dict[str, Any]] = []
        self.last_gallery_frame_by_track: dict[int, int] = {}
        self.last_gallery_bbox_by_track: dict[int, list[float]] = {}

        if not self.adl_enabled and str(cfg.get("adl", {}).get("fallback", "rules")).lower() == "rules":
            try:
                from src.action.rule_adl import classify_rule_adl

                self._rule_adl_fn = classify_rule_adl
                self._rule_adl_buffers: dict[int, list[dict[str, np.ndarray]]] = {}
                self._adl_min_frames = int(cfg.get("adl", {}).get("min_frames", 30))
            except Exception as exc:
                self.adl_error = f"{self.adl_error or ''}; rule fallback unavailable: {exc}".strip("; ")

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _load_mode_config(self) -> dict[str, Any]:
        path = Path("configs/pipeline_mode.yaml")
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}
        except Exception as exc:
            print(f"[Config] pipeline_mode.yaml unavailable: {exc}")
        return {"adl": {"realtime": "efficientgcn", "research": "posec3d"}}

    def _print_device_allocation_header(self) -> None:
        gpu_state = "enabled" if self.use_gpu else "disabled"
        print("╔══════════════════════════════════════════════╗")
        print("║          CPose Pipeline Configuration        ║")
        print("╠══════════════╦═══════════════╦═══════════════╣")
        print("║ Module       ║ Model         ║ Device        ║")
        print("╠══════════════╬═══════════════╬═══════════════╣")
        print("║ Detect       ║ yolo11n       ║ CPU           ║")
        print("║ Pose         ║ yolo11n-pose  ║ CPU           ║")
        print("║ Tracking     ║ ByteTrack     ║ CPU           ║")
        print("║ ReID         ║ OSNet-x0.25   ║ CPU           ║")
        print("║ ADL          ║ EfficientGCN  ║ CPU           ║")
        print("║ RTSP         ║ CAP_FFMPEG    ║ CPU           ║")
        print("║ WebSocket    ║ binary bytes  ║ CPU           ║")
        print("╚══════════════╩═══════════════╩═══════════════╝")
        print(f"Mode: {self.mode} | GPU: {gpu_state}")

    # ── Public: module switching ───────────────────────────────────────────────

    def set_modules(self, modules: set[str]) -> None:
        self.modules = set(modules)
        if not self.modules:
            self.person_active = False
            self.no_person_frames = 0

    # ── Lazy model loaders ────────────────────────────────────────────────────

    def _ensure_pose_model(self) -> bool:
        if self.pose_model is not None:
            return True
        self._pose_model_error = self.registry.error_for("pose")
        return False

    def _ensure_reid(self) -> bool:
        if self.reid is not None:
            return True
        self.reid_error = self.registry.error_for("reid")
        return False

    def _ensure_adl(self) -> bool:
        """Lazy-load ADL engine.  Moved import inside to avoid top-level crash."""
        if self.adl is not None:
            return True
        if self.adl_error:
            return hasattr(self, "_rule_adl_fn")
        adl_cfg = self.cfg.get("adl", {})
        mode_adl = self.mode_cfg.get("adl", {}).get(self.mode)
        backend = str(mode_adl or adl_cfg.get("model_type", "efficientgcn")).lower()

        if self.mode == "realtime" and backend == "posec3d":
            raise RuntimeError(
                "PoseC3D không được phép chạy realtime. "
                "Dùng CPOSE_MODE=research để chạy offline."
            )

        if backend == "disabled" or backend == "posec3d":
            self.adl_error = f"ADL backend={backend} not supported in web runtime"
            print(f"[ADL] {self.adl_error}")
            return False

        if backend == "efficientgcn":
            try:
                from src.action.efficientgcn_adl import EfficientGCNADL

                self.adl = EfficientGCNADL(
                    weight_path=adl_cfg.get("weights", "models/2015_EfficientGCN-B0_ntu-xsub120.pth.tar"),
                    window=int(adl_cfg.get("min_frames", 30)),
                    stride=int(adl_cfg.get("infer_every_n_frames", 15)),
                    device="cpu",
                )
                self.adl_enabled = True
                if getattr(self.adl, "load_error", None):
                    print(f"[ADL] EfficientGCN fallback=unknown: {self.adl.load_error}")
                else:
                    print("[ADL] EfficientGCN loaded, device=CPU")
                return True
            except Exception as exc:
                self.adl_error = f"EfficientGCN: {type(exc).__name__}: {exc}"
                print(f"[ADL] EfficientGCN unavailable: {self.adl_error}")
                return False

        self.adl_error = f"ADL backend={backend} not supported"
        return False

    # ── Temp GID helpers ──────────────────────────────────────────────────────

    def _next_temp_gid(self, track_id: int) -> str:
        if track_id in self.temp_gid_by_track:
            return self.temp_gid_by_track[track_id]
        self.temp_gid_counter += 1
        gid = f"gid_{self.temp_gid_counter:05d}"
        self.temp_gid_by_track[track_id] = gid
        return gid

    # ── Overlay helpers ───────────────────────────────────────────────────────

    def _track_adl_status(self, track_id: int) -> dict[str, Any] | None:
        status = self.adl_status_by_track.get(int(track_id))
        if not status:
            return None
        if not status.get("label") or status.get("label") == "unknown":
            return None
        return status

    def _track_overlay_label(self, det: dict[str, Any]) -> str:
        tid = int(det.get("track_id", -1))
        cached = bool(det.get("cached", False))
        conf = float(det.get("score", 0.0))
        parts = [f"track={tid}"]
        if cached:
            parts.append("cached")
        parts.append(f"conf={conf:.2f}")
        adl_status = self._track_adl_status(tid)
        if adl_status:
            label = adl_status.get("label", "")
            score = float(adl_status.get("score", 0.0))
            parts.append(f"ADL={label} {score:.2f}")
        return " ".join(parts)

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def _build_track_metrics(self, display_dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tracks = []
        for det in display_dets:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            adl_status = self._track_adl_status(tid)
            tracks.append({
                "track_id": tid,
                "conf": round(float(det.get("score", 0.0)), 3),
                "bbox": [round(float(v), 2) for v in det.get("bbox", [0, 0, 0, 0])],
                "cached": bool(det.get("cached", False)),
                "age": int(det.get("age", 0)),
                "gid": self.reid_gid_cache.get(tid, {}).get("gid", "unknown"),
                "reid_score": round(float(self.reid_gid_cache.get(tid, {}).get("score", 0.0)), 3),
                "adl_label": adl_status.get("label") if adl_status else "collecting",
                "adl_score": round(float(adl_status.get("score", 0.0)), 3) if adl_status else 0.0,
            })
        return tracks

    @staticmethod
    def _avg(values: deque[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _smooth_pose_keypoints(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply OneEuro smoothing to fresh pose detections."""
        active_ids: set[int] = set()
        for det in detections:
            tid = int(det.get("track_id", -1))
            keypoints = det.get("keypoints")
            if tid < 0 or keypoints is None:
                continue
            active_ids.add(tid)
            try:
                kps = np.asarray(keypoints, dtype=np.float32)
                if kps.shape == (17, 3):
                    smoothed = self.keypoint_smoother.smooth(tid, kps)
                    det["keypoints"] = smoothed
                    det["keypoint_scores"] = smoothed[:, 2].tolist()
            except Exception as exc:
                print(f"[POSE] smoother warning track={tid}: {exc}")
        self.keypoint_smoother.cleanup(active_ids)
        return detections

    # ── Main process ──────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray):
        self.frame_idx += 1
        fps = round(float(self.fps_counter.tick()), 2)
        logs: list[tuple[str, str]] = []
        h, w = frame.shape[:2]

        base_metrics: dict[str, Any] = {
            "camera_id": self.camera_id,
            "frame_idx": self.frame_idx,
            "fps": fps,
            "modules": sorted(self.modules),
            "detections": 0,
            "tracked": 0,
            "live_tracked": 0,
            "display_tracked": 0,
            "filtered": 0,
            "persons": 0,
            "valid_skeletons": 0,
            "avg_kpt": 0.0,
            "tracks": [],
            "best_track_conf": 0.0,
            "mean_track_conf": 0.0,
            "num_tracks": 0,
            "detect_ms": 0.0,
            "pose_ms": 0.0,
            "reid_ms": 0.0,
            "adl_ms": 0.0,
            "total_ms": 0.0,
        }

        # ── No module selected: raw pass-through ─────────────────────────
        if not self.modules:
            base_metrics["ai_active"] = False
            return frame, [], base_metrics

        need_pose = bool({"pose", "adl"} & self.modules)

        t_total = time.perf_counter()
        detect_ms = pose_ms = reid_ms = adl_ms = 0.0

        # ── Detection + Tracking ──────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            if self.detector is None:
                raise RuntimeError(self.registry.error_for("detect") or "detector unavailable")
            with self.registry.locks["detect"]:
                bboxes = self.detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            fresh_tracked: list[dict[str, Any]] = self.tracker.update(bboxes, frame, frame_stride=1, timestamp=time.time())
        except Exception as exc:
            logs.append(("warning", f"Detection failed: {type(exc).__name__}: {exc}"))
            return frame, logs, base_metrics

        # Update gate state
        if bboxes:
            self.person_active = True
            self.no_person_frames = 0
            self.last_person_count = len(bboxes)
            self.last_gate_best_conf = max(float(b[4]) for b in bboxes)
            self.last_gate_status = "person_detected"
        else:
            self.no_person_frames += 1
            if self.no_person_frames > int(self.cfg.get("web", {}).get("lost_person_grace_frames", 30)):
                self.person_active = False
            self.last_gate_status = "waiting"

        # ── Pose estimation (only when needed) ───────────────────────────
        pose_interval = max(1, int(self.cfg.get("pose", {}).get("infer_every_n_frames", 4)))
        run_pose = need_pose and fresh_tracked and (self.frame_idx % pose_interval == 0)
        if run_pose:
            if self._ensure_pose_model():
                try:
                    t1 = time.perf_counter()
                    with self.registry.locks["pose"]:
                        fresh_detections: list[dict[str, Any]] = self.pose_model.estimate(frame, fresh_tracked)
                    fresh_detections = self._smooth_pose_keypoints(fresh_detections)
                    pose_ms = (time.perf_counter() - t1) * 1000.0
                except Exception as exc:
                    logs.append(("warning", f"Pose failed: {type(exc).__name__}: {exc}"))
                    fresh_detections = fresh_tracked
            else:
                fresh_detections = fresh_tracked
                logs.append(("warning", f"Pose model unavailable: {self._pose_model_error}"))
        else:
            fresh_detections = fresh_tracked

        # ── TrackCache: update with FRESH detections only ─────────────────
        self.track_cache.update(fresh_detections, self.frame_idx)
        # display_dets may include stale cached entries – used for drawing only
        cached_dets = [d for d in self.track_cache.active(self.frame_idx) if bool(d.get("cached", False))]
        # fresh_only: tracks actually seen THIS frame – used for AI inference
        fresh_ids = {int(d.get("track_id", -1)) for d in fresh_detections if d.get("track_id", -1) >= 0}
        fresh_only = [
            {**d, "cached": False, "age": 0}
            for d in fresh_detections
            if int(d.get("track_id", -1)) in fresh_ids
        ]
        fresh_boxes = [d.get("bbox", [0, 0, 0, 0]) for d in fresh_only]
        cached_dets = [
            d for d in cached_dets
            if not fresh_boxes
            or max(bbox_iou(d.get("bbox", [0, 0, 0, 0]), fb) for fb in fresh_boxes) <= 0.5
        ]
        display_dets = fresh_only + cached_dets

        live_tracked = sum(1 for d in fresh_only if int(d.get("track_id", -1)) >= 0)
        display_tracked = sum(1 for d in display_dets if int(d.get("track_id", -1)) >= 0)
        skeleton_scores = [
            keypoint_quality(d.get("keypoint_scores"), 0.0)[1]
            for d in fresh_only
            if d.get("keypoints") is not None
        ]
        valid_skeletons = len(skeleton_scores)
        avg_kpt = float(sum(skeleton_scores) / len(skeleton_scores)) if skeleton_scores else 0.0

        base_metrics.update({
            "detections": len(bboxes),
            "tracked": display_tracked,
            "live_tracked": live_tracked,
            "display_tracked": display_tracked,
            "persons": len(display_dets),
            "valid_skeletons": valid_skeletons,
            "avg_kpt": round(avg_kpt, 2),
            "selected_modules": sorted(self.modules),
            "ai_active": True,
        })

        # ── Module-specific pipelines (fresh detections only!) ────────────

        adl_interval = max(1, int(self.cfg.get("adl", {}).get("infer_every_n_frames", 15)))
        if "adl" in self.modules and self.frame_idx % adl_interval == 0:
            t_adl = time.perf_counter()
            self._process_adl(frame, fresh_only, logs, (h, w))  # ← fresh_only, NOT display_dets
            adl_ms = (time.perf_counter() - t_adl) * 1000.0

        if "reid" in self.modules:
            t_reid = time.perf_counter()
            self._process_reid(frame, fresh_only, logs)  # ← fresh_only, NOT display_dets
            reid_ms = (time.perf_counter() - t_reid) * 1000.0

        # ── Draw all display tracks (including cached) ────────────────────
        include_pose_vis = bool({"pose", "adl"} & self.modules)
        for det in display_dets:
            draw_det = det if include_pose_vis else {**det, "keypoints": None, "keypoint_scores": None}
            draw_detection(frame, draw_det, label=self._track_overlay_label(det))

        # ── Per-module log lines ──────────────────────────────────────────
        tracks_metric = self._build_track_metrics(display_dets)
        track_confs = [float(t["conf"]) for t in tracks_metric]
        mean_conf = float(sum(track_confs) / len(track_confs)) if track_confs else 0.0
        best_conf = max(track_confs, default=0.0)

        if "track" in self.modules and self.frame_idx % 15 == 0:
            logs.append(("ai", f"TRACK: frame={self.frame_idx} det={len(bboxes)} live={live_tracked} display={display_tracked} fps={fps}"))

        if "pose" in self.modules and self.frame_idx % 15 == 0:
            logs.append(("ai", f"POSE: frame={self.frame_idx} persons={len(fresh_only)} valid={valid_skeletons} avg_kpt={avg_kpt:.2f}"))

        # ── Gallery events (fresh_only, with interval dedup) ─────────────
        self._generate_gallery_events(frame, fresh_only)  # ← fresh_only

        # ── Info panel ───────────────────────────────────────────────────
        draw_info_panel(frame, {
            "Camera": self.camera_id,
            "Frame": self.frame_idx,
            "Modules": ",".join(sorted(self.modules)) or "none",
            "Detections": len(bboxes),
            "Tracked": display_tracked,
            "FPS": f"{fps:.1f}",
        })

        total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_pose_ms = pose_ms
        self.last_reid_ms = reid_ms
        self.last_adl_ms = adl_ms
        self.last_total_ms = total_ms
        self._pose_ms_window.append(pose_ms)
        self._reid_ms_window.append(reid_ms)
        self._adl_ms_window.append(adl_ms)
        self._total_ms_window.append(total_ms)
        id_max = max(int(getattr(self.tracker, "next_id", 1)) - 1, 0)

        base_metrics.update({
            "tracks": tracks_metric,
            "best_track_conf": round(best_conf, 3),
            "mean_track_conf": round(mean_conf, 3),
            "num_tracks": len(tracks_metric),
            "detect_ms": round(detect_ms, 2),
            "pose_ms": round(pose_ms, 2),
            "reid_ms": round(reid_ms, 2),
            "adl_ms": round(adl_ms, 2),
            "total_ms": round(total_ms, 2),
            "timing": {
                "detect_ms": round(detect_ms, 2),
                "pose_ms": round(pose_ms, 2),
                "reid_ms": round(reid_ms, 2),
                "adl_ms": round(adl_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "id_max": id_max,
        })

        if self.frame_idx % 100 == 0:
            avg_pose = self._avg(self._pose_ms_window)
            avg_reid = self._avg(self._reid_ms_window)
            avg_adl = self._avg(self._adl_ms_window)
            avg_total = self._avg(self._total_ms_window)
            print(
                f"[METRIC] fps={fps:.1f} | det={len(bboxes)} | live={live_tracked} display={display_tracked} | "
                f"id_max={id_max} | pose={avg_pose:.1f}ms | reid={avg_reid:.1f}ms | "
                f"adl={avg_adl:.1f}ms | total={avg_total:.1f}ms"
            )

        return frame, logs, base_metrics

    # ── ReID ──────────────────────────────────────────────────────────────────

    def _process_reid(self, frame: np.ndarray, detections: list[dict[str, Any]], logs: list) -> None:
        """Run ReID on fresh (non-cached) detections only."""
        reid_cfg = self.cfg.get("reid", {})
        min_track_age = int(reid_cfg.get("min_track_age", 10))
        min_crop_area = float(reid_cfg.get("min_crop_area", 2500))
        min_track_conf = float(reid_cfg.get("min_track_conf", 0.60))
        min_hits = int(reid_cfg.get("min_hits", min_track_age))
        log_interval = int(reid_cfg.get("reid_log_interval", 45))

        if not self._ensure_reid():
            if not self._reid_warned:
                logs.append(("warning", f"ReID unavailable: {self.reid_error}"))
                self._reid_warned = True
            return

        gallery_disabled_reason = getattr(self.reid, "gallery_disabled_reason", None)
        gallery_empty = not bool(getattr(self.reid, "gallery", {})) or bool(gallery_disabled_reason)
        if gallery_empty and not self._reid_warned:
            logs.append(("warning", gallery_disabled_reason or "ReID gallery empty; matching disabled."))
            self._reid_warned = True

        for det in detections:
            if det.get("cached", False):
                continue  # never ReID stale cached detections

            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue

            track_age = int(det.get("track_age", det.get("age", 0)))
            hit_count = int(det.get("hit_count", 0))
            conf = float(det.get("score", 0.0))
            if track_age < min_track_age or hit_count < min_hits or conf < min_track_conf:
                continue

            # Min crop area guard
            x1, y1, x2, y2 = map(float, det["bbox"])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_crop_area:
                continue

            # Interval guard: skip if re-queried too recently
            last_reid = self.reid_last_frame.get(f"reid_{tid}", -999999)
            interval = int(getattr(self.reid, "interval", reid_cfg.get("reid_interval", 15)))
            if isinstance(last_reid, int) and self.frame_idx - last_reid < interval:
                continue  # use cached result silently

            crop = clipped_crop(frame, det["bbox"])
            if crop is None:
                continue

            if gallery_empty:
                gid, score = self._next_temp_gid(tid), 0.0
                self.reid_gid_cache[tid] = {"gid": "unknown", "score": score}
                self.reid_last_frame[f"reid_{tid}"] = self.frame_idx
            else:
                try:
                    prev_gid = self.reid_gid_cache.get(tid, {}).get("gid")
                    gid, score = self.reid.identify(crop, area)
                    self.reid_gid_cache[tid] = {"gid": gid, "score": score}
                    self.reid_last_frame[f"reid_{tid}"] = self.frame_idx

                    # Log only on gid change or every 30 frames
                    last_log = self._reid_gid_log_frame.get(tid, -999999)
                    if prev_gid != gid or self.frame_idx - last_log >= log_interval:
                        logs.append(("ai", f"ReID: track={tid} gid={gid} score={score:.2f}"))
                        self._reid_gid_log_frame[tid] = self.frame_idx

                except Exception as exc:
                    gid = self._next_temp_gid(tid)
                    score = 0.0
                    logs.append(("warning", f"ReID error track={tid}: {exc}"))

            self.reid_last_frame[tid] = self.frame_idx

    # ── ADL ───────────────────────────────────────────────────────────────────

    def _process_adl(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        logs: list,
        img_shape: tuple,
    ) -> None:
        """Run ADL on fresh detections only. Supports EfficientGCN or rule-based fallback."""
        if not self._ensure_adl():
            if not self._adl_warned:
                logs.append(("warning", f"ADL unavailable: {self.adl_error or 'backend not loaded'}"))
                self._adl_warned = True
            return
        if self.adl is None and hasattr(self, "_rule_adl_fn") and not self._adl_warned:
            logs.append(("warning", f"ADL unavailable: {self.adl_error or 'EfficientGCN disabled'}; using rule fallback."))
            self._adl_warned = True

        active_ids: set[int] = set()
        first_status = None
        adl_cfg = self.cfg.get("adl", {})

        for det in detections:
            if det.get("cached", False):
                continue  # never feed stale cached frames to ADL buffer

            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            active_ids.add(tid)
            self.adl_last_seen_frame[tid] = self.frame_idx

            keypoints = det.get("keypoints")
            action_label = "unknown"

            # ── EfficientGCN path ──────────────────────────────────────────
            if self.adl is not None and hasattr(self.adl, "update"):
                if keypoints is not None:
                    kps = np.asarray(keypoints, dtype=np.float32)
                    scores = det.get("keypoint_scores")
                    if kps.shape == (17, 2):
                        score_arr = (
                            np.ones((17, 1), dtype=np.float32)
                            if scores is None
                            else np.asarray(scores, dtype=np.float32).reshape(17, 1)
                        )
                        kps = np.concatenate([kps, score_arr], axis=1)
                    if kps.shape == (17, 3):
                        action_label, action_score = self.adl.update(tid, kps, self.frame_idx)
                    else:
                        action_score = 0.0
                else:
                    action_score = 0.0

                current_len = len(getattr(self.adl, "buffers", {}).get(tid, []))
                min_frames = int(adl_cfg.get("min_frames", 30))
                status: dict[str, Any] = {
                    "status": "inferred" if action_label != "unknown" else "collecting",
                    "label": action_label,
                    "score": float(action_score),
                    "current_len": current_len,
                    "seq_len": min_frames,
                }

            # ── Rule-based path ────────────────────────────────────────────
            elif hasattr(self, "_rule_adl_fn"):
                if keypoints is not None:
                    kps = np.asarray(keypoints, dtype=np.float32)
                    sc = det.get("keypoint_scores")
                    sc_arr = np.ones(17, dtype=np.float32) if sc is None else np.asarray(sc, dtype=np.float32)
                    buf = self._rule_adl_buffers.setdefault(tid, [])
                    buf.append({"kps": kps, "scores": sc_arr})
                    if len(buf) > self._adl_min_frames * 2:
                        buf = buf[-self._adl_min_frames * 2:]
                        self._rule_adl_buffers[tid] = buf
                    current_len = len(buf)
                    min_frames = self._adl_min_frames
                    if current_len >= min_frames:
                        window_kps = np.stack([e["kps"] for e in buf[-min_frames:]], axis=0)
                        window_sc = np.stack([e["scores"] for e in buf[-min_frames:]], axis=0)
                        rule_result = self._rule_adl_fn({
                            "keypoints": window_kps,
                            "scores": window_sc,
                            "frame_idx": list(range(current_len)),
                            "img_shape": img_shape,
                        })
                        status = {
                            "status": rule_result.get("status", "inferred"),
                            "label": rule_result.get("label", "unknown"),
                            "score": float(rule_result.get("score", 0.0)),
                            "current_len": current_len,
                            "seq_len": min_frames,
                        }
                        action_label = status["label"]
                    else:
                        status = {
                            "status": "collecting",
                            "label": "collecting",
                            "score": 0.0,
                            "current_len": current_len,
                            "seq_len": min_frames,
                        }
                else:
                    status = {"status": "collecting", "label": "collecting", "score": 0.0,
                              "current_len": 0, "seq_len": self._adl_min_frames}
            else:
                status = {"status": "collecting", "label": "collecting", "score": 0.0,
                          "current_len": 0, "seq_len": 30}

            self.adl_status_by_track[tid] = status
            first_status = first_status or status

            if action_label != "unknown":
                draw_action_label(frame, det.get("bbox", [0, 0, 0, 0]), action_label)
                if self.frame_idx % 30 == 0:
                    logs.append(("ai", f"ADL: track={tid} action={action_label} score={status.get('score', 0):.2f}"))

        # GC lost tracks
        for tid, last_seen in list(self.adl_last_seen_frame.items()):
            if tid not in active_ids and self.frame_idx - int(last_seen) > self.lost_track_threshold:
                if self.adl and hasattr(self.adl, "cleanup_track"):
                    self.adl.cleanup_track(tid)
                if hasattr(self, "_rule_adl_buffers"):
                    self._rule_adl_buffers.pop(tid, None)
                self.adl_last_seen_frame.pop(tid, None)
                self.adl_status_by_track.pop(tid, None)

        if first_status:
            draw_adl_status(frame, first_status, pos=(10, 150))

    # ── Gallery ───────────────────────────────────────────────────────────────

    def _make_crop_event(self, frame: np.ndarray, det: dict[str, Any]) -> dict[str, Any] | None:
        x1, y1, x2, y2 = map(int, det["bbox"])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        tid = int(det.get("track_id", -1))
        adl_status = self._track_adl_status(tid)
        reid_info = self.reid_gid_cache.get(tid, {})
        return {
            "type": "gallery",
            "track_id": tid,
            "global_id": reid_info.get("gid", "unknown"),
            "reid_score": round(float(reid_info.get("score", 0.0)), 3),
            "conf": round(float(det.get("score", 0.0)), 2),
            "track_conf": round(float(det.get("score", 0.0)), 2),
            "adl_label": adl_status.get("label") if adl_status else None,
            "adl_score": round(float(adl_status.get("score", 0.0)), 3) if adl_status else 0.0,
            "frame_idx": self.frame_idx,
            "crop_jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
            "ts": time.strftime("%H:%M:%S"),
        }

    def _generate_gallery_events(
        self, frame: np.ndarray, fresh_dets: list[dict[str, Any]]
    ) -> None:
        """Generate gallery crop events; only from fresh (non-cached) detections."""
        self.gallery_events = []
        web_cfg = self.cfg.get("web", {})
        if not web_cfg.get("gallery_enabled", True):
            return
        gallery_interval = int(web_cfg.get("gallery_interval_frames", 30))
        max_items = int(web_cfg.get("max_gallery_items", 20))

        for det in fresh_dets:
            if det.get("cached", False):
                continue  # redundant guard

            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue

            last = self.last_gallery_frame_by_track.get(tid, -(gallery_interval + 1))
            if self.frame_idx - last < gallery_interval:
                continue
            last_bbox = self.last_gallery_bbox_by_track.get(tid)
            if (
                last_bbox is not None
                and self.frame_idx - last < gallery_interval * 3
                and bbox_iou(det.get("bbox", [0, 0, 0, 0]), last_bbox) > 0.9
            ):
                continue

            if len(self.gallery_events) >= max_items:
                break

            event = self._make_crop_event(frame, det)
            if event:
                self.gallery_events.append(event)
                self.last_gallery_frame_by_track[tid] = self.frame_idx
                self.last_gallery_bbox_by_track[tid] = list(det.get("bbox", [0, 0, 0, 0]))
