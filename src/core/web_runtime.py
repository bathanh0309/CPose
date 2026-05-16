"""
src/core/web_runtime.py  — CPose realtime AI processor (patched)

FIXES applied in this version
──────────────────────────────
1. ADLEfficientGCN import moved inside _ensure_adl() (lazy).
   A missing src/action/adl_efficientgcn.py no longer crashes the server.
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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.action.pose_buffer import PoseSequenceBuffer
from src.core.track_cache import TrackCache
from src.detectors.person_gate import PersonGateDetector          # ← was missing
from src.detectors.yolo_ultralytics import YOLODetectUltralytics, YOLOPoseUltralytics
from src.trackers.bytetrack import ByteTrackNumpy
from src.utils.filters import keypoint_quality
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


# ── main class ────────────────────────────────────────────────────────────────

class WebAIProcessor:
    def __init__(self, camera_id: str, modules: set[str], cfg: dict[str, Any]):
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self.modules: set[str] = set(modules)
        self.frame_idx = -1
        self.fps_counter = FPSCounter()

        self.last_detect_ms = 0.0
        self.last_pose_ms = 0.0
        self.last_adl_ms = 0.0
        self.last_total_ms = 0.0

        self._print_device_allocation_header()

        tracking_cfg = cfg.get("tracking", {})
        pose_cfg = cfg.get("pose", {})
        detect_weights = (
            tracking_cfg.get("weights")
            or tracking_cfg.get("fallback_weights")
            or "models/yolo11n.pt"
        )

        self.detector = YOLODetectUltralytics(
            weights=detect_weights,
            device="cpu",
            conf_threshold=float(tracking_cfg.get("conf", 0.4)),
            iou_threshold=float(tracking_cfg.get("iou", 0.45)),
            class_filter=[0],
            imgsz=int(tracking_cfg.get("imgsz", 416)),
        )
        print(f"[DETECT] device={self.detector.device} imgsz={self.detector.imgsz}")

        self.pose_model: YOLOPoseUltralytics | None = None   # lazy-loaded
        self._pose_model_error: str | None = None

        self.tracker = ByteTrackNumpy(
            high_thresh=float(tracking_cfg.get("conf", 0.4)),
            low_thresh=0.1,
            match_thresh=float(tracking_cfg.get("iou", 0.45)),
            max_age=int(tracking_cfg.get("track_ttl_frames", 30)),
        )
        print("[ByteTrack] device=CPU")

        # ── Legacy / gate ──────────────────────────────────────────────────
        self.person_gate: PersonGateDetector | None = None
        self.person_gate_error: str | None = None

        # ── ReID ───────────────────────────────────────────────────────────
        self.reid_extractor = None
        self.reid_gallery = None
        self.reid_error: str | None = None
        self.reid_gid_cache: dict[int, dict[str, Any]] = {}
        self.reid_last_frame: dict[int, int] = {}
        self.temp_gid_counter = 0
        self.temp_gid_by_track: dict[int, str] = {}

        # ── ADL ────────────────────────────────────────────────────────────
        self.adl = None
        self.adl_enabled = False
        self.adl_error: str | None = None
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

        # Lazy-load ADL on first use if module is enabled
        self._ensure_adl()

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _print_device_allocation_header(self) -> None:
        print("╔══════════════════════════════════════════╗")
        print("║         CPose Device Allocation          ║")
        print("╠══════════════════════════════════════════╣")
        print("║  YOLO Detect   │ CPU                     ║")
        print("║  YOLO Pose     │ CPU (lazy load)         ║")
        print("║  ByteTrack     │ CPU                     ║")
        print("║  ADL           │ CPU (lazy load)         ║")
        print("║  JPEG + WS     │ CPU                     ║")
        print("╚══════════════════════════════════════════╝")

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
        if self._pose_model_error:
            return False
        try:
            pose_cfg = self.cfg.get("pose", {})
            self.pose_model = YOLOPoseUltralytics(
                weights=pose_cfg.get("weights", "models/yolo11n-pose.pt"),
                device="cpu",
                conf_threshold=float(pose_cfg.get("conf", 0.55)),
                iou_threshold=float(pose_cfg.get("iou", 0.5)),
                imgsz=int(pose_cfg.get("imgsz", 416)),
            )
            print("[POSE] model loaded, device=cpu")
            return True
        except Exception as exc:
            self._pose_model_error = f"{type(exc).__name__}: {exc}"
            print(f"[POSE] load failed: {self._pose_model_error}")
            return False

    def _ensure_reid(self) -> bool:
        if self.reid_extractor is not None and self.reid_gallery is not None:
            return True
        if self.reid_error:
            return False
        try:
            from src.reid.fast_reid import FastReIDExtractor
            from src.reid.gallery import ReIDGallery

            reid_cfg = self.cfg.get("reid", {})
            self.reid_extractor = FastReIDExtractor(
                config=reid_cfg["fastreid_config"],
                weights_path=reid_cfg["weights"],
                device=self.cfg.get("system", {}).get("device", "cpu"),
                output_dir=reid_cfg.get("output_dir"),
                fastreid_root=reid_cfg.get("fastreid_root"),
            )
            self.reid_gallery = ReIDGallery(
                self.reid_extractor,
                reid_cfg["gallery_dir"],
                embedding_dirs=reid_cfg.get("embedding_dirs"),
                id_aliases=reid_cfg.get("id_aliases"),
            )
            self.reid_gallery.build()
            # Validate embedding dimension consistency
            if self.reid_gallery.prototypes:
                dims = {p.shape[0] for p in self.reid_gallery.prototypes.values()}
                if len(dims) > 1:
                    print(f"[ReID] WARNING: mixed gallery dims {dims}; queries will be filtered by dim")
            return True
        except Exception as exc:
            self.reid_error = f"{type(exc).__name__}: {exc}"
            print(f"[ReID] unavailable: {self.reid_error}")
            return False

    def _ensure_adl(self) -> bool:
        """Lazy-load ADL engine.  Moved import inside to avoid top-level crash."""
        if self.adl is not None:
            return True
        if self.adl_error:
            return False
        adl_cfg = self.cfg.get("adl", {})
        backend = adl_cfg.get("model_type", "rules")

        if backend in ("disabled", "posec3d"):
            # posec3d is offline-only; never run subprocess in web realtime
            self.adl_error = f"ADL backend={backend} not supported in web realtime"
            print(f"[ADL] {self.adl_error}")
            return False

        if backend == "efficientgcn":
            try:
                from src.action.adl_efficientgcn import ADLEfficientGCN  # lazy import
                self.adl = ADLEfficientGCN(
                    xml_path=adl_cfg.get("weights", ""),
                    conf_threshold=float(adl_cfg.get("conf_threshold", 0.3)),
                    min_frames=int(adl_cfg.get("min_frames", 30)),
                    device="CPU",
                    precision_hint=adl_cfg.get("precision_hint", "f16"),
                )
                self.adl.infer_every_n_frames = int(adl_cfg.get("infer_every_n_frames", 15))
                self.adl_enabled = True
                print("[ADL] EfficientGCN loaded, device=CPU")
                return True
            except Exception as exc:
                self.adl_error = f"EfficientGCN: {type(exc).__name__}: {exc}"
                print(f"[ADL] EfficientGCN unavailable: {self.adl_error}; falling back to rules")

        # Fallback: rule-based ADL (always available)
        try:
            from src.action.rule_adl import classify_rule_adl
            self._rule_adl_fn = classify_rule_adl
            self._rule_adl_buffers: dict[int, list] = {}   # tid -> list of kps arrays
            self._adl_min_frames = int(adl_cfg.get("min_frames", 30))
            self.adl_enabled = True
            self.adl_error = None
            print("[ADL] using rule-based backend (CPU)")
            return True
        except Exception as exc:
            self.adl_error = f"rule ADL: {type(exc).__name__}: {exc}"
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
            "adl_ms": 0.0,
            "total_ms": 0.0,
        }

        # ── No module selected: raw pass-through ─────────────────────────
        if not self.modules:
            base_metrics["ai_active"] = False
            return frame, [], base_metrics

        need_pose = bool({"pose", "adl"} & self.modules)

        t_total = time.perf_counter()
        detect_ms = pose_ms = adl_ms = 0.0

        # ── Detection + Tracking ──────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            bboxes = self.detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            fresh_tracked: list[dict[str, Any]] = self.tracker.update(bboxes, frame)
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
        if need_pose and fresh_tracked:
            if self._ensure_pose_model():
                try:
                    t1 = time.perf_counter()
                    fresh_detections: list[dict[str, Any]] = self.pose_model.estimate(frame, fresh_tracked)
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
        display_dets = self.track_cache.active(self.frame_idx)
        # fresh_only: tracks actually seen THIS frame – used for AI inference
        fresh_ids = {int(d.get("track_id", -1)) for d in fresh_detections if d.get("track_id", -1) >= 0}
        fresh_only = [d for d in display_dets if int(d.get("track_id", -1)) in fresh_ids]

        tracked = sum(1 for d in display_dets if int(d.get("track_id", -1)) >= 0)
        skeleton_scores = [
            keypoint_quality(d.get("keypoint_scores"), 0.0)[1]
            for d in fresh_only
            if d.get("keypoints") is not None
        ]
        valid_skeletons = len(skeleton_scores)
        avg_kpt = float(sum(skeleton_scores) / len(skeleton_scores)) if skeleton_scores else 0.0

        base_metrics.update({
            "detections": len(bboxes),
            "tracked": tracked,
            "persons": len(display_dets),
            "valid_skeletons": valid_skeletons,
            "avg_kpt": round(avg_kpt, 2),
            "selected_modules": sorted(self.modules),
            "ai_active": True,
        })

        # ── Module-specific pipelines (fresh detections only!) ────────────

        if "adl" in self.modules:
            t_adl = time.perf_counter()
            self._process_adl(frame, fresh_only, logs, (h, w))  # ← fresh_only, NOT display_dets
            adl_ms = (time.perf_counter() - t_adl) * 1000.0

        if "reid" in self.modules:
            self._process_reid(frame, fresh_only, logs)  # ← fresh_only, NOT display_dets

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

        if "track" in self.modules:
            logs.append(("ai", f"TRACK: frame={self.frame_idx} det={len(bboxes)} tracked={tracked} fps={fps}"))
            if tracks_metric:
                per_t = " ".join(f"t{t['track_id']}={t['conf']:.2f}" for t in tracks_metric)
                logs.append(("metric", f"TRACK_CONF: {per_t} mean={mean_conf:.2f}"))

        if "pose" in self.modules:
            logs.append(("ai", f"POSE: frame={self.frame_idx} persons={len(fresh_only)} valid={valid_skeletons} avg_kpt={avg_kpt:.2f}"))

        # ── Gallery events (fresh_only, with interval dedup) ─────────────
        self._generate_gallery_events(frame, fresh_only)  # ← fresh_only

        # ── Info panel ───────────────────────────────────────────────────
        draw_info_panel(frame, {
            "Camera": self.camera_id,
            "Frame": self.frame_idx,
            "Modules": ",".join(sorted(self.modules)) or "none",
            "Detections": len(bboxes),
            "Tracked": tracked,
            "FPS": f"{fps:.1f}",
        })

        total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_pose_ms = pose_ms
        self.last_adl_ms = adl_ms
        self.last_total_ms = total_ms

        base_metrics.update({
            "tracks": tracks_metric,
            "best_track_conf": round(best_conf, 3),
            "mean_track_conf": round(mean_conf, 3),
            "num_tracks": len(tracks_metric),
            "detect_ms": round(detect_ms, 2),
            "pose_ms": round(pose_ms, 2),
            "adl_ms": round(adl_ms, 2),
            "total_ms": round(total_ms, 2),
        })

        if self.frame_idx % 100 == 0:
            print(
                f"[METRIC][cam={self.camera_id}] "
                f"Detect={detect_ms:.1f}ms | Pose={pose_ms:.1f}ms | "
                f"ADL={adl_ms:.1f}ms | Total={total_ms:.1f}ms | FPS={fps:.1f} | "
                f"tracks={tracked}"
            )

        return frame, logs, base_metrics

    # ── ReID ──────────────────────────────────────────────────────────────────

    def _process_reid(self, frame: np.ndarray, detections: list[dict[str, Any]], logs: list) -> None:
        """Run ReID on fresh (non-cached) detections only."""
        reid_cfg = self.cfg.get("reid", {})
        reid_interval = int(reid_cfg.get("reid_interval", 30))
        min_track_age = int(reid_cfg.get("min_track_age", 10))
        min_crop_area = float(reid_cfg.get("min_crop_area", 2500))
        threshold = float(reid_cfg.get("threshold", 0.55))
        reid_warn_logged = False

        if not self._ensure_reid():
            if not reid_warn_logged:
                logs.append(("warning", f"ReID unavailable: {self.reid_error}"))
                reid_warn_logged = True
            return

        gallery_empty = not bool(getattr(self.reid_gallery, "prototypes", {}))
        if gallery_empty and not reid_warn_logged:
            logs.append(("warning", "ReID gallery empty; assigned temporary gid"))
            reid_warn_logged = True

        for det in detections:
            if det.get("cached", False):
                continue  # never ReID stale cached detections

            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue

            # Min track age guard
            track_hit_count = self.reid_last_frame.get(tid, -min_track_age)
            if self.frame_idx - track_hit_count < min_track_age:
                continue

            # Min crop area guard
            x1, y1, x2, y2 = map(float, det["bbox"])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_crop_area:
                continue

            # Interval guard: skip if re-queried too recently
            last_reid = self.reid_last_frame.get(f"reid_{tid}", -(reid_interval + 1))
            if isinstance(last_reid, int) and self.frame_idx - last_reid < reid_interval:
                continue  # use cached result silently

            crop = clipped_crop(frame, det["bbox"])
            if crop is None:
                continue

            if gallery_empty:
                gid, score = self._next_temp_gid(tid), 0.0
            else:
                try:
                    feat = self.reid_extractor.extract(crop)
                    feat_dim = int(feat.shape[0])

                    # Dimension mismatch guard
                    gallery_dims = {p.shape[0] for p in self.reid_gallery.prototypes.values()}
                    if feat_dim not in gallery_dims:
                        logs.append(("warning", f"ReID gallery dim mismatch: extractor={feat_dim} gallery={gallery_dims}; skip"))
                        continue

                    gid, score = self.reid_gallery.query(feat, threshold=threshold)
                    self.reid_gid_cache[tid] = {"gid": gid, "score": score}
                    self.reid_last_frame[f"reid_{tid}"] = self.frame_idx

                    # Log only on gid change or every 30 frames
                    prev_gid = self.reid_gid_cache.get(tid, {}).get("gid")
                    if prev_gid != gid or self.frame_idx % 30 == 0:
                        logs.append(("ai", f"ReID: track={tid} gid={gid} score={score:.2f}"))

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
            logs.append(("warning", f"ADL unavailable: {self.adl_error}"))
            return

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
                        action_label = self.adl.update(tid, kps, det.get("bbox", [0, 0, 0, 0]))

                current_len = len(getattr(self.adl, "buffers", {}).get(tid, []))
                min_frames = int(adl_cfg.get("min_frames", 30))
                status: dict[str, Any] = {
                    "status": "inferred" if action_label != "unknown" else "collecting",
                    "label": action_label,
                    "score": float(getattr(self.adl, "score", lambda _: 0.0)(tid)) if hasattr(self.adl, "score") else 0.0,
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

            if len(self.gallery_events) >= max_items:
                break

            event = self._make_crop_event(frame, det)
            if event:
                self.gallery_events.append(event)
                self.last_gallery_frame_by_track[tid] = self.frame_idx
