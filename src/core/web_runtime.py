from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.action.pose_buffer import PoseSequenceBuffer
from src.core.track_cache import TrackCache
from src.detectors.yolo_openvino import YOLODetectOpenVINO, YOLOPoseOpenVINO
from src.device_manager import resolve_detect_device
from src.trackers.bytetrack import ByteTrackNumpy
from src.utils.filters import bbox_area, keypoint_quality
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel
from src.action.adl_efficientgcn import ADLEfficientGCN


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
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y1 - th - 10)
    x_end = min(w - 1, x1 + tw + 8)
    cv2.rectangle(frame, (x1, y0), (x_end, y1), (0, 80, 160), -1)
    cv2.putText(frame, text, (x1 + 3, max(th + 2, y1 - 7)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


class WebAIProcessor:
    def __init__(self, camera_id: str, modules: set[str], cfg: dict[str, Any]):
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self.modules = set(modules)
        self.frame_idx = -1
        self.fps_counter = FPSCounter()
        self.last_detect_ms = 0.0
        self.last_pose_ms = 0.0
        self.last_adl_ms = 0.0
        self.last_total_ms = 0.0

        self._print_device_allocation_header()
        detect_device = resolve_detect_device()
        self.detector = YOLODetectOpenVINO(
            xml_path="D:/Capstone_Project/models/yolo11n_openvino_model/yolo11n.xml",
            device=detect_device,
            conf_threshold=0.4,
            iou_threshold=0.45,
            class_filter=[0],
        )
        print(f"[DETECT] device={self.detector.device}")
        self.pose_model = YOLOPoseOpenVINO(
            model_path="D:/Capstone_Project/models/yolo11n-pose_openvino_model",
            device="CPU",
        )
        print("[POSE] device=CPU")
        self.tracker = ByteTrackNumpy(high_thresh=0.4, low_thresh=0.1, match_thresh=0.3, max_age=30)
        print("[ByteTrack] device=CPU")

        # Legacy lazy fields kept for older helper methods, but web inference uses
        # the OpenVINO/numpy path above.
        self.tracking_tracker = None
        self.tracking_tracker_error = None
        self.pose_tracker = None
        self.pose_tracker_error = None
        self.openvino_devices = []
        self.openvino_device = self.detector.device
        self.openvino_enabled = True

        self.person_gate = None
        self.person_gate_error = None

        self.reid_extractor = None
        self.reid_gallery = None
        self.reid_error = None
        self.reid_gid_cache: dict[int, dict[str, Any]] = {}
        self.temp_gid_counter = 0
        self.temp_gid_by_track: dict[int, str] = {}

        self.pose_buffer = None
        self.adl = None
        self.adl_enabled = False
        self.adl_error = None
        self.adl_status_by_track: dict[int, dict[str, Any]] = {}
        self.adl_last_seen_frame: dict[int, int] = {}
        self.lost_track_threshold = int(cfg.get("adl", {}).get("lost_track_threshold", 30))

        self.person_active = False
        self.no_person_frames = 0
        self.last_wait_log_frame = -999999
        self.last_gate_log_frame = -999999
        self.last_person_count = 0
        self.last_gate_detections: list[dict[str, Any]] = []
        self.last_gate_best_conf: float = 0.0
        self.last_gate_status = "disabled"

        # TrackCache for anti-flicker
        track_ttl = int(cfg.get("tracking", {}).get("track_ttl_frames", 25))
        self.track_cache = TrackCache(ttl_frames=track_ttl)

        # Gallery event support
        self.gallery_events: list[dict[str, Any]] = []
        self.last_gallery_frame_by_track: dict[int, int] = {}

        self._ensure_adl()

    def _print_device_allocation_header(self) -> None:
        print("╔══════════════════════════════════════════╗")
        print("║         CPose Device Allocation          ║")
        print("╠══════════════════════════════════════════╣")
        print("║  YOLO Detect   │ GPU.0 (Intel Iris Xe)   ║")
        print("║  YOLO Pose     │ CPU                     ║")
        print("║  ByteTrack     │ CPU                     ║")
        print("║  EfficientGCN  │ CPU                     ║")
        print("║  JPEG + WS     │ CPU                     ║")
        print("╚══════════════════════════════════════════╝")

    def set_modules(self, modules: set[str]):
        self.modules = set(modules)
        if not self.modules:
            self.person_active = False
            self.no_person_frames = 0
            self.last_person_count = 0
            self.last_gate_detections = []
            self.last_gate_status = "disabled"

    def _select_openvino_device(self) -> str | None:
        web_cfg = self.cfg.get("web", {})
        env_enabled = os.environ.get("CPOSE_OPENVINO_ENABLED")
        if env_enabled is not None and env_enabled.strip() in {"0", "false", "False"}:
            return None
        if env_enabled is None and not web_cfg.get("openvino_enabled", False):
            return None
        if not self.openvino_devices:
            return None

        preferred = str(os.environ.get("CPOSE_OPENVINO_DEVICE") or web_cfg.get("openvino_device", "GPU.0"))
        fallback = str(web_cfg.get("openvino_fallback_device", "CPU"))
        return select_openvino_device(self.openvino_devices, preferred=preferred, fallback=fallback)

    def _is_openvino_gpu_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("igc_check", "cisa", "gpu compiler", "openvino", "cldnn"))

    def _fallback_openvino_cpu(self, exc: Exception) -> bool:
        if not self.openvino_enabled or not is_openvino_gpu_device(self.openvino_device):
            return False
        if not self._is_openvino_gpu_error(exc):
            return False

        fallback = str(self.cfg.get("web", {}).get("openvino_fallback_device", "CPU")).upper()
        if fallback != "CPU" or "CPU" not in self.openvino_devices:
            return False

        print(
            f"[WebAIProcessor] cam={self.camera_id} Warning: OpenVINO GPU failed "
            f"({type(exc).__name__}: {exc}); falling back to CPU"
        )
        self.openvino_device = "CPU"
        self.openvino_enabled = True
        self.pose_tracker = None
        self.pose_tracker_error = None
        self.tracking_tracker = None
        self.tracking_tracker_error = None
        self.person_gate = None
        self.person_gate_error = None
        return True

    def _select_tracking_runtime(self) -> tuple[str, str | None, str]:
        """Select lightweight detection-only model for tracking (NOT pose model)."""
        tracking_cfg = self.cfg.get("tracking", {})
        web_cfg = self.cfg.get("web", {})
        pt_weights = str(tracking_cfg.get("weights", "models/tracking.pt"))

        if self.openvino_enabled:
            ov_weights = web_cfg.get("openvino_tracking_weights")
            if ov_weights and Path(str(ov_weights)).exists():
                device = self.openvino_device
                yolo_device = to_ultralytics_openvino_device(device)
                return str(ov_weights), yolo_device, "OpenVINO"

        fallback = str(tracking_cfg.get("fallback_weights", pt_weights))
        if Path(fallback).exists():
            return fallback, self.cfg.get("system", {}).get("device"), "PyTorch-fallback"
        return pt_weights, self.cfg.get("system", {}).get("device"), "PyTorch"

    def _select_pose_runtime(self) -> tuple[str, str | None, str]:
        """Select pose model — only used when Pose module is active."""
        pose_cfg = self.cfg.get("pose", {})
        web_cfg = self.cfg.get("web", {})
        pt_weights = str(pose_cfg["weights"])

        if self.openvino_enabled:
            ov_weights = web_cfg.get("openvino_pose_weights")
            if ov_weights and Path(str(ov_weights)).exists():
                device = self.openvino_device
                yolo_device = to_ultralytics_openvino_device(device)
                return str(ov_weights), yolo_device, "OpenVINO"

        return pt_weights, self.cfg.get("system", {}).get("device"), "PyTorch"

    def _ensure_tracking_tracker(self):
        """Ensure the lightweight tracking-only detector is loaded."""
        if self.tracking_tracker is not None:
            return True
        if self.tracking_tracker_error:
            return False
        try:
            tracking_cfg = self.cfg.get("tracking", {})
            weights, device, runtime = self._select_tracking_runtime()
            print(
                f"[WebAIProcessor] cam={self.camera_id} TRACKING runtime={runtime} "
                f"weights={weights} device={device}"
            )
            self.tracking_tracker = PedestrianYoloTracker(
                weights=weights,
                conf=tracking_cfg.get("conf", tracking_cfg.get("person_conf", 0.35)),
                iou=tracking_cfg.get("iou", 0.5),
                tracker=tracking_cfg.get("tracker_yaml", "bytetrack.yaml"),
                device=device,
                classes=tracking_cfg.get("classes", [0]),
                tracking_cfg=tracking_cfg,
            )
            return True
        except Exception as exc:
            self.tracking_tracker_error = f"{type(exc).__name__}: {exc}"
            print(f"[WebAIProcessor] tracking_tracker failed cam={self.camera_id}: {self.tracking_tracker_error}")
            return False

    def _ensure_pose_tracker(self):
        """Ensure the pose model is loaded — only when Pose module is active."""
        if self.pose_tracker is not None:
            return True
        if self.pose_tracker_error:
            return False
        try:
            pose_cfg = self.cfg.get("pose", {})
            tracking_cfg = self.cfg.get("tracking", {})
            weights, device, runtime = self._select_pose_runtime()
            print(
                f"[WebAIProcessor] cam={self.camera_id} POSE runtime={runtime} "
                f"weights={weights} device={device}"
            )
            self.pose_tracker = YoloPoseTracker(
                weights=weights,
                conf=pose_cfg.get("conf", 0.6),
                iou=pose_cfg.get("iou", 0.5),
                tracker=tracking_cfg.get("tracker_yaml", "bytetrack.yaml"),
                device=device,
                classes=tracking_cfg.get("classes", [0]),
                tracking_cfg=tracking_cfg,
            )
            return True
        except Exception as exc:
            self.pose_tracker_error = f"{type(exc).__name__}: {exc}"
            return False

    def _select_person_gate_runtime(self) -> tuple[str, str | None]:
        gate_cfg = self.cfg.get("person_gate", {})
        web_cfg = self.cfg.get("web", {})

        weights = gate_cfg.get("weights") or web_cfg.get("openvino_detect_weights")
        fallback = gate_cfg.get("fallback_weights") or self.cfg.get("object", {}).get("weights")
        weights = weights or fallback

        if self.openvino_enabled and weights and "openvino_model" in str(weights):
            device = to_ultralytics_openvino_device(self.openvino_device)
        else:
            device = self.cfg.get("system", {}).get("device")

        return str(weights), device

    def _ensure_person_gate(self) -> bool:
        if self.person_gate is not None:
            return True
        if self.person_gate_error:
            return False

        try:
            gate_cfg = self.cfg.get("person_gate", {})
            weights, device = self._select_person_gate_runtime()
            print(f"[PersonGate] cam={self.camera_id} weights={weights} device={device}")
            self.person_gate = PersonGateDetector(
                weights=weights,
                fallback_weights=gate_cfg.get("fallback_weights"),
                conf=gate_cfg.get("conf", 0.25),
                iou=gate_cfg.get("iou", 0.5),
                imgsz=gate_cfg.get("imgsz", 640),
                classes=gate_cfg.get("classes", [0]),
                min_box_area=gate_cfg.get("min_box_area", 800),
                device=device,
            )
            return True
        except Exception as exc:
            self.person_gate_error = f"{type(exc).__name__}: {exc}"
            print(f"[PersonGate] unavailable cam={self.camera_id}: {self.person_gate_error}")
            return False

    def _ensure_reid(self):
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
                device=self.cfg.get("system", {}).get("device"),
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
            return True
        except Exception as exc:
            self.reid_error = f"{type(exc).__name__}: {exc}"
            return False

    def _ensure_adl(self):
        if self.adl is not None:
            return True
        if self.adl_error:
            return False
        try:
            adl_cfg = self.cfg.get("adl", {})
            self.adl = ADLEfficientGCN(
                xml_path="D:/Capstone_Project/models/efficientgcn_b0_ntu_xsub120_openvino_model/efficientgcn_b0_ntu_xsub120.xml",
                conf_threshold=float(adl_cfg.get("conf_threshold", 0.3)),
                min_frames=int(adl_cfg.get("min_frames", 30)),
                device="CPU",
                cache_dir=adl_cfg.get("openvino_cache_dir") or self.cfg.get("web", {}).get("openvino_cache_dir", "openvino_cache"),
                precision_hint=adl_cfg.get("precision_hint", "f16"),
            )
            self.adl.infer_every_n_frames = int(adl_cfg.get("infer_every_n_frames", 15))
            self.adl_enabled = True
            print("[ADL] device=CPU")
            return True
        except Exception as exc:
            self.adl_error = f"{type(exc).__name__}: {exc}"
            return False

    def _next_temp_gid(self, track_id: int) -> str:
        if track_id in self.temp_gid_by_track:
            return self.temp_gid_by_track[track_id]
        self.temp_gid_counter += 1
        gid = f"gid_{self.temp_gid_counter:05d}"
        self.temp_gid_by_track[track_id] = gid
        return gid

    def _track_label_det(self, det: dict[str, Any], include_pose: bool) -> dict[str, Any]:
        if include_pose:
            return det
        return {**det, "keypoints": None, "keypoint_scores": None}

    def _draw_waiting_panel(self, frame: np.ndarray, fps: float, metrics: dict[str, Any]) -> None:
        web_cfg = self.cfg.get("web", {})
        if not web_cfg.get("draw_waiting_panel", True):
            return

        draw_info_panel(frame, {
            "Camera": self.camera_id,
            "Frame": self.frame_idx,
            "Status": "Waiting for person",
            "Selected": ",".join(sorted(self.modules)) or "none",
            "AI Active": "no",
            "Gate det": metrics.get("gate_detections", 0),
            "Gate conf": metrics.get("gate_best_conf", 0.0),
            "FPS": f"{fps:.1f}",
        })

    def _handle_person_detected(self, logs: list[tuple[str, str]], person_count: int) -> None:
        # Mark that a person was detected and reset grace counters.
        self.person_active = True
        self.no_person_frames = 0
        self.last_person_count = person_count

    def _run_person_gate(self, frame: np.ndarray, logs: list[tuple[str, str]]) -> bool:
        gate_cfg = self.cfg.get("person_gate", {})
        interval = int(gate_cfg.get("interval_frames", 2))
        lost_grace = int(gate_cfg.get("lost_grace_frames", 15))
        log_interval = int(gate_cfg.get("log_interval_frames", 10))
        draw_gate_box = bool(gate_cfg.get("draw_gate_box", True))

        should_check = self.person_active or self.frame_idx % max(interval, 1) == 0
        if not should_check:
            self.last_gate_status = "idle_wait"
            return False

        if not self._ensure_person_gate():
            logs.append(("warning", f"PERSON_GATE: unavailable ({self.person_gate_error}); allowing selected AI modules"))
            self.last_gate_status = "unavailable_allow"
            return True

        try:
            detected, gate_dets = self.person_gate.detect(frame)
            self.last_gate_detections = gate_dets
        except Exception as exc:
            logs.append(("warning", f"PERSON_GATE: failed {type(exc).__name__}: {exc}; allowing selected AI modules"))
            self.last_gate_status = "failed_allow"
            return True

        best_conf = max([float(d.get("score", 0.0)) for d in gate_dets], default=0.0)
        self.last_gate_best_conf = round(float(best_conf), 3)

        if detected:
            self._handle_person_detected(logs, len(gate_dets))
            self.person_active = True
            self.no_person_frames = 0
            self.last_gate_status = "person_detected"

            if draw_gate_box:
                self.person_gate.draw_gate_detections(frame, gate_dets)

            # Log to UI/terminal at most once per `log_interval` frames
            if self.frame_idx - self.last_gate_log_frame >= log_interval:
                logs.append(("ai", f"PERSON_GATE: detected={len(gate_dets)} best_conf={best_conf:.2f}"))
                self.last_gate_log_frame = self.frame_idx

            return True

        # No detection
        self.no_person_frames += 1
        if self.no_person_frames >= lost_grace:
            self.person_active = False

        self.last_gate_status = "grace_wait" if self.person_active else "waiting"
        return self.person_active

    def process(self, frame: np.ndarray):
        self.frame_idx += 1
        fps = round(float(self.fps_counter.tick()), 2)
        logs: list[tuple[str, str]] = []
        h, w = frame.shape[:2]

        metrics = {
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
        }

        if not self.modules:
            metrics.update({
                "ai_active": False,
                "person_gate": "disabled",
            })
            return frame, [], metrics

        need_pose_model = bool({"track", "pose", "reid", "adl"} & self.modules)
        detections: list[dict[str, Any]] = []
        filter_stats: dict[str, int] = {}
        t_total = time.perf_counter()
        detect_ms = pose_ms = adl_ms = 0.0

        if need_pose_model:
            try:
                t0 = time.perf_counter()
                bboxes = self.detector.detect(frame)
                detect_ms = (time.perf_counter() - t0) * 1000.0
                tracked = self.tracker.update(bboxes, frame)
                if {"pose", "adl"} & self.modules:
                    t1 = time.perf_counter()
                    detections = self.pose_model.estimate(frame, tracked)
                    pose_ms = (time.perf_counter() - t1) * 1000.0
                else:
                    detections = tracked
            except Exception as exc:
                logs.append(("warning", f"OpenVINO/numpy pipeline failed: {type(exc).__name__}: {exc}"))
                return frame, logs, metrics

            self.last_gate_detections = [{"bbox": b[:4], "score": b[4], "class_id": 0} for b in bboxes]
            self.last_gate_best_conf = max([float(b[4]) for b in bboxes], default=0.0)
            if bboxes:
                self._handle_person_detected(logs, len(bboxes))
                self.last_gate_status = "person_detected"
            else:
                self.last_gate_status = "waiting"
                self.no_person_frames += 1
                metrics.update({
                    "ai_active": False,
                    "person_gate": "waiting",
                    "person_gate_detected": False,
                    "gate_detections": 0,
                    "gate_best_conf": 0.0,
                    "selected_modules": sorted(self.modules),
                })
                self._draw_waiting_panel(frame, fps, metrics)
                return frame, logs, metrics

        # --- TrackCache anti-flicker ---
        self.track_cache.update(detections, self.frame_idx)
        display_dets = self.track_cache.active(self.frame_idx)

        filtered = int(sum(filter_stats.values())) if filter_stats else 0
        tracked = sum(1 for det in display_dets if int(det.get("track_id", -1)) >= 0)
        skeleton_scores = [
            keypoint_quality(det.get("keypoint_scores"), 0.0)[1]
            for det in display_dets
            if det.get("keypoints") is not None
        ]
        valid_skeletons = len(skeleton_scores)
        avg_kpt = float(sum(skeleton_scores) / len(skeleton_scores)) if skeleton_scores else 0.0

        metrics.update({
            "detections": len(detections),
            "tracked": tracked,
            "filtered": filtered,
            "persons": len(display_dets),
            "valid_skeletons": valid_skeletons,
            "avg_kpt": round(avg_kpt, 2),
            "selected_modules": sorted(self.modules),
        })

        if len(display_dets) > 0:
            self._handle_person_detected(logs, len(display_dets))
            best_conf = max([float(d.get("score", 0.0)) for d in display_dets], default=0.0)
            metrics.update({
                "ai_active": True,
                "person_gate_detected": True,
                "gate_detections": len(display_dets),
                "gate_best_conf": round(float(best_conf), 3),
            })

        if "track" in self.modules:
            for det in display_dets:
                tid = int(det.get("track_id", -1))
                cached = det.get("cached", False)
                score = float(det.get("score", 0.0))
                label = f"track={tid} {'cached ' if cached else ''}conf={score:.2f}"
                draw_detection(
                    frame,
                    self._track_label_det(det, include_pose=("pose" in self.modules or "adl" in self.modules)),
                    label=label,
                )
            logs.append(("ai", f"TRACK: frame={self.frame_idx} det={len(detections)} tracked={tracked} filtered={filtered} fps={fps}"))

        if "pose" in self.modules:
            for det in display_dets:
                if "track" not in self.modules:
                    draw_detection(frame, det, label=f"pose conf={float(det.get('score', 0.0)):.2f}")
            logs.append(("ai", f"POSE: frame={self.frame_idx} persons={len(display_dets)} valid={valid_skeletons} avg_kpt={avg_kpt:.2f}"))

        if "reid" in self.modules:
            self._process_reid(frame, display_dets, logs)

        if "adl" in self.modules:
            t_adl = time.perf_counter()
            self._process_adl(frame, display_dets, logs, (h, w))
            adl_ms = (time.perf_counter() - t_adl) * 1000.0

        # --- Gallery events ---
        self._generate_gallery_events(frame, display_dets)

        draw_info_panel(frame, {
            "Camera": self.camera_id,
            "Frame": self.frame_idx,
            "Modules": ",".join(sorted(self.modules)) or "none",
            "Detections": len(detections),
            "Tracked": tracked,
            "Filtered": filtered,
            "FPS": f"{fps:.1f}",
        })

        total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_detect_ms = detect_ms
        self.last_pose_ms = pose_ms
        self.last_adl_ms = adl_ms
        self.last_total_ms = total_ms
        metrics.update({
            "detect_ms": round(detect_ms, 2),
            "pose_ms": round(pose_ms, 2),
            "adl_ms": round(adl_ms, 2),
            "total_ms": round(total_ms, 2),
            "detect_device": self.detector.device,
            "pose_device": "CPU",
            "adl_device": "CPU",
        })
        if self.frame_idx % 100 == 0:
            print(
                f"[METRIC] Detect={detect_ms:.1f}ms ({self.detector.device}) | "
                f"Pose={pose_ms:.1f}ms (CPU) | ADL={adl_ms:.1f}ms (CPU, throttle) | "
                f"Total={total_ms:.1f}ms | FPS={fps:.1f}"
            )

        return frame, logs, metrics

    def _process_reid(self, frame, detections, logs):
        if not self._ensure_reid():
            logs.append(("warning", f"ReID: unavailable ({self.reid_error})"))
            return

        reid_cfg = self.cfg.get("reid", {})
        interval = int(reid_cfg.get("reid_interval", 20))
        threshold = float(reid_cfg.get("threshold", 0.55))
        gallery_empty = not bool(getattr(self.reid_gallery, "prototypes", {}))
        if gallery_empty:
            logs.append(("warning", "[ReID] gallery empty, assigned temporary gid_00001"))

        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            crop = clipped_crop(frame, det["bbox"])
            if crop is None:
                continue

            if gallery_empty:
                gid, score, status = self._next_temp_gid(tid), 0.0, "gallery_empty"
            else:
                cached = self.reid_gid_cache.get(tid)
                if cached and self.frame_idx % interval != 0:
                    gid, score, status = cached["gid"], cached["score"], "cache"
                else:
                    try:
                        feat = self.reid_extractor.extract(crop)
                        gid, score = self.reid_gallery.query(feat, threshold=threshold)
                        status = "match" if gid != "unknown" else "unknown"
                        self.reid_gid_cache[tid] = {"gid": gid, "score": score}
                    except Exception as exc:
                        gid, score, status = self._next_temp_gid(tid), 0.0, f"failed:{type(exc).__name__}"

            draw_detection(
                frame,
                self._track_label_det(det, include_pose=("pose" in self.modules or "adl" in self.modules)),
                label=f"track={tid} gid={gid} score={float(score):.2f}",
            )
            logs.append(("ai" if status not in {"gallery_empty"} else "warning", f"ReID: track={tid} gid={gid} score={float(score):.2f}"))

    def _process_adl(self, frame, detections, logs, img_shape):
        if not self._ensure_adl():
            logs.append(("warning", f"ADL: unavailable ({self.adl_error})"))
            return

        active_ids: set[int] = set()
        first_status = None
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            active_ids.add(tid)
            self.adl_last_seen_frame[tid] = self.frame_idx

            keypoints = det.get("keypoints")
            scores = det.get("keypoint_scores")
            action_label = "unknown"
            if keypoints is not None:
                kps = np.asarray(keypoints, dtype=np.float32)
                if kps.shape == (17, 2):
                    score_arr = np.ones((17, 1), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32).reshape(17, 1)
                    kps = np.concatenate([kps, score_arr], axis=1)
                if kps.shape == (17, 3):
                    action_label = self.adl.update(tid, kps, det.get("bbox", [0, 0, 0, 0]))

            status = {
                "status": "inferred" if action_label != "unknown" else "collecting",
                "label": action_label,
                "score": 0.0,
                "current_len": len(self.adl.buffers.get(tid, [])) if self.adl else 0,
                "seq_len": int(self.cfg.get("adl", {}).get("sequence_len", 300)),
            }
            self.adl_status_by_track[tid] = status
            first_status = first_status or status

            if action_label != "unknown":
                draw_action_label(frame, det.get("bbox", [0, 0, 0, 0]), action_label)
                if self.frame_idx % 30 == 0:
                    logs.append(("ai", f"ADL: track={tid} action={action_label}"))

        for tid, last_seen in list(self.adl_last_seen_frame.items()):
            if tid not in active_ids and self.frame_idx - int(last_seen) > self.lost_track_threshold:
                if self.adl:
                    self.adl.cleanup_track(tid)
                self.adl_last_seen_frame.pop(tid, None)
                self.adl_status_by_track.pop(tid, None)

        if first_status:
            draw_adl_status(frame, first_status, pos=(10, 150))

        if self.adl and self.frame_idx % 100 == 0:
            m = self.adl.metrics()
            logs.append((
                "ai",
                f"ADL_METRICS: infer_fps={m['infer_fps']} active_tracks={m['active_tracks']} buffer_frames={m['buffer_frames']}",
            ))

    def _make_crop_event(self, frame: np.ndarray, det: dict[str, Any]) -> dict[str, Any] | None:
        """Create a gallery crop event from a detection."""
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
        return {
            "type": "gallery",
            "track_id": int(det.get("track_id", -1)),
            "global_id": det.get("global_id", "unknown"),
            "conf": round(float(det.get("score", 0.0)), 2),
            "crop_jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
            "ts": time.strftime("%H:%M:%S"),
        }

    def _generate_gallery_events(self, frame: np.ndarray, display_dets: list[dict[str, Any]]) -> None:
        """Generate gallery crop events at configured intervals."""
        self.gallery_events = []
        web_cfg = self.cfg.get("web", {})
        if not web_cfg.get("gallery_enabled", True):
            return
        gallery_interval = int(web_cfg.get("gallery_interval_frames", 20))
        for det in display_dets:
            if det.get("cached", False):
                continue  # Don't gallery-crop cached (stale) detections
            tid = int(det.get("track_id", -1))
            last = self.last_gallery_frame_by_track.get(tid, -999999)
            if self.frame_idx - last >= gallery_interval:
                event = self._make_crop_event(frame, det)
                if event:
                    self.gallery_events.append(event)
                    self.last_gallery_frame_by_track[tid] = self.frame_idx
