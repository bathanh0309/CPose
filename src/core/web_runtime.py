from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import os

from src.action.pose_buffer import PoseSequenceBuffer
from src.detectors.person_gate import PersonGateDetector
from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.filters import bbox_area, keypoint_quality
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel


def available_openvino_devices() -> list[str]:
    try:
        try:
            from openvino.runtime import Core
        except ModuleNotFoundError:
            from openvino import Core

        return list(Core().available_devices)
    except Exception:
        return []


def clipped_crop(frame: np.ndarray, bbox: list[float]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class WebAIProcessor:
    def __init__(self, camera_id: str, modules: set[str], cfg: dict[str, Any]):
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self.modules = set(modules)
        self.frame_idx = -1
        self.fps_counter = FPSCounter()

        self.pose_tracker = None
        self.pose_tracker_error = None
        self.openvino_devices = available_openvino_devices()
        self.openvino_device = self._select_openvino_device()
        self.openvino_enabled = bool(self.openvino_device)

        self.person_gate = None
        self.person_gate_error = None

        self.reid_extractor = None
        self.reid_gallery = None
        self.reid_error = None
        self.reid_gid_cache: dict[int, dict[str, Any]] = {}
        self.temp_gid_counter = 0
        self.temp_gid_by_track: dict[int, str] = {}

        self.pose_buffer = None
        self.adl_error = None
        self.adl_status_by_track: dict[int, dict[str, Any]] = {}

        self.person_active = False
        self.no_person_frames = 0
        self.last_wait_log_frame = -999999
        self.last_gate_log_frame = -999999
        self.last_person_count = 0
        self.last_gate_detections: list[dict[str, Any]] = []
        self.last_gate_best_conf: float = 0.0
        self.last_gate_status = "disabled"

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

        preferred_raw = str(os.environ.get("CPOSE_OPENVINO_DEVICE") or web_cfg.get("openvino_device", "GPU"))
        preferred = preferred_raw.replace("intel:", "").upper()
        fallback = str(web_cfg.get("openvino_fallback_device", "CPU")).upper()
        if preferred in self.openvino_devices:
            return preferred
        if fallback in self.openvino_devices:
            return fallback
        return None

    def _is_openvino_gpu_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("igc_check", "cisa", "gpu compiler", "openvino", "cldnn"))

    def _fallback_openvino_cpu(self, exc: Exception) -> bool:
        if not self.openvino_enabled or str(self.openvino_device).upper() != "GPU":
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
        return True

    def _select_yolo_pose_runtime(self) -> tuple[str, str | None, str]:
        pose_cfg = self.cfg.get("pose", {})
        web_cfg = self.cfg.get("web", {})
        pt_weights = str(pose_cfg["weights"])

        if self.openvino_enabled:
            ov_weights = web_cfg.get("openvino_pose_weights")
            if ov_weights and Path(str(ov_weights)).exists():
                device = self.openvino_device
                yolo_device = device if str(device).lower().startswith("intel:") else f"intel:{str(device).lower()}"
                return str(ov_weights), yolo_device, "OpenVINO"

        return pt_weights, self.cfg.get("system", {}).get("device"), "PyTorch"

    def _ensure_pose_tracker(self):
        if self.pose_tracker is not None:
            return True
        if self.pose_tracker_error:
            return False
        try:
            pose_cfg = self.cfg.get("pose", {})
            tracking_cfg = self.cfg.get("tracking", {})
            weights, device, runtime = self._select_yolo_pose_runtime()
            print(
                f"[WebAIProcessor] cam={self.camera_id} YOLO runtime={runtime} "
                f"weights={weights} device={device}"
            )
            self.pose_tracker = YoloPoseTracker(
                weights=weights,
                conf=tracking_cfg.get("person_conf", pose_cfg.get("conf", 0.6)),
                iou=tracking_cfg.get("iou", pose_cfg.get("iou", 0.5)),
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
            device = self.openvino_device
            if device and not str(device).lower().startswith("intel:"):
                device = f"intel:{str(device).lower()}"
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
        if self.pose_buffer is not None:
            return True
        if self.adl_error:
            return False
        try:
            adl_cfg = self.cfg.get("adl", {})
            self.pose_buffer = PoseSequenceBuffer(
                seq_len=adl_cfg.get("seq_len", 48),
                stride=adl_cfg.get("stride", 12),
                output_dir=adl_cfg.get("export_dir", "data/output/clips_pkl"),
                default_label=adl_cfg.get("default_label", 0),
                max_idle_frames=adl_cfg.get("max_idle_frames", 150),
                export_enabled=False,
            )
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

        if self.frame_idx - self.last_wait_log_frame >= log_interval:
            logs.append(("warning", "PERSON_GATE: waiting for person"))
            self.last_wait_log_frame = self.frame_idx

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

        web_cfg = self.cfg.get("web", {})
        gate_cfg = self.cfg.get("person_gate", {})
        start_ai_on_person_only = bool(web_cfg.get("start_ai_on_person_only", True))
        person_gate_enabled = bool(gate_cfg.get("enabled", True))
        need_pose_model = bool({"track", "pose", "reid", "adl"} & self.modules)
        detections: list[dict[str, Any]] = []
        filter_stats = {}

        if start_ai_on_person_only and person_gate_enabled and need_pose_model:
            ai_allowed = self._run_person_gate(frame, logs)
            metrics.update({
                "ai_active": bool(ai_allowed),
                "person_gate_detected": bool(ai_allowed),
                "gate_detections": len(self.last_gate_detections),
                "gate_best_conf": float(self.last_gate_best_conf),
                "selected_modules": sorted(self.modules),
            })
            if not ai_allowed:
                metrics.update({
                    "detections": 0,
                    "tracked": 0,
                    "persons": 0,
                    "ai_active": False,
                })
                self._draw_waiting_panel(frame, fps, metrics)
                return frame, logs, metrics

        if need_pose_model:
            if not self._ensure_pose_tracker():
                logs.append(("warning", f"AI runtime unavailable: {self.pose_tracker_error}"))
                return frame, logs, metrics
            try:
                detections, _ = self.pose_tracker.infer(frame, persist=True)
                filter_stats = self.pose_tracker.last_filter_stats.as_dict()
            except Exception as exc:
                retry_ok = False
                if self._fallback_openvino_cpu(exc) and self._ensure_pose_tracker():
                    try:
                        detections, _ = self.pose_tracker.infer(frame, persist=True)
                        filter_stats = self.pose_tracker.last_filter_stats.as_dict()
                        retry_ok = True
                    except Exception as retry_exc:
                        exc = retry_exc
                if not retry_ok:
                    logs.append(("warning", f"AI inference failed: {type(exc).__name__}: {exc}"))
                    return frame, logs, metrics

        filtered = int(sum(filter_stats.values())) if filter_stats else 0
        tracked = sum(1 for det in detections if int(det.get("track_id", -1)) >= 0)
        skeleton_scores = [
            keypoint_quality(det.get("keypoint_scores"), 0.0)[1]
            for det in detections
            if det.get("keypoints") is not None
        ]
        valid_skeletons = len(skeleton_scores)
        avg_kpt = float(sum(skeleton_scores) / len(skeleton_scores)) if skeleton_scores else 0.0

        metrics.update({
            "detections": len(detections),
            "tracked": tracked,
            "filtered": filtered,
            "persons": len(detections),
            "valid_skeletons": valid_skeletons,
            "avg_kpt": round(avg_kpt, 2),
            "selected_modules": sorted(self.modules),
        })

        if not (start_ai_on_person_only and person_gate_enabled) and len(detections) > 0:
            self._handle_person_detected(logs, len(detections))
            best_conf = max([float(d.get("score", 0.0)) for d in detections], default=0.0)
            metrics.update({
                "ai_active": True,
                "person_gate_detected": True,
                "gate_detections": len(detections),
                "gate_best_conf": round(float(best_conf), 3),
            })
        elif start_ai_on_person_only and person_gate_enabled:
            metrics.update({
                "ai_active": True,
                "person_gate_detected": (self.last_gate_status == "person_detected"),
                "gate_detections": len(self.last_gate_detections),
                "gate_best_conf": float(self.last_gate_best_conf),
            })

        if "track" in self.modules:
            for det in detections:
                tid = int(det.get("track_id", -1))
                draw_detection(
                    frame,
                    self._track_label_det(det, include_pose=("pose" in self.modules or "adl" in self.modules)),
                    label=f"track={tid} conf={float(det.get('score', 0.0)):.2f}",
                )
            logs.append(("ai", f"TRACK: frame={self.frame_idx} det={len(detections)} tracked={tracked} filtered={filtered} fps={fps}"))

        if "pose" in self.modules:
            for det in detections:
                if "track" not in self.modules:
                    draw_detection(frame, det, label=f"pose conf={float(det.get('score', 0.0)):.2f}")
            logs.append(("ai", f"POSE: frame={self.frame_idx} persons={len(detections)} valid={valid_skeletons} avg_kpt={avg_kpt:.2f}"))

        if "reid" in self.modules:
            self._process_reid(frame, detections, logs)

        if "adl" in self.modules:
            self._process_adl(frame, detections, logs, (h, w))

        draw_info_panel(frame, {
            "Camera": self.camera_id,
            "Frame": self.frame_idx,
            "Modules": ",".join(sorted(self.modules)) or "none",
            "Detections": len(detections),
            "Tracked": tracked,
            "Filtered": filtered,
            "FPS": f"{fps:.1f}",
        })

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

        first_status = None
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            status = self.pose_buffer.update(
                self.camera_id,
                tid,
                f"track_{tid}",
                self.frame_idx,
                det.get("keypoints"),
                det.get("keypoint_scores"),
                img_shape,
            )
            self.adl_status_by_track[tid] = status
            first_status = first_status or status

            state = status.get("status")
            if state == "collecting":
                logs.append(("ai", f"ADL: track={tid} collecting={status.get('current_len', 0)}/{status.get('seq_len', 0)}"))
            elif state == "exported":
                logs.append(("ai", f"ADL: track={tid} clip exported"))
            elif state == "disabled":
                logs.append(("warning", f"ADL: track={tid} inference disabled"))
            elif state == "inferred":
                logs.append(("ai", f"ADL: track={tid} {status.get('label')} score={float(status.get('score', 0.0)):.2f}"))
            elif state:
                logs.append(("warning", f"ADL: track={tid} {state}"))

            draw_detection(frame, det, label=f"track={tid} ADL:{state}")

        if first_status:
            draw_adl_status(frame, first_status, pos=(10, 150))
