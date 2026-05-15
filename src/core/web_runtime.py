from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.action.pose_buffer import PoseSequenceBuffer
from src.detectors.yolo_pose import YoloPoseTracker
from src.utils.filters import bbox_area, keypoint_quality
from src.utils.vis import FPSCounter, draw_adl_status, draw_detection, draw_info_panel


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

        self.reid_extractor = None
        self.reid_gallery = None
        self.reid_error = None
        self.reid_gid_cache: dict[int, dict[str, Any]] = {}
        self.temp_gid_counter = 0
        self.temp_gid_by_track: dict[int, str] = {}

        self.pose_buffer = None
        self.adl_error = None
        self.adl_status_by_track: dict[int, dict[str, Any]] = {}

    def set_modules(self, modules: set[str]):
        self.modules = set(modules)

    def _ensure_pose_tracker(self):
        if self.pose_tracker is not None:
            return True
        if self.pose_tracker_error:
            return False
        try:
            pose_cfg = self.cfg.get("pose", {})
            tracking_cfg = self.cfg.get("tracking", {})
            self.pose_tracker = YoloPoseTracker(
                weights=pose_cfg["weights"],
                conf=tracking_cfg.get("person_conf", pose_cfg.get("conf", 0.6)),
                iou=tracking_cfg.get("iou", pose_cfg.get("iou", 0.5)),
                tracker=tracking_cfg.get("tracker_yaml", "bytetrack.yaml"),
                device=self.cfg.get("system", {}).get("device"),
                classes=tracking_cfg.get("classes", [0]),
                tracking_cfg=tracking_cfg,
            )
            return True
        except Exception as exc:
            self.pose_tracker_error = f"{type(exc).__name__}: {exc}"
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
            return frame, [], metrics

        need_pose_model = bool({"track", "pose", "reid", "adl"} & self.modules)
        detections: list[dict[str, Any]] = []
        filter_stats = {}

        if need_pose_model:
            if not self._ensure_pose_tracker():
                logs.append(("warning", f"AI runtime unavailable: {self.pose_tracker_error}"))
                return frame, logs, metrics
            try:
                detections, _ = self.pose_tracker.infer(frame, persist=True)
                filter_stats = self.pose_tracker.last_filter_stats.as_dict()
            except Exception as exc:
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
