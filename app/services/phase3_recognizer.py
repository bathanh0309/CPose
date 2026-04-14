"""Phase 3 offline pose estimation and rule-based ADL recognition."""

from __future__ import annotations

from dataclasses import dataclass
import math
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.utils.file_handler import (
    MULTICAM_CAMERA_ORDER,
    extract_multicam_camera_id,
    multicam_sort_key,
    sort_multicam_clips,
)
from app.utils.runtime_config import get_runtime_section
from app.utils.pose_utils import draw_skeleton, rule_based_adl
from app.core.cross_camera_merger import CrossCameraIDMerger

logger = logging.getLogger("[Phase3]")

_PHASE3_CFG = get_runtime_section("phase3")

CONF_THRESHOLD_P3 = float(_PHASE3_CFG.get("conf_threshold", 0.45))
KP_CONF_MIN = float(_PHASE3_CFG.get("keypoint_conf_min", 0.30))
WINDOW_SIZE = int(_PHASE3_CFG.get("window_size", 30))
PROGRESS_EVERY = int(_PHASE3_CFG.get("progress_every", 10))
PERSON_CLASS_ID = int(_PHASE3_CFG.get("person_class_id", 0))


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: np.ndarray
    last_seen_frame: int
    missed_frames: int = 0


@dataclass(slots=True)
class _TrackedDetection:
    track_id: int
    bbox: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    detection_conf: float


class _SequentialTracker:
    """Tiny IoU-based tracker to keep local IDs stable inside one clip."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 15,
        center_distance_ratio: float = 0.18,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.center_distance_ratio = float(center_distance_ratio)
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_shape: tuple[int, int] | None,
        frame_id: int,
    ) -> tuple[list[_TrackedDetection], list[int]]:
        if not detections:
            expired_track_ids = self._age_tracks(set())
            return [], expired_track_ids

        frame_diag = math.hypot(float(frame_shape[1]), float(frame_shape[0])) if frame_shape else 1.0

        matches: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(detections):
                score = self._match_score(track.bbox, det["bbox"], frame_diag)
                if score is not None:
                    matches.append((score, track_id, det_idx))

        matches.sort(key=lambda item: item[0], reverse=True)

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        det_to_track: dict[int, int] = {}

        for score, track_id, det_idx in matches:
            if track_id in assigned_tracks or det_idx in assigned_dets:
                continue

            det = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox = np.asarray(det["bbox"], dtype=float)
            track.last_seen_frame = frame_id
            track.missed_frames = 0

            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)
            det_to_track[det_idx] = track_id

        expired_track_ids = self._age_tracks(assigned_tracks)

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue

            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=np.asarray(det["bbox"], dtype=float),
                last_seen_frame=frame_id,
            )
            det_to_track[det_idx] = track_id

        tracked: list[_TrackedDetection] = []
        for det_idx, det in enumerate(detections):
            tracked.append(
                _TrackedDetection(
                    track_id=det_to_track[det_idx],
                    bbox=np.asarray(det["bbox"], dtype=float),
                    keypoints_xy=np.asarray(det["keypoints_xy"], dtype=float),
                    keypoints_conf=np.asarray(det["keypoints_conf"], dtype=float),
                    detection_conf=float(det.get("detection_conf", 0.0)),
                )
            )

        return tracked, expired_track_ids

    def _age_tracks(self, active_track_ids: set[int]) -> list[int]:
        expired_track_ids: list[int] = []
        for track_id in list(self._tracks.keys()):
            if track_id in active_track_ids:
                continue

            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                expired_track_ids.append(track_id)
                del self._tracks[track_id]

        return expired_track_ids

    def _match_score(self, track_bbox: np.ndarray, det_bbox: np.ndarray, frame_diag: float) -> float | None:
        iou = _bbox_iou(track_bbox, det_bbox)
        if iou >= self.iou_threshold:
            return 1.0 + iou

        if frame_diag <= 0:
            return None

        center_distance = _bbox_center_distance(track_bbox, det_bbox) / frame_diag
        if center_distance <= self.center_distance_ratio:
            return 0.75 - center_distance

        return None


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _bbox_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _draw_track_label(frame, bbox: np.ndarray, track_id: int, adl_label: tuple[str, float] | None) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    label = f"ID {track_id}"
    if adl_label is not None:
        label_name, label_conf = adl_label
        if label_name and label_name != "unknown":
            label = f"{label} {label_name} {label_conf:.2f}"
        else:
            label = f"{label} pending"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 140, 255), 2)
    text_origin = (x1, max(18, y1 - 8))
    cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)


class PoseADLRecognizer:
    """Run offline pose extraction and ADL classification over MP4 clips."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._config: dict[str, Any] = {}
        self._state: dict[str, Any] = {
            "running": False,
            "current_clip": "",
            "active_camera": "",
            "clips_total": 0,
            "clips_done": 0,
            "keypoints_written": 0,
            "adl_events": 0,
            "progress_pct": 0,
            "save_overlay": True,
            "lamp_state": self._default_lamp_state(),
            "clip_queue": [],
            "error": None,
        }

    def start(
        self,
        clips: list[Path],
        output_dir: Path,
        model_path: Path,
        config_path: Path,
        save_overlay: bool = True,
    ) -> None:
        if self.is_running():
            logger.warning("Phase 3 is already running")
            return

        ordered_clips = sort_multicam_clips(clips)
        clip_queue = self._build_clip_queue(ordered_clips)
        self._stop_evt.clear()
        self._config = self._load_config(config_path)
        self._update_state(
            running=True,
            current_clip="",
            active_camera="",
            clips_total=len(ordered_clips),
            clips_done=0,
            keypoints_written=0,
            adl_events=0,
            progress_pct=0,
            save_overlay=save_overlay,
            lamp_state=self._default_lamp_state(),
            clip_queue=clip_queue,
            error=None,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(ordered_clips, output_dir, model_path, save_overlay),
            daemon=True,
            name="phase3-recognizer",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("Phase 3 stop requested")

    def get_snapshot(self, view: str) -> bytes | None:
        with self._lock:
            return self._state.get(f"snapshot_{view}")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            # Filter out bytes to prevent JSON serialization errors
            return {
                k: v 
                for k, v in self._state.items() 
                if not k.startswith("snapshot_")
            }

    @staticmethod
    def _default_lamp_state() -> dict[str, str]:
        return {cam_id: "IDLE" for cam_id in MULTICAM_CAMERA_ORDER}

    def _build_clip_queue(self, clips: list[Path]) -> list[dict[str, str]]:
        queue: list[dict[str, str]] = []
        for clip in clips:
            cam_id = extract_multicam_camera_id(clip) or "unknown"
            clip_dt, _, _ = multicam_sort_key(clip)
            queue.append(
                {
                    "clip_name": clip.name,
                    "clip_stem": clip.stem,
                    "cam_id": cam_id,
                    "clip_time": clip_dt.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        return queue

    def _count_clips_per_camera(self, clips: list[Path]) -> dict[str, int]:
        """Count how many clips each camera has for tracking multi-clip completion."""
        counts: dict[str, int] = {cam: 0 for cam in MULTICAM_CAMERA_ORDER}
        for clip in clips:
            cam_id = extract_multicam_camera_id(clip) or "unknown"
            if cam_id in counts:
                counts[cam_id] += 1
        return counts

    def _emit_lamp_state(self, socketio, **kwargs) -> None:
        state = self.status()
        payload = {
            "lamp_state": state.get("lamp_state", self._default_lamp_state()),
            "active_camera": state.get("active_camera", ""),
            "current_clip": state.get("current_clip", ""),
            "clips_done": state.get("clips_done", 0),
            "clips_total": state.get("clips_total", 0),
            "clip_queue": state.get("clip_queue", []),
        }
        payload.update(kwargs)
        socketio.emit("pose_lamp_state", payload)

    def _run(self, clips: list[Path], output_dir: Path, model_path: Path, save_overlay: bool) -> None:
        from app import socketio

        started_at = time.time()
        try:
            model = self._load_model(model_path)
        except Exception as exc:
            self._update_state(running=False, error=str(exc))
            socketio.emit("error", {"source": "phase3", "message": str(exc)})
            return

        self._emit_lamp_state(socketio)

        clips_done = 0
        keypoints_written = 0
        adl_events = 0
        
        # Track clips per camera for proper DONE state (only when ALL clips for that camera done)
        clips_per_camera = self._count_clips_per_camera(clips)
        clips_processed_per_camera: dict[str, int] = {cam: 0 for cam in MULTICAM_CAMERA_ORDER}

        for clip in clips:
            if self._stop_evt.is_set():
                logger.info("Phase 3 interrupted by user")
                break

            cam_id = extract_multicam_camera_id(clip) or ""
            lamp_state = dict(self.status().get("lamp_state", self._default_lamp_state()))
            if cam_id in lamp_state:
                lamp_state[cam_id] = "ACTIVE"
            self._update_state(current_clip=clip.name, active_camera=cam_id, lamp_state=lamp_state)
            self._emit_lamp_state(socketio, current_clip=clip.name, active_camera=cam_id)
            logger.info("Processing pose clip: %s", clip.name)

            try:
                clip_keypoints, clip_adl = self._process_clip(model, clip, output_dir, save_overlay, socketio)
                keypoints_written += clip_keypoints
                adl_events += clip_adl
            except Exception as exc:
                lamp_state = dict(self.status().get("lamp_state", self._default_lamp_state()))
                if cam_id in lamp_state:
                    lamp_state[cam_id] = "ALERT"
                self._update_state(lamp_state=lamp_state, error=str(exc))
                self._emit_lamp_state(socketio, current_clip=clip.name, active_camera=cam_id)
                logger.exception("Error while processing pose clip %s", clip.name)
                socketio.emit("error", {"source": "phase3", "clip": clip.name, "message": str(exc)})

            clips_done += 1
            if cam_id in clips_processed_per_camera:
                clips_processed_per_camera[cam_id] += 1
            
            lamp_state = dict(self.status().get("lamp_state", self._default_lamp_state()))
            # Only mark camera DONE if all its clips are processed and no errors
            if cam_id in lamp_state and lamp_state[cam_id] != "ALERT":
                if clips_processed_per_camera.get(cam_id, 0) >= clips_per_camera.get(cam_id, 1):
                    lamp_state[cam_id] = "DONE"
                else:
                    lamp_state[cam_id] = "ACTIVE"
            
            self._update_state(
                clips_done=clips_done,
                keypoints_written=keypoints_written,
                adl_events=adl_events,
                progress_pct=round((clips_done / max(len(clips), 1)) * 100),
                lamp_state=lamp_state,
            )
            self._emit_lamp_state(socketio, current_clip=clip.name, active_camera=cam_id)

        elapsed_s = round(time.time() - started_at, 2)
        self._update_state(running=False, progress_pct=100 if clips else 0, active_camera="")
        self._emit_lamp_state(socketio, active_camera="")
        
        # ── Run cross-camera ID merger to reduce IDs across clips ──────────────────────
        if clips and output_dir.exists():
            try:
                logger.info("Running cross-camera ID merger...")
                merger = CrossCameraIDMerger(output_dir, clips)
                merger.merge()
                logger.info(f"ID merger complete: reduced to {merger.next_global_id - 1} global IDs")
            except Exception as e:
                logger.warning(f"Cross-camera ID merger failed: {e}")

        socketio.emit(
            "pose_complete",
            {
                "clips_done": clips_done,
                "keypoints_written": keypoints_written,
                "adl_events": adl_events,
                "elapsed_s": elapsed_s,
                "lamp_state": self.status().get("lamp_state", self._default_lamp_state()),
                "clip_queue": self.status().get("clip_queue", []),
            },
        )
        # Clear snapshot on complete
        self._update_state(snapshot_original=None, snapshot_processed=None)

        logger.info(
            "Phase 3 complete: %d clips, %d keypoint rows, %d ADL events",
            clips_done,
            keypoints_written,
            adl_events,
        )

    @staticmethod
    def _load_model(model_path: Path):
        from ultralytics import YOLO

        logger.info("Loading Phase 3 model from %s", model_path)
        return YOLO(str(model_path))

    def _process_clip(self, model, clip: Path, output_dir: Path, save_overlay: bool, socketio) -> tuple[int, int]:
        started_at = time.time()
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise OSError(f"Cannot open clip: {clip}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        clip_output_dir = output_dir / clip.stem
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        
        # New: Video output process
        from app import BASE_DIR
        out_process_dir = BASE_DIR / "data" / "output_process" / clip.stem
        out_process_dir.mkdir(parents=True, exist_ok=True)
        out_vid_path = out_process_dir / f"{clip.stem}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

        keypoints_file = clip_output_dir / f"{clip.stem}_keypoints.txt"
        adl_file = clip_output_dir / f"{clip.stem}_adl.txt"

        window_size = max(1, int(self._config.get("window_size", WINDOW_SIZE)))
        min_conf = float(self._config.get("keypoint_conf_min", KP_CONF_MIN))
        person_windows = defaultdict(lambda: deque(maxlen=window_size))
        tracker = _SequentialTracker(
            iou_threshold=float(self._config.get("track_iou_threshold", 0.20)),
            max_missed_frames=int(self._config.get("track_max_missed_frames", max(15, window_size // 2))),
            center_distance_ratio=float(self._config.get("track_center_distance_ratio", 0.18)),
        )
        overlay_every = max(1, int(self._config.get("overlay_every", 30)))
        latest_adl_by_track: dict[int, tuple[str, float]] = {}

        keypoints_written = 0
        adl_events = 0
        frame_id = 0

        with keypoints_file.open("w", encoding="utf-8") as kp_handle, adl_file.open("w", encoding="utf-8") as adl_handle:
            kp_handle.write(
                "# frame_id track_id "
                "kp0_x kp0_y kp0_conf kp1_x kp1_y kp1_conf ... kp16_x kp16_y kp16_conf\n"
            )
            adl_handle.write("# frame_id track_id adl_label confidence\n")

            while True:
                try:
                    if self._stop_evt.is_set():
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    detections = self._predict_people(model, frame)
                    tracked_people, expired_track_ids = tracker.update(detections, frame.shape[:2], frame_id)
                    for expired_track_id in expired_track_ids:
                        person_windows.pop(expired_track_id, None)
                        latest_adl_by_track.pop(expired_track_id, None)

                    tracked_people.sort(key=lambda item: item.track_id)
                    persons_detected = len(tracked_people)

                    should_save_overlay = save_overlay and (frame_id % overlay_every == 0)
                    overlay_frame = frame.copy() if should_save_overlay else None

                    for tracked_person in tracked_people:
                        track_id = tracked_person.track_id
                        person_xy = tracked_person.keypoints_xy
                        person_conf = tracked_person.keypoints_conf
                        flattened = []
                        for idx in range(len(person_xy)):
                            flattened.extend(
                                [
                                    f"{float(person_xy[idx][0]):.1f}",
                                    f"{float(person_xy[idx][1]):.1f}",
                                    f"{float(person_conf[idx]):.2f}",
                                ]
                            )
                        kp_handle.write(f"{frame_id} {track_id} {' '.join(flattened)}\n")
                        keypoints_written += 1

                        track_window = person_windows[track_id]
                        track_window.append((np.asarray(person_xy), np.asarray(person_conf)))
                        if len(track_window) == window_size:
                            label, confidence = rule_based_adl(list(track_window), self._config)
                            latest_adl_by_track[track_id] = (label, confidence)
                            adl_handle.write(f"{frame_id} {track_id} {label} {confidence:.2f}\n")
                            adl_events += 1
                        else:
                            latest_adl_by_track.setdefault(track_id, ("unknown", 0.0))

                        if overlay_frame is not None:
                            overlay_frame = draw_skeleton(overlay_frame, person_xy, person_conf, min_conf=min_conf)
                            _draw_track_label(
                                overlay_frame,
                                tracked_person.bbox,
                                track_id,
                                latest_adl_by_track.get(track_id),
                            )

                    process_frame_out = overlay_frame if overlay_frame is not None else frame.copy()

                    if overlay_frame is not None and persons_detected > 0:
                        overlay_path = clip_output_dir / f"{clip.stem}_overlay_{frame_id:04d}.png"
                        cv2.imwrite(str(overlay_path), process_frame_out)
                        
                    if out_video is not None:
                        out_video.write(process_frame_out)

                    jpeg_orig = cv2.imencode('.jpg', frame)[1].tobytes()
                    jpeg_proc = cv2.imencode('.jpg', process_frame_out)[1].tobytes()
                    self._update_state(snapshot_original=jpeg_orig, snapshot_processed=jpeg_proc)

                    if frame_id % PROGRESS_EVERY == 0:
                        pct = round(((frame_id + 1) / max(total_frames, 1)) * 100)
                        now_time = time.time()
                        fps_val = round(frame_id / max(now_time - started_at, 0.1), 1)
                        pose_state = self.status()
                        socketio.emit(
                            "pose_progress",
                            {
                                "clip": clip.stem,
                                "current_clip": pose_state.get("current_clip", clip.name),
                                "frame_id": frame_id,
                                "total_frames": total_frames,
                                "pct": pct,
                                "fps": fps_val,
                                "persons_detected": persons_detected,
                                "active_tracks": persons_detected,
                                "keypoints_written": keypoints_written,
                                "adl_events": adl_events,
                                "lamp_state": pose_state.get("lamp_state", self._default_lamp_state()),
                                "active_camera": pose_state.get("active_camera", ""),
                                "clips_done": pose_state.get("clips_done", 0),
                                "clips_total": pose_state.get("clips_total", 0),
                                "clip_queue": pose_state.get("clip_queue", []),
                            },
                        )
                        self._update_state(progress_pct=pct)

                    frame_id += 1

                except Exception as e:
                    logger.error(f"Error processing frame {frame_id} of {clip.name}: {e}")
                    # Skip the bad frame and continue
                    frame_id += 1
                    continue

        if out_video:
            out_video.release()
        cap.release()
        logger.info("%s -> %d keypoint rows, %d ADL rows", clip.stem, keypoints_written, adl_events)
        return keypoints_written, adl_events

    def _predict_people(self, model, frame) -> list[dict[str, Any]]:
        results = model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=float(self._config.get("conf_threshold", CONF_THRESHOLD_P3)),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []

        keypoints_xy = result.keypoints.xy.cpu().numpy()
        conf_tensor = result.keypoints.conf
        if conf_tensor is None:
            keypoints_conf = np.ones((len(keypoints_xy), 17), dtype=float)
        else:
            keypoints_conf = conf_tensor.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        box_conf_tensor = result.boxes.conf
        box_conf = box_conf_tensor.cpu().numpy() if box_conf_tensor is not None else np.ones(len(boxes), dtype=float)

        limit = min(len(keypoints_xy), len(keypoints_conf), len(boxes))
        detections: list[dict[str, Any]] = []
        for idx in range(limit):
            detections.append(
                {
                    "bbox": boxes[idx],
                    "keypoints_xy": keypoints_xy[idx],
                    "keypoints_conf": keypoints_conf[idx],
                    "detection_conf": float(box_conf[idx]) if idx < len(box_conf) else 0.0,
                }
            )

        return detections

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}

        # Unified config format: extract phase3 section.
        phase3_cfg = data.get("phase3")
        if isinstance(phase3_cfg, dict):
            merged = dict(phase3_cfg)

            # Backward-compatible bridge from old pose_adl-style flat keys.
            if "save_overlay" in data and "save_overlay" not in merged:
                merged["save_overlay"] = bool(data.get("save_overlay"))
            if isinstance(data.get("adl_classes"), list) and "adl_classes" not in merged:
                merged["adl_classes"] = list(data["adl_classes"])
            if isinstance(data.get("thresholds"), dict) and "thresholds" not in merged:
                merged["thresholds"] = dict(data["thresholds"])
            return merged

        # Legacy pose_adl format: return whole dict.
        return data

    def _update_state(self, **kwargs) -> None:
        with self._lock:
            self._state.update(kwargs)
