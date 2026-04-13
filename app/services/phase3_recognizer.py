"""Phase 3 offline pose estimation and rule-based ADL recognition."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.utils.runtime_config import get_runtime_section
from app.utils.pose_utils import draw_skeleton, rule_based_adl

logger = logging.getLogger("[Phase3]")

_PHASE3_CFG = get_runtime_section("phase3")

CONF_THRESHOLD_P3 = float(_PHASE3_CFG.get("conf_threshold", 0.45))
KP_CONF_MIN = float(_PHASE3_CFG.get("keypoint_conf_min", 0.30))
WINDOW_SIZE = int(_PHASE3_CFG.get("window_size", 30))
PROGRESS_EVERY = int(_PHASE3_CFG.get("progress_every", 10))
PERSON_CLASS_ID = int(_PHASE3_CFG.get("person_class_id", 0))


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
            "clips_total": 0,
            "clips_done": 0,
            "keypoints_written": 0,
            "adl_events": 0,
            "progress_pct": 0,
            "save_overlay": True,
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

        self._stop_evt.clear()
        self._config = self._load_config(config_path)
        self._update_state(
            running=True,
            current_clip="",
            clips_total=len(clips),
            clips_done=0,
            keypoints_written=0,
            adl_events=0,
            progress_pct=0,
            save_overlay=save_overlay,
            error=None,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(list(clips), output_dir, model_path, save_overlay),
            daemon=True,
            name="phase3-recognizer",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("Phase 3 stop requested")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _run(self, clips: list[Path], output_dir: Path, model_path: Path, save_overlay: bool) -> None:
        from app import socketio

        started_at = time.time()
        try:
            model = self._load_model(model_path)
        except Exception as exc:
            self._update_state(running=False, error=str(exc))
            socketio.emit("error", {"source": "phase3", "message": str(exc)})
            return

        clips_done = 0
        keypoints_written = 0
        adl_events = 0

        for clip in sorted(clips):
            if self._stop_evt.is_set():
                logger.info("Phase 3 interrupted by user")
                break

            self._update_state(current_clip=clip.name)
            logger.info("Processing pose clip: %s", clip.name)

            try:
                clip_keypoints, clip_adl = self._process_clip(model, clip, output_dir, save_overlay, socketio)
                keypoints_written += clip_keypoints
                adl_events += clip_adl
            except Exception as exc:
                logger.exception("Error while processing pose clip %s", clip.name)
                socketio.emit("error", {"source": "phase3", "clip": clip.name, "message": str(exc)})

            clips_done += 1
            self._update_state(
                clips_done=clips_done,
                keypoints_written=keypoints_written,
                adl_events=adl_events,
                progress_pct=round((clips_done / max(len(clips), 1)) * 100),
            )

        elapsed_s = round(time.time() - started_at, 2)
        self._update_state(running=False, progress_pct=100 if clips else 0)
        socketio.emit(
            "pose_complete",
            {
                "clips_done": clips_done,
                "keypoints_written": keypoints_written,
                "adl_events": adl_events,
                "elapsed_s": elapsed_s,
            },
        )
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
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise OSError(f"Cannot open clip: {clip}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        clip_output_dir = output_dir / clip.stem
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        keypoints_file = clip_output_dir / f"{clip.stem}_keypoints.txt"
        adl_file = clip_output_dir / f"{clip.stem}_adl.txt"

        window_size = int(self._config.get("window_size", WINDOW_SIZE))
        min_conf = float(self._config.get("keypoint_conf_min", KP_CONF_MIN))
        person_windows = defaultdict(lambda: deque(maxlen=window_size))

        keypoints_written = 0
        adl_events = 0
        frame_id = 0

        with keypoints_file.open("w", encoding="utf-8") as kp_handle, adl_file.open("w", encoding="utf-8") as adl_handle:
            kp_handle.write(
                "# frame_id person_id "
                "kp0_x kp0_y kp0_conf kp1_x kp1_y kp1_conf ... kp16_x kp16_y kp16_conf\n"
            )
            adl_handle.write("# frame_id person_id adl_label confidence\n")

            while True:
                if self._stop_evt.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                overlay_frame = frame.copy() if save_overlay else None
                people_xy, people_conf = self._predict_people(model, frame)
                persons_detected = len(people_xy)

                for person_id, (person_xy, person_conf) in enumerate(zip(people_xy, people_conf)):
                    flattened = []
                    for idx in range(len(person_xy)):
                        flattened.extend(
                            [
                                f"{float(person_xy[idx][0]):.1f}",
                                f"{float(person_xy[idx][1]):.1f}",
                                f"{float(person_conf[idx]):.2f}",
                            ]
                        )
                    kp_handle.write(f"{frame_id} {person_id} {' '.join(flattened)}\n")
                    keypoints_written += 1

                    person_windows[person_id].append((np.asarray(person_xy), np.asarray(person_conf)))
                    if len(person_windows[person_id]) == window_size:
                        label, confidence = rule_based_adl(list(person_windows[person_id]), self._config)
                        adl_handle.write(f"{frame_id} {person_id} {label} {confidence:.2f}\n")
                        adl_events += 1

                    if overlay_frame is not None:
                        overlay_frame = draw_skeleton(overlay_frame, person_xy, person_conf, min_conf=min_conf)

                if overlay_frame is not None and persons_detected > 0:
                    overlay_path = clip_output_dir / f"{clip.stem}_overlay_{frame_id:04d}.png"
                    cv2.imwrite(str(overlay_path), overlay_frame)

                if frame_id % PROGRESS_EVERY == 0:
                    pct = round(((frame_id + 1) / max(total_frames, 1)) * 100)
                    socketio.emit(
                        "pose_progress",
                        {
                            "clip": clip.stem,
                            "frame_id": frame_id,
                            "total_frames": total_frames,
                            "pct": pct,
                            "persons_detected": persons_detected,
                        },
                    )
                    self._update_state(progress_pct=pct)

                frame_id += 1

        cap.release()
        logger.info("%s -> %d keypoint rows, %d ADL rows", clip.stem, keypoints_written, adl_events)
        return keypoints_written, adl_events

    def _predict_people(self, model, frame) -> tuple[list[np.ndarray], list[np.ndarray]]:
        results = model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=float(self._config.get("conf_threshold", CONF_THRESHOLD_P3)),
            verbose=False,
        )
        if not results:
            return [], []

        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return [], []

        keypoints_xy = result.keypoints.xy.cpu().numpy()
        conf_tensor = result.keypoints.conf
        if conf_tensor is None:
            keypoints_conf = np.ones((len(keypoints_xy), 17), dtype=float)
        else:
            keypoints_conf = conf_tensor.cpu().numpy()
        return list(keypoints_xy), list(keypoints_conf)

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
