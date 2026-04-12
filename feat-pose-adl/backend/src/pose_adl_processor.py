"""
Single-clip pose + ADL processor used by the realtime collection backend.

This module keeps the pose/ADL logic inside `backend/src` so the old
computer-vision code remains the source of truth, while the realtime backend
can import and run it as a background job for each saved clip.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

import cv2
import torch
import yaml
from ultralytics import YOLO

from adl import ADLConfig, TrackState, classify_posture
from visualize import POSTURE_COLORS, draw_skeleton

SRC_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SRC_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "pose_adl.yaml"
DEFAULT_POSE_MODEL_PATH = REPO_ROOT / "models" / "yolo11n-pose.pt"


@dataclass(slots=True)
class ClipProcessingResult:
    input_path: Path
    output_path: Path
    total_frames: int
    processed_frames: int
    posture_counts: dict[str, int]


class PoseAdlClipProcessor:
    """Annotate a single MP4 clip with pose skeletons and ADL labels."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config()

        ADLConfig.from_dict(self._config.get("adl", {}))

        yolo_cfg = self._config.get("yolo", {})
        self._model_path = self._resolve_path(
            yolo_cfg.get("model_path"),
            default=DEFAULT_POSE_MODEL_PATH,
            fallback=BACKEND_DIR / "models" / "yolo11n-pose.pt",
        )
        self._conf_threshold = float(yolo_cfg.get("conf_threshold", 0.5))
        self._iou_threshold = float(yolo_cfg.get("iou_threshold", 0.7))
        self._device = self._resolve_device(yolo_cfg.get("device"))

        reid_cfg = self._config.get("reid", {})
        self._min_keypoints = int(reid_cfg.get("min_keypoints", 8))

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        stop_event: Optional[Event] = None,
        frame_callback=None,
    ) -> ClipProcessingResult:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Input clip not found: {input_path}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open input clip: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to create output clip: {output_path}")

        model = YOLO(str(self._model_path))
        track_states: dict[int, TrackState] = {}
        posture_counts: Counter[str] = Counter()
        processed_frames = 0
        frame_idx = 0

        try:
            while True:
                if stop_event and stop_event.is_set():
                    raise RuntimeError("Pose/ADL processing interrupted")

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                annotated = frame.copy()
                current_track_ids: set[int] = set()

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=self._conf_threshold,
                    iou=self._iou_threshold,
                    device=self._device,
                )

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints = results[0].keypoints
                    current_track_ids = set(track_ids.tolist())

                    for idx, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[idx]
                        bbox = (float(x1), float(y1), float(x2), float(y2))
                        state = track_states.setdefault(track_id, TrackState(track_id, height))
                        state.update_position(bbox)

                        posture = ""
                        kpts = None
                        if keypoints is not None:
                            kpts = keypoints.data[idx].cpu().numpy()
                            visible_keypoints = sum(1 for kp in kpts if kp[2] > ADLConfig.KEYPOINT_CONF)
                            if visible_keypoints >= self._min_keypoints:
                                old_posture = state.current_posture
                                posture = classify_posture(kpts, bbox, state, height)
                                state.add_posture(posture)
                                if state.current_posture and state.current_posture != old_posture:
                                    posture_counts[state.current_posture] += 1

                        label = state.current_posture or "TRACKING"
                        color = POSTURE_COLORS.get(state.current_posture, (80, 200, 80))
                        p1 = (max(0, int(x1)), max(0, int(y1)))
                        p2 = (min(width - 1, int(x2)), min(height - 1, int(y2)))

                        cv2.rectangle(annotated, p1, p2, color, 2)
                        if kpts is not None:
                            draw_skeleton(annotated, kpts, colorful=True)

                        tag = f"G{track_id} | {label}"
                        (text_w, text_h), _ = cv2.getTextSize(
                            tag,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            2,
                        )
                        text_y = p1[1] - 10 if p1[1] - 10 > text_h else p1[1] + text_h + 10
                        cv2.rectangle(
                            annotated,
                            (p1[0], text_y - text_h - 8),
                            (p1[0] + text_w + 10, text_y + 2),
                            color,
                            -1,
                        )
                        cv2.putText(
                            annotated,
                            tag,
                            (p1[0] + 5, text_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )

                lost_tracks = set(track_states.keys()) - current_track_ids
                for track_id in lost_tracks:
                    del track_states[track_id]

                self._draw_header(
                    frame=annotated,
                    clip_name=input_path.name,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    posture_counts=posture_counts,
                )
                if frame_callback is not None:
                    try:
                        frame_callback(annotated, frame_idx, total_frames, posture_counts)
                    except Exception:
                        pass
                writer.write(annotated)
                processed_frames += 1
        except Exception:
            writer.release()
            cap.release()
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            raise

        writer.release()
        cap.release()

        return ClipProcessingResult(
            input_path=input_path,
            output_path=output_path,
            total_frames=total_frames,
            processed_frames=processed_frames,
            posture_counts=dict(posture_counts),
        )

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _resolve_path(
        configured_path: Optional[str],
        default: Path,
        fallback: Optional[Path] = None,
    ) -> Path:
        candidates: list[Path] = []
        if configured_path:
            path = Path(configured_path)
            candidates.append(path if path.is_absolute() else REPO_ROOT / path)
        candidates.append(default)
        if fallback is not None:
            candidates.append(fallback)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _resolve_device(requested_device: Optional[str]) -> Optional[str]:
        if not requested_device:
            return None

        device = requested_device.lower()
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"

        if device == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not has_mps:
                return "cpu"

        return requested_device

    @staticmethod
    def _draw_header(
        frame,
        clip_name: str,
        frame_idx: int,
        total_frames: int,
        posture_counts: Counter[str],
    ) -> None:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 84), (24, 28, 36), -1)
        cv2.putText(
            frame,
            "POSE + ADL PIPELINE",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (78, 204, 163),
            2,
        )
        cv2.putText(
            frame,
            clip_name,
            (18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (228, 232, 240),
            1,
        )

        progress_text = f"Frame {frame_idx}/{total_frames if total_frames else '?'}"
        cv2.putText(
            frame,
            progress_text,
            (width - 260, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (228, 232, 240),
            2,
        )

        counts_text = " | ".join(
            f"{name}: {count}" for name, count in sorted(posture_counts.items())
        ) or "No ADL labels yet"
        cv2.putText(
            frame,
            counts_text[:100],
            (width - 470, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 190, 200),
            1,
        )
