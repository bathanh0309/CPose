"""
CPose — app/services/phase2_analyzer.py
Phase 2: Offline high-accuracy analysis with YOLOv8l.

For each clip:
  - Read every frame
  - Run YOLOv8l detection (person only)
  - If detections found: save PNG + append to label file
  
Label format (space-separated, one bbox per line):
  frame_id  x_min  y_min  x_max  y_max
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger("[Phase2]")

PERSON_CLASS_ID = 0
CONF_THRESHOLD  = 0.50    # higher confidence for precise analysis


class Analyzer:
    """Offline Phase 2 analysis manager."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_evt  = threading.Event()
        self._lock      = threading.Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "current_clip": "",
            "clips_total": 0,
            "clips_done": 0,
            "frames_saved": 0,
            "labels_written": 0,
            "progress_pct": 0,
            "error": None,
        }

    # ──────────────────────────────────────────────────────────────────────
    def start(self, clips: list[Path], output_dir: Path, model_path: Path):
        if self._thread and self._thread.is_alive():
            logger.warning("Analyzer already running")
            return

        self._stop_evt.clear()
        self._update_state(
            running=True,
            clips_total=len(clips),
            clips_done=0,
            frames_saved=0,
            labels_written=0,
            progress_pct=0,
            error=None,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(clips, output_dir, model_path),
            daemon=True,
            name="phase2-analyzer",
        )
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        logger.info("Analyzer stop requested")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    # ──────────────────────────────────────────────────────────────────────
    def _run(self, clips: list[Path], output_dir: Path, model_path: Path):
        from app import socketio

        try:
            model = self._load_model(model_path)
        except Exception as exc:
            self._update_state(running=False, error=str(exc))
            socketio.emit("error", {"source": "phase2", "message": str(exc)})
            return

        clips_done   = 0
        frames_saved = 0
        labels_total = 0

        for clip in clips:
            if self._stop_evt.is_set():
                logger.info("Phase 2 interrupted by user")
                break

            self._update_state(current_clip=clip.name)
            logger.info("Analyzing clip: %s", clip.name)

            try:
                f_saved, l_written = self._analyze_clip(
                    model, clip, output_dir, socketio
                )
                frames_saved += f_saved
                labels_total += l_written
            except Exception as exc:
                logger.error("Error analyzing %s: %s", clip.name, exc)
                socketio.emit("error", {"source": "phase2", "clip": clip.name, "message": str(exc)})

            clips_done += 1
            pct = round(clips_done / len(clips) * 100)
            self._update_state(
                clips_done=clips_done,
                frames_saved=frames_saved,
                labels_written=labels_total,
                progress_pct=pct,
            )
            socketio.emit("analysis_progress", {
                "clip": clip.name,
                "clips_done": clips_done,
                "clips_total": len(clips),
                "frames_saved": frames_saved,
                "pct": pct,
            })

        self._update_state(running=False, progress_pct=100)
        socketio.emit("analysis_complete", {
            "clips_done": clips_done,
            "frames_saved": frames_saved,
            "labels_written": labels_total,
        })
        logger.info(
            "Phase 2 complete — %d clips, %d frames, %d labels",
            clips_done, frames_saved, labels_total,
        )

    # ──────────────────────────────────────────────────────────────────────
    def _load_model(self, model_path: Path):
        from ultralytics import YOLO
        logger.info("Loading heavy model from %s …", model_path)
        model = YOLO(str(model_path))
        logger.info("Heavy model loaded")
        return model

    # ──────────────────────────────────────────────────────────────────────
    def _analyze_clip(
        self,
        model,
        clip: Path,
        output_dir: Path,
        socketio,
    ) -> tuple[int, int]:
        """
        Process one clip.
        Returns (frames_saved, labels_written).
        """
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise IOError(f"Cannot open clip: {clip}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Destination folder: output_labels/<clip_stem>/
        clip_out = output_dir / clip.stem
        clip_out.mkdir(parents=True, exist_ok=True)
        label_file = clip_out / f"{clip.stem}_labels.txt"

        frames_saved = 0
        labels_written = 0
        frame_id = 0

        with open(label_file, "w", encoding="utf-8") as lf:
            lf.write("# frame_id  x_min  y_min  x_max  y_max\n")

            while True:
                if self._stop_evt.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    frame,
                    classes=[PERSON_CLASS_ID],
                    conf=CONF_THRESHOLD,
                    verbose=False,
                )

                detections = []
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == PERSON_CLASS_ID:
                            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                            detections.append((x1, y1, x2, y2))

                if detections:
                    # Save frame image
                    frame_name = f"{clip.stem}_frame_{frame_id:04d}.png"
                    cv2.imwrite(str(clip_out / frame_name), frame)
                    frames_saved += 1

                    # Write labels
                    for (x1, y1, x2, y2) in detections:
                        lf.write(f"{frame_id}  {x1}  {y1}  {x2}  {y2}\n")
                        labels_written += 1

                    # Emit per-frame progress every 30 frames
                    if frames_saved % 30 == 0:
                        socketio.emit("analysis_progress", {
                            "clip": clip.stem,
                            "frame": frame_id,
                            "total_frames": total_frames,
                            "pct": round(frame_id / max(total_frames, 1) * 100),
                        })

                frame_id += 1

        cap.release()
        logger.info(
            "%s → %d frames saved, %d bbox labels",
            clip.stem, frames_saved, labels_written,
        )
        return frames_saved, labels_written

    # ──────────────────────────────────────────────────────────────────────
    def _update_state(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)
