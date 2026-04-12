"""Phase 2 offline analyzer for PNG and bounding-box label export."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger("[Phase2]")

PERSON_CLASS_ID = 0
CONF_THRESHOLD_P2 = 0.50
PROGRESS_EVERY = 10


class Analyzer:
    """Run offline analysis over saved MP4 clips."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
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

    def start(self, clips: list[Path], output_dir: Path, model_path: Path) -> None:
        if self.is_running():
            logger.warning("Analyzer is already running")
            return

        self._stop_evt.clear()
        self._update_state(
            running=True,
            current_clip="",
            clips_total=len(clips),
            clips_done=0,
            frames_saved=0,
            labels_written=0,
            progress_pct=0,
            error=None,
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(list(clips), output_dir, model_path),
            daemon=True,
            name="phase2-analyzer",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        logger.info("Phase 2 stop requested")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _run(self, clips: list[Path], output_dir: Path, model_path: Path) -> None:
        from app import socketio

        started_at = time.time()
        try:
            model = self._load_model(model_path)
        except Exception as exc:
            self._update_state(running=False, error=str(exc))
            socketio.emit("error", {"source": "phase2", "message": str(exc)})
            return

        clips_done = 0
        frames_saved = 0
        labels_written = 0

        for clip in sorted(clips):
            if self._stop_evt.is_set():
                logger.info("Phase 2 interrupted by user")
                break

            self._update_state(current_clip=clip.name)
            logger.info("Analyzing clip: %s", clip.name)

            try:
                clip_frames, clip_labels = self._analyze_clip(model, clip, output_dir, socketio)
                frames_saved += clip_frames
                labels_written += clip_labels
            except Exception as exc:
                logger.exception("Error while analyzing %s", clip.name)
                socketio.emit("error", {"source": "phase2", "clip": clip.name, "message": str(exc)})

            clips_done += 1
            overall_pct = round((clips_done / max(len(clips), 1)) * 100)
            self._update_state(
                clips_done=clips_done,
                frames_saved=frames_saved,
                labels_written=labels_written,
                progress_pct=overall_pct,
            )

        elapsed_s = round(time.time() - started_at, 2)
        self._update_state(running=False, progress_pct=100 if clips else 0)
        socketio.emit(
            "analysis_complete",
            {
                "clips_done": clips_done,
                "frames_saved": frames_saved,
                "labels_written": labels_written,
                "elapsed_s": elapsed_s,
            },
        )
        logger.info(
            "Phase 2 complete: %d clips, %d frames saved, %d labels written",
            clips_done,
            frames_saved,
            labels_written,
        )

    @staticmethod
    def _load_model(model_path: Path):
        from ultralytics import YOLO

        logger.info("Loading Phase 2 model from %s", model_path)
        return YOLO(str(model_path))

    def _analyze_clip(self, model, clip: Path, output_dir: Path, socketio) -> tuple[int, int]:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise OSError(f"Cannot open clip: {clip}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        clip_output_dir = output_dir / clip.stem
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        label_file = clip_output_dir / f"{clip.stem}_labels.txt"

        frames_saved = 0
        labels_written = 0
        frame_id = 0

        with label_file.open("w", encoding="utf-8") as handle:
            handle.write("# frame_id x_min y_min x_max y_max\n")

            while True:
                if self._stop_evt.is_set():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                detections: list[tuple[int, int, int, int]] = []
                results = model.predict(
                    frame,
                    classes=[PERSON_CLASS_ID],
                    conf=CONF_THRESHOLD_P2,
                    verbose=False,
                )
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) != PERSON_CLASS_ID:
                            continue
                        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
                        detections.append((x1, y1, x2, y2))

                if detections:
                    frame_name = f"{clip.stem}_frame_{frame_id:04d}.png"
                    cv2.imwrite(str(clip_output_dir / frame_name), frame)
                    frames_saved += 1
                    for x1, y1, x2, y2 in detections:
                        handle.write(f"{frame_id} {x1} {y1} {x2} {y2}\n")
                        labels_written += 1

                if frame_id % PROGRESS_EVERY == 0:
                    pct = round(((frame_id + 1) / max(total_frames, 1)) * 100)
                    socketio.emit(
                        "analysis_progress",
                        {
                            "clip": clip.stem,
                            "frame_id": frame_id,
                            "total_frames": total_frames,
                            "pct": pct,
                            "frames_saved": frames_saved,
                            "labels_written": labels_written,
                        },
                    )

                frame_id += 1

        cap.release()
        logger.info("%s -> %d PNG frames, %d labels", clip.stem, frames_saved, labels_written)
        return frames_saved, labels_written

    def _update_state(self, **kwargs) -> None:
        with self._lock:
            self._state.update(kwargs)
