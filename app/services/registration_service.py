from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from app.setting import FACE_DATA_PATH, FACE_MIN_SIZE

logger = logging.getLogger("[Registration]")


class RegistrationSession:
    def __init__(
        self,
        session_id: str,
        source: str,
        user_name: str,
        user_id: str,
        user_age: str,
        on_progress: Callable[[Dict[str, Any]], None],
        on_done: Callable[[Dict[str, Any]], None],
        detector_model: Any = None,
    ):
        self.session_id = session_id
        self.source = source
        self.user_name = user_name
        self.user_id = user_id
        self.user_age = user_age
        self.on_progress = on_progress
        self.on_done = on_done

        self.running = False
        self.stop_requested = False
        self.done = False
        self.error: str | None = None
        self.thread = None
        self.cap = None

        self.target_count = 3
        self.angles = ["center", "left", "right", "up", "down"]
        self.counts = {angle: 0 for angle in self.angles}
        self.embeddings = {angle: [] for angle in self.angles}

        self.current_frame = None
        self.current_output_frame = None
        self.latest_jpeg = None
        self.progress = 0
        self.last_instruction = ""

        if detector_model is not None:
            self.model = detector_model
        else:
            if FaceAnalysis is None:
                raise RuntimeError("FaceAnalysis is not available")
            self.model = FaceAnalysis(name="buffalo_s")
            self.model.prepare(ctx_id=0, det_size=(640, 640))

        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.running = True
            self.stop_requested = False
            self.done = False
            self.error = None
            self.progress = 0
            self.latest_jpeg = self._encode_jpeg(
                self._build_status_frame("Starting registration...")
            )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(
            "Registration session %s started for %s (ID: %s, Age: %s)",
            self.session_id,
            self.user_name,
            self.user_id,
            self.user_age,
        )

    def stop(self):
        self.running = False
        self.stop_requested = True
        if self.thread and threading.current_thread() != self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.done = True
        logger.info("Registration session %s stopped", self.session_id)

    def get_snapshot(self):
        with self.lock:
            if self.latest_jpeg is not None:
                return self.latest_jpeg
            if self.current_output_frame is not None:
                self.latest_jpeg = self._encode_jpeg(self.current_output_frame)
                return self.latest_jpeg
        return None

    def _estimate_angle(self, face):
        if not hasattr(face, "pose"):
            return "unknown"

        yaw, pitch, roll = face.pose

        yaw_thres = 25
        pitch_thres = 18

        if abs(yaw) < yaw_thres and abs(pitch) < pitch_thres:
            return "center"
        if yaw < -yaw_thres:
            return "left"
        if yaw > yaw_thres:
            return "right"
        if pitch < -pitch_thres:
            return "up"
        if pitch > pitch_thres:
            return "down"

        return "mid"

    def _is_blurry(self, image, threshold=60):
        if image is None or image.size == 0:
            return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def _build_status_frame(self, message: str):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (639, 479), (40, 40, 40), 2)
        cv2.putText(frame, "Registration", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 180), 2)
        cv2.putText(frame, f"ID: {self.user_id}", (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Name: {self.user_name}", (24, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Age: {self.user_age}", (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, message, (24, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 180), 2)
        return frame

    def _open_capture(self, source_val):
        candidates = []
        if isinstance(source_val, int):
            if hasattr(cv2, "CAP_DSHOW"):
                candidates.append(lambda: cv2.VideoCapture(source_val, cv2.CAP_DSHOW))
            candidates.append(lambda: cv2.VideoCapture(source_val))
        else:
            candidates.append(lambda: cv2.VideoCapture(source_val))

        for opener in candidates:
            cap = opener()
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return None

    def _run(self):
        try:
            source_val = int(self.source) if self.source.isdigit() else (0 if self.source == "local" else self.source)
        except Exception:
            source_val = self.source

        self.cap = self._open_capture(source_val)
        if not self.cap or not self.cap.isOpened():
            logger.error("Failed to open source %s", self.source)
            error_msg = f"Cannot open camera source: {self.source}"
            with self.lock:
                self.error = error_msg
                self.latest_jpeg = self._encode_jpeg(self._build_status_frame(error_msg))
                self.done = True
            self.running = False
            self.on_done(
                {
                    "status": "error",
                    "message": error_msg,
                    "session_id": self.session_id,
                    "person_id": self.user_id,
                    "name": self.user_name,
                    "age": self.user_age,
                }
            )
            return

        consecutive_failures = 0
        while self.running and not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    with self.lock:
                        self.latest_jpeg = self._encode_jpeg(self._build_status_frame("Waiting for camera..."))
                if consecutive_failures >= 120:
                    error_msg = f"Cannot read frames from source: {self.source}"
                    logger.error(error_msg)
                    with self.lock:
                        self.error = error_msg
                        self.latest_jpeg = self._encode_jpeg(self._build_status_frame(error_msg))
                        self.done = True
                    self.running = False
                    self.on_done(
                        {
                            "status": "error",
                            "message": error_msg,
                            "session_id": self.session_id,
                            "person_id": self.user_id,
                            "name": self.user_name,
                            "age": self.user_age,
                        }
                    )
                    break
                time.sleep(0.03)
                continue

            consecutive_failures = 0

            h, w = frame.shape[:2]
            box_size = min(h, w) // 2
            x1, y1 = (w - box_size) // 2, (h - box_size) // 2
            x2, y2 = x1 + box_size, y1 + box_size

            try:
                faces = self.model.get(frame)
            except Exception as exc:
                logger.warning(
                    "Registration detector failed on session %s: %s",
                    self.session_id,
                    exc,
                    exc_info=True,
                )
                faces = []

            output_frame = frame.copy()
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            instruction = "Waiting for face..."
            target_angle = self._get_next_needed_angle()

            if not target_angle:
                instruction = "Registration complete..."
                self.running = False
                self._finalize()
                break

            if len(faces) == 0:
                instruction = "Please center your face"
            elif len(faces) > 1:
                instruction = "Only one person in frame"
            else:
                face = faces[0]
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                cv2.rectangle(output_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                info_str = f"ID:{self.user_id} | {self.user_name} ({self.user_age})"
                cv2.putText(output_frame, info_str, (fx1, max(20, fy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                is_inside = fx1 > x1 and fy1 > y1 and fx2 < x2 and fy2 < y2
                if not is_inside:
                    instruction = "Move face into the box"
                elif self._is_blurry(frame[fy1:fy2, fx1:fx2]):
                    instruction = "Hold still, face is blurry"
                else:
                    angle = self._estimate_angle(face)
                    if angle == target_angle:
                        emb = face.embedding
                        emb = emb / np.linalg.norm(emb)
                        self.embeddings[angle].append(emb)
                        self.counts[angle] += 1

                        instruction = f"Capturing {angle.upper()}: {self.counts[angle]}/{self.target_count}"
                        total_samples = sum(self.counts.values())
                        max_samples = self.target_count * len(self.angles)
                        progress_pct = int((total_samples / max_samples) * 100)
                        self.progress = progress_pct

                        self.on_progress(
                            {
                                "session_id": self.session_id,
                                "progress": progress_pct,
                                "message": instruction,
                                "instruction": instruction,
                                "conf": float(face.det_score) if hasattr(face, "det_score") else 0.95,
                                "angle": angle,
                                "count": self.counts[angle],
                            }
                        )
                    else:
                        instruction = self._get_instruction_for_angle(target_angle)

            with self.lock:
                self.current_frame = frame
                self.current_output_frame = output_frame
                self.last_instruction = instruction
                self.latest_jpeg = self._encode_jpeg(output_frame)

            time.sleep(0.01)

        if self.cap:
            self.cap.release()
            self.cap = None

    def _get_next_needed_angle(self):
        for angle in self.angles:
            if self.counts[angle] < self.target_count:
                return angle
        return None

    def _get_instruction_for_angle(self, angle):
        instructions = {
            "center": "Look straight",
            "left": "Turn face LEFT",
            "right": "Turn face RIGHT",
            "up": "Look UP",
            "down": "Look DOWN",
        }
        return instructions.get(angle, "Look straight")

    def _finalize(self):
        logger.info("Finalizing registration for %s", self.user_name)

        all_embs = []
        for angle in self.angles:
            all_embs.extend(self.embeddings[angle])

        if not all_embs:
            self.done = True
            self.on_done(
                {
                    "status": "error",
                    "session_id": self.session_id,
                    "person_id": self.user_id,
                    "name": self.user_name,
                    "age": self.user_age,
                    "message": "No embeddings captured",
                }
            )
            return

        mean_emb = np.mean(all_embs, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        user_dir = Path(FACE_DATA_PATH) / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        save_path = user_dir / "embedding.npy"
        np.save(str(save_path), mean_emb)

        meta_path = user_dir / "info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": self.user_id,
                    "name": self.user_name,
                    "age": self.user_age,
                    "timestamp": int(time.time()),
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        self.done = True
        self.progress = 100
        self.on_done(
            {
                "status": "success",
                "session_id": self.session_id,
                "person_id": self.user_id,
                "user_id": self.user_id,
                "name": self.user_name,
                "user_name": self.user_name,
                "age": self.user_age,
                "message": f"Registration success: {self.user_name} (ID: {self.user_id})",
            }
        )


class RegistrationManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RegistrationManager, cls).__new__(cls)
            cls._instance.sessions = {}
            cls._instance.detector = None
            cls._instance.lock = threading.RLock()
        return cls._instance

    def _get_detector(self):
        if self.detector is None:
            try:
                if FaceAnalysis is None:
                    raise RuntimeError("FaceAnalysis is not available")
                self.detector = FaceAnalysis(name="buffalo_s")
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as e:
                logger.error("Failed to initialize FaceAnalysis: %s", e)
        return self.detector

    def get_next_id(self):
        """Sequential auto-increment ID generator."""
        try:
            path = Path(FACE_DATA_PATH)
            if not path.exists():
                return "0001"

            ids = []
            for d in path.iterdir():
                if d.is_dir():
                    try:
                        ids.append(int(d.name))
                    except Exception:
                        pass

            if not ids:
                return "0001"
            return f"{max(ids) + 1:04d}"
        except Exception:
            return "0001"

    def start_session(self, source, user_name, user_age, person_id, on_progress, on_done):
        with self.lock:
            existing_sessions = list(self.sessions.keys())

        for sid in existing_sessions:
            self.stop_session(sid)

        session_id = f"reg_{int(time.time())}"

        session = RegistrationSession(
            session_id=session_id,
            source=source,
            user_name=user_name,
            user_id=person_id,
            user_age=user_age,
            on_progress=on_progress,
            on_done=on_done,
            detector_model=self._get_detector(),
        )

        with self.lock:
            self.sessions[session_id] = session

        session.start()
        return session_id

    def stop_session(self, session_id):
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session:
            session.stop()

    def get_snapshot(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        if session:
            return session.get_snapshot()
        return None
