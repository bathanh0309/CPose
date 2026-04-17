import logging
import threading
import time
import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from app.setting import FACE_DATA_PATH, FACE_MIN_SIZE
from app.detectors.face import FaceRecognition

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
        detector_model: Any = None
    ):
        self.session_id = session_id
        self.source = source
        self.user_name = user_name
        self.user_id = user_id
        self.user_age = user_age
        self.on_progress = on_progress
        self.on_done = on_done
        
        self.running = False
        self.thread = None
        self.cap = None
        
        # Target samples per angle
        self.target_count = 5
        self.angles = ["center", "left", "right", "up", "down"]
        self.counts = {angle: 0 for angle in self.angles}
        self.embeddings = {angle: [] for angle in self.angles}
        
        self.current_frame = None
        self.current_output_frame = None
        
        # Detector
        if detector_model:
            self.model = detector_model
        else:
            # Fallback if no model provided
            self.model = FaceAnalysis(name="buffalo_s")
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Registration session {self.session_id} started for {self.user_name} (ID: {self.user_id}, Age: {self.user_age})")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Registration session {self.session_id} stopped")

    def get_snapshot(self):
        with self.lock:
            if self.current_output_frame is not None:
                _, jpeg = cv2.imencode('.jpg', self.current_output_frame)
                return jpeg.tobytes()
        return None

    def _estimate_angle(self, face):
        if not hasattr(face, 'pose'):
            return "unknown"
        
        yaw, pitch, roll = face.pose
        
        # Thresholds
        YAW_THRES = 20
        PITCH_THRES = 15
        
        if abs(yaw) < YAW_THRES and abs(pitch) < PITCH_THRES:
            return "center"
        if yaw < -YAW_THRES:
            return "left"
        if yaw > YAW_THRES:
            return "right"
        if pitch < -PITCH_THRES:
            return "up"
        if pitch > PITCH_THRES:
            return "down"
        
        return "mid"

    def _is_blurry(self, image, threshold=80):
        if image is None or image.size == 0: return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def _run(self):
        # Open camera
        try:
            source_val = int(self.source) if self.source.isdigit() else (0 if self.source == "local" else self.source)
        except:
            source_val = self.source
            
        self.cap = cv2.VideoCapture(source_val)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open source {self.source}")
            self.on_done({"status": "error", "message": f"Cannot open camera source: {self.source}"})
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            h, w = frame.shape[:2]
            # Define bounding box (center square)
            box_size = min(h, w) // 2
            x1, y1 = (w - box_size) // 2, (h - box_size) // 2
            x2, y2 = x1 + box_size, y1 + box_size
            
            # Detect faces
            faces = self.model.get(frame)
            
            output_frame = frame.copy()
            # Draw UI Box (Yellow-like Cyan)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            instruction = "Đang chờ mặt..."
            target_angle = self._get_next_needed_angle()
            
            if not target_angle:
                instruction = "Đang hoàn tất..."
                self.running = False
                self._finalize()
                break

            if len(faces) == 0:
                instruction = "Vui lòng đưa mặt vào giữa khung hình"
            elif len(faces) > 1:
                instruction = "Chỉ để một người trong khung hình"
            else:
                face = faces[0]
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                
                # Draw face bbox overlay with info
                cv2.rectangle(output_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                # Overlay Info above bbox
                info_str = f"ID:{self.user_id} | {self.user_name} ({self.user_age})"
                cv2.putText(output_frame, info_str, (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Check if face is within the square box
                is_inside = fx1 > x1 and fy1 > y1 and fx2 < x2 and fy2 < y2
                
                if not is_inside:
                    instruction = "Đưa mặt vào trong ô vuông"
                elif self._is_blurry(frame[fy1:fy2, fx1:fx2]):
                    instruction = "Giữ yên, mặt bị mờ"
                else:
                    angle = self._estimate_angle(face)
                    
                    if angle == target_angle:
                        # Capture embedding
                        emb = face.embedding
                        emb = emb / np.linalg.norm(emb)
                        self.embeddings[angle].append(emb)
                        self.counts[angle] += 1
                        
                        instruction = f"Đang chụp góc {angle.upper()}: {self.counts[angle]}/{self.target_count}"
                        
                        # Trigger UI update
                        self.on_progress({
                            "session_id": self.session_id,
                            "angle": angle,
                            "count": self.counts[angle],
                            "total": self.target_count,
                            "instruction": instruction,
                            "counts": self.counts
                        })
                    else:
                        instruction = self._get_instruction_for_angle(target_angle)

            with self.lock:
                self.current_output_frame = output_frame
            
            time.sleep(0.01) # Low latency

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
            "center": "Nhìn thẳng",
            "left": "Nghiêng mặt sang TRÁI",
            "right": "Nghiêng mặt sang PHẢI",
            "up": "NGẨNG mặt lên",
            "down": "CÚI mặt xuống"
        }
        return instructions.get(angle, "Nhìn thẳng")

    def _finalize(self):
        logger.info(f"Finalizing registration for {self.user_name}")
        
        all_embs = []
        for angle in self.angles:
            all_embs.extend(self.embeddings[angle])
        
        if not all_embs:
            self.on_done({"status": "error", "message": "No embeddings captured"})
            return

        mean_emb = np.mean(all_embs, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        
        # Save to database using user_id as directory
        user_dir = Path(FACE_DATA_PATH) / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = user_dir / "embedding.npy"
        np.save(str(save_path), mean_emb)
        
        # Save metadata info.json
        meta_path = user_dir / "info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": self.user_id,
                "name": self.user_name,
                "age": self.user_age,
                "timestamp": int(time.time())
            }, f, indent=4, ensure_ascii=False)
        
        self.on_done({
            "status": "success",
            "user_id": self.user_id,
            "user_name": self.user_name,
            "message": f"Đăng ký thành công: {self.user_name} (ID: {self.user_id})"
        })

class RegistrationManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RegistrationManager, cls).__new__(cls)
            cls._instance.sessions = {}
            cls._instance.detector = None
            cls._instance.lock = threading.Lock()
        return cls._instance

    def _get_detector(self):
        if self.detector is None:
            try:
                self.detector = FaceAnalysis(name="buffalo_s")
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as e:
                logger.error(f"Failed to initialize FaceAnalysis: {e}")
        return self.detector

    def get_next_id(self):
        """Sequential auto-increment ID generator"""
        try:
            path = Path(FACE_DATA_PATH)
            if not path.exists(): return "0001"
            
            # List all directories that are numeric-ish
            ids = []
            for d in path.iterdir():
                if d.is_dir():
                    try:
                        ids.append(int(d.name))
                    except:
                        pass
            
            if not ids: return "0001"
            return f"{max(ids) + 1:04d}"
        except:
            return "0001"

    def start_session(self, source, user_name, user_age, person_id, on_progress, on_done):
        with self.lock:
            # Check for existing sessions
            if len(self.sessions) > 0:
                # Stop existing sessions to avoid camera lock
                for sid in list(self.sessions.keys()):
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
                detector_model=self._get_detector()
            )
            self.sessions[session_id] = session
            session.start()
            return session_id

    def stop_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                del self.sessions[session_id]

    def get_snapshot(self, session_id):
        session = self.sessions.get(session_id)
        if session:
            return session.get_snapshot()
        return None
