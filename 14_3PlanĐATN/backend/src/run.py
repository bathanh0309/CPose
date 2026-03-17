"""
HAVEN Multi-Camera Sequential Runner
Full features: Pose + ADL + ReID + Registration

Cam1 = Registration Master (detect + register profile + create G-ID)
Cam2/3/4 = Slaves (match only)
"""
import sys
import cv2
import csv
import yaml
import numpy as np
import sqlite3
import json
import threading
try:
    import imageio
except ImportError:
    imageio = None
from pathlib import Path
from datetime import datetime

# Paths
SRC_DIR = Path(__file__).parent       # backend/src/
BACKEND_DIR = SRC_DIR.parent          # backend/
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from reid import EnhancedReID
from core.global_id_manager import GlobalIDManager
from storage.persistence import PersistenceManager
from storage.vector_db import VectorDatabase
from adl import TrackState, classify_posture, ADLConfig
from visualize import draw_skeleton, draw_ui_panel, get_color_for_id, POSTURE_COLORS


class SequentialRunner:
    """Sequential multi-camera runner."""
    
    def __init__(self, config_path=None, headless=False):
        """Initialize."""
        self.headless = headless
        # Default to local config.yaml
        if config_path is None:
            config_path = SRC_DIR / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*60)
        print("HAVEN Sequential: Pose + ADL + ReID + Registration")
        print("="*60)
        
        # 1. CSV Logging
        self.output_dir = BACKEND_DIR / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / "log.csv"
        
        mode = 'a' if self.csv_path.exists() else 'w'
        self.csv_file = open(self.csv_path, mode, newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if mode == 'w':
            self.csv_writer.writerow(['timestamp', 'camera', 'frame', 'track_id', 'global_id', 'posture', 'bbox', 'keypoints', 'objects'])
        print(f"CSV Log: {self.csv_path}")
        
        # 2. Database Logging (SQLite) for Events
        self.db_path = BACKEND_DIR / "database" / "haven_reid.db"
        
        # CLEANUP FOR DEMO: Remove old persistence state to ensure G1, G2...
        persist_dir = BACKEND_DIR / "database"
        if persist_dir.exists():
            try:
                import shutil
                # Remove PersistenceManager files
                if (persist_dir / "haven_state.db").exists():
                    (persist_dir / "haven_state.db").unlink()
                if (persist_dir / "embeddings.npy").exists():
                    (persist_dir / "embeddings.npy").unlink()
                if (persist_dir / "embeddings.new.npy").exists():
                    (persist_dir / "embeddings.new.npy").unlink()
                print("DEBUG: Cleared old persistence state for fresh demo run.")
            except Exception as e:
                print(f"Warning: Could not clear old DB: {e}")
        
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create table if not exists
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                camera TEXT,
                frame INTEGER,
                track_id INTEGER,
                global_id INTEGER,
                posture TEXT,
                bbox TEXT,
                objects TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
        self.session_id = timestamp
        print(f"Database: {self.db_path}")

        # Config values
        self.display_w = self.config['display']['width']
        self.display_h = self.config['display']['height']
        reid_cfg = self.config['reid']
        reid_threshold = reid_cfg['threshold']
        self.reid_update = reid_cfg['update_interval']
        confirm_frames = reid_cfg.get('confirm_frames', 3)
        max_features = reid_cfg.get('max_features', 100)
        self.min_keypoints = reid_cfg.get('min_keypoints', 10)
        
        # Load ADL config
        ADLConfig.from_dict(self.config.get('adl', {}))
        
        # Load Person Profiles from config
        self.profile_queue = list(self.config.get('person_profiles', []))
        self.person_profiles = {}  # global_id -> {name, age, gender}
        print(f"Person Profiles to register: {len(self.profile_queue)}")
        
        # YOLO Pose (Keypoints)
        yolo_cfg = self.config['yolo']
        pose_model_path = BACKEND_DIR / "models" / yolo_cfg['model']
        self.pose_model = YOLO(str(pose_model_path))
        self.yolo_conf = yolo_cfg['conf_threshold']
        print(f"Pose Model: {yolo_cfg['model']}")
        
        # YOLO Detect (Objects - Danger)
        det_model_path = BACKEND_DIR / "models" / "yolo11n.pt"
        self.det_model = YOLO(str(det_model_path)) if det_model_path.exists() else YOLO("yolo11n.pt")
        # Classes: 34 (baseball bat), 38 (tennis racket), 43 (knife)
        self.danger_classes = [34, 38, 43] 
        print(f"Object Model: yolo11n.pt (Danger items)")
        
        # Cameras
        self.cameras = [c for c in self.config['cameras'] if c['enabled']]
        print(f"Cameras: {len(self.cameras)}")

        # ReID Components
        # 1. Persistence Layer
        self.persistence = PersistenceManager(persist_path=str(BACKEND_DIR / "database"), embedding_dim=176)
        
        # 2. Vector Database
        self.vector_db = VectorDatabase(embedding_dim=176)
        
        # 3. Feature Extractor
        self.feature_extractor = EnhancedReID()
        
        # 4. GlobalIDManager
        master_cam = self.config.get('master_camera', 'cam1')
        # Identify slave cameras (not master)
        slave_cams = list(set(c['id'] for c in self.cameras if c['id'] != master_cam))
        
        gid_config = {
            'strong_threshold': reid_threshold,
            'weak_threshold': 0.45,
            'confirm_frames': confirm_frames,
            'unk_namespace': 'global'
        }
        
        self.global_id_manager = GlobalIDManager(
            master_camera=master_cam,
            slave_cameras=slave_cams,
            persistence=self.persistence,
            vector_db=self.vector_db,
            config=gid_config
        )
        
        # Video Recording
        self.recording = False
        self.video_writer = None
        
        # Web streaming state (thread-safe)
        self._frame_lock = threading.Lock()
        self._latest_frame = None      # numpy array (BGR)
        self._stop_flag = False
        self._running = False
        self._current_camera = ""
        self._log_messages = []        # last N log entries
        self._progress = {"current": 0, "total": len(self.cameras), "camera": ""}
        
        print("="*60)
    
    def _get_profile_label(self, gid):
        """Get profile label for a global ID."""
        if gid in self.person_profiles:
            p = self.person_profiles[gid]
            gender_short = "M" if p['gender'] == 'male' else "F"
            return f"{p['name']}, {p['age']}, {gender_short}"
        return None
    
    def _add_log(self, msg):
        """Add a log message (thread-safe, last 100)."""
        self._log_messages.append(msg)
        if len(self._log_messages) > 100:
            self._log_messages = self._log_messages[-100:]
        print(msg)
    
    def get_latest_frame(self):
        """Get latest frame as JPEG bytes (for MJPEG streaming)."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self._latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes()
    
    def get_status(self):
        """Get current system status for web dashboard."""
        return {
            "running": self._running,
            "camera": self._current_camera,
            "progress": self._progress,
            "profiles": self.person_profiles,
            "logs": self._log_messages[-30:],
        }
    
    def set_profiles(self, profiles):
        """Set person profiles for registration (called from web API)."""
        self.profile_queue = list(profiles)
        self.person_profiles = {}
    
    def stop(self):
        """Signal to stop processing."""
        self._stop_flag = True
    
    def process_camera(self, cam_config, cam_index):
        """Process one camera."""
        cam_id = cam_config['id']
        video_path = cam_config['video_path']
        is_registration = cam_config.get('registration', False)
        
        self._current_camera = cam_id
        self._progress["camera"] = cam_id
        self._progress["current"] = cam_index + 1
        
        self._add_log(f"\n{cam_id.upper()}")
        self._add_log(f"   {video_path}")
        if is_registration:
            self._add_log(f"   MODE: REGISTRATION (detect + register profile)")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._add_log(f"   Error opening video: {video_path}")
            return True
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._add_log(f"   {total_frames} frames, {fps:.1f} FPS, {orig_w}x{orig_h}")
        
        # Tracking state
        track_states = {}
        local_to_global = {}
        global_ids_seen = []
        
        frame_idx = 0
        display = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
        paused = False
        
        # Display scale
        scale_x = self.display_w / orig_w
        scale_y = self.display_h / orig_h
        
        while not self._stop_flag:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                # Copy for drawing
                display = cv2.resize(frame, (self.display_w, self.display_h))
                
                # 1. Detect Danger Objects (Knife, Bat, Racket)
                danger_objects = []
                det_results = self.det_model(frame, verbose=False, conf=0.1)
                for box in det_results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.danger_classes:
                        conf = float(box.conf[0])
                        label = det_results[0].names[cls_id]
                        if label == "tennis racket": label = "pickleball paddle"
                        danger_objects.append(f"{label}")
                        
                        # Draw warning
                        bx = box.xyxy[0].cpu().numpy()
                        d_x1, d_y1, d_x2, d_y2 = map(int, bx)
                        d_x1_s = int(d_x1 * scale_x)
                        d_y1_s = int(d_y1 * scale_y)
                        d_x2_s = int(d_x2 * scale_x)
                        d_y2_s = int(d_y2 * scale_y)
                        cv2.rectangle(display, (d_x1_s, d_y1_s), (d_x2_s, d_y2_s), (0, 0, 255), 2)
                        cv2.putText(display, f"{label.upper()}", (d_x1_s, d_y1_s-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                objects_str = str(danger_objects) if danger_objects else ""

                # 2. Track Pose
                results = self.pose_model.track(frame, persist=True, verbose=False, conf=self.yolo_conf)
                
                current_track_ids = set()
                
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints = results[0].keypoints
                    
                    # Get confidence scores
                    if results[0].boxes.conf is not None:
                        conf_scores = results[0].boxes.conf.cpu().numpy()
                    else:
                        conf_scores = [1.0] * len(track_ids)
                    
                    current_track_ids = set(track_ids)
                    
                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        bbox = (x1, y1, x2, y2)
                        
                        # Get confidence
                        conf = float(conf_scores[i]) if i < len(conf_scores) else 1.0
                        
                        kpts = None
                        if keypoints is not None:
                            kpts = keypoints.data[i].cpu().numpy()
                        
                        # Check keypoint quality
                        visible_kpts = 0
                        if kpts is not None:
                            visible_kpts = sum(1 for kp in kpts if kp[2] > 0.2)
                        
                        # Skip if not enough keypoints
                        if visible_kpts < self.min_keypoints:
                            continue
                        
                        # Track state
                        if track_id not in track_states:
                            track_states[track_id] = TrackState(track_id, orig_h)
                        
                        state = track_states[track_id]
                        state.update_position(bbox)
                       
                        # ADL
                        if kpts is not None:
                            posture = classify_posture(kpts, bbox, state, orig_h)
                            state.add_posture(posture)
                            hand_event = state.check_hand_raise(kpts)
                            if hand_event:
                                state.add_event(hand_event)

                        # LOGGING (CSV + DB)
                        kpts_str = str(kpts.tolist()) if kpts is not None else "[]"
                        bbox_str = str(bbox)
                        
                        # CSV
                        self.csv_writer.writerow([
                            datetime.now().strftime("%H:%M:%S.%f"),
                            cam_id, frame_idx, track_id, state.global_id, 
                            state.current_posture, bbox_str, kpts_str, objects_str
                        ])
                        
                        # DB
                        try:
                            self.cursor.execute('''
                                INSERT INTO event_log 
                                (session_id, camera, frame, track_id, global_id, posture, bbox, objects)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (self.session_id, cam_id, frame_idx, int(track_id), 
                                  int(state.global_id) if state.global_id else None,
                                  state.current_posture, bbox_str, objects_str))
                        except Exception as e:
                            print(f"DB Error: {e}")

                        # === ReID ===
                        if track_id not in local_to_global or frame_idx % self.reid_update == 0:
                            x1_int = max(0, int(x1))
                            y1_int = max(0, int(y1))
                            x2_int = min(orig_w, int(x2))
                            y2_int = min(orig_h, int(y2))
                            person_crop = frame[y1_int:y2_int, x1_int:x2_int]
                            
                            if person_crop.size > 0:
                                embedding = self.feature_extractor.extract(person_crop)
                                
                                if embedding is not None:
                                    assigned_id, metadata = self.global_id_manager.assign_id(
                                        camera=cam_id,
                                        track_id=track_id,
                                        embedding=embedding,
                                        bbox=[x1_int, y1_int, x2_int, y2_int],
                                        frame_time=frame_idx / fps,
                                        quality_score=conf
                                    )
                                    
                                    # Parse result (G1, UNK2, etc.)
                                    if assigned_id.startswith("G"):
                                        gid_num = int(assigned_id[1:].replace('*', ''))
                                        local_to_global[track_id] = gid_num
                                        if gid_num not in global_ids_seen:
                                            global_ids_seen.append(gid_num)
                                        state.global_id = gid_num
                                        
                                        # === REGISTRATION: Attach profile on NEW_ID ===
                                        if metadata.get('state') == 'NEW_ID' and is_registration:
                                            if self.profile_queue:
                                                profile = self.profile_queue.pop(0)
                                                self.person_profiles[gid_num] = profile
                                                self.persistence.register_profile(
                                                    gid_num,
                                                    profile['name'],
                                                    profile.get('age'),
                                                    profile.get('gender')
                                                )
                                                print(f"    G-ID {gid_num} REGISTERED in {cam_id.upper()}: "
                                                      f"{profile['name']}, {profile.get('age')}, {profile.get('gender')}")
                                            else:
                                                print(f"    G-ID {gid_num} created in {cam_id.upper()} (no profile available)")
                                        elif metadata.get('state') == 'NEW_ID':
                                            print(f"    G-ID {gid_num} created in {cam_id.upper()}")
                                        elif metadata.get('state') == 'CONFIRMED':
                                            profile_label = self._get_profile_label(gid_num)
                                            if profile_label:
                                                print(f"    G-ID {gid_num} confirmed in {cam_id.upper()} [{profile_label}]")
                                            else:
                                                print(f"    G-ID {gid_num} confirmed in {cam_id.upper()}")
                                    
                                    else:
                                        # UNK assignment
                                        state.global_id = None
                        
                        # === Draw ===
                        x1_s = int(x1 * scale_x)
                        y1_s = int(y1 * scale_y)
                        x2_s = int(x2 * scale_x)
                        y2_s = int(y2 * scale_y)
                        
                        # === BBOX COLOR LOGIC ===
                        is_intruder = False
                        if state.current_posture == "FALL_DOWN":
                            bbox_color = (0, 0, 255)
                        elif state.global_id:
                            bbox_color = get_color_for_id(state.global_id)
                        else:
                            bbox_color = (0, 0, 255)
                            if "cam4" in cam_id.lower():
                                is_intruder = True
                        
                        # Bbox
                        cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), bbox_color, 3)
                        
                        # Skeleton (COLORFUL)
                        if kpts is not None:
                            scaled_kpts = [[kx * scale_x, ky * scale_y, kc] for kx, ky, kc in kpts]
                            draw_skeleton(display, scaled_kpts, colorful=True)
                        
                        # === LABEL (with profile info) ===
                        if state.global_id:
                            id_text = f"G{state.global_id}"
                            profile_label = self._get_profile_label(state.global_id)
                            if profile_label:
                                id_text = f"G{state.global_id} | {profile_label}"
                        else:
                            # Simple UNK label - no spiraling counter
                            id_text = "UNK"
                        
                        if state.current_posture:
                            label = f"{id_text} | {state.current_posture}"
                        else:
                            label = id_text
                        
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display, (x1_s, y1_s-th-10), (x1_s+tw+10, y1_s), bbox_color, -1)
                        cv2.putText(display, label, (x1_s+5, y1_s-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # === INTRUDER ALERT (CAM4 ONLY) ===
                        if is_intruder:
                            if frame_idx % 10 < 5:
                                cv2.rectangle(display, (0, 0), (self.display_w-1, self.display_h-1), (0, 0, 255), 8)
                            warning_text = "!! INTRUDER DETECTED !!"
                            (wt, ht), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                            wx = (self.display_w - wt) // 2
                            wy = 80
                            cv2.rectangle(display, (wx-10, wy-ht-10), (wx+wt+10, wy+10), (0, 0, 200), -1)
                            cv2.putText(display, warning_text, (wx, wy),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Lost tracks
                lost_tracks = set(track_states.keys()) - current_track_ids
                for track_id in lost_tracks:
                    del track_states[track_id]
                
                # UI
                is_master_cam = (cam_id == self.config.get('master_camera', 'cam1'))
                draw_ui_panel(display, cam_id, frame_idx, total_frames, is_master_cam, 
                             global_ids_seen, self.person_profiles)
                
                # Video REC indicator & Capture
                if self.recording:
                    cv2.circle(display, (self.display_w - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (self.display_w - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.video_writer:
                        self.video_writer.write(display)
                
                # Store frame for web streaming
                with self._frame_lock:
                    self._latest_frame = display.copy()
                
                # Show (only in GUI mode)
                if not self.headless:
                    cv2.imshow("HAVEN Sequential", display)
                    
                    wait_time = int(1000 / fps) if not paused else 50
                    key = cv2.waitKey(wait_time) & 0xFF
                    
                    if key == ord('q'):
                        cap.release()
                        return False
                    elif key == ord(' '):
                        paused = not paused
                    elif key == ord('n'):
                        break
                    elif key == ord('g'):
                        if not self.recording:
                            self.recording = True
                            out_path = BACKEND_DIR.parent / "output.mp4"
                            self.video_writer = cv2.VideoWriter(
                                str(out_path), 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (self.display_w, self.display_h)
                            )
                            print(f"\n[REC] Started recording: {out_path}")
                        else:
                            self.recording = False
                            if self.video_writer:
                                self.video_writer.release()
                                self.video_writer = None
                            print("\n[REC] Stop. Video saved.")
                else:
                    # Headless: brief sleep to control frame rate
                    import time
                    time.sleep(max(0.001, 1.0 / fps))
            else:
                # Paused handling (GUI only)
                if not self.headless:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        return False
                    elif key == ord(' '):
                        paused = not paused
                    elif key == ord('n'):
                        break
                    elif key == ord('g'):
                        if not self.recording:
                            self.recording = True
                            out_path = BACKEND_DIR.parent / "output.mp4"
                            self.video_writer = cv2.VideoWriter(
                                str(out_path), 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (self.display_w, self.display_h)
                            )
                            print(f"\n[REC] Started recording: {out_path}")
                        else:
                            self.recording = False
                            if self.video_writer:
                                self.video_writer.release()
                                self.video_writer = None
                            print("\n[REC] Stop. Video saved.")
        
        cap.release()
        self._add_log(f"   Finished. Global IDs: {global_ids_seen}")
        self.csv_file.flush()
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop recording if active
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("\n[REC] Video saved on cleanup.")
        
        if hasattr(self, 'global_id_manager'):
            stats = self.global_id_manager.get_statistics()
            persist_stats = stats.get('persistence', stats)
            print("\n" + "="*60)
            print(" GLOBAL ID STATISTICS")
            print("="*60)
            print(f" Total Global IDs: {persist_stats.get('total_global_ids', 0)}")
            print(f" Next ID: {persist_stats.get('next_global_id', 1)}")
            
            # Print registered profiles
            if self.person_profiles:
                print(f"\n Registered Profiles:")
                for gid, p in self.person_profiles.items():
                    gender_short = "M" if p.get('gender') == 'male' else "F"
                    print(f"   G{gid}: {p['name']}, {p.get('age')}, {gender_short}")
        
        if hasattr(self, 'persistence'):
            self.persistence.close()
            
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            print(f"CSV Log saved: {self.csv_path}")

    def run(self, headless_override=False):
        """Run sequential processing."""
        self._running = True
        self._stop_flag = False
        use_gui = not self.headless and not headless_override
        
        self._add_log("Starting...")
        
        if use_gui:
            cv2.namedWindow("HAVEN Sequential", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HAVEN Sequential", self.display_w, self.display_h)
        
        for i, cam_config in enumerate(self.cameras):
            if self._stop_flag:
                break
            should_continue = self.process_camera(cam_config, i)
            if not should_continue:
                break
            
            # Transition
            if use_gui and i < len(self.cameras) - 1:
                transition = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
                next_cam = self.cameras[i + 1]
                cv2.putText(transition, f"Next: {next_cam['id'].upper()}",
                           (self.display_w//2 - 150, self.display_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("HAVEN Sequential", transition)
                cv2.waitKey(1500)
            elif not use_gui and i < len(self.cameras) - 1:
                # Brief pause for web transition
                import time
                transition = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
                next_cam = self.cameras[i + 1]
                cv2.putText(transition, f"Next: {next_cam['id'].upper()}",
                           (self.display_w//2 - 150, self.display_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                with self._frame_lock:
                    self._latest_frame = transition
                time.sleep(1.5)
        
        if use_gui:
            cv2.destroyAllWindows()
        
        # Summary
        self.cleanup()
        self._running = False
        self._add_log("Complete!")


if __name__ == "__main__":
    runner = SequentialRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        runner.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        runner.cleanup()
        raise
