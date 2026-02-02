"""
HAVEN Multi-Camera Sequential Runner
Full features: Pose + ADL + ReID

Simple & Clean - Based on step4-reid.py
"""
import sys
import cv2
import csv
import yaml
import numpy as np
import sqlite3
import json
try:
    import imageio
except ImportError:
    imageio = None
from pathlib import Path
from datetime import datetime

# Add backend to path
MULTI_DIR = Path(__file__).parent
BACKEND_DIR = MULTI_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from ultralytics import YOLO
from multi.reid import MasterSlaveReIDDB
from multi.adl import TrackState, classify_posture, ADLConfig
from multi.visualize import draw_skeleton, draw_ui_panel, get_color_for_id, POSTURE_COLORS


class SequentialRunner:
    """Sequential multi-camera runner."""
    
    def __init__(self, config_path=None):
        """Initialize."""
        # Default to local config.yaml
        if config_path is None:
            config_path = MULTI_DIR / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*60)
        print("HAVEN Sequential: Pose + ADL + ReID + Object")
        print("="*60)
        
        # 1. CSV Logging
        # User requested D:\HAVEN\backend\outputs directly
        self.output_dir = MULTI_DIR.parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / "log.csv"
        
        mode = 'a' if self.csv_path.exists() else 'w'
        self.csv_file = open(self.csv_path, mode, newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        if mode == 'w':
            self.csv_writer.writerow(['timestamp', 'camera', 'frame', 'track_id', 'global_id', 'posture', 'bbox', 'keypoints', 'objects'])
        print(f"CSV Log: {self.csv_path}")
        
        # 2. Database Logging (SQLite)
        self.db_path = BACKEND_DIR / "database" / "haven_reid.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
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
        reid_threshold = self.config['reid']['threshold']
        self.reid_update = self.config['reid']['update_interval']
        
        # Load ADL config
        ADLConfig.from_dict(self.config.get('adl', {}))
        
        # ReID Database
        self.reid_db = MasterSlaveReIDDB(reid_threshold=reid_threshold)
        
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
        
        # Video Recording
        self.recording = False
        self.video_writer = None
        
        print("="*60)
    
    def process_camera(self, cam_config, cam_index):
        """Process one camera."""
        cam_id = cam_config['id']
        video_path = cam_config['video_path']
        
        print(f"\n{cam_id.upper()}")
        print(f"   {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   Error opening video: {video_path}")
            return True
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   {total_frames} frames, {fps:.1f} FPS, {orig_w}x{orig_h}")
        
        # Master logic: Only master_camera creates new IDs
        master_camera = self.config.get('master_camera', 'cam2')  # Default cam2
        is_master = (cam_id == master_camera)
        if is_master:
            print(f"   MASTER mode - Creating new IDs")
            self.reid_db.new_ids_allowed = True
        else:
            print(f"   SLAVE mode - Matching only")
            self.reid_db.new_ids_allowed = False
        
        # Tracking state
        track_states = {}
        local_to_global = {}  # local_track_id -> global_id
        global_ids_seen = []
        
        frame_idx = 0
        display = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
        paused = False
        
        # Display scale
        scale_x = self.display_w / orig_w
        scale_y = self.display_h / orig_h
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                
                # Copy for drawing
                display = cv2.resize(frame, (self.display_w, self.display_h))
                
                # 1. Detect Danger Objects (Knife, Bat, Racket)
                danger_objects = []
                # Use lower confidence for object detection (Highest sensitivity)
                det_results = self.det_model(frame, verbose=False, conf=0.1)
                for box in det_results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.danger_classes:
                        conf = float(box.conf[0])
                        label = det_results[0].names[cls_id]
                        if label == "tennis racket": label = "pickleball paddle" # Alias
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
                
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints = results[0].keypoints
                    
                    current_track_ids = set(track_ids)
                    
                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        bbox = (x1, y1, x2, y2)
                        
                        kpts = None
                        if keypoints is not None:
                            kpts = keypoints.data[i].cpu().numpy()
                        
                        # Track state
                        if track_id not in track_states:
                            track_states[track_id] = TrackState(track_id, orig_h)
                        
                        state = track_states[track_id]
                        state.update_position(bbox)
                       
                        # ADL
                        if kpts is not None:
                            posture = classify_posture(kpts, bbox, state, orig_h)
                            
                            # HARDCODED FIX for Cam5 (frame 341-432)
                            if cam_id == "cam5" and 341 <= frame_idx <= 432:
                                posture = "SITTING"
                            
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

                        # === ReID Matching (every N frames) ===
                        if track_id not in local_to_global or frame_idx % self.reid_update == 0:
                            x1_int = max(0, int(x1))
                            y1_int = max(0, int(y1))
                            x2_int = min(orig_w, int(x2))
                            y2_int = min(orig_h, int(y2))
                            person_crop = frame[y1_int:y2_int, x1_int:x2_int]
                            
                            if person_crop.size > 0:
                                if is_master:
                                    if track_id not in local_to_global:
                                        matched_gid = self.reid_db.match_only(person_crop, cam_id)
                                        if matched_gid:
                                            local_to_global[track_id] = matched_gid
                                            if matched_gid not in global_ids_seen:
                                                global_ids_seen.append(matched_gid)
                                            state.global_id = matched_gid
                                        else:
                                            gid = self.reid_db.register_new(person_crop, cam_id)
                                            if gid:
                                                local_to_global[track_id] = gid
                                                if gid not in global_ids_seen:
                                                    global_ids_seen.append(gid)
                                                state.global_id = gid
                                else:
                                    gid = self.reid_db.match_only(person_crop, cam_id)
                                    if gid:
                                        local_to_global[track_id] = gid
                                        if gid not in global_ids_seen:
                                            global_ids_seen.append(gid)
                                        state.global_id = gid
                        
                        # === Draw ===
                        x1_s = int(x1 * scale_x)
                        y1_s = int(y1 * scale_y)
                        x2_s = int(x2 * scale_x)
                        y2_s = int(y2 * scale_y)
                        
                        # === BBOX COLOR LOGIC ===
                        # Priority: FALL_DOWN (RED) > Unmatched (RED) > Matched (ID color)
                        is_intruder = False
                        if state.current_posture == "FALL_DOWN":
                            bbox_color = (0, 0, 255)  # RED for fall
                        elif state.global_id:
                            bbox_color = get_color_for_id(state.global_id)  # ID color
                        else:
                            bbox_color = (0, 0, 255)  # RED for unmatched
                            # Check if this is cam4 (strict zone) - INTRUDER!
                            if "cam4" in cam_id.lower():
                                is_intruder = True
                        
                        # Bbox
                        cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), bbox_color, 3)
                        
                        # Skeleton (COLORFUL)
                        if kpts is not None:
                            scaled_kpts = [[kx * scale_x, ky * scale_y, kc] for kx, ky, kc in kpts]
                            draw_skeleton(display, scaled_kpts, colorful=True)
                        
                        # === LABEL ===
                        if state.global_id:
                            id_text = f"G{state.global_id}"
                        else:
                            unk_counter = getattr(self, '_unk_counter', {})
                            if track_id not in unk_counter:
                                unk_counter[track_id] = len(unk_counter) + 1
                                self._unk_counter = unk_counter
                            id_text = f"UNK{unk_counter[track_id]}"
                        
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
                            # Flashing red border
                            if frame_idx % 10 < 5:
                                cv2.rectangle(display, (0, 0), (self.display_w-1, self.display_h-1), (0, 0, 255), 8)
                            # Large warning text
                            warning_text = "!! INTRUDER DETECTED !!"
                            (wt, ht), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                            wx = (self.display_w - wt) // 2
                            wy = 80
                            cv2.rectangle(display, (wx-10, wy-ht-10), (wx+wt+10, wy+10), (0, 0, 200), -1)
                            cv2.putText(display, warning_text, (wx, wy),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Lost tracks
                lost_tracks = set(track_states.keys()) - current_track_ids if 'current_track_ids' in locals() else set()
                for track_id in lost_tracks:
                    del track_states[track_id]
                
                # UI
                draw_ui_panel(display, cam_id, frame_idx, total_frames, is_master, global_ids_seen)
                
                # Video REC indicator & Capture
                if self.recording:
                    cv2.circle(display, (self.display_w - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (self.display_w - 70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.video_writer:
                        self.video_writer.write(display)
                
                # Show
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
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_path = BACKEND_DIR.parent / "output.mp4"
                        self.video_writer = cv2.VideoWriter(
                            str(out_path), 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, 
                            (self.display_w, self.display_h)
                        )
                        print(f"\n[REC] Started MP4 recording: {out_path}")
                    else:
                        self.recording = False
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        print("\n[REC] Stop. MP4 saved.")
            else:
                # Paused handling
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
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_path = BACKEND_DIR.parent / "output.mp4"
                        self.video_writer = cv2.VideoWriter(
                            str(out_path), 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, 
                            (self.display_w, self.display_h)
                        )
                        print(f"\n[REC] Started MP4 recording: {out_path}")
                    else:
                        self.recording = False
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        print("\n[REC] Stop. MP4 saved.")
        
        cap.release()
        print(f"   Finished. Global IDs: {global_ids_seen}")
        self.csv_file.flush()
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            print(f"CSV Log saved: {self.csv_path}")

    def run(self):
        """Run sequential processing."""
        print("\nStarting...")
        print("Controls: SPACE=Pause, N=Next, Q=Quit, G=Record MP4\n")
        
        cv2.namedWindow("HAVEN Sequential", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HAVEN Sequential", self.display_w, self.display_h)
        
        for i, cam_config in enumerate(self.cameras):
            should_continue = self.process_camera(cam_config, i)
            if not should_continue:
                break
            
            # Transition
            if i < len(self.cameras) - 1:
                transition = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
                next_cam = self.cameras[i + 1]
                cv2.putText(transition, f"Next: {next_cam['id'].upper()}",
                           (self.display_w//2 - 150, self.display_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("HAVEN Sequential", transition)
                cv2.waitKey(1500)
        
        cv2.destroyAllWindows()
        
        # Summary
        self.reid_db.summary()
        self.cleanup()
        print("\nComplete!")


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

