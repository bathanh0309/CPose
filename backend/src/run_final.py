"""
HAVEN Final - 4 Camera Sequential Runner
=========================================
- Cam1: Show only (không xử lý)
- Cam2: Master (tạo Global ID, Pose, ADL)
- Cam3/Cam4: Slave (chỉ match ID)

Controls:
- N: Skip to next video
- G: Start/Stop recording
- Q: Quit
"""
import sys
import cv2
import csv
import yaml
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from glob import glob

# Add paths
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO
from multi.reid import MasterSlaveReIDDB
from multi.adl import TrackState, classify_posture, ADLConfig
from multi.visualize import draw_skeleton, get_color_for_id


class VideoRecorder:
    """Simple video recorder."""
    
    def __init__(self, output_dir, fps=25, size=(640, 480)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.writer = None
        self.recording = False
        self.current_file = None
    
    def toggle(self, cam_name="recording"):
        """Toggle recording on/off."""
        if self.recording:
            self.stop()
            return False
        else:
            self.start(cam_name)
            return True
    
    def start(self, cam_name="recording"):
        """Start recording - simple name like cam1.mp4"""
        self.current_file = self.output_dir / f"{cam_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.current_file), fourcc, self.fps, self.size)
        self.recording = True
        print(f"  🔴 Recording: {self.current_file.name}")
    
    def stop(self):
        """Stop recording."""
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        if self.current_file:
            print(f"  ⏹️ Saved: {self.current_file.name}")
    
    def write(self, frame):
        """Write frame to video."""
        if self.recording and self.writer:
            resized = cv2.resize(frame, self.size)
            self.writer.write(resized)


class DatabaseLogger:
    """SQLite database logger."""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create tables if not exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                camera TEXT,
                frame INTEGER,
                track_id INTEGER,
                global_id TEXT,
                posture TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                camera TEXT
            )
        """)
        self.conn.commit()
    
    def log_detection(self, camera, frame, track_id, global_id, posture):
        """Log a detection."""
        self.conn.execute(
            "INSERT INTO detections (timestamp, camera, frame, track_id, global_id, posture) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera, frame, track_id, str(global_id), posture)
        )
    
    def log_recording(self, filename, camera):
        """Log a recording."""
        self.conn.execute(
            "INSERT INTO recordings (timestamp, filename, camera) VALUES (?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, camera)
        )
        self.conn.commit()
    
    def close(self):
        """Close connection."""
        self.conn.commit()
        self.conn.close()


class FinalRunner:
    """4-Camera Sequential Runner with Recording."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = BACKEND_DIR.parent / "configs" / "unified_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("HAVEN Final - 4 Camera Sequential")
        print("=" * 60)
        print("  Cam1: SHOW ONLY")
        print("  Cam2: MASTER (ID, Pose, ADL)")
        print("  Cam3/Cam4: SLAVE (match only)")
        print("  Controls: N=Next, G=Record, Q=Quit")
        print("=" * 60)
        
        # Display
        self.win_w = self.config['display']['width']
        self.win_h = self.config['display']['height']
        
        # Database
        db_path = self.config['output']['database_path']
        self.db = DatabaseLogger(db_path)
        
        # Recorder
        # Recorder - Save to root 'recordings' folder
        recordings_dir = Path("recordings")
        self.recorder = VideoRecorder(recordings_dir, fps=25, size=(self.win_w, self.win_h))
        
        # ReID
        reid_cfg = self.config['reid']
        self.reid_db = MasterSlaveReIDDB(
            reid_threshold=reid_cfg['threshold'],
            max_features=reid_cfg.get('max_features', 1000)
        )
        
        # ADL
        ADLConfig.from_dict(self.config.get('adl', {}))
        
        # YOLO
        yolo_cfg = self.config['yolo']
        models_dir = BACKEND_DIR / "models"
        pose_path = models_dir / yolo_cfg['pose_model']
        self.pose_model = YOLO(str(pose_path)) if pose_path.exists() else YOLO(yolo_cfg['pose_model'])
        self.yolo_conf = yolo_cfg['conf_threshold']
        
        # Data
        self.data_root = Path(self.config['data_source']['data_root'])
        self.master_camera = self.config.get('master_camera', 'cam2')
        self.total_global_ids = 0
        self.unknown_counter = 0  # Counter for UNK1, UNK2, etc.
        self.track_to_unknown = {}  # track_id -> unknown_id mapping
        
        print(f"Pose Model: {yolo_cfg['pose_model']} (conf={self.yolo_conf})")
        print(f"ReID Threshold: {reid_cfg['threshold']}")
        print(f"Data: {self.data_root}")
        print("=" * 60)
    
    def get_videos(self, cam_id):
        """Get videos for a camera."""
        cam_dir = self.data_root / cam_id
        if not cam_dir.exists():
            return []
        exts = self.config['data_source'].get('video_extensions', ['.mp4', '.avi'])
        videos = []
        for ext in exts:
            videos.extend(glob(str(cam_dir / f"*{ext}")))
        return sorted(videos)
    
    def create_waiting_frame(self, cam_id, text="Waiting..."):
        """Create waiting frame."""
        frame = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        cv2.putText(frame, f"{cam_id.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, text, ((self.win_w - tw) // 2, self.win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        return frame
    
    def handle_key(self, key, current_cam):
        """Handle keyboard input. Returns: (continue, skip)"""
        if key == ord('q'):
            return False, False
        elif key == ord('n'):
            return True, True
        elif key == ord('g'):
            is_recording = self.recorder.toggle(current_cam)
            if not is_recording and self.recorder.current_file:
                self.db.log_recording(self.recorder.current_file.name, current_cam)
        return True, False
    
    def process_camera(self, cam_id, videos, is_master=False, show_only=False):
        """Process a camera's videos."""
        mode = "MASTER" if is_master else ("SHOW ONLY" if show_only else "SLAVE")
        print(f"\n[{cam_id.upper()}] {mode} mode")
        
        self.reid_db.new_ids_allowed = is_master
        
        for video_path in videos:
            print(f"  Video: {Path(video_path).name}")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale_x, scale_y = self.win_w / orig_w, self.win_h / orig_h
            
            track_states, local_to_global = {}, {}
            global_ids_seen = set()
            frame_idx = 0
            
            # === TEMPORAL CONSISTENCY ===
            match_history = {}  # tid -> {gid: count, ...}
            CONFIRM_FRAMES = 3 if not is_master else 1
            
            # === PER-VIDEO UNK MANAGEMENT ===
            # UNK counter resets per video, max = number of unique people detected
            video_unk_counter = 0
            video_track_to_unk = {}  # tid -> unk_id (video-local)
            
            # === SPATIAL PERSISTENCE ===
            # Remember last bbox for each track to handle frame shake
            last_bbox = {}  # tid -> (x1, y1, x2, y2)
            BBOX_IOU_THRESHOLD = 0.3  # If new bbox overlaps old, keep same ID
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                display = cv2.resize(frame, (self.win_w, self.win_h))
                
                if not show_only:
                    results = self.pose_model.track(frame, persist=True, verbose=False, conf=self.yolo_conf)
                    
                    if results[0].boxes and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        keypoints = results[0].keypoints
                        
                        # Per-frame exclusive assignment
                        used_gids = set(local_to_global.values())
                        pending = []
                        
                        for i, tid in enumerate(track_ids):
                            bbox = tuple(boxes[i])
                            kpts = keypoints.data[i].cpu().numpy() if keypoints else None
                            
                            if tid not in track_states:
                                track_states[tid] = TrackState(tid, orig_h)
                            state = track_states[tid]
                            state.update_position(bbox)
                            
                            if kpts is not None:
                                posture = classify_posture(kpts, bbox, state, orig_h)
                                state.add_posture(posture)
                            
                            # Only process if has valid keypoints (at least 12 visible)
                            valid_kpts = 0
                            if kpts is not None:
                                for kp in kpts:
                                    if len(kp) >= 3 and kp[2] > 0.3:  # confidence > 0.3
                                        valid_kpts += 1
                            
                            if valid_kpts >= 12 and tid not in local_to_global:
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                crop = frame[max(0,y1):min(orig_h,y2), max(0,x1):min(orig_w,x2)]
                                if crop.size > 0:
                                    pending.append((tid, bbox, kpts, crop, state, valid_kpts))
                        
                        # Assign IDs with confirmation voting
                        for tid, bbox, kpts, crop, state, valid_kpts in pending:
                            matched_gid = self.reid_db.match_exclusive(crop, cam_id, exclude_ids=used_gids)
                            
                            if is_master:
                                # Master: assign immediately
                                if matched_gid:
                                    local_to_global[tid] = matched_gid
                                    used_gids.add(matched_gid)
                                else:
                                    gid = self.reid_db.register_new(crop, cam_id)
                                    if gid:
                                        local_to_global[tid] = gid
                                        used_gids.add(gid)
                                        self.total_global_ids = max(self.total_global_ids, gid)
                            else:
                                # Slave: use voting confirmation
                                if tid not in match_history:
                                    match_history[tid] = {}
                                
                                if matched_gid:
                                    # Increment vote for this GID
                                    match_history[tid][matched_gid] = match_history[tid].get(matched_gid, 0) + 1
                                    
                                    # Check if any GID has enough votes
                                    best_gid = max(match_history[tid], key=match_history[tid].get)
                                    vote_count = match_history[tid][best_gid]
                                    
                                    if vote_count >= CONFIRM_FRAMES:
                                        # Confirmed! Assign this ID
                                        local_to_global[tid] = best_gid
                                        used_gids.add(best_gid)
                                        # Clear history for this track
                                        match_history[tid] = {best_gid: CONFIRM_FRAMES}
                        
                        # Draw
                        for i, tid in enumerate(track_ids):
                            bbox = tuple(boxes[i])
                            x1, y1, x2, y2 = bbox
                            kpts = keypoints.data[i].cpu().numpy() if keypoints else None
                            state = track_states[tid]
                            
                            # Count valid keypoints
                            valid_kpts = 0
                            if kpts is not None:
                                for kp in kpts:
                                    if len(kp) >= 3 and kp[2] > 0.3:
                                        valid_kpts += 1
                            
                            if tid in local_to_global:
                                state.global_id = local_to_global[tid]
                                global_ids_seen.add(state.global_id)
                            
                            # Draw box
                            x1_s, y1_s = int(x1 * scale_x), int(y1 * scale_y)
                            x2_s, y2_s = int(x2 * scale_x), int(y2 * scale_y)
                            
                            if state.global_id:
                                color = get_color_for_id(state.global_id)
                                label = f"G{state.global_id}"
                            elif valid_kpts >= 12:
                                # === SMART UNK ASSIGNMENT (per-video, limited) ===
                                # Only assign UNK if this track doesn't have one already
                                if tid not in video_track_to_unk:
                                    # Check if we can reuse an existing UNK based on spatial overlap
                                    assigned_unk = None
                                    
                                    # Check spatial overlap with existing UNK tracks
                                    for old_tid, old_unk in video_track_to_unk.items():
                                        if old_tid in last_bbox:
                                            old_box = last_bbox[old_tid]
                                            # Calculate IoU
                                            x1i = max(bbox[0], old_box[0])
                                            y1i = max(bbox[1], old_box[1])
                                            x2i = min(bbox[2], old_box[2])
                                            y2i = min(bbox[3], old_box[3])
                                            
                                            if x2i > x1i and y2i > y1i:
                                                inter = (x2i - x1i) * (y2i - y1i)
                                                area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                                area2 = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
                                                iou = inter / (area1 + area2 - inter + 1e-6)
                                                
                                                if iou > BBOX_IOU_THRESHOLD:
                                                    # Reuse this UNK
                                                    assigned_unk = old_unk
                                                    break
                                    
                                    if assigned_unk:
                                        video_track_to_unk[tid] = assigned_unk
                                    else:
                                        # Create new UNK, max 10 per video
                                        if video_unk_counter < 10:
                                            video_unk_counter += 1
                                            video_track_to_unk[tid] = video_unk_counter
                                        else:
                                            # Max UNK reached, skip this person
                                            continue
                                
                                # Update last_bbox for this track
                                last_bbox[tid] = bbox
                                
                                unk_id = video_track_to_unk[tid]
                                color = (0, 165, 255)  # Orange
                                label = f"UNK{unk_id}"
                            else:
                                # Skip drawing if not enough keypoints
                                continue
                            
                            # Log to database
                            log_id = state.global_id or f"UNK{video_track_to_unk.get(tid, 0)}"
                            self.db.log_detection(cam_id, frame_idx, tid, log_id, state.current_posture or "")
                            
                            if state.current_posture:
                                label += f" | {state.current_posture}"
                            
                            cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(display, (x1_s, y1_s - th - 6), (x1_s + tw + 4, y1_s), color, -1)
                            cv2.putText(display, label, (x1_s + 2, y1_s - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            if kpts is not None:
                                scaled_kpts = [[kx * scale_x, ky * scale_y, kc] for kx, ky, kc in kpts]
                                draw_skeleton(display, scaled_kpts, colorful=True)
                
                # UI overlay
                cv2.putText(display, f"{cam_id.upper()} ({mode})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_master else (255, 165, 0), 2)
                cv2.putText(display, f"Frame: {frame_idx} | IDs: {sorted(global_ids_seen)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                if self.recorder.recording:
                    cv2.circle(display, (self.win_w - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (self.win_w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Record if active
                self.recorder.write(display)
                
                cv2.imshow(cam_id.upper(), display)
                
                # Update other windows
                for other in ["CAM1", "CAM2", "CAM3", "CAM4"]:
                    if other != cam_id.upper():
                        done_text = "Done" if other < cam_id.upper() else "Waiting..."
                        cv2.imshow(other, self.create_waiting_frame(other.lower(), done_text))
                
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                cont, skip = self.handle_key(key, cam_id)
                if not cont:
                    cap.release()
                    return False
                if skip:
                    break
            
            cap.release()
            print(f"    IDs: {sorted(global_ids_seen)}")
        
        return True
    
    def run(self):
        """Run sequential processing."""
        print("\nStarting...\n")
        
        # Create windows
        for cam in ["CAM1", "CAM2", "CAM3", "CAM4"]:
            cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam, self.win_w, self.win_h)
        cv2.moveWindow("CAM1", 0, 0)
        cv2.moveWindow("CAM2", self.win_w + 10, 0)
        cv2.moveWindow("CAM3", 0, self.win_h + 50)
        cv2.moveWindow("CAM4", self.win_w + 10, self.win_h + 50)
        
        # Initialize
        for cam in ["CAM1", "CAM2", "CAM3", "CAM4"]:
            cv2.imshow(cam, self.create_waiting_frame(cam.lower()))
        cv2.waitKey(100)
        
        # Process cameras sequentially
        for cam_id, is_master, show_only in [("cam1", False, True), ("cam2", True, False), ("cam3", False, False), ("cam4", False, False)]:
            videos = self.get_videos(cam_id)
            if videos:
                if not self.process_camera(cam_id, videos, is_master, show_only):
                    break
        
        # Summary
        print("\n" + "=" * 60)
        print(f"COMPLETE! Total Global IDs: {self.total_global_ids}")
        print("=" * 60)
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.recorder.stop()
        self.db.close()
        self.reid_db.summary()
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HAVEN Final Runner")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    runner = FinalRunner(config_path)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
        runner.cleanup()


if __name__ == "__main__":
    main()
