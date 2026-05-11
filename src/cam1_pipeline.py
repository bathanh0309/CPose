import cv2
from collections import defaultdict, Counter

from src.core.detector import FaceRecognition
from src.core.face_recognizer import crop, infer, load_model, process_with_logits
from src.core.tracker import Tracker

from src.setting import *


class Cam1Pipeline:
    def __init__(self):
        # models
        self.person_tracker = Tracker()
        self.face_detection = FaceRecognition(data_path= DATA_DIR / "face")
        
        self.face_detection.load_face_database()
        
        # anti spoofing
        self.session, self.input_name = load_model(LIVENESS_MODEL)
        
        self.spoof_threshold = 0.6
        
        # state
        self.track_votes = defaultdict(list)
        self.track_final = {} # need to change the structure
        self.unknown_count = 0
        
        print("[Cam 1] Ready!")
        
    def process(self, frame, timestamp):
        results = []
        
        faces = self.face_detection.face_detect(frame)
        
        detections = []
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                detections.append([[x1, y1, x2-x1, y2-y1], 1.0, 0])
                
        tracks = self.person_tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
                        
            # init match face variables
            best_face = None
            best_iou = 0
            
            if faces is not None:
                for face in faces:
                    iou = compute_iou((x1, y1, x2, y2), tuple(face["bbox"]))
                    if iou > best_iou:
                        best_iou = iou
                        best_face = face
            
            # === process ===
            if best_face is not None and best_iou > 0.2:
                fx1, fy1, fx2, fy2 = best_face["bbox"]
                emb = best_face["embedding"]
                
                # === anti spoofing ===
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_crop = crop(frame_rgb, (fx1, fy1, fx2, fy2), 1.5)
                face_crop = cv2.resize(face_crop, (128, 128))
                
                preds = infer([face_crop], self.session, self.input_name, 128)
                res = process_with_logits(preds[0], self.spoof_threshold)

                spoof_label = 1 if res["is_real"] else 0
                spoof_conf = float(res["prob_real"])

                # === face recognition ===
                person_id, sim = self.face_detection.match_face(embedding = emb)
                
                if person_id is not None and sim >= TH_SIM:
                    recog_id = person_id
                else:
                    recog_id = "UNKNOWN"
                
                # === voting ===
                self.track_votes[track_id].append({
                    "recog_id": recog_id, 
                    "spoof_label": spoof_label, 
                    "spoof_conf": spoof_conf,
                    "time": timestamp
                })
                
                # === finalize tracking ID ===
                if track_id not in self.track_final:
                    if len(self.track_votes[track_id]) >= VOTE_LEN:
                        votes = self.track_votes[track_id]
                        
                        spoof_scores = [v["spoof_conf"] for v in votes]
                        avg_real = sum(spoof_scores) / len(spoof_scores)
                        
                        recog_votes = [v["recog_id"] for v in votes]
                        recog_final = Counter(recog_votes).most_common(1)[0][0]
                        
                        is_real = avg_real >= self.spoof_threshold
                        
                        # if identity is face or unknown
                        if not is_real or recog_final == "UNKNOWN":
                            print(f"[CAM 1] Face verify FAILED. Please try again.")
                            # reset voting
                            self.track_votes[track_id] = []
                            
                            label = "FAILED"
                            color = (0,0,255)
                            continue
                        
                            # lưu lại thông tin của người chưa đc verified

                        # verification success
                        final_id = f"{recog_final} | REAL"
                        
                        print(f"[CAM 1] Verified successfully. Welcome {recog_final}.")
                        
                        self.track_final[track_id] = {
                            "id": final_id,
                            "is_real": True,
                            "status": "verified",
                            "last_seen": timestamp
                        }
                        
                        self.track_votes.pop(track_id, None)
            
            # === draw info ===
            if track_id in self.track_final:
                self.track_final[track_id]["last_seen"] = timestamp
                
                info = self.track_final[track_id]
                
                if not info["is_real"]:
                    label = info["id"]
                    color = (0,0,255)
                else:
                    label = info["id"]
                    color = (0,255,0)
            else:
                label = "Detecting..."
                color = (0,255,255)
            
            foot_x = (x1 + x2) // 2
            foot_y = (y1 + y2) // 2

            results.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                
                "identity": self.track_final[track_id]["id"] if track_id in self.track_final else None,
                "is_real": info["is_real"]
                    if track_id in self.track_final else None,
                    
                "label": label,
                "color": color,
                
                "footpoint": (foot_x, foot_y),
                
                "timestamp": timestamp
            })
            
        self.cleanup_tracks(timestamp)
        self.cleanup_votes(timestamp)
            
        return results
    
    
    def cleanup_votes(self, now):
        for track_id in list(self.track_votes.keys()):
            votes = self.track_votes[track_id]
            
            votes = [v for v in votes if now - v["time"] < 3]
            
            if len(votes) == 0:
                del self.track_votes[track_id]
            else:
                self.track_votes[track_id] = votes
                
    # remove a tracking queue if a person not visible for few seconds
    def cleanup_tracks(self, now):
        remove_keys = []
        
        for track_id, info in self.track_final.items():
            if now - info["last_seen"] > 3: # TTL is 5
                remove_keys.append(track_id)
                
        for k in remove_keys:
            del self.track_final[k]
            print(f"[CAM 1] Remove track {k}")
