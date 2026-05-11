import time
import cv2
import numpy as np

class MultiCameraMatcher:
    def __init__(self):
        self.memory = []
        
        self.track_id_map = {}
        self.time_to_live = 3.0
        
        self.cam1_locked = {}
        
    # === class operations ===
    def update_cam1(self, cam1_results):
        
        for r in cam1_results:
            identity = r["identity"]
            track_id = r["track_id"]
            
            if identity is None or "FAKE" in identity:
                continue

            if track_id not in self.cam1_locked:
                self.cam1_locked[track_id] = identity
                
            self.memory.append({
                "identity": self.cam1_locked[track_id],
                "timestamp": r["timestamp"]
            })
            
        # keep memory short, not overflow
        self.memory = self.memory[-50:]

    
    def match(self, cam2_results):
        mapping = {}
        now = time.time()
        used_ids = set()
        
        for t2 in cam2_results:
            track_id = t2["track_id"]
            
            # == lock id ==
            if track_id in self.track_id_map:
                identity = self.track_id_map[track_id]["id"]
                
                # === check duplicate  ====
                if identity in used_ids:
                    # keep old ID if already locked
                    mapping[track_id] = self.track_id_map[track_id]["id"]
                    continue
                
                used_ids.add(identity)
                self.track_id_map[track_id]["last_seen"] = now
                mapping[track_id] = identity
                continue
            
            # === matching ID ===
            best = None
            best_score = 0
            
            for t1 in self.memory:
                identity = t1["identity"]
                
                if identity is None or "FAKE" in identity:
                    continue
                                                
                proj_x, proj_y = t1["projected_point"]
                cam2_x, cam2_y = t2["footpoint"]
                
                dist = ((proj_x - cam2_x)**2 + (proj_y - cam2_y)**2) ** 0.5
                
                # if dist > 200:
                #     continue

                dt = abs(t2["timestamp"] - t1["timestamp"])
                
                if dt > 2:                
                    continue
                
                time_score = 1 / (1 + dt)
                space_score = 1 / (1 + dist)
                score = (0.3 * space_score + 0.7 * time_score)
                
                print(f"[Score] Time score: {time_score}")
                print(f"[Score] Space score: {space_score}")
                print(f"[Score] Score: {score}")
                
                if score > best_score:
                    best_score = score
                    best = t1
                    
            # === match success ===  
            if best:
                identity = best["identity"]
                
                # === check duplicate ===
                if identity is None:
                    continue
                
                if identity in used_ids:
                    if track_id in self.track_id_map:
                        mapping[track_id] = self.track_id_map[track_id]["id"]
                    continue
                
                used_ids.add(identity)
                
                # lock id here
                self.track_id_map[track_id] = {
                    "id": identity,
                    "last_seen": now
                }
                
                mapping[track_id] = identity
        
        self.cleanup(now)
        return mapping
    
    def cleanup(self, now):
        remove_keys = []
        
        for track_id, info in self.track_id_map.items():
            if now - info["last_seen"] > self.time_to_live:
                remove_keys.append(track_id)
                
        for k in remove_keys:
            del self.track_id_map[k]