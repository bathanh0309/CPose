from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, embedder_name = 'mobilenet'):
        self.model = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.4,
            embedder=embedder_name,
            half=True
        )
        
    def update_tracks(self, detections, frame):
        return self.model.update_tracks(detections, frame=frame)
    
from collections import defaultdict, Counter
import numpy as np

class FaceTrackerSystem:
    def __init__(self, vote_len = 5, emb_cache_len = 5, sim_th = 0.4):
        self.vote_len = vote_len
        self.emb_cache_len = emb_cache_len
        self.sim_th = sim_th
        
        self.track_cache = {}
        self.track_votes = defaultdict(list)
        self.track_final = {}
        self.active_ids = {}
        
        self.track_spoof_votes = defaultdict(list)
    
    def update_embedding(self, tid, emb):
        if tid not in self.track_cache:
            self.track_cache[tid] = {
                "emb_list": [],
                "avg_emb": None
            }
            
        cache = self.track_cache[tid]
        
        if cache["avg_emb"] is None:
            cache["emb_list"].append(emb)
            
            if len(cache["emb_list"]) >= self.emb_cache_len:
                avg = np.mean(cache["emb_list"], axis=0)
                avg = avg / np.linalg.norm(avg)
                cache["avg_emb"] = avg
                
            return emb
        
        else:
            avg = cache["avg_emb"]
            avg = 0.9 * avg + 0.1 * emb
            avg = avg / np.linalg.norm(avg)
            cache["avg_emb"] = avg
            return avg
        
    def check_id_conflict(self, tid, pid, sim):
        if pid not in self.active_ids:
            self.active_ids[pid] = {"tid": tid, "sim":sim}
            return pid
        
        info =  self.active_ids[pid]
        
        # same track -> ok
        if info["tid"] == tid:
            return pid
        
        # higher confidence => replace
        if sim > info["sim"]:
            self.active_ids[pid] = {"tid": tid, "sim": sim}
            return pid
        
        # conflict
        return "UNKOWN"
    

    def update_vote(self, tid, pid):
        if tid in self.track_final:
            return self.track_final[tid]
        
        self.track_votes[tid].append(pid)
        
        if len(self.track_votes[tid]) >= self.vote_len:
            final = Counter(self.track_votes[tid]).most_common(1)[0][0]
            self.track_final[tid] = final
            self.track_votes.pop(tid, None)
            
            return final

        return None
    
    # Main process
    def process(self, tid, emb, face_detection):
        # step 1: caching face embeddings
        emb = self.update_embedding(tid, emb)
        
        # step 2: match
        pid, sim = face_detection.match_face(embedding=emb)
        
        if pid is None or sim < self.sim_th:
            pid = "UNKNOWN"
            
        else:
            # step 3: exclusivity
            pid = self. check_id_conflict(tid, pid, sim)
            
        # step 4: voting
        final = self.update_vote(tid, pid)
        
        return final if final is not None else pid
    
    def cleanup(self, active_tids):
        # remove lost tracks from active_tids
        remove_keys = []
        
        for pid, info in self.active_ids.items():
            if info["tid"] not in active_tids:
                remove_keys.append(pid)
        
        for k in remove_keys:
            self.active_ids.pop(k ,None)