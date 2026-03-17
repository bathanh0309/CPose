import numpy as np
import os

from insightface.app import FaceAnalysis
from setting import *

class FaceRecognition:
    def __init__(self, model_name = "buffalo_s", threshold = 0.4, data_path=FACE_DATA_PATH):
        self.model = FaceAnalysis(name=model_name)
        self.threshold = threshold
        self.data_path = data_path
        self.data = None

        self.model.prepare(ctx_id=0, det_size=(640, 640))

    # Load user's facial data
    def load_face_database(self):
        db = {}
        for pid in os.listdir(self.data_path):
            pdir = os.path.join(self.data_path, pid)
            if not os.path.isdir(pdir):
                continue
            embs = []
            for f in os.listdir(pdir):
                if f.endswith(".npy"):
                    emb = np.load(os.path.join(pdir, f))
                    emb = emb / np.linalg.norm(emb)
                    embs.append(emb)
            if embs:
                db[pid] = np.vstack(embs)

        self.data = db
        print("Loaded Face embedding data")
        return db
    
    # Detect users' face in the frame and return index locations
    def face_detect(self, frame):
        faces = self.model.get(frame)
        emb = None
        
        if len(faces) == 1:
            face = faces[0]
            fx1, fy1, fx2, fy2 = map(int, face.bbox)

            if (fx2 - fx1) > FACE_MIN_SIZE:
                emb = face.embedding
                emb = emb / np.linalg.norm(emb)
                
        return faces, emb # return both for processing
        

    # Calculate cosine similarity to match with the database and return best possible results
    def match_face(self, embedding):
        best_id = None
        best_sim = -1.0

        for person_id, embs in self.data.items():
            sims = embs @ embedding
            max_sim = np.max(sims)
            if max_sim > best_sim:
                best_sim = max_sim
                best_id = person_id
            
        return best_id, best_sim


def age_to_group(age):
    if not isinstance(age, (int, float)):
        return "?"
    if age < 20:
        return "Youth"
    elif age < 50:
        return "Adult"
    else:
        return "Elder"