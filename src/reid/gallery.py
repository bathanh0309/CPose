from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReIDGallery:
    def __init__(self, extractor, gallery_dir):
        self.extractor = extractor
        self.gallery_dir = Path(gallery_dir)
        self.prototypes = {}
        self.memory = {}

    @staticmethod
    def cosine_similarity(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def build(self):
        self.prototypes = {}
        self.memory = {}

        if not self.gallery_dir.exists():
            logger.warning(f"ReID gallery directory does not exist: {self.gallery_dir}")
            return

        logger.info(f"Building ReID gallery from: {self.gallery_dir}")
        for person_dir in self.gallery_dir.iterdir():
            if not person_dir.is_dir():
                continue

            feats = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in person_dir.glob(ext):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    feats.append(self.extractor.extract(img))

            if feats:
                feats = np.stack(feats, axis=0)
                self.memory[person_dir.name] = feats
                self.prototypes[person_dir.name] = feats.mean(axis=0).astype(np.float32)
                logger.info(f"Loaded {len(feats)} ReID embeddings for {person_dir.name}")

    def query(self, feat, threshold=0.55):
        if not self.prototypes:
            return "unknown", -1.0

        best_id = "unknown"
        best_score = -1.0
        for person_id, proto in self.prototypes.items():
            score = self.cosine_similarity(feat, proto)
            if score > best_score:
                best_score = score
                best_id = person_id

        if best_score < threshold:
            return "unknown", best_score
        return best_id, best_score

    def add_embedding(self, person_id, feat):
        if person_id not in self.memory:
            self.memory[person_id] = np.expand_dims(feat, axis=0)
        else:
            self.memory[person_id] = np.vstack([self.memory[person_id], feat[None, ...]])

        self.prototypes[person_id] = self.memory[person_id].mean(axis=0).astype(np.float32)

    def add_crop(self, person_id, crop_bgr):
        feat = self.extractor.extract(crop_bgr)
        self.add_embedding(person_id, feat)
        return feat
