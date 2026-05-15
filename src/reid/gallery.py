import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReIDGallery:
    def __init__(self, extractor, gallery_dir, embedding_dirs=None):
        self.extractor = extractor
        self.gallery_dir = Path(gallery_dir)
        self.embedding_dirs = [Path(path) for path in (embedding_dirs or [])]
        self.prototypes = {}
        self.memory = {}
        self.initial_empty = True
        self._dimension_warnings = set()
        self.index_ids = []
        self.index_matrix = None

    @staticmethod
    def cosine_similarity(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _person_id_from_dir(person_dir: Path) -> str:
        meta_path = person_dir / "meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                person_id = str(meta.get("person_id", "")).strip()
                if person_id:
                    return person_id
            except Exception as exc:
                logger.warning(f"Cannot read embedding metadata {meta_path}: {exc}")
        return person_dir.name

    def _build_from_image_gallery(self):
        if not self.gallery_dir.exists():
            logger.warning(f"ReID gallery directory does not exist: {self.gallery_dir}")
            return

        logger.info(f"Building ReID gallery images from: {self.gallery_dir}")
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
                self._add_feature_stack(person_dir.name, feats, source="image gallery")

    def _build_from_npy_dirs(self):
        for embedding_dir in self.embedding_dirs:
            if not embedding_dir.exists():
                logger.warning(f"Embedding directory does not exist: {embedding_dir}")
                continue

            logger.info(f"Inspecting external embeddings from: {embedding_dir}")
            for person_dir in embedding_dir.iterdir():
                if not person_dir.is_dir():
                    continue

                person_id = self._person_id_from_dir(person_dir)
                feats = []
                for npy_path in sorted(person_dir.glob("*.npy")):
                    try:
                        feat = np.load(npy_path).astype(np.float32).reshape(-1)
                    except Exception as exc:
                        logger.warning(f"Cannot load embedding {npy_path}: {exc}")
                        continue
                    feats.append(feat)

                if feats:
                    self._add_feature_stack(person_id, feats, source=f"npy embeddings {person_dir}")

    def _add_feature_stack(self, person_id, feats, source):
        feats = np.stack(feats, axis=0).astype(np.float32)
        if person_id in self.memory:
            if self.memory[person_id].shape[1] != feats.shape[1]:
                logger.warning(
                    f"Skipping {source} for {person_id}: dim {feats.shape[1]} != existing dim {self.memory[person_id].shape[1]}"
                )
                return
            self.memory[person_id] = np.vstack([self.memory[person_id], feats])
        else:
            self.memory[person_id] = feats

        self.prototypes[person_id] = self.memory[person_id].mean(axis=0).astype(np.float32)
        self._rebuild_index()
        logger.info(f"Loaded {len(feats)} embeddings for {person_id} from {source}; dim={feats.shape[1]}")

    def _rebuild_index(self):
        self.index_ids = []
        rows = []
        for person_id, proto in self.prototypes.items():
            norm = np.linalg.norm(proto) + 1e-12
            self.index_ids.append(person_id)
            rows.append((proto / norm).astype(np.float32))
        self.index_matrix = np.stack(rows, axis=0) if rows else None

    def build(self):
        self.prototypes = {}
        self.memory = {}
        self._dimension_warnings = set()

        self._build_from_image_gallery()
        self._build_from_npy_dirs()

        if not self.prototypes:
            logger.warning(f"ReID gallery empty: {self.gallery_dir}")
        self.initial_empty = not bool(self.prototypes)
        self._rebuild_index()

    def query(self, feat, threshold=0.55):
        if not self.prototypes:
            return "unknown", -1.0

        if self.index_matrix is not None and self.index_matrix.shape[1] == feat.shape[0]:
            query = feat.astype(np.float32) / (np.linalg.norm(feat) + 1e-12)
            scores = self.index_matrix @ query
            best_pos = int(np.argmax(scores))
            best_id = self.index_ids[best_pos]
            best_score = float(scores[best_pos])
        else:
            best_id = "unknown"
            best_score = -1.0
            for person_id, proto in self.prototypes.items():
                if proto.shape != feat.shape:
                    if person_id not in self._dimension_warnings:
                        logger.warning(
                            f"Skipping gallery id={person_id}: embedding dim {proto.shape[0]} != FastReID dim {feat.shape[0]}"
                        )
                        self._dimension_warnings.add(person_id)
                    continue
                score = self.cosine_similarity(feat, proto)
                if score > best_score:
                    best_score = score
                    best_id = person_id

        if best_score < threshold:
            return "unknown", best_score
        return best_id, best_score

    def get_top_matches(self, feat, topk=3):
        matches = []
        for person_id, proto in self.prototypes.items():
            if proto.shape != feat.shape:
                continue
            matches.append((person_id, self.cosine_similarity(feat, proto), None))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:topk]

    def add_embedding(self, person_id, feat):
        if person_id == "unknown":
            raise ValueError('Refusing to add ReID embedding under person_id="unknown"')
        if person_id not in self.memory:
            self.memory[person_id] = np.expand_dims(feat, axis=0)
        else:
            self.memory[person_id] = np.vstack([self.memory[person_id], feat[None, ...]])

        self.prototypes[person_id] = self.memory[person_id].mean(axis=0).astype(np.float32)
        self._rebuild_index()

    def add_crop(self, person_id, crop_bgr):
        feat = self.extractor.extract(crop_bgr)
        self.add_embedding(person_id, feat)
        return feat
