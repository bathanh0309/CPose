import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReIDGallery:
    def __init__(self, extractor, gallery_dir, embedding_dirs=None, id_aliases=None):
        self.extractor = extractor
        self.gallery_dir = Path(gallery_dir)
        self.embedding_dirs = [Path(path) for path in (embedding_dirs or [])]
        self.id_aliases = {str(k): str(v) for k, v in (id_aliases or {}).items()}
        self.prototypes = {}
        self.memory = {}
        self.initial_empty = True
        self._dimension_warnings = set()
        self.index_ids = []
        self.index_matrix = None
        self.index_by_dim = {}

    @staticmethod
    def _standardize_dim(feat, target_dim=512):
        feat = feat.flatten()
        if feat.shape[0] == target_dim:
            return feat
        feat_2d = feat.reshape(1, -1)
        import cv2
        resized = cv2.resize(feat_2d, (target_dim, 1), interpolation=cv2.INTER_LINEAR).flatten()
        return resized / (np.linalg.norm(resized) + 1e-12)

    @staticmethod
    def cosine_similarity(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def _canonical_id(self, person_id: str) -> str:
        return self.id_aliases.get(str(person_id), str(person_id))

    def _person_id_from_dir(self, person_dir: Path) -> str:
        meta_path = person_dir / "meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                person_id = str(meta.get("person_id", "")).strip()
                if person_id:
                    return self._canonical_id(person_id)
            except Exception as exc:
                logger.warning(f"Cannot read embedding metadata {meta_path}: {exc}")
        return self._canonical_id(person_dir.name)

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
                    feat = self.extractor.extract(img)
                    feats.append(self._standardize_dim(feat))

            if feats:
                self._add_feature_stack(self._canonical_id(person_dir.name), feats, source="image gallery")

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
                        feat = self._standardize_dim(feat)
                        feat = self._standardize_dim(feat)
                    except Exception as exc:
                        logger.warning(f"Cannot load embedding {npy_path}: {exc}")
                        continue
                    feats.append(feat)

                if feats:
                    grouped = {}
                    for feat in feats:
                        grouped.setdefault(int(feat.shape[0]), []).append(feat)
                    for dim, dim_feats in grouped.items():
                        if len(grouped) > 1:
                            logger.warning(
                                f"Mixed embedding dims in {person_dir}; loading dim={dim} count={len(dim_feats)} separately"
                            )
                        self._add_feature_stack(person_id, dim_feats, source=f"npy embeddings {person_dir}")

    def _add_feature_stack(self, person_id, feats, source):
        feats = np.stack(feats, axis=0).astype(np.float32)
        key = (str(person_id), int(feats.shape[1]))
        if key in self.memory:
            if self.memory[key].shape[1] != feats.shape[1]:
                logger.warning(
                    f"Skipping {source} for {person_id}: dim {feats.shape[1]} != existing dim {self.memory[key].shape[1]}"
                )
                return
            self.memory[key] = np.vstack([self.memory[key], feats])
        else:
            self.memory[key] = feats

        proto_id = self._prototype_id(person_id, feats.shape[1])
        self.prototypes[proto_id] = self.memory[key].mean(axis=0).astype(np.float32)
        self._rebuild_index()
        logger.info(f"Loaded {len(feats)} embeddings for {person_id} from {source}; dim={feats.shape[1]}")

    @staticmethod
    def _prototype_id(person_id, dim):
        return str(person_id) if int(dim) == 2048 else f"{person_id}#d{int(dim)}"

    @staticmethod
    def _display_id(proto_id):
        return str(proto_id).split("#d", 1)[0]

    def _rebuild_index(self):
        self.index_ids = []
        rows = []
        by_dim = {}
        for person_id, proto in self.prototypes.items():
            norm = np.linalg.norm(proto) + 1e-12
            dim = int(proto.shape[0])
            row = (proto / norm).astype(np.float32)
            self.index_ids.append(person_id)
            rows.append(row)
            by_dim.setdefault(dim, {"ids": [], "rows": []})
            by_dim[dim]["ids"].append(person_id)
            by_dim[dim]["rows"].append(row)
        self.index_matrix = np.stack(rows, axis=0) if rows and len({row.shape[0] for row in rows}) == 1 else None
        self.index_by_dim = {
            dim: (payload["ids"], np.stack(payload["rows"], axis=0))
            for dim, payload in by_dim.items()
        }

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

        feat = self._standardize_dim(feat)
        dim = int(feat.shape[0])
        if dim in self.index_by_dim:
            ids, matrix = self.index_by_dim[dim]
            query = feat.astype(np.float32) / (np.linalg.norm(feat) + 1e-12)
            scores = matrix @ query
            best_pos = int(np.argmax(scores))
            best_id = self._display_id(ids[best_pos])
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
                    best_id = self._display_id(person_id)

        if best_score < threshold:
            return "unknown", best_score
        return best_id, best_score

    def get_top_matches(self, feat, topk=3):
        feat = self._standardize_dim(feat)
        matches = []
        for person_id, proto in self.prototypes.items():
            if proto.shape != feat.shape:
                continue
            matches.append((self._display_id(person_id), self.cosine_similarity(feat, proto), None))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:topk]

    def add_embedding(self, person_id, feat):
        if person_id == "unknown":
            raise ValueError('Refusing to add ReID embedding under person_id="unknown"')
        feat = self._standardize_dim(feat)
        key = (str(person_id), int(feat.shape[0]))
        if key not in self.memory:
            self.memory[key] = np.expand_dims(feat, axis=0)
        else:
            self.memory[key] = np.vstack([self.memory[key], feat[None, ...]])

        proto_id = self._prototype_id(person_id, feat.shape[0])
        self.prototypes[proto_id] = self.memory[key].mean(axis=0).astype(np.float32)
        self._rebuild_index()

    def add_crop(self, person_id, crop_bgr):
        feat = self.extractor.extract(crop_bgr)
        self.add_embedding(person_id, feat)
        return feat
