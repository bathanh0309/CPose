"""CPU OSNet-x0.25 ReID wrapper for realtime CPose."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OSNetReID:
    """Extract and match OSNet-x0.25 body embeddings on CPU."""

    def __init__(
        self,
        weight_path: str,
        threshold: float = 0.65,
        reid_interval: int = 15,
        max_gallery: int = 10,
        min_crop_area: float = 2500.0,
        min_gallery_size: int = 5,
    ) -> None:
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x0_25",
            model_path=str(weight_path),
            device="cpu",
            image_size=(256, 128),
        )
        self.threshold = float(threshold)
        self.interval = int(reid_interval)
        self.max_gallery = int(max_gallery)
        self.min_crop_area = float(min_crop_area)
        self.min_gallery_size = int(min_gallery_size)
        self.extractor_dim = 512
        self.gallery: dict[str, dict[str, np.ndarray]] = {}
        self._frame_count = 0
        self.gallery_disabled_reason: str | None = None
        self.load_stats = {
            "loaded": 0,
            "persons": [],
            "skipped_face_arcface": 0,
            "skipped_fastreid": 0,
            "skipped_dim": 0,
            "skipped_missing_meta": 0,
            "skipped_other": 0,
        }

    def register(self, person_id: str, crop_bgr: np.ndarray) -> None:
        """Register a person from a BGR crop."""
        feat = self._extract(crop_bgr)
        if feat is not None:
            self.gallery.setdefault(str(person_id), {})["body_osnet"] = feat
            self._trim_gallery()

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract one normalized OSNet feature from a BGR crop."""
        feat = self._extract(crop_bgr)
        if feat is None:
            raise ValueError("OSNetReID received an empty crop")
        return feat

    def load_gallery_embeddings(
        self,
        embedding_dirs: Iterable[str] | None,
        id_aliases: dict[str, str] | None = None,
    ) -> int:
        """Load only body OSNet embeddings into the gallery.

        Embeddings must be accompanied by meta.json declaring:
        embedding_type=body, model=osnet_x0_25, dim=extractor_dim.
        Face and FastReID galleries are different feature spaces and are skipped.
        """
        aliases = id_aliases or {}
        loaded = 0
        self.gallery = {}
        self.load_stats = {
            "loaded": 0,
            "persons": [],
            "skipped_face_arcface": 0,
            "skipped_fastreid": 0,
            "skipped_dim": 0,
            "skipped_missing_meta": 0,
            "skipped_other": 0,
        }
        if isinstance(embedding_dirs, (str, Path)):
            embedding_dirs = [str(embedding_dirs)]

        for root_value in embedding_dirs or []:
            loaded += self._load_gallery_path(Path(root_value), aliases)
        self._trim_gallery()
        self.load_stats["loaded"] = loaded
        if loaded < self.min_gallery_size:
            self.gallery_disabled_reason = "ReID gallery too small; matching disabled."
        else:
            self.gallery_disabled_reason = None
        if loaded:
            logger.info(f"Loaded OSNet body gallery: {', '.join(sorted(self.gallery))}")
        skipped = self.load_stats
        if skipped["skipped_face_arcface"] or skipped["skipped_fastreid"]:
            logger.warning(
                "Skipped incompatible embeddings: "
                f"face_arcface={skipped['skipped_face_arcface']}, "
                f"fastreid={skipped['skipped_fastreid']}"
            )
        return loaded

    def _load_gallery_path(self, path: Path, aliases: dict[str, str]) -> int:
        if not path.exists():
            logger.warning(f"OSNet gallery path not found: {path}")
            return 0

        if path.is_file():
            if path.suffix.lower() == ".pkl":
                return self._load_gallery_pickle(path, aliases)
            logger.warning(f"Skip unsupported OSNet gallery file: {path}")
            return 0

        pkl_files = sorted(path.glob("*.pkl"))
        if pkl_files:
            return sum(self._load_gallery_pickle(pkl_path, aliases) for pkl_path in pkl_files)

        pkl_files = sorted(path.glob("*/*_embeddings.pkl"))
        if pkl_files:
            return sum(self._load_gallery_pickle(pkl_path, aliases) for pkl_path in pkl_files)

        loaded = 0
        for person_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            loaded += self._load_gallery_person_dir(person_dir, aliases)
        return loaded

    def _load_gallery_pickle(self, pkl_path: Path, aliases: dict[str, str]) -> int:
        try:
            with pkl_path.open("rb") as file:
                data = pickle.load(file)
        except Exception as exc:
            logger.warning(f"Skip unreadable OSNet gallery pkl: {pkl_path}: {exc}")
            self.load_stats["skipped_other"] += 1
            return 0

        body = data.get("body") if isinstance(data, dict) else None
        if not isinstance(body, dict):
            logger.warning(f"Skip OSNet gallery pkl without body section: {pkl_path}")
            self.load_stats["skipped_other"] += 1
            return 0

        model = str(body.get("model", "")).lower()
        if model and model != "osnet_x0_25":
            logger.warning(f"Skip pkl with incompatible body model={model}: {pkl_path}")
            self.load_stats["skipped_other"] += 1
            return 0

        proto = body.get("prototype")
        if proto is not None:
            arr = np.asarray(proto, dtype=np.float32).reshape(-1)
        else:
            embeddings = np.asarray(body.get("embeddings", []), dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                logger.warning(f"Skip pkl without body embeddings: {pkl_path}")
                self.load_stats["skipped_other"] += 1
                return 0
            if embeddings.shape[1] != self.extractor_dim:
                self.load_stats["skipped_dim"] += int(embeddings.shape[0])
                logger.warning(
                    f"Skip pkl body dim={embeddings.shape[1]}, expected={self.extractor_dim}: {pkl_path}"
                )
                return 0
            arr = np.mean(embeddings, axis=0).astype(np.float32)

        if arr.size != self.extractor_dim:
            self.load_stats["skipped_dim"] += 1
            logger.warning(f"Skip pkl prototype dim={arr.size}, expected={self.extractor_dim}: {pkl_path}")
            return 0

        pid_value = str(data.get("person_id") or pkl_path.stem.replace("_embeddings", ""))
        pid = aliases.get(pid_value, pid_value)
        proto = arr / (np.linalg.norm(arr) + 1e-12)
        self.gallery.setdefault(str(pid), {})["body_osnet"] = proto.astype(np.float32)
        self.load_stats["persons"].append(str(pid))
        logger.info(f"Loaded OSNet body gallery pkl: {pid} <- {pkl_path}")
        return 1

    def _load_gallery_person_dir(self, person_dir: Path, aliases: dict[str, str]) -> int:
        meta = self._read_meta(person_dir)
        if not self._is_compatible_body_osnet(meta, person_dir):
            return 0
        vectors: list[np.ndarray] = []
        for npy_path in sorted(person_dir.glob("body_*.npy")):
            try:
                arr = np.load(str(npy_path)).astype(np.float32).reshape(-1)
            except Exception:
                continue
            if arr.size != self.extractor_dim:
                self.load_stats["skipped_dim"] += 1
                continue
            norm = np.linalg.norm(arr) + 1e-12
            vectors.append(arr / norm)
        if not vectors:
            return 0
        pid_value = str(meta.get("person_id") or person_dir.name)
        pid = aliases.get(pid_value, pid_value)
        proto = np.mean(np.stack(vectors, axis=0), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        self.gallery.setdefault(str(pid), {})["body_osnet"] = proto.astype(np.float32)
        self.load_stats["persons"].append(str(pid))
        return 1

    def identify(self, crop_bgr: np.ndarray, bbox_area: float) -> tuple[str, float]:
        """Return the best matching person id and cosine similarity."""
        if float(bbox_area) < self.min_crop_area:
            return "too_small", 0.0
        feat = self._extract(crop_bgr)
        if feat is None or not self.gallery or self.gallery_disabled_reason:
            return "unknown", 0.0
        best_id, best_sim = "unknown", 0.0
        for pid, entry in self.gallery.items():
            gfeat = entry.get("body_osnet")
            if gfeat is None:
                continue
            if gfeat.shape != feat.shape:
                continue
            sim = float(np.dot(feat, gfeat))
            if sim > best_sim:
                best_sim, best_id = sim, pid
        if best_sim < self.threshold:
            return "unknown", best_sim
        return best_id, best_sim

    def get_top_matches(self, crop_bgr: np.ndarray, topk: int = 3) -> list[tuple[str, float, None]]:
        """Return top gallery matches for UI panels."""
        feat = self._extract(crop_bgr)
        if feat is None:
            return []
        matches: list[tuple[str, float, None]] = []
        for pid, entry in self.gallery.items():
            gfeat = entry.get("body_osnet")
            if gfeat is None:
                continue
            if gfeat.shape == feat.shape:
                matches.append((pid, float(np.dot(feat, gfeat)), None))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:topk]

    def _extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            feat = self.extractor(crop_rgb)
            feat = F.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def should_run(self, frame_idx: int) -> bool:
        """Return whether ReID should run for this frame index."""
        return int(frame_idx) % max(1, self.interval) == 0

    def _trim_gallery(self) -> None:
        if self.max_gallery <= 0:
            return
        while len(self.gallery) > self.max_gallery:
            oldest_key = next(iter(self.gallery))
            self.gallery.pop(oldest_key, None)

    def _read_meta(self, person_dir: Path) -> dict:
        meta_path = person_dir / "meta.json"
        if not meta_path.exists():
            self.load_stats["skipped_missing_meta"] += len(list(person_dir.glob("*.npy")))
            logger.warning(f"Skip OSNet gallery without meta.json: {person_dir}")
            return {}
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as exc:
            logger.warning(f"Skip unreadable OSNet gallery meta: {meta_path}: {exc}")
            self.load_stats["skipped_other"] += len(list(person_dir.glob("*.npy")))
            return {}

    def _is_compatible_body_osnet(self, meta: dict, person_dir: Path) -> bool:
        npy_count = len(list(person_dir.glob("*.npy")))
        embedding_type = str(meta.get("embedding_type", "")).lower()
        model = str(meta.get("model", "")).lower()
        note = str(meta.get("note", "")).lower()
        path_text = str(person_dir).replace("\\", "/").lower()
        dim = int(meta.get("dim", -1) or -1)
        if embedding_type == "face" or "/face/" in path_text or path_text.endswith("/face"):
            self.load_stats["skipped_face_arcface"] += npy_count
            logger.warning(f"Skip face embedding for OSNet body matcher: {person_dir}")
            return False
        if "fastreid" in model or "fastreid" in note or "/body_fastreid/" in path_text:
            self.load_stats["skipped_fastreid"] += npy_count
            logger.warning(f"Skip FastReID embedding for OSNet matcher: {person_dir}")
            return False
        if embedding_type != "body" or model != "osnet_x0_25":
            self.load_stats["skipped_other"] += npy_count
            logger.warning(f"Skip incompatible embedding for OSNet matcher: {person_dir}")
            return False
        if dim != self.extractor_dim:
            self.load_stats["skipped_dim"] += npy_count
            logger.warning(f"Skip OSNet gallery with dim={dim}, expected={self.extractor_dim}: {person_dir}")
            return False
        return True
