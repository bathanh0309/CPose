"""CPU OSNet-x0.25 ReID wrapper for realtime CPose."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid


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
        self.gallery: dict[str, np.ndarray] = {}
        self._frame_count = 0
        self.gallery_disabled_reason: str | None = None

    def register(self, person_id: str, crop_bgr: np.ndarray) -> None:
        """Register a person from a BGR crop."""
        feat = self._extract(crop_bgr)
        if feat is not None:
            self.gallery[str(person_id)] = feat
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
        """Load precomputed .npy embeddings into the gallery."""
        aliases = id_aliases or {}
        loaded = 0
        for root_value in embedding_dirs or []:
            root = Path(root_value)
            if not root.exists():
                continue
            for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                vectors: list[np.ndarray] = []
                for npy_path in sorted(person_dir.glob("*.npy")):
                    try:
                        arr = np.load(str(npy_path)).astype(np.float32).reshape(-1)
                    except Exception:
                        continue
                    if arr.size != 512:
                        continue
                    norm = np.linalg.norm(arr) + 1e-12
                    vectors.append(arr / norm)
                if not vectors:
                    continue
                pid = aliases.get(person_dir.name, person_dir.name)
                proto = np.mean(np.stack(vectors, axis=0), axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-12)
                self.gallery[str(pid)] = proto.astype(np.float32)
                loaded += 1
        self._trim_gallery()
        if loaded < self.min_gallery_size:
            self.gallery_disabled_reason = "ReID gallery too small; matching disabled."
        else:
            self.gallery_disabled_reason = None
        return loaded

    def identify(self, crop_bgr: np.ndarray, bbox_area: float) -> tuple[str, float]:
        """Return the best matching person id and cosine similarity."""
        if float(bbox_area) < self.min_crop_area:
            return "too_small", 0.0
        feat = self._extract(crop_bgr)
        if feat is None or not self.gallery or self.gallery_disabled_reason:
            return "unknown", 0.0
        best_id, best_sim = "unknown", 0.0
        for pid, gfeat in self.gallery.items():
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
        for pid, gfeat in self.gallery.items():
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
