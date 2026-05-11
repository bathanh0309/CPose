"""Face/body gallery utilities with stacked embeddings and optional FAISS."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class GalleryEntry:
    person_id: str
    embeddings: np.ndarray
    meta: dict[str, Any]


class FaceGallery:
    def __init__(self, root: str | Path = "data/face") -> None:
        self.root = Path(root)
        self.entries: dict[str, GalleryEntry] = {}
        self.index: Any | None = None
        self.index_ids: list[str] = []

    def load(self) -> dict[str, GalleryEntry]:
        self.entries.clear()
        if not self.root.exists():
            return self.entries
        for person_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            embeddings_path = person_dir / "embeddings.npy"
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path).astype("float32")
            else:
                rows = [np.load(path).astype("float32") for path in sorted(person_dir.glob("emb_*.npy"))]
                if not rows:
                    continue
                embeddings = np.vstack(rows)
                np.save(embeddings_path, embeddings)
            meta_path = person_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta.setdefault("name", person_dir.name)
            meta["n_samples"] = int(embeddings.shape[0])
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self.entries[person_dir.name] = GalleryEntry(person_dir.name, self._normalize(embeddings), meta)
        return self.entries

    def build_faiss(self, output_index: str | Path = "data/face_gallery.faiss", output_ids: str | Path = "data/face_gallery_ids.json") -> bool:
        self.load()
        vectors: list[np.ndarray] = []
        ids: list[str] = []
        for person_id, entry in self.entries.items():
            vectors.append(entry.embeddings)
            ids.extend([person_id] * len(entry.embeddings))
        if not vectors:
            return False
        matrix = np.vstack(vectors).astype("float32")
        try:
            import faiss
        except Exception:
            return False
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(output_index))
        Path(output_ids).write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        self.index = index
        self.index_ids = ids
        return True

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


__all__ = ["FaceGallery", "GalleryEntry"]
