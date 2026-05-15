"""
src/reid/fusion.py

Stage 2 — Dynamic Weighted Fusion (Face + Body ReID)

Nguyên tắc:
  - face_conf cao (> 0.8)  → face chiếm 80%, body 20%
  - face_conf thấp (< 0.3) → body chiếm 90%, face 10%
  - Khoảng giữa            → nội suy tuyến tính

Đầu vào:
  face_feat   : np.ndarray shape (D,) hoặc None
  body_feat   : np.ndarray shape (D,) hoặc None
  face_conf   : float 0–1 (ArcFace / anti-spoofing confidence)

Đầu ra:
  fusion_score: float (cosine similarity tổng hợp)
  weights     : dict {"face": float, "body": float}
"""

from __future__ import annotations

import numpy as np


# ── Ngưỡng trọng số ──────────────────────────────────────────────────────────
FACE_HIGH_CONF = 0.8       # face_conf ≥ này → face dominant
FACE_LOW_CONF  = 0.3       # face_conf ≤ này → body dominant

WEIGHT_FACE_HIGH  = 0.80   # trọng số face khi face_conf cao
WEIGHT_BODY_HIGH  = 0.20

WEIGHT_FACE_LOW   = 0.10   # trọng số face khi face_conf thấp
WEIGHT_BODY_LOW   = 0.90


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vector 1-D."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _interpolate_weights(face_conf: float) -> tuple[float, float]:
    """
    Nội suy tuyến tính trọng số theo face_conf.

    Returns:
        (w_face, w_body) tổng = 1.0
    """
    face_conf = float(np.clip(face_conf, 0.0, 1.0))

    if face_conf >= FACE_HIGH_CONF:
        return WEIGHT_FACE_HIGH, WEIGHT_BODY_HIGH

    if face_conf <= FACE_LOW_CONF:
        return WEIGHT_FACE_LOW, WEIGHT_BODY_LOW

    # Nội suy tuyến tính trong khoảng [LOW_CONF, HIGH_CONF]
    t = (face_conf - FACE_LOW_CONF) / (FACE_HIGH_CONF - FACE_LOW_CONF)  # 0→1
    w_face = WEIGHT_FACE_LOW + t * (WEIGHT_FACE_HIGH - WEIGHT_FACE_LOW)
    w_body = 1.0 - w_face
    return w_face, w_body


def fusion_similarity(
    query_face: np.ndarray | None,
    query_body: np.ndarray | None,
    gallery_face: np.ndarray | None,
    gallery_body: np.ndarray | None,
    face_conf: float = 0.0,
) -> tuple[float, dict]:
    """
    Tính Fusion Similarity Score giữa query và một gallery entry.

    Args:
        query_face:   face embedding của query (None nếu không thấy mặt)
        query_body:   body embedding của query (None nếu crop không hợp lệ)
        gallery_face: face prototype trong gallery
        gallery_body: body prototype trong gallery
        face_conf:    confidence rằng mặt hiện tại là hợp lệ (0–1)

    Returns:
        (fusion_score, weights_dict)
        weights_dict = {"face": w_face, "body": w_body, "mode": str}
    """
    has_face = (query_face is not None and gallery_face is not None
                and query_face.shape == gallery_face.shape)
    has_body = (query_body is not None and gallery_body is not None
                and query_body.shape == gallery_body.shape)

    # Chỉ có 1 modal → trả thẳng cosine của modal đó
    if has_face and not has_body:
        score = _cosine_sim(query_face, gallery_face)
        return score, {"face": 1.0, "body": 0.0, "mode": "face_only"}

    if has_body and not has_face:
        score = _cosine_sim(query_body, gallery_body)
        return score, {"face": 0.0, "body": 1.0, "mode": "body_only"}

    if not has_face and not has_body:
        return 0.0, {"face": 0.0, "body": 0.0, "mode": "no_modal"}

    # Cả 2 modal → Dynamic Weighted Fusion
    w_face, w_body = _interpolate_weights(face_conf)
    sim_face = _cosine_sim(query_face, gallery_face)
    sim_body = _cosine_sim(query_body, gallery_body)
    score = w_face * sim_face + w_body * sim_body

    mode = (
        "face_dominant" if w_face >= WEIGHT_FACE_HIGH
        else "body_dominant" if w_body >= WEIGHT_BODY_LOW
        else "balanced"
    )

    return float(score), {
        "face": round(w_face, 3),
        "body": round(w_body, 3),
        "sim_face": round(sim_face, 4),
        "sim_body": round(sim_body, 4),
        "mode": mode,
    }


class MultiModalGallery:
    """
    Gallery lưu prototype face + body cho từng person_id.

    Load từ:
      data/face/{person_id}/*.npy  → face embeddings
      data/body/{person_id}/*.npy  → body embeddings

    Query:
      scores = gallery.query_all(query_face, query_body, face_conf)
      → list of (person_id, fusion_score, weights_dict) sorted descending
    """

    def __init__(self, face_dir: str = "data/face", body_dir: str = "data/body", id_aliases=None):
        self.face_dir = _to_path(face_dir)
        self.body_dir = _to_path(body_dir)
        self.id_aliases = {str(k): str(v) for k, v in (id_aliases or {}).items()}
        self.face_prototypes: dict[str, np.ndarray] = {}
        self.body_prototypes: dict[str, np.ndarray] = {}
        self._dim_warnings: set[str] = set()

    def _canonical_id(self, person_id: str) -> str:
        return self.id_aliases.get(str(person_id), str(person_id))

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        """Load và tính prototype (mean) cho mỗi người."""
        self.face_prototypes = self._load_prototypes(self.face_dir, label="face")
        self.body_prototypes = self._load_prototypes(self.body_dir, label="body")
        all_ids = set(self.face_prototypes) | set(self.body_prototypes)
        return self

    def _load_prototypes(self, root, label: str) -> dict[str, np.ndarray]:
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        prototypes: dict[str, np.ndarray] = {}
        if not root.exists():
            log.warning(f"Gallery dir not found: {root}")
            return prototypes
        for person_dir in root.iterdir():
            if not person_dir.is_dir():
                continue
            person_id = self._canonical_id(_resolve_person_id(person_dir))
            feats = []
            for npy in sorted(person_dir.glob("*.npy")):
                try:
                    f = np.load(str(npy)).astype(np.float32).reshape(-1)
                    feats.append(f)
                except Exception as exc:
                    log.warning(f"Cannot load {npy}: {exc}")
            if feats:
                proto = np.mean(feats, axis=0).astype(np.float32)
                prototypes[person_id] = proto
                log.info(f"  [{label}] {person_id}: {len(feats)} embeddings, dim={proto.shape[0]}")
        return prototypes

    # ── Update prototype (EMA) ─────────────────────────────────────────────────

    def update_ema(
        self,
        person_id: str,
        face_feat: np.ndarray | None = None,
        body_feat: np.ndarray | None = None,
        alpha: float = 0.1,
    ):
        """
        Cập nhật prototype bằng Exponential Moving Average.
        alpha=0.1 → nghiêng nhẹ về embedding mới nhất.
        """
        if face_feat is not None:
            if person_id in self.face_prototypes:
                self.face_prototypes[person_id] = (
                    (1 - alpha) * self.face_prototypes[person_id] + alpha * face_feat
                ).astype(np.float32)
            else:
                self.face_prototypes[person_id] = face_feat.astype(np.float32)

        if body_feat is not None:
            if person_id in self.body_prototypes:
                self.body_prototypes[person_id] = (
                    (1 - alpha) * self.body_prototypes[person_id] + alpha * body_feat
                ).astype(np.float32)
            else:
                self.body_prototypes[person_id] = body_feat.astype(np.float32)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_all(
        self,
        query_face: np.ndarray | None,
        query_body: np.ndarray | None,
        face_conf: float = 0.0,
    ) -> list[tuple[str, float, dict]]:
        """
        So khớp query với toàn bộ gallery.

        Returns:
            List of (person_id, fusion_score, weights_dict) sorted descending.
        """
        all_ids = set(self.face_prototypes) | set(self.body_prototypes)
        results = []
        for pid in all_ids:
            gf = self.face_prototypes.get(pid)
            gb = self.body_prototypes.get(pid)
            score, weights = fusion_similarity(query_face, query_body, gf, gb, face_conf)
            results.append((pid, score, weights))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def query_top1(
        self,
        query_face: np.ndarray | None,
        query_body: np.ndarray | None,
        face_conf: float = 0.0,
        threshold: float = 0.55,
    ) -> tuple[str, float, dict]:
        """
        Returns:
            (matched_person_id, score, weights) hoặc ("unknown", score, weights)
        """
        results = self.query_all(query_face, query_body, face_conf)
        if not results:
            return "unknown", 0.0, {"mode": "no_gallery"}
        best_id, best_score, best_weights = results[0]
        if best_score < threshold:
            return "unknown", best_score, best_weights
        return best_id, best_score, best_weights

    @property
    def is_empty(self) -> bool:
        return not (self.face_prototypes or self.body_prototypes)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_path(p):
    from pathlib import Path
    return Path(p)


def _resolve_person_id(person_dir) -> str:
    import json
    meta = person_dir / "meta.json"
    if meta.exists():
        try:
            with meta.open("r", encoding="utf-8") as f:
                data = json.load(f)
            pid = str(data.get("person_id", "")).strip()
            if pid:
                return pid
        except Exception:
            pass
    return person_dir.name
