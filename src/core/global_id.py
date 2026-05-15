"""
src/core/global_id.py

Stage 3 — Multi-Modal GlobalIDManager

Thay thế logic cấp ID tĩnh thành quản lý ID toàn cục xuyên camera,
kết hợp Face + Body embedding với Dynamic Weighted Fusion.

Luồng xử lý mỗi lần assign():
  1. Kiểm tra cache (track đã biết + chưa đến reid_interval) → trả cache.
  2. Trích xuất face_feat (nếu thấy mặt) + body_feat từ bbox.
  3. MultiModalGallery.query_top1() với face_conf động.
  4. Nếu score >= MATCH_THRESHOLD → trả global_id đã biết + EMA update.
  5. Nếu người mới → tạo global_id mới + lưu vào EMA cache.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Ngưỡng ───────────────────────────────────────────────────────────────────
MATCH_THRESHOLD   = 0.65   # fusion score tối thiểu để nhận dạng
EMA_ALPHA         = 0.10   # tốc độ cập nhật EMA prototype
MIN_HITS_NEW_ID   = 3      # số frame tối thiểu trước khi cấp ID mới
MIN_CROP_H        = 32
MIN_CROP_W        = 16


class GlobalIDManager:
    """
    Quản lý Global ID xuyên camera bằng Face + Body fusion.

    Args:
        gallery:        MultiModalGallery (face + body)
        threshold:      fusion score để nhận dạng (default MATCH_THRESHOLD)
        reid_interval:  tính ReID mỗi N frame (default 10)
        new_prefix:     prefix cho global_id mới (default "gid")
        ema_alpha:      tốc độ EMA update prototype
    """

    def __init__(
        self,
        gallery,
        threshold: float = MATCH_THRESHOLD,
        reid_interval: int = 10,
        new_prefix: str = "gid",
        ema_alpha: float = EMA_ALPHA,
        min_hits_before_new: int = MIN_HITS_NEW_ID,
    ):
        self.gallery = gallery
        self.threshold = float(threshold)
        self.reid_interval = max(1, int(reid_interval))
        self.new_prefix = str(new_prefix)
        self.ema_alpha = float(ema_alpha)
        self.min_hits = int(min_hits_before_new)

        # (camera_id, track_id) → {global_id, score, status, face_feat, body_feat}
        self._cache: dict[tuple, dict] = {}
        # (camera_id, track_id) → số lần thấy
        self._hit_count: dict[tuple, int] = {}

        self._next_id = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        frame: object,                      # np.ndarray BGR
        bbox,
        frame_idx: int = 0,
        face_feat=None,                     # np.ndarray | None  (từ ArcFace)
        face_conf: float = 0.0,             # confidence mặt hợp lệ
        body_extractor=None,                # BodyExtractor instance | None
    ) -> tuple[str, float, str, dict]:
        """
        Gán global_id cho (camera_id, local_track_id).

        Args:
            camera_id:       ID camera.
            local_track_id:  track_id từ ByteTrack.
            frame:           frame BGR hiện tại.
            bbox:            [x1, y1, x2, y2] của track.
            frame_idx:       index frame hiện tại.
            face_feat:       embedding khuôn mặt (None nếu không nhìn thấy mặt).
            face_conf:       confidence khuôn mặt (0–1).
            body_extractor:  BodyExtractor để lấy body embedding từ bbox.

        Returns:
            (global_id, fusion_score, status, weights_dict)
        """
        key = (str(camera_id), int(local_track_id))
        hits = self._hit_count.get(key, 0) + 1
        self._hit_count[key] = hits

        # ── Cache hit (chưa đến reid_interval) ───────────────────────────────
        if key in self._cache and hits % self.reid_interval != 0:
            c = self._cache[key]
            return c["global_id"], c["score"], "cache_hit", c.get("weights", {})

        # ── Lấy body embedding ────────────────────────────────────────────────
        body_feat = None
        if body_extractor is not None and frame is not None and bbox is not None:
            try:
                body_feat = body_extractor.extract_from_bbox(frame, bbox)
            except Exception as exc:
                logger.warning(f"body_extractor failed track={local_track_id}: {exc}")

        # ── Kiểm tra crop body có đủ kích thước không ────────────────────────
        if body_feat is None and face_feat is None:
            if key in self._cache:
                c = self._cache[key]
                return c["global_id"], c["score"], "no_feat_cache", c.get("weights", {})
            return f"pending_{local_track_id}", 0.0, "no_feat", {}

        # ── Query gallery ─────────────────────────────────────────────────────
        matched_id, score, weights = self.gallery.query_top1(
            query_face=face_feat,
            query_body=body_feat,
            face_conf=face_conf,
            threshold=self.threshold,
        )

        if matched_id != "unknown":
            # Nhận dạng thành công → EMA update
            status = "gallery_match"
            global_id = matched_id
            self.gallery.update_ema(
                global_id,
                face_feat=face_feat,
                body_feat=body_feat,
                alpha=self.ema_alpha,
            )
        else:
            # Người lạ → dùng cache nếu có, hoặc chờ đủ hits rồi cấp ID mới
            if key in self._cache:
                global_id = self._cache[key]["global_id"]
                status = "gallery_miss_cache"
            elif hits < self.min_hits:
                return f"pending_{local_track_id}", score, "pending", weights
            else:
                global_id = self._new_global_id()
                status = "new_global_id"
                self.gallery.update_ema(
                    global_id,
                    face_feat=face_feat,
                    body_feat=body_feat,
                    alpha=1.0,          # khởi tạo trực tiếp, không EMA
                )
                logger.info(
                    f"New global_id={global_id} "
                    f"cam={camera_id} track={local_track_id} "
                    f"face_conf={face_conf:.2f} "
                    f"mode={weights.get('mode', '?')}"
                )

        # ── Lưu cache ─────────────────────────────────────────────────────────
        self._cache[key] = {
            "global_id": global_id,
            "score": float(score),
            "status": status,
            "weights": weights,
            "face_feat": face_feat,
            "body_feat": body_feat,
        }
        return global_id, float(score), status, weights

    def forget_track(self, camera_id: str, local_track_id: int):
        """Gọi khi track bị mất — giải phóng cache."""
        key = (str(camera_id), int(local_track_id))
        self._cache.pop(key, None)
        self._hit_count.pop(key, None)

    def get_active_ids(self) -> dict[tuple, str]:
        """Trả dict {(camera_id, track_id): global_id} của tất cả track đang active."""
        return {k: v["global_id"] for k, v in self._cache.items()}

    # ── Private ───────────────────────────────────────────────────────────────

    def _new_global_id(self) -> str:
        gid = f"{self.new_prefix}_{self._next_id:05d}"
        self._next_id += 1
        return gid
