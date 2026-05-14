from src.utils.logger import get_logger

logger = get_logger(__name__)


class GlobalIDManager:
    def __init__(
        self,
        gallery,
        threshold=0.55,
        new_prefix="gid",
        reid_interval=10,
        min_hits_before_new=3,
        min_crop_size=(32, 16),
    ):
        self.gallery = gallery
        self.threshold = float(threshold)
        self.new_prefix = new_prefix
        self.reid_interval = max(1, int(reid_interval))
        self.min_hits_before_new = max(1, int(min_hits_before_new))
        self.min_crop_h, self.min_crop_w = map(int, min_crop_size)
        # map: (camera_id, local_track_id) -> (global_id: str, score: float, status: str)
        self.track_to_global = {}
        self.track_frame_count = {}
        self.next_id = 1

    def _new_global_id(self):
        gid = f"{self.new_prefix}_{self.next_id:05d}"
        self.next_id += 1
        return gid

    def assign(self, camera_id, local_track_id, crop_bgr, frame_idx=None):
        key = (str(camera_id), int(local_track_id))
        count = self.track_frame_count.get(key, 0)
        seen_count = count + 1
        self.track_frame_count[key] = seen_count

        should_reid = count == 0 or count % self.reid_interval == 0
        if key in self.track_to_global and not should_reid:
            gid, cached_score, _ = self.track_to_global[key]
            return gid, float(cached_score), "cache_hit"

        h, w = crop_bgr.shape[:2]
        if h < self.min_crop_h or w < self.min_crop_w:
            existing = self.track_to_global.get(key)
            if existing is not None:
                gid, cached_score, _ = existing
                return gid, float(cached_score), "bad_crop_cache"
            return f"pending_{local_track_id}", 0.0, "bad_crop"

        feat = self.gallery.extractor.extract(crop_bgr)
        matched_id, score = self.gallery.query(feat, threshold=self.threshold)

        if matched_id == "unknown":
            existing = self.track_to_global.get(key)
            if existing is not None:
                matched_id = existing[0]
                status = "cache_hit"
            elif seen_count < self.min_hits_before_new:
                return f"pending_{local_track_id}", float(score), "pending"
            else:
                matched_id = self._new_global_id()
                status = "new_global_id"
                try:
                    self.gallery.add_embedding(matched_id, feat)
                except Exception as exc:
                    logger.warning(f"Failed to add embedding for {matched_id}: {exc}", exc_info=True)
            logger.info(f"Assigned new global_id={matched_id} for camera={camera_id} track={local_track_id}")
        else:
            status = "gallery_match"

        result = (matched_id, float(score), status)
        self.track_to_global[key] = result
        return result

    def forget_track(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        if key in self.track_to_global:
            del self.track_to_global[key]
        self.track_frame_count.pop(key, None)
