from src.utils.logger import get_logger

logger = get_logger(__name__)


class GlobalIDManager:
    def __init__(self, gallery, threshold=0.55, new_prefix="gid", reid_interval=10):
        self.gallery = gallery
        self.threshold = threshold
        self.new_prefix = new_prefix
        self.reid_interval = int(reid_interval)
        # map: (camera_id, local_track_id) -> (global_id: str, score: float)
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
        self.track_frame_count[key] = count + 1

        should_reid = self.reid_interval > 0 and count % self.reid_interval == 0
        # If we have a cached mapping and we don't need to re-run ReID, return cached score
        if key in self.track_to_global and not should_reid:
            gid, cached_score = self.track_to_global[key]
            return gid, float(cached_score)

        feat = self.gallery.extractor.extract(crop_bgr)
        matched_id, score = self.gallery.query(feat, threshold=self.threshold)

        if matched_id == "unknown":
            # reuse existing mapping if present, otherwise create a new global id
            existing = self.track_to_global.get(key)
            if existing is not None:
                matched_id = existing[0]
            else:
                matched_id = self._new_global_id()
                try:
                    # only add embedding for a concrete global id, not for the literal "unknown"
                    self.gallery.add_embedding(matched_id, feat)
                except Exception as exc:
                    logger.warning(f"Failed to add embedding for {matched_id}: {exc}", exc_info=True)
            logger.info(f"Assigned new global_id={matched_id} for camera={camera_id} track={local_track_id}")

        # cache both id and score so cached hits can return a realistic confidence
        self.track_to_global[key] = (matched_id, float(score))
        return matched_id, float(score)

    def forget_track(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        if key in self.track_to_global:
            del self.track_to_global[key]
        self.track_frame_count.pop(key, None)
