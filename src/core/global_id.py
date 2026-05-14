from src.utils.logger import get_logger

logger = get_logger(__name__)


class GlobalIDManager:
    def __init__(self, gallery, threshold=0.55, new_prefix="gid", reid_interval=10):
        self.gallery = gallery
        self.threshold = threshold
        self.new_prefix = new_prefix
        self.reid_interval = int(reid_interval)
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
        if key in self.track_to_global and not should_reid:
            return self.track_to_global[key], 1.0

        feat = self.gallery.extractor.extract(crop_bgr)
        matched_id, score = self.gallery.query(feat, threshold=self.threshold)

        if matched_id == "unknown":
            matched_id = self.track_to_global.get(key) or self._new_global_id()
            self.gallery.add_embedding(matched_id, feat)
            logger.info(f"Assigned new global_id={matched_id} for camera={camera_id} track={local_track_id}")

        self.track_to_global[key] = matched_id
        return matched_id, float(score)

    def forget_track(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        if key in self.track_to_global:
            del self.track_to_global[key]
        self.track_frame_count.pop(key, None)
