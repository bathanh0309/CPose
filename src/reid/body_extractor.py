"""
src/reid/body_extractor.py

Stage 1 — Body Embedding Extractor
Nhận bbox từ ByteTrack + frame gốc → crop toàn thân → FastReID embedding 2048d → lưu .npy

Cấu trúc output:
  data/body/{person_id}/emb_{STT:02d}.npy
  data/body/{person_id}/meta.json
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CROP_H = 48
MIN_CROP_W = 20


def clip_bbox(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Cắt crop toàn thân từ frame, trả None nếu crop không hợp lệ."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < MIN_CROP_W or y2 - y1 < MIN_CROP_H:
        return None
    return frame[y1:y2, x1:x2]


class BodyExtractor:
    """
    Bọc FastReIDExtractor để trích xuất body embedding từ bbox.

    Args:
        extractor: instance của FastReIDExtractor (đã khởi tạo sẵn)
    """

    def __init__(self, extractor):
        if not hasattr(extractor, "extract"):
            raise TypeError("BodyExtractor cần FastReIDExtractor có phương thức extract(image_bgr)")
        self.extractor = extractor

    def extract_from_bbox(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """
        Cắt crop từ bbox rồi trích xuất embedding.

        Returns:
            np.ndarray shape (D,) dtype float32, hoặc None nếu crop không hợp lệ.
        """
        crop = clip_bbox(frame, bbox)
        if crop is None:
            return None
        try:
            feat = self.extractor.extract(crop)
            return feat.astype(np.float32)
        except Exception as exc:
            logger.warning(f"BodyExtractor.extract_from_bbox failed: {exc}", exc_info=True)
            return None

    def extract_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] < MIN_CROP_H or crop.shape[1] < MIN_CROP_W:
            return None
        try:
            return self.extractor.extract(crop).astype(np.float32)
        except Exception as exc:
            logger.warning(f"BodyExtractor.extract_from_crop failed: {exc}", exc_info=True)
            return None


class BodyGalleryBuilder:
    """
    Script chạy tự động trích xuất body gallery từ video mẫu.

    Cách dùng:
        builder = BodyGalleryBuilder(extractor, output_dir="data/body")
        builder.build_from_video(
            video_path="data/input/cam2_...mp4",
            person_id="APhu",
            max_frames=300,
            sample_every=10,
        )

    Output:
        data/body/APhu/emb_00.npy
        data/body/APhu/emb_01.npy
        ...
        data/body/APhu/meta.json
    """

    def __init__(self, extractor, output_dir: str = "data/body"):
        self.body_extractor = BodyExtractor(extractor)
        self.output_dir = Path(output_dir)

    def _save_embedding(self, person_dir: Path, feat: np.ndarray, idx: int):
        person_dir.mkdir(parents=True, exist_ok=True)
        out = person_dir / f"emb_{idx:02d}.npy"
        np.save(str(out), feat)
        return out

    def _save_meta(self, person_dir: Path, meta: dict):
        person_dir.mkdir(parents=True, exist_ok=True)
        meta_path = person_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta_path

    def build_from_video(
        self,
        video_path: str,
        person_id: str,
        max_frames: int = 500,
        sample_every: int = 10,
        detector=None,
    ) -> list[str]:
        """
        Trích xuất body embeddings từ video.

        Args:
            video_path:   đường dẫn video.
            person_id:    tên thư mục lưu (ví dụ: "APhu").
            max_frames:   số frame tối đa xử lý.
            sample_every: lấy mẫu mỗi N frame để đa dạng pose.
            detector:     YoloPoseTracker instance (nếu None thì dùng full-frame crop).

        Returns:
            Danh sách đường dẫn các file .npy đã lưu.
        """
        person_dir = self.output_dir / person_id
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        saved_paths = []
        scores = []
        frame_idx = 0
        emb_idx = 0
        t0 = time.time()

        logger.info(f"Building body gallery for '{person_id}' from {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break
                frame_idx += 1

                if frame_idx % sample_every != 0:
                    continue

                if detector is not None:
                    try:
                        detections, _ = detector.infer(frame, persist=False)
                    except Exception as exc:
                        logger.warning(f"[frame {frame_idx}] detector failed: {exc}")
                        continue

                    for det in detections:
                        feat = self.body_extractor.extract_from_bbox(frame, det["bbox"])
                        if feat is None:
                            continue
                        out = self._save_embedding(person_dir, feat, emb_idx)
                        saved_paths.append(str(out))
                        scores.append(float(det.get("score", 0.0)))
                        emb_idx += 1
                        logger.info(f"  Saved body emb #{emb_idx}: {out.name}")
                else:
                    # Không có detector: dùng toàn bộ frame
                    feat = self.body_extractor.extract_from_crop(frame)
                    if feat is None:
                        continue
                    out = self._save_embedding(person_dir, feat, emb_idx)
                    saved_paths.append(str(out))
                    emb_idx += 1
                    logger.info(f"  Saved body emb #{emb_idx}: {out.name}")

        finally:
            cap.release()

        meta = {
            "person_id": person_id,
            "source_video": str(video_path),
            "total_embeddings": emb_idx,
            "avg_detection_score": float(np.mean(scores)) if scores else 0.0,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 2),
            "note": "body embeddings via FastReID",
        }
        self._save_meta(person_dir, meta)
        logger.info(f"Body gallery built: {emb_idx} embeddings → {person_dir}")
        return saved_paths
