"""
Build body embeddings for a person gallery.

Face embeddings are expected to already exist in the output folder
as face_*.npy files. This script does not import or run any face model;
it only creates body_*.npy files in the same folder.

Usage:
  python data/build-body-gallery.py

  python data/build-body-gallery.py \
    --person_id APhu \
    --video_dir "D:/Capstone_Project/data/body/APhu" \
    --out_dir   "D:/Capstone_Project/data/embeddings/APhu" \
    --sample_every 15 \
    --max_crops 80
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
import torchreid
from ultralytics import YOLO

DEFAULT_PERSON_ID = "Huy"
DEFAULT_MEDIA_DIR = "D:/Capstone_Project/data/body/Huy"
DEFAULT_OUT_DIR = "D:/Capstone_Project/data/embeddings/Huy"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_body_extractor(osnet_weight: str):
    return torchreid.utils.FeatureExtractor(
        model_name="osnet_x0_25",
        model_path=osnet_weight,
        device="cpu",
        image_size=(256, 128),
    )


def extract_body_feat(extractor, crop_bgr: np.ndarray) -> np.ndarray | None:
    """Return one L2-normalized OSNet feature vector."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h * w < 2500:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    feat = extractor(crop_rgb)  # torch Tensor [1, 512]
    feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy().flatten().astype(np.float32)


def next_body_index(out_dir: Path) -> int:
    indexes: list[int] = []
    for path in out_dir.glob("body_*.npy"):
        try:
            indexes.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indexes, default=-1) + 1


def extract_largest_person_body(
    frame: np.ndarray,
    detector: YOLO,
    body_extractor,
) -> np.ndarray | None:
    results = detector.predict(
        frame,
        imgsz=640,
        conf=0.40,
        classes=[0],
        device="cpu",
        verbose=False,
    )
    if not results or not results[0].boxes:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, boxes[best_idx])

    crop = frame[y1:y2, x1:x2]
    return extract_body_feat(body_extractor, crop)


def save_body_embedding(out_dir: Path, body_feat: np.ndarray, index: int) -> Path:
    out_path = out_dir / f"body_{index:04d}.npy"
    np.save(str(out_path), body_feat)
    return out_path


def process_video(
    video_path: Path,
    detector: YOLO,
    body_extractor,
    sample_every: int,
    max_crops: int,
    out_dir: Path,
    start_index: int,
    existing_body_count: int,
) -> tuple[int, int]:
    """
    Process one video and save body embeddings.

    Returns:
        (saved_count, next_index)
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0, start_index

    frame_idx = 0
    body_count = 0

    print(f"  -> Processing: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        if (existing_body_count + body_count) >= max_crops:
            print(f"     [DONE] Enough body embeddings: {max_crops}")
            break

        body_feat = extract_largest_person_body(frame, detector, body_extractor)
        if body_feat is None:
            continue

        save_body_embedding(out_dir, body_feat, start_index + body_count)
        body_count += 1

    cap.release()
    return body_count, start_index + body_count


def process_image(
    image_path: Path,
    detector: YOLO,
    body_extractor,
    max_crops: int,
    out_dir: Path,
    start_index: int,
    existing_body_count: int,
) -> tuple[int, int]:
    if existing_body_count >= max_crops:
        return 0, start_index

    print(f"  -> Processing: {image_path.name}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return 0, start_index

    body_feat = extract_largest_person_body(frame, detector, body_extractor)
    if body_feat is None:
        return 0, start_index

    save_body_embedding(out_dir, body_feat, start_index)
    return 1, start_index + 1


def find_media_files(media_dir: Path) -> tuple[list[Path], list[Path]]:
    if not media_dir.exists():
        raise FileNotFoundError(f"Media folder does not exist: {media_dir}")

    files = sorted(path for path in media_dir.iterdir() if path.is_file())
    videos = [path for path in files if path.suffix.lower() in VIDEO_EXTS]
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTS]
    return videos, images


def clear_body_outputs(out_dir: Path, person_id: str) -> None:
    for path in out_dir.glob("body_*.npy"):
        path.unlink()
    pkl_path = out_dir / f"{person_id}_embeddings.pkl"
    if pkl_path.exists():
        pkl_path.unlink()


def load_existing_meta(out_dir: Path) -> dict:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Cannot read existing meta.json: {exc}")
        return {}


def write_meta(
    out_dir: Path,
    person_id: str,
    media_dir: Path,
    sample_every: int,
    total_body_saved: int,
) -> None:
    meta = load_existing_meta(out_dir)
    existing_face = len(list(out_dir.glob("face_*.npy")))
    total_body_files = len(list(out_dir.glob("body_*.npy")))

    meta.update(
        {
            "person_id": person_id,
            "face_embeddings_existing": existing_face,
            "body_embeddings": total_body_files,
            "body_embeddings_added": total_body_saved,
            "body_source_dir": str(media_dir),
            "body_sample_every": sample_every,
            "body_embedding_type": "body",
            "body_model": "osnet_x0_25",
            "body_dim": 512,
            "body_updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--person_id",
        default=DEFAULT_PERSON_ID,
        help=f"Person ID, default={DEFAULT_PERSON_ID}",
    )
    parser.add_argument(
        "--video_dir",
        default=DEFAULT_MEDIA_DIR,
        help=f"Folder containing videos/images, default={DEFAULT_MEDIA_DIR}",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Embedding folder that already contains face_*.npy, default={DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--osnet_weight",
        default="D:/Capstone_Project/models/osnet_x0_25_msmt17.pth",
    )
    parser.add_argument(
        "--yolo_model",
        default="D:/Capstone_Project/models/yolo11n.pt",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=15,
        help="Use one frame every N frames",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=80,
        help="Maximum total body embeddings in out_dir",
    )
    parser.add_argument(
        "--overwrite_body",
        action="store_true",
        help="Delete existing body_*.npy before rebuilding; face_*.npy is kept",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_body:
        clear_body_outputs(out_dir, args.person_id)

    media_dir = Path(args.video_dir)
    videos, images = find_media_files(media_dir)

    print("[1/2] Load YOLO detector...")
    detector = YOLO(args.yolo_model)

    print("[2/2] Load OSNet body extractor...")
    body_extractor = load_body_extractor(args.osnet_weight)

    face_count = len(list(out_dir.glob("face_*.npy")))
    body_index = next_body_index(out_dir)
    existing_body_count = len(list(out_dir.glob("body_*.npy")))

    print(f"\nFound {len(videos)} videos and {len(images)} images in {media_dir}")
    print(f"Output folder: {out_dir}")
    print(f"Existing face embeddings: {face_count}")
    print(f"Existing body embeddings: {existing_body_count}")

    total_body_saved = 0
    for video_path in videos:
        saved, body_index = process_video(
            video_path=video_path,
            detector=detector,
            body_extractor=body_extractor,
            sample_every=args.sample_every,
            max_crops=args.max_crops,
            out_dir=out_dir,
            start_index=body_index,
            existing_body_count=existing_body_count + total_body_saved,
        )
        total_body_saved += saved
        print(f"     body={saved}")

        if (existing_body_count + total_body_saved) >= args.max_crops:
            break

    for image_path in images:
        saved, body_index = process_image(
            image_path=image_path,
            detector=detector,
            body_extractor=body_extractor,
            max_crops=args.max_crops,
            out_dir=out_dir,
            start_index=body_index,
            existing_body_count=existing_body_count + total_body_saved,
        )
        total_body_saved += saved
        print(f"     body={saved}")

        if (existing_body_count + total_body_saved) >= args.max_crops:
            break

    write_meta(
        out_dir=out_dir,
        person_id=args.person_id,
        media_dir=media_dir,
        sample_every=args.sample_every,
        total_body_saved=total_body_saved,
    )

    print(f"\nDone. Saved to {out_dir}")
    print(f"   Body embeddings added : {total_body_saved}")
    print(f"   Body embeddings total : {len(list(out_dir.glob('body_*.npy')))}")
    print(f"   Face embeddings kept  : {len(list(out_dir.glob('face_*.npy')))}")
    print(f"   meta.json             : updated")


if __name__ == "__main__":
    main()
