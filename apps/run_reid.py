import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reid.fast_reid import FastReIDExtractor
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug FastReID query against a gallery")
    parser.add_argument("--query", type=str, required=True, help="query image path")
    parser.add_argument("--gallery", type=str, default=None, help="gallery folder")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def extract_feature(extractor, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return extractor.extract(img)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    extractor = FastReIDExtractor(
        fastreid_root=cfg["reid"]["fastreid_root"],
        config_path=cfg["reid"]["config"],
        weights_path=cfg["reid"]["weights"],
        device=cfg["system"]["device"],
    )

    query_feat = extract_feature(extractor, args.query)
    gallery_dir = Path(args.gallery) if args.gallery else Path(cfg["reid"]["gallery_dir"])

    gallery_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        gallery_files.extend(gallery_dir.rglob(ext))

    scores = []
    for img_path in gallery_files:
        gallery_feat = extract_feature(extractor, img_path)
        scores.append((str(img_path), cosine_similarity(query_feat, gallery_feat)))

    scores.sort(key=lambda item: item[1], reverse=True)

    logger.info("Top matches:")
    for path, score in scores[:args.topk]:
        logger.info(f"{score:.4f}  {path}")


if __name__ == "__main__":
    main()
