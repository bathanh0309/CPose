import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.config import normalize_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FastReIDExtractor:
    def __init__(self, fastreid_root, config_path, weights_path, device=None):
        self.fastreid_root = Path(fastreid_root).resolve()
        if not self.fastreid_root.exists():
            raise FileNotFoundError(f"fastreid_root not found: {self.fastreid_root}")

        self.config_path = Path(config_path).resolve()
        self.weights_path = Path(weights_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"FastReID config not found: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"FastReID weights not found: {self.weights_path}")
        self.device = normalize_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")

        fastreid_path = str(self.fastreid_root)
        if fastreid_path not in sys.path:
            sys.path.insert(0, fastreid_path)

        logger.info(f"Loading FastReID model: {self.weights_path}")
        try:
            from fastreid.config import get_cfg
            from demo.predictor import FeatureExtractionDemo
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"FastReID dependency missing: {exc.name}. "
                "Install requirements.txt and the FastReID dependencies before enabling ReID."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.merge_from_list([
            "MODEL.WEIGHTS", str(self.weights_path),
            "MODEL.DEVICE", self.device
        ])
        cfg.freeze()

        self.demo = FeatureExtractionDemo(cfg, parallel=False)

    def extract(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("FastReID received an empty crop")
        feat = self.demo.run_on_image(image_bgr)
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy()[0].astype(np.float32)

    def extract_from_path(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return self.extract(img)
