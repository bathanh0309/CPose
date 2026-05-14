import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.device import resolve_torch_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FastReIDExtractor:
    def __init__(self, fastreid_root, config_path, weights_path, device=None):
        self.fastreid_root = Path(fastreid_root).resolve()
        if not self.fastreid_root.exists():
            raise FileNotFoundError(f"fastreid_root not found: {self.fastreid_root}")

        self.config_path = str(config_path)
        self.weights_path = str(weights_path)
        self.device = resolve_torch_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")

        fastreid_path = str(self.fastreid_root)
        if fastreid_path not in sys.path:
            sys.path.insert(0, fastreid_path)

        logger.info(f"Loading FastReID model: {self.weights_path}")
        from fastreid.config import get_cfg
        from demo.predictor import FeatureExtractionDemo

        cfg = get_cfg()
        cfg.merge_from_file(self.config_path)
        cfg.merge_from_list([
            "MODEL.WEIGHTS", self.weights_path,
            "MODEL.DEVICE", self.device
        ])
        cfg.freeze()

        self.demo = FeatureExtractionDemo(cfg, parallel=False)

    def extract(self, image_bgr):
        feat = self.demo.run_on_image(image_bgr)
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy()[0].astype(np.float32)

    def extract_from_path(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return self.extract(img)
