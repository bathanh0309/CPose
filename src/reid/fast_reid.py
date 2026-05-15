import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.utils.config import normalize_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FastReIDExtractor:
    def __init__(self, config, weights_path, device=None, output_dir=None, fastreid_root=None):
        self.fastreid_root = Path(fastreid_root).resolve() if fastreid_root else None
        if self.fastreid_root is not None and not self.fastreid_root.exists():
            raise FileNotFoundError(f"FastReID root not found: {self.fastreid_root}")

        if not isinstance(config, dict):
            raise ValueError("FastReID config must be an inline mapping from pipeline.yaml")
        self.config = config
        self.weights_path = Path(weights_path).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        if not self.weights_path.exists():
            raise FileNotFoundError(f"FastReID weights not found: {self.weights_path}")
        self.device = normalize_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.fastreid_root is not None:
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
                "Install requirements.txt and FastReID dependencies, or set reid.fastreid_root in configs/system/pipeline.yaml."
            ) from exc

        cfg_path = self._write_temp_config()
        try:
            cfg = get_cfg()
            cfg.merge_from_file(str(cfg_path))
            overrides = [
                "MODEL.WEIGHTS", str(self.weights_path),
                "MODEL.DEVICE", self.device,
            ]
            if self.output_dir is not None:
                overrides.extend(["OUTPUT_DIR", str(self.output_dir)])
            cfg.merge_from_list(overrides)
            cfg.freeze()
        finally:
            cfg_path.unlink(missing_ok=True)

        self.demo = FeatureExtractionDemo(cfg, parallel=False)

    def _write_temp_config(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yml", mode="w", encoding="utf-8")
        yaml.safe_dump(self.config, tmp, sort_keys=False)
        tmp.close()
        return Path(tmp.name)

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
