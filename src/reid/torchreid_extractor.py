import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid

from src.utils.config import normalize_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TorchreidExtractor:
    def __init__(self, model_name, weights_path, device=None):
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = normalize_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading Torchreid model: {self.model_name} from {self.weights_path}")
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=self.model_name,
            model_path=str(self.weights_path),
            device=self.device
        )

    def extract(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Torchreid received an empty crop")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Extractor returns a batch tensor. Since we pass a list of 1 image, it's (1, dim)
        feat = self.extractor([image_rgb])
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy()[0].astype(np.float32)

    def extract_from_path(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return self.extract(img)
