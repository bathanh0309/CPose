from __future__ import annotations
import os
import logging
from typing import List, Tuple

import numpy as np
import torch

# Note: These require mmcv and pyskl installed in the environment
try:
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from pyskl.models import build_model
    PYSKL_AVAILABLE = True
except ImportError:
    PYSKL_AVAILABLE = False
    Config = None
    load_checkpoint = None
    build_model = None

logger = logging.getLogger("[ADL-Model]")

class ADLModelWrapper:
    """
    Wrapper CTR-GCN (via PYSKL RecognizerGCN) for CPose.
    - Input from CPose: (T, V, 2) or (T, V, 3)
    - Auto-builds tensor with shape: (N, C, T, V, M)
    """

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str = "cuda",
        class_names: List[str] | None = None,
        num_person: int = 1,
        use_conf: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_person = num_person
        self.use_conf = use_conf
        self.class_names = class_names or [
            "standing", "sitting", "walking", "falling", 
            "lying", "reaching", "bending", "unknown"
        ]

        if not PYSKL_AVAILABLE:
            logger.error("pyskl or mmcv not installed. ADLModelWrapper will run in dummy mode.")
            self.model = None
            return

        if not os.path.exists(cfg_path):
            logger.error(f"Config not found: {cfg_path}")
            self.model = None
            return

        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            self.model = None
            return

        try:
            cfg = Config.fromfile(cfg_path)
            if not hasattr(cfg, "model"):
                raise RuntimeError("Invalid PYSKL config: no `model` field")
            
            cfg.model.backbone.pretrained = None
            cfg.model.train_cfg = None
            cfg.model.test_cfg = dict(average_clips="prob")

            # Build RecognizerGCN model (CTR-GCN backbone)
            self.model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.model.test_cfg)
            self.model.to(self.device)
            self.model.eval()

            load_checkpoint(self.model, ckpt_path, map_location=self.device)
            logger.info(f"Loaded ADL Model from {ckpt_path}")
            
            # Update class names if not provided and model has cls_head
            if class_names is None and hasattr(self.model, 'cls_head'):
                num_classes = self.model.cls_head.num_classes
                if num_classes != len(self.class_names):
                    self.class_names = [f"class_{i}" for i in range(num_classes)]

        except Exception as e:
            logger.exception(f"Failed to initialize ADL model: {e}")
            self.model = None

    @staticmethod
    def _normalize_skeleton(feat: np.ndarray) -> np.ndarray:
        """
        feat: (T, V, C)
        Standard normalization to [-1, 1] range.
        Note: Align this with your specific training data normalization.
        """
        xy = feat[..., :2]
        # Avoid issues with all zeros
        if np.all(xy == 0):
            return feat
            
        min_xy = xy.min(axis=(0, 1), keepdims=True)
        max_xy = xy.max(axis=(0, 1), keepdims=True)
        scale = (max_xy - min_xy).max(axis=-1, keepdims=True)
        
        scale[scale < 1e-6] = 1.0
        xy_norm = (xy - min_xy) / scale * 2.0 - 1.0
        
        feat_norm = feat.copy()
        feat_norm[..., :2] = xy_norm
        return feat_norm

    def _build_gcn_input(
        self,
        xy_seq: np.ndarray,
        conf_seq: np.ndarray | None,
    ) -> torch.Tensor:
        """
        xy_seq: (T, V, 2)
        conf_seq: (T, V) or None
        return: torch.Tensor (N, C, T, V, M)
        """
        T, V, _ = xy_seq.shape
        if self.use_conf and conf_seq is not None:
            feat = np.concatenate([xy_seq, conf_seq[..., None]], axis=-1)  # (T, V, 3)
        else:
            feat = xy_seq  # (T, V, 2)

        feat = self._normalize_skeleton(feat)

        C = feat.shape[-1]
        # (N, M, T, V, C)
        keypoint = np.zeros((1, self.num_person, T, V, C), dtype=np.float32)
        keypoint[0, 0, :, :, :] = feat

        # Transpose: (N, M, T, V, C) -> (N, C, T, V, M)
        keypoint = np.transpose(keypoint, (0, 4, 2, 3, 1))

        return torch.from_numpy(keypoint).float().to(self.device)

    @torch.no_grad()
    def infer_sequence(
        self,
        xy_seq: np.ndarray,
        conf_seq: np.ndarray | None = None,
    ) -> Tuple[str, float]:
        """
        Receive skeleton sequence (T,V,2)/(T,V,3) -> return (label, confidence).
        """
        if self.model is None:
            # Dummy fallback if pyskl is not installed or model failed to load
            return "unknown", 0.0

        data = self._build_gcn_input(xy_seq, conf_seq)  # (1, C, T, V, M)
        data_batch = dict(keypoint=data, label=None)

        # RecognizerGCN forward
        cls_score = self.model(return_loss=False, **data_batch)

        if isinstance(cls_score, (list, tuple)):
            cls_score = cls_score[0]
            
        prob = torch.softmax(cls_score, dim=1)[0]
        cls_idx = int(torch.argmax(prob).item())
        score = float(prob[cls_idx].item())
        
        if cls_idx < len(self.class_names):
            label = self.class_names[cls_idx]
        else:
            label = f"class_{cls_idx}"
            
        return label, score
