"""OSNet body ReID embedding support."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class BodyEmbedder:
    """Body ReID embedder with ONNX Runtime preferred and torchreid fallback."""

    def __init__(
        self,
        onnx_path: str | Path = "models/global_reid/osnet_x0_25_market.onnx",
        model_name: str = "osnet_x0_25",
        device: str = "cpu",
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.model_name = model_name
        self.device = device
        self.session: Any | None = None
        self.model: Any | None = None
        self._backend = "none"
        if self.onnx_path.exists():
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self._backend = "onnx"
        else:
            self._load_torchreid()

    @property
    def backend(self) -> str:
        return self._backend

    def _load_torchreid(self) -> None:
        try:
            import torch
            import torchreid

            self.model = torchreid.models.build_model(self.model_name, num_classes=0, pretrained=True)
            self.model.eval()
            self._torch = torch
            self._backend = "torchreid"
        except Exception as exc:
            self._backend = f"unavailable:{exc}"

    def embed(self, crop_bgr: Any) -> np.ndarray | None:
        if crop_bgr is None:
            return None
        if self.session is not None:
            tensor = self._preprocess(crop_bgr)
            result = self.session.run(None, {self.input_name: tensor})[0]
            return self._normalize(result.reshape(-1))
        if self.model is not None:
            tensor = self._torch.from_numpy(self._preprocess(crop_bgr))
            with self._torch.no_grad():
                result = self.model(tensor).detach().cpu().numpy()
            return self._normalize(result.reshape(-1))
        return None

    @staticmethod
    def _preprocess(crop_bgr: Any) -> np.ndarray:
        import cv2

        resized = cv2.resize(crop_bgr, (128, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        chw = ((rgb - mean) / std).transpose(2, 0, 1)
        return chw[None, ...].astype("float32")

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-12)


__all__ = ["BodyEmbedder"]
