from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import numpy as np
import torch

ArrayLike = np.ndarray

@dataclass
class FaceEmbedding:
    vector: ArrayLike          # shape (D,)
    model_name: str            # "arcface_r50"...
    normed: bool = True

class FaceRecognizer:
    """
    Handles face recognition backbones (ArcFace/CosFace/...).
    Converts cropped faces to embeddings and computes similarities.
    """

    def __init__(self, 
                 model_path: str,
                 device: str = "cuda",
                 model_name: str = "arcface_r50"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        # self.model = self._load_model(model_path).to(self.device).eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        # TODO: Implement actual model loading (onnx or pt)
        return torch.nn.Identity()

    def embed_face(self, face_img: ArrayLike) -> FaceEmbedding:
        """
        Input: aligned face image HxWxC.
        Output: FaceEmbedding vector.
        """
        # TODO: preprocess -> inference -> L2 normalize
        # dummy vector
        d = 512
        vector = np.random.randn(d).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return FaceEmbedding(vector=vector, model_name=self.model_name)

    def embed_batch(self, faces: Iterable[ArrayLike]) -> List[FaceEmbedding]:
        return [self.embed_face(img) for img in faces]

    @staticmethod
    def cosine_similarity(emb1: FaceEmbedding, emb2: FaceEmbedding) -> float:
        v1, v2 = emb1.vector, emb2.vector
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    @staticmethod
    def cosine_distance(emb1: FaceEmbedding, emb2: FaceEmbedding) -> float:
        return 1.0 - FaceRecognizer.cosine_similarity(emb1, emb2)
