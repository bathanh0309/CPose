"""EfficientGCN-B0 ADL wrapper with safe unknown fallback."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


NTU120_DISPLAY_LABELS = [
    "falling_down",
    "sitting_down",
    "standing_up",
    "walking",
    "drinking_water",
    "phone_call",
    "reading",
    "writing",
    "eat_meal",
]


class EfficientGCNADL:
    """Maintain skeleton windows and run EfficientGCN-B0 when loadable."""

    def __init__(
        self,
        weight_path: str,
        window: int = 30,
        stride: int = 15,
        device: str = "cpu",
    ) -> None:
        self.weight_path = str(weight_path)
        self.window = int(window)
        self.stride = int(stride)
        self.device = "cpu"
        self.buffers: dict[int, deque[np.ndarray]] = {}
        self.last_action: dict[int, tuple[str, float]] = {}
        self.model: Any | None = None
        self.load_error: str | None = None
        self._load_model()

    def update(self, track_id: int, keypoints: np.ndarray, frame_idx: int) -> tuple[str, float]:
        """Append one [17,3] or [17,2] skeleton and infer on stride frames."""
        tid = int(track_id)
        kps = np.asarray(keypoints, dtype=np.float32)
        if kps.shape == (17, 3):
            xy = kps[:, :2]
        elif kps.shape == (17, 2):
            xy = kps
        else:
            return self.last_action.get(tid, ("unknown", 0.0))

        buf = self.buffers.setdefault(tid, deque(maxlen=self.window))
        buf.append(xy.astype(np.float32))

        if len(buf) < self.window:
            return self.last_action.get(tid, ("unknown", 0.0))
        if int(frame_idx) % max(1, self.stride) != 0:
            return self.last_action.get(tid, ("unknown", 0.0))
        if self.model is None:
            self.last_action[tid] = ("unknown", 0.0)
            return self.last_action[tid]

        label, score = self._infer(np.stack(list(buf), axis=0))
        self.last_action[tid] = (label, score)
        return label, score

    def score(self, track_id: int) -> float:
        """Return the last ADL confidence for a track."""
        return float(self.last_action.get(int(track_id), ("unknown", 0.0))[1])

    def cleanup_track(self, track_id: int) -> None:
        """Forget buffered state for a lost track."""
        tid = int(track_id)
        self.buffers.pop(tid, None)
        self.last_action.pop(tid, None)

    def _load_model(self) -> None:
        path = Path(self.weight_path)
        if not path.exists():
            self.load_error = f"weights not found: {path}"
            return
        try:
            import torch

            try:
                self.model = torch.jit.load(str(path), map_location="cpu")
            except Exception:
                try:
                    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
                except TypeError:
                    checkpoint = torch.load(str(path), map_location="cpu")
                if hasattr(checkpoint, "eval") and callable(checkpoint):
                    self.model = checkpoint
                else:
                    self.model = None
                    self.load_error = "checkpoint is not a TorchScript/callable model"
                    return
            self.model.eval()
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _infer(self, sequence_tvc: np.ndarray) -> tuple[str, float]:
        try:
            import torch
            import torch.nn.functional as F

            # [T,V,C] -> [1,C,T,V,M]
            data = np.transpose(sequence_tvc.astype(np.float32), (2, 0, 1))
            data = data[None, :, :, :, None]
            tensor = torch.from_numpy(data).to("cpu")
            with torch.inference_mode():
                output = self.model(tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            label = NTU120_DISPLAY_LABELS[idx] if idx < len(NTU120_DISPLAY_LABELS) else "unknown"
            return label, score
        except Exception as exc:
            self.load_error = f"infer {type(exc).__name__}: {exc}"
            return "unknown", 0.0
