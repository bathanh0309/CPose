"""EfficientGCN-B0 ADL wrapper with safe unknown fallback."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

COCO17_TO_NTU25 = {
    3: 0,    # head <- nose
    4: 5,    # left shoulder
    5: 7,    # left elbow
    6: 9,    # left wrist
    7: 9,    # left hand
    8: 6,    # right shoulder
    9: 8,    # right elbow
    10: 10,  # right wrist
    11: 10,  # right hand
    12: 11,  # left hip
    13: 13,  # left knee
    14: 15,  # left ankle
    15: 15,  # left foot
    16: 12,  # right hip
    17: 14,  # right knee
    18: 16,  # right ankle
    19: 16,  # right foot
    21: 9,   # left hand tip
    22: 9,   # left thumb
    23: 10,  # right hand tip
    24: 10,  # right thumb
}


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
        self.input_shape = [3, 6, self.window, 25, 2]
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
                elif isinstance(checkpoint, dict) and callable(checkpoint.get("model", None)):
                    self.model = checkpoint["model"]
                elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
                    self.model = self._build_efficientgcn_model(torch)
                    self.model.load_state_dict(checkpoint["model"], strict=True)
                elif isinstance(checkpoint, dict) and (
                    "state_dict" in checkpoint or "model_state_dict" in checkpoint
                ):
                    self.model = self._build_efficientgcn_model(torch)
                    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
                    self.model.load_state_dict(state, strict=True)
                else:
                    self.model = None
                    self.load_error = "checkpoint is not a TorchScript/callable model"
                    return
            self.model.eval()
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _build_efficientgcn_model(self, torch):
        from src.action.efficientgcn_v1 import create_efficientgcn_b0
        from src.action.efficientgcn_v1.graphs import Graph

        graph = Graph("ntu-xsub120")
        self.input_shape = [3, 6, self.window, 25, 2]
        return create_efficientgcn_b0(
            data_shape=self.input_shape,
            num_class=120,
            A=torch.tensor(graph.A, dtype=torch.float32),
            parts=graph.parts,
        ).to("cpu")

    @staticmethod
    def _coco17_to_ntu25(sequence_tvc: np.ndarray) -> np.ndarray:
        """Map YOLO COCO-17 [T,17,2/3] skeletons to normalized NTU-25 [3,T,25,2]."""
        seq = np.asarray(sequence_tvc, dtype=np.float32)
        t_len = seq.shape[0]
        ntu = np.zeros((3, t_len, 25, 2), dtype=np.float32)
        xy = seq[:, :, :2].copy()

        for t in range(t_len):
            pts = xy[t]
            visible = np.isfinite(pts).all(axis=1)
            if not visible.any():
                continue
            center = pts[visible].mean(axis=0)
            height = max(float(pts[visible, 1].max() - pts[visible, 1].min()), 1.0)
            norm = (pts - center) / height

            shoulders = norm[[5, 6]].mean(axis=0)
            hips = norm[[11, 12]].mean(axis=0)
            spine_mid = (shoulders + hips) * 0.5

            synthetic = {
                0: hips,
                1: spine_mid,
                2: shoulders,
                20: shoulders,
            }
            for ntu_idx, value in synthetic.items():
                ntu[0:2, t, ntu_idx, 0] = value
            for ntu_idx, coco_idx in COCO17_TO_NTU25.items():
                ntu[0:2, t, ntu_idx, 0] = norm[coco_idx]
        return ntu

    @staticmethod
    def _multi_input(data: np.ndarray) -> np.ndarray:
        """Official EfficientGCNv1 J/V/B input transform: [3,T,25,2] -> [3,6,T,25,2]."""
        connect_joint = np.array(
            [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12],
            dtype=np.int64,
        ) - 1
        c, t_len, joints, persons = data.shape
        joint = np.zeros((c * 2, t_len, joints, persons), dtype=np.float32)
        velocity = np.zeros_like(joint)
        bone = np.zeros_like(joint)

        joint[:c] = data
        for i in range(joints):
            joint[c:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(max(0, t_len - 2)):
            velocity[:c, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
            velocity[c:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
        for i, parent in enumerate(connect_joint):
            bone[:c, :, i, :] = data[:, :, i, :] - data[:, :, parent, :]
        bone_length = np.sqrt(np.sum(bone[:c] ** 2, axis=0, keepdims=True)) + 0.0001
        bone[c:] = np.arccos(np.clip(bone[:c] / bone_length, -1.0, 1.0))
        return np.stack([joint, velocity, bone], axis=0).astype(np.float32)

    def _infer(self, sequence_tvc: np.ndarray) -> tuple[str, float]:
        try:
            import torch
            import torch.nn.functional as F

            data = self._coco17_to_ntu25(sequence_tvc)
            data = self._multi_input(data)
            data = data[None, :, :, :, :, :]
            tensor = torch.from_numpy(data).to("cpu")
            with torch.inference_mode():
                output = self.model(tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
            idx = int(np.argmax(probs))
            score = float(probs[idx])
            label = str(idx)
            return label, score
        except Exception as exc:
            self.load_error = f"infer {type(exc).__name__}: {exc}"
            return "unknown", 0.0
