from collections import deque
from pathlib import Path

import numpy as np

from src.utils.io import ensure_dir, save_pickle
from src.utils.logger import get_logger
from src.utils.naming import make_clip_id

logger = get_logger(__name__)

EXPECTED_KEYPOINTS = 17


class PoseSequenceBuffer:
    def __init__(
        self,
        seq_len=48,
        stride=12,
        output_dir="data/output/clips_pkl",
        default_label=0,
        max_idle_frames=150,
        export_enabled=False,
    ):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.output_dir = Path(output_dir)
        self.default_label = int(default_label)
        self.max_idle_frames = int(max_idle_frames)
        self.export_enabled = bool(export_enabled)
        self.states = {}
        self.last_seen = {}
        if self.export_enabled:
            ensure_dir(self.output_dir)

    def _get_state(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        if key not in self.states:
            self.states[key] = {
                "frame_idx": deque(maxlen=self.seq_len),
                "keypoints": deque(maxlen=self.seq_len),
                "scores": deque(maxlen=self.seq_len),
                "img_shape": None,
                "last_export_end": -10**9,
                "export_count": 0
            }
        return self.states[key]

    def latest_window(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        state = self.states.get(key)
        if not state or len(state["keypoints"]) < self.seq_len:
            return None
        return {
            "frame_idx": list(state["frame_idx"]),
            "keypoints": np.stack(list(state["keypoints"]), axis=0),
            "scores": np.stack(list(state["scores"]), axis=0),
            "img_shape": state["img_shape"],
        }

    def update(self, camera_id, local_track_id, global_id, frame_idx, keypoints_xy, keypoint_scores, img_shape):
        key = (str(camera_id), int(local_track_id))
        self.last_seen[key] = int(frame_idx)
        self._gc(int(frame_idx))

        if keypoints_xy is None:
            return {"status": "waiting", "current_len": 0, "seq_len": self.seq_len, "pkl_path": None}

        keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
        if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2:
            return {
                "status": "skipped",
                "reason": "invalid_keypoint_shape",
                "current_len": 0,
                "seq_len": self.seq_len,
                "pkl_path": None,
            }
        if keypoints_xy.shape[0] != EXPECTED_KEYPOINTS:
            logger.warning(
                f"Unexpected keypoint count {keypoints_xy.shape[0]} != {EXPECTED_KEYPOINTS}, skipping"
            )
            return {
                "status": "skipped",
                "reason": "invalid_keypoint_count",
                "current_len": 0,
                "seq_len": self.seq_len,
                "pkl_path": None,
            }

        num_kp = keypoints_xy.shape[0]
        if keypoint_scores is None:
            keypoint_scores = np.ones((num_kp,), dtype=np.float32)
        else:
            keypoint_scores = np.asarray(keypoint_scores, dtype=np.float32)
            if keypoint_scores.ndim != 1 or keypoint_scores.shape[0] != EXPECTED_KEYPOINTS:
                return {
                    "status": "skipped",
                    "reason": "invalid_keypoint_score_shape",
                    "current_len": 0,
                    "seq_len": self.seq_len,
                    "pkl_path": None,
                }

        state = self._get_state(camera_id, local_track_id)
        state["frame_idx"].append(int(frame_idx))
        state["keypoints"].append(keypoints_xy)
        state["scores"].append(keypoint_scores)
        state["img_shape"] = tuple(map(int, img_shape))

        if len(state["keypoints"]) < self.seq_len:
            return {
                "status": "collecting",
                "current_len": len(state["keypoints"]),
                "seq_len": self.seq_len,
                "pkl_path": None,
            }

        end_idx = state["frame_idx"][-1]
        if end_idx - state["last_export_end"] < self.stride:
            return {
                "status": "collecting",
                "current_len": self.seq_len,
                "seq_len": self.seq_len,
                "pkl_path": None,
            }

        state["export_count"] += 1
        state["last_export_end"] = end_idx

        if not self.export_enabled:
            return {
                "status": "disabled",
                "reason": "clip_export_disabled",
                "current_len": self.seq_len,
                "seq_len": self.seq_len,
                "pkl_path": None,
            }

        pkl_path = self._export_current_window(
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_id=global_id,
            state=state
        )
        return {
            "status": "exported",
            "current_len": self.seq_len,
            "seq_len": self.seq_len,
            "pkl_path": str(pkl_path),
        }

    def _gc(self, current_frame_idx):
        dead = [
            key for key, last_seen in self.last_seen.items()
            if current_frame_idx - last_seen > self.max_idle_frames
        ]
        for key in dead:
            self.states.pop(key, None)
            self.last_seen.pop(key, None)

    def _export_current_window(self, camera_id, local_track_id, global_id, state):
        frame_idx = list(state["frame_idx"])
        keypoints = np.stack(list(state["keypoints"]), axis=0)          # [T, V, 2]
        scores = np.stack(list(state["scores"]), axis=0)                # [T, V]
        h, w = state["img_shape"]

        clip_index = int(state["export_count"])
        sample_id = make_clip_id(
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_id=global_id,
            clip_index=clip_index,
        )

        ann = {
            "frame_dir": sample_id,
            "total_frames": int(keypoints.shape[0]),
            "img_shape": (int(h), int(w)),
            "original_shape": (int(h), int(w)),
            "label": int(self.default_label),
            "camera_id": str(camera_id),
            "local_track_id": int(local_track_id),
            "global_id": str(global_id),
            "frame_start": int(frame_idx[0]),
            "frame_end": int(frame_idx[-1]),
            "clip_index": clip_index,
            "keypoint": keypoints[None, ...].astype(np.float32),        # [M, T, V, C], M=1
            "keypoint_score": scores[None, ...].astype(np.float32)      # [M, T, V]
        }

        dataset_pkl = {
            "split": {"test": [sample_id]},
            "annotations": [ann]
        }

        out_path = self.output_dir / f"{sample_id}.pkl"
        save_pickle(dataset_pkl, out_path)
        logger.info(f"Exported pose clip: {out_path}")
        return out_path

    def reset_track(self, camera_id, local_track_id):
        key = (str(camera_id), int(local_track_id))
        if key in self.states:
            del self.states[key]
        self.last_seen.pop(key, None)
