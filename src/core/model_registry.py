from __future__ import annotations

import threading
from typing import Any
from pathlib import Path

from src.detectors.yolo_ultralytics import YOLODetectUltralytics, YOLOPoseUltralytics


class ModelRegistry:
    """Server-wide model registry.

    Model instances are shared across sessions; mutable runtime state such as
    trackers, ReID caches, ADL buffers, and TrackCache must stay per session.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.detector = None
        self.pose_model = None
        self.reid_model = None
        self.adl_model = None
        self.errors: dict[str, str] = {}
        self.locks = {
            "detect": threading.Lock(),
            "pose": threading.Lock(),
            "reid": threading.Lock(),
            "adl": threading.Lock(),
        }

    @staticmethod
    def _select_weight(primary: str | None, fallback: str | None, default: str) -> str:
        for value in (primary, fallback, default):
            if not value:
                continue
            path = Path(str(value))
            if path.exists():
                return str(path)
        return str(primary or fallback or default)

    def preload(self, modules: set[str] | None = None) -> None:
        modules = set(modules or {"detect", "pose", "reid"})
        if "detect" in modules or "track" in modules:
            self.get_detector()
        if "pose" in modules or "adl" in modules:
            self.get_pose_model()
        if "reid" in modules:
            self.get_reid_model()
        if "adl" in modules:
            self.get_adl_model()

    def get_detector(self):
        if self.detector is not None:
            return self.detector
        if "detect" in self.errors:
            return None
        try:
            cfg = self.cfg.get("tracking", {})
            weights = self._select_weight(
                cfg.get("weights"),
                cfg.get("fallback_weights"),
                "models/yolo11n.pt",
            )
            self.detector = YOLODetectUltralytics(
                weights=weights,
                device="cpu",
                conf_threshold=float(cfg.get("conf", 0.40)),
                iou_threshold=float(cfg.get("iou", 0.50)),
                class_filter=[0],
                imgsz=int(cfg.get("imgsz", 416)),
            )
            print(f"[DETECT] model preloaded, device={self.detector.device}, imgsz={self.detector.imgsz}")
        except Exception as exc:
            self.errors["detect"] = f"{type(exc).__name__}: {exc}"
            print(f"[DETECT] preload failed: {self.errors['detect']}")
        return self.detector

    def get_pose_model(self):
        if self.pose_model is not None:
            return self.pose_model
        if "pose" in self.errors:
            return None
        try:
            cfg = self.cfg.get("pose", {})
            weights = self._select_weight(
                cfg.get("weights"),
                cfg.get("fallback_weights"),
                "models/yolo11n-pose.pt",
            )
            self.pose_model = YOLOPoseUltralytics(
                weights=weights,
                device="cpu",
                conf_threshold=float(cfg.get("conf", 0.55)),
                iou_threshold=float(cfg.get("iou", 0.50)),
                imgsz=int(cfg.get("imgsz", 416)),
            )
            print("[POSE] model preloaded, device=cpu")
        except Exception as exc:
            self.errors["pose"] = f"{type(exc).__name__}: {exc}"
            print(f"[POSE] preload failed: {self.errors['pose']}")
        return self.pose_model

    def get_reid_model(self):
        if self.reid_model is not None:
            return self.reid_model
        if "reid" in self.errors:
            return None
        try:
            from src.reid.osnet_reid import OSNetReID

            cfg = self.cfg.get("reid", {})
            weights = self._select_weight(
                cfg.get("weights"),
                cfg.get("fallback_weights"),
                "models/osnet_x0_25_msmt17.pth",
            )
            self.reid_model = OSNetReID(
                weight_path=weights,
                threshold=float(cfg.get("threshold", 0.45)),
                reid_interval=int(cfg.get("reid_interval", 45)),
                max_gallery=int(cfg.get("max_gallery", 10)),
                min_crop_area=float(cfg.get("min_crop_area", 3500)),
                min_gallery_size=int(cfg.get("min_gallery_size", 5)),
            )
            loaded = self.reid_model.load_gallery_embeddings(
                cfg.get("embedding_dirs"),
                cfg.get("id_aliases"),
            )
            print(f"[ReID] OSNet-x0.25 preloaded on CPU; gallery={loaded}")
        except Exception as exc:
            self.errors["reid"] = f"{type(exc).__name__}: {exc}"
            print(f"[ReID] preload failed: {self.errors['reid']}")
        return self.reid_model

    def get_adl_model(self):
        if self.adl_model is not None:
            return self.adl_model
        if "adl" in self.errors:
            return None
        try:
            from src.action.efficientgcn_adl import EfficientGCNADL

            cfg = self.cfg.get("adl", {})
            weights = self._select_weight(
                cfg.get("weights"),
                cfg.get("fallback_weights"),
                "models/2015_EfficientGCN-B0_ntu-xsub120.pth.tar",
            )
            self.adl_model = EfficientGCNADL(
                weight_path=weights,
                window=int(cfg.get("min_frames", 30)),
                stride=int(cfg.get("infer_every_n_frames", 15)),
                device="cpu",
            )
            if getattr(self.adl_model, "load_error", None):
                self.errors["adl"] = str(self.adl_model.load_error)
                print(f"[ADL] unavailable: {self.errors['adl']}")
                if bool(cfg.get("disable_if_load_failed", True)):
                    self.adl_model = None
            else:
                print("[ADL] EfficientGCN preloaded, device=CPU")
        except Exception as exc:
            self.errors["adl"] = f"{type(exc).__name__}: {exc}"
            print(f"[ADL] preload failed: {self.errors['adl']}")
        return self.adl_model

    def error_for(self, module: str) -> str | None:
        return self.errors.get(module)
