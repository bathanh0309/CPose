import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoseC3DRunner:
    def __init__(self, config, checkpoint, work_dir, num_classes=60, mmaction_root=None):
        self.mmaction_root = Path(mmaction_root).resolve() if mmaction_root else None
        if not isinstance(config, dict):
            raise ValueError("PoseC3D config must be an inline mapping from pipeline.yaml")
        self.config = config
        self.checkpoint = Path(checkpoint).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.num_classes = int(num_classes)

        if self.mmaction_root is None:
            raise ValueError("Set adl.mmaction_root in configs/system/pipeline.yaml before enabling PoseC3D inference.")
        if not self.mmaction_root.exists():
            raise FileNotFoundError(f"MMAction root not found: {self.mmaction_root}")
        if not (self.mmaction_root / "tools" / "test.py").exists():
            raise FileNotFoundError(f"MMAction2 test.py not found under: {self.mmaction_root}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"PoseC3D checkpoint not found: {self.checkpoint}")

        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_temp_test_config(self, ann_file):
        cfg = self._prepare_config(ann_file)
        lines = [
            "# Auto-generated from configs/system/pipeline.yaml",
            f"load_from = r'{self.checkpoint.as_posix()}'",
            f"work_dir = r'{self.work_dir.as_posix()}'",
        ]
        for key, value in cfg.items():
            lines.append(f"{key} = {repr(value)}")
        lines.append(
            "test_dataloader = dict(batch_size=1, num_workers=0, "
            "dataset=dict(type=dataset_type, ann_file=r'"
            f"{Path(ann_file).as_posix()}"
            "', split='test', pipeline=test_pipeline, test_mode=True))"
        )
        content = "\n".join(lines) + "\n"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
        tmp.write(content)
        tmp.close()
        return tmp.name

    def _prepare_config(self, ann_file):
        import copy

        cfg = copy.deepcopy(self.config)
        cfg["model"]["cls_head"]["num_classes"] = self.num_classes
        cfg["dataset_type"] = cfg.get("dataset_type", "PoseDataset")
        cfg["test_pipeline"] = cfg.get("test_pipeline", [])
        cfg["data"]["test"]["ann_file"] = str(Path(ann_file).as_posix())
        cfg["data"]["test"]["pipeline"] = cfg["test_pipeline"]
        cfg["data"]["test"]["split"] = "test"
        return cfg

    def run_test(self, ann_file):
        cfg_path = self.build_temp_test_config(ann_file)
        result_path = self.work_dir / f"posec3d_result_{uuid.uuid4().hex}.pkl"
        try:
            cmd = [
                sys.executable,
                str(self.mmaction_root / "tools" / "test.py"),
                cfg_path,
                "-C",
                str(self.checkpoint),
                "--out",
                str(result_path),
            ]
            logger.info("Running PoseC3D subprocess")
            # capture output so we can try to parse results
            cp = subprocess.run(cmd, cwd=str(self.mmaction_root), check=False, capture_output=True, text=True)

            stdout = cp.stdout or ""
            stderr = cp.stderr or ""
            combined = stdout + "\n" + stderr

            # Try to find a results file under work_dir or parse stdout
            try:
                import re
                import pickle

                result_candidates = [result_path]

                # look for a saved results path in the combined output
                for pattern in (r"saved to (.+)", r"writing results to (.+)"):
                    m = re.search(pattern, combined, re.IGNORECASE)
                    if not m:
                        continue
                    path_str = m.group(1).strip()
                    p = Path(path_str)
                    if not p.is_absolute():
                        p = (self.work_dir / p).resolve()
                    result_candidates.append(p)

                for p in result_candidates:
                    if p.exists():
                        try:
                            with open(p, "rb") as f:
                                res = pickle.load(f)
                            # res may be a list/array of predictions
                            if isinstance(res, (list, tuple)) and len(res) > 0:
                                first = res[0]
                                # If first is an int label
                                if isinstance(first, int):
                                    label = int(first)
                                    return {"status": "inferred", "label": str(label), "score": 1.0, "label_id": label}
                                if hasattr(first, "argmax"):
                                    label = int(first.argmax())
                                    score = float(first[label]) if hasattr(first, "__getitem__") else 1.0
                                    return {"status": "inferred", "label": str(label), "score": score, "label_id": label}
                                # If first is (label, score)
                                if isinstance(first, (list, tuple)) and len(first) >= 1:
                                    try:
                                        label = int(first[0])
                                        score = float(first[1]) if len(first) > 1 else 1.0
                                        return {"status": "inferred", "label": str(label), "score": score, "label_id": label}
                                    except Exception:
                                        pass
                        except Exception as exc:
                            logger.warning(f"Failed to load PoseC3D results file {p}: {exc}", exc_info=True)

                # fallback: try parse simple predicted class lines
                m2 = re.search(r"Predicted.*class[:=]\s*(\w+)", combined, re.IGNORECASE)
                if m2:
                    label_name = m2.group(1)
                    try:
                        label = int(label_name)
                        return {"status": "inferred", "label": label_name, "score": 1.0, "label_id": label}
                    except Exception:
                        return {"status": "inferred", "label": label_name, "score": 1.0, "label_id": None}
            except Exception as exc:
                logger.debug(f"Error while parsing PoseC3D output: {exc}", exc_info=True)

            logger.info("PoseC3D subprocess finished; no ADL result parsed.")
            if cp.returncode != 0:
                logger.warning(f"PoseC3D subprocess failed: {combined[-1000:]}")
                return {"status": "failed", "label": None, "score": 0.0, "message": combined[-1000:]}
            return {"status": "completed_no_result", "label": None, "score": 0.0, "message": "No parseable ADL output"}
        finally:
            Path(cfg_path).unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
