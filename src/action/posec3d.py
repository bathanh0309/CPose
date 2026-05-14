import subprocess
import sys
import tempfile
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoseC3DRunner:
    def __init__(self, mmaction_root, base_config, checkpoint, work_dir):
        self.mmaction_root = Path(mmaction_root).resolve()
        self.base_config = Path(base_config).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.work_dir = Path(work_dir).resolve()

        if not self.mmaction_root.exists():
            raise FileNotFoundError(f"mmaction_root not found: {self.mmaction_root}")
        if not (self.mmaction_root / "tools" / "test.py").exists():
            raise FileNotFoundError(f"MMAction2 test.py not found under: {self.mmaction_root}")
        if not self.base_config.exists():
            raise FileNotFoundError(f"PoseC3D base config not found: {self.base_config}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"PoseC3D checkpoint not found: {self.checkpoint}")

        self.work_dir.mkdir(parents=True, exist_ok=True)

    def build_temp_test_config(self, ann_file):
        content = f"""
_base_ = r"{self.base_config.as_posix()}"
load_from = r"{self.checkpoint.as_posix()}"
work_dir = r"{self.work_dir.as_posix()}"

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        ann_file=r"{Path(ann_file).as_posix()}",
        split='test',
        test_mode=True
    )
)
"""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
        tmp.write(content)
        tmp.close()
        return tmp.name

    def run_test(self, ann_file):
        cfg_path = self.build_temp_test_config(ann_file)
        try:
            cmd = [
                sys.executable,
                str(self.mmaction_root / "tools" / "test.py"),
                cfg_path,
                str(self.checkpoint)
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

                # look for a saved results path in the combined output
                m = re.search(r"saved to (.+)", combined, re.IGNORECASE)
                if m:
                    path_str = m.group(1).strip()
                    p = Path(path_str)
                    if not p.is_absolute():
                        p = (self.work_dir / p).resolve()
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
                return {"status": "failed", "label": None, "score": 0.0, "message": combined[-1000:]}
            return {"status": "completed_no_result", "label": None, "score": 0.0, "message": "No parseable ADL output"}
        finally:
            Path(cfg_path).unlink(missing_ok=True)
