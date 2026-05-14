import subprocess
import sys
import tempfile
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoseC3DRunner:
    def __init__(self, mmaction_root, base_config, checkpoint, work_dir):
        self.mmaction_root = Path(mmaction_root)
        self.base_config = Path(base_config)
        self.checkpoint = Path(checkpoint)
        self.work_dir = Path(work_dir)

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
            return subprocess.run(cmd, cwd=str(self.mmaction_root), check=False)
        finally:
            Path(cfg_path).unlink(missing_ok=True)
