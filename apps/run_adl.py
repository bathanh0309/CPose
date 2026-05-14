import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action.posec3d import PoseC3DRunner
from src.utils.config import load_pipeline_cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug PoseC3D inference")
    parser.add_argument("--input", type=str, required=True, help="MMAction2 pose annotation pkl")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "system" / "pipeline.yaml"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_pipeline_cfg(Path(args.config), ROOT)
    runner = PoseC3DRunner(
        mmaction_root=cfg["adl"]["mmaction_root"],
        base_config=cfg["adl"]["base_config"],
        checkpoint=cfg["adl"]["weights"],
        work_dir=cfg["adl"]["work_dir"],
    )
    logger.info(f"Running PoseC3D on {args.input}")
    result = runner.run_test(args.input)
    if result.returncode != 0:
        raise RuntimeError(f"PoseC3D failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
