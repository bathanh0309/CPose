"""src.modules.pose_estimation.main — forwards to src.pose_estimation.main."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.pose_estimation.main import main

if __name__ == "__main__":
    main()
