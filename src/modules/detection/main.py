"""src.modules.detection.main — forwards to src.human_detection.main."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.human_detection.main import main

if __name__ == "__main__":
    main()
