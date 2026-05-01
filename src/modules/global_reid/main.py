"""src.modules.global_reid.main — forwards to src.global_reid.main."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.global_reid.main import main

if __name__ == "__main__":
    main()
