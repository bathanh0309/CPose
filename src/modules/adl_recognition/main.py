"""src.modules.adl_recognition.main — forwards to src.adl_recognition.main."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.adl_recognition.main import main

if __name__ == "__main__":
    main()
