"""src.modules.detection — shim forwarding to src.human_detection."""
from src.human_detection import *  # noqa: F401, F403
from src.human_detection.api import process_folder, process_video  # noqa: F401
