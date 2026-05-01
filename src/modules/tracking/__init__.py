"""src.modules.tracking — shim forwarding to src.human_tracking."""
from src.human_tracking import *  # noqa: F401, F403
from src.human_tracking.api import process_folder, process_video  # noqa: F401
