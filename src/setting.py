import numpy as np
# from collections import defaultdict, Counter, deque
from pathlib import Path
import os
import argparse

from src.config import (
    FACE_GALLERY_DIR,
    HOMOGRAPHY_PATH,
    LIVENESS_MODEL as CONFIG_LIVENESS_MODEL,
    PERSON_DETECTOR_MODEL,
    RTSP_CAM1,
    RTSP_CAM2,
)


VID1_SRC = ""
VID2_SRC = ""

# paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR.parent / "data"
CONF_DIR = BASE_DIR.parent / "configs"
MODEL_DIR = BASE_DIR.parent / "models"
WEIGHT_DIR = MODEL_DIR

# data / face = registered, unknonw images, test videos

TEST_VIDEO = DATA_DIR / "video"
RECORD_PATH = DATA_DIR / "records"
UNKNOWN_DIR = DATA_DIR / "unknown"

# models = anti-spoofing + face detect + face recognition + tracking

# models = models neural blocks code + weights
# weights path

LIVENESS_MODEL = CONFIG_LIVENESS_MODEL
YOLO_PATH = PERSON_DETECTOR_MODEL
FACE_ANALYSIS_PATH = ""
TRACKER_PATH = ""

# others parameters + variables
# face parameters
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)


# model processing parameters
# body
CONF_THRES = 0.4
PERSON_CLASS_ID = 0

ROI_RATIO_W = 0.4
ROI_RATIO_H = 1.0

# face
VERIFY_TH = 0.5
TH_LOW_CONF = 0.35

# face regconize
TH_SIM = 0.3    
VOTE_LEN = 5
FACE_MIN_SIZE = 25


# compute iou score
def compute_iou(boxA, boxB):
    x1a, y1a, x2a, y2a = boxA
    x1b, y1b, x2b, y2b = boxB
    
    inter_w = max(0, min(x2a, x2b) - max(x1a, x1b))
    inter_h = max(0, min(y2a, y2b) - max(y1a, y1b))
    inter = inter_w * inter_h
    
    areaA = (x2a - x1a) * (y2a - y1a)
    areaB = (x2b - x1b) * (y2b - y1b)
    
    return inter / (areaA + areaB - inter + 1e-6)

# parse command line when run code
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="rtsp",
                        choices=["rtsp", "video"],
                        help="Input source mode")
    
    parser.add_argument("--cam1", type=str, default=None,
                        help="Directory path to cam1 video")
    
    parser.add_argument("--cam2", type=str, default=None,
                        help="Directory path to cam2 video")
    
    return parser.parse_args()
