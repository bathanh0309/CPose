import numpy as np
from collections import defaultdict, Counter, deque

# paths
VIDEO_PATH = "D:\\ModA_FaceReg\\data\\video\\APhu\\inside.mp4"
DB_DIR = "./data/face"
UNKNOWN_DIR = "./data/face/unknown"

RTSP_IMOU = "rtsp://admin:L2D2E11E@192.168.1.18:554/cam/realmonitor?channel=1&subtype=0"
RTSP_EZVIZ = "rtsp://admin:YESXYL@192.168.1.2:554/streaming/channels/101/"


YOLO_PATH = "D:\\ModA_FaceReg\\models\\yolov8n.pt"
FACE_ANALYSIS_PATH = ""
TRACKER_PATH = ""

FACE_DATA_PATH = "./data/face"


# face parameters
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)


# video processing parameters
COLOR_DETECTING = (0, 165, 255)
COLOR_FINAL  = (0, 255, 0)


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
