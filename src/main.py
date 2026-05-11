from models.camera.camera_worker import CameraWorker
from src.cam1_pipeline import Cam1Pipeline
from src.cam2_pipeline import Cam2Pipeline
from src.matcher import MultiCameraMatcher
from src.global_state import GlobalState
from models.camera.video_src import VideoSource

from src.setting import *
import cv2
import time

matcher = MultiCameraMatcher()

# === homo graphy projection ===

# === waiting for cmd input ===
args = parse_args()

if args.mode == "rtsp":
    CAM1_SOURCE = RTSP_CAM1
    CAM2_SOURCE = RTSP_CAM2

    # init cameras
    cam1_src = CameraWorker("cam1", CAM1_SOURCE, Cam1Pipeline(), target_fps=10)
    cam2_src = CameraWorker("cam2", CAM2_SOURCE, Cam2Pipeline(), target_fps=10)

    cam1_src.start()
    cam2_src.start()

    while True:
        frame1, res1 = cam1_src.get_results()
        frame2, res2 = cam2_src.get_results()
        
        # === update matcher ====
        matcher.update_cam1(res1)
        mapping = matcher.match(res2)
        
        # === draw cam 1 ===
        if frame1 is not None:
            for obj in res1:
                (x1, y1, x2, y2), label, color = obj["bbox"], obj["label"], obj["color"]
                            
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame1, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                px, py = matcher.project_point((x1+x2)//2, (y1+y2)//2)
                cv2.circle(frame2,(px, py), 6, (0,0,255), -1)
                
            cv2.imshow("CAM1 - Front Door", frame1)
        
        # === draw cam 2 ===
        if frame2 is not None:
            for obj in res2:
                track_id = obj["track_id"]
                x1, y1, x2, y2 = obj["bbox"]
                                    
                if track_id in mapping:
                    label = mapping[track_id]
                    color = (0, 255, 0)
                else:
                    label = obj["label"]
                    color = obj["color"]
                
                cv2.rectangle(frame2, (x1, y1), (x2, y2), color , 2)
                cv2.putText(frame2, label, (x1, y1 -10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
                
            cv2.imshow("CAM 2 - Overhead", frame2)
                
        # === stream exit ===
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


elif args.mode == "video":
    if args.cam1 is None or args.cam2 is None:
        raise ValueError("❌ Please provide --cam1 and --cam2 for video mode")

    CAM1_SOURCE = str(RECORD_PATH / args.cam1)
    CAM2_SOURCE = str(RECORD_PATH / args.cam2)

    # ===== INIT =====
    cam1_src = VideoSource(CAM1_SOURCE, "cam1")
    cam2_src = VideoSource(CAM2_SOURCE, "cam2")

    cam1 = Cam1Pipeline()
    cam2 = Cam2Pipeline()

    # ===== LOOP =====
    while True:
        data1 = cam1_src.read()
        data2 = cam2_src.read()

        if data1 is None or data2 is None:
            print("End stream")
            break

        frame1, ts1 = data1
        frame2, ts2 = data2

        res1 = cam1.process(frame1, ts1)
        matcher.update_cam1(res1)

        res2 = cam2.process(frame2, ts2)
        mapping = matcher.match(res2)
            
        # draw cam1
        for obj in res1:
            (x1, y1, x2, y2), label, color = obj["bbox"], obj["label"], obj["color"]
            cv2.rectangle(frame1, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame1, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw cam2
        if frame2 is not None:
            for obj in res2:
                track_id = obj["track_id"]
    
                x1, y1, x2, y2 = obj["bbox"]
                                    
                if track_id in mapping:
                    label = mapping[track_id]
                    color = (0, 255, 0)
                else:
                    label = obj["label"]
                    color = obj["color"]
                
                cv2.rectangle(frame2, (x1, y1), (x2, y2), color , 2)
                cv2.putText(frame2, label, (x1, y1 -10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
                        
        cv2.imshow("Cam1", frame1)
        cv2.imshow("Cam2", frame2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break