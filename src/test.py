import cv2
import numpy as np
import threading

from src.setting import *
from src.core.detector import FaceRecognition
from src.core.detector import SimplePersonDetector as PersonDetector

# ===== LOAD =====
H = np.load("homography.npy")

face_detector = FaceRecognition()
human_detector = PersonDetector()

prev_frame1 = None
prev_frame2 = None

frame1 = None
frame2 = None

lock1 = threading.Lock()
lock2 = threading.Lock()

# ===== CAMERA THREAD =====
def cam_reader(rtsp, cam_id):
    global frame1, frame2

    cap = cv2.VideoCapture(rtsp)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if cam_id == 1:
            with lock1:
                frame1 = frame.copy()
        else:
            with lock2:
                frame2 = frame.copy()

# ===== PROJECT =====
def project_points(points, H):
    results = []
    
    for (x, y) in points:
        p = np.array([x, y, 1.0]).reshape(3, 1)
        p2 = H @ p

        if p2[2] == 0:
            results.append(None)
            continue

        p2 = p2 / p2[2]
        results.append((int(p2[0]), int(p2[1])))

    return results


# ===== MAIN =====
def main():
    global frame1, frame2
    # global prev_frame1, prev_frame2

    t1 = threading.Thread(target=cam_reader, args=(RTSP_CAM1,1), daemon=True)
    t2 = threading.Thread(target=cam_reader, args=(RTSP_CAM2,2), daemon=True)

    t1.start()
    t2.start()

    while True:
        with lock1:
            f1 = frame1.copy() if frame1 is not None else None
        with lock2:
            f2 = frame2.copy() if frame2 is not None else None

        if f1 is None or f2 is None:
            continue
        
        if f1 is None or f2 is None:
            continue
        
        
        faces_f1 = face_detector.face_detect(frame=f1)
        people_f2 = human_detector.person_detect(frame=f2)
        
        # draw human bbox in frame 2
        for person in people_f2:
            x1, y1, x2, y2 = person[0]
            
            cv2.rectangle(f2, (x1,y1),(x2,y2),(100,155,0),2)
            
        # draw face bbox in frame1
        if faces_f1 is not None:
            for face in faces_f1:
                x1,y1,x2,y2 = face["bbox"]

                cv2.rectangle(f1, (x1,y1),(x2,y2),(0,255,0),2)

                # ===== face center =====
                fx = (x1+x2)//2
                fy = (y1+y2)//2

                cv2.circle(f1,(fx,fy),5,(0,0,255),-1)

                px_py = project_points([(fx,fy)], H)

                if px_py is not None:
                    for pxpy in px_py:
                        px, py = pxpy
                        # py += 100 # offset value

                        cv2.circle(f2,(px,py),6,(255,0,0),-1)

                
        cv2.imshow("Cam1", f1)
        cv2.imshow("Cam2", f2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
