import cv2
import numpy as np
import threading

from src.setting import *
from src.core.detector import FaceRecognition  # bạn đang dùng
# giả định bạn có person detector
from src.core.detector import SimplePersonDetector as PersonDetector

# ===== INIT =====
face_model = FaceRecognition()   # có bbox + embedding
person_detect = PersonDetector()


# person_detector = PersonDetector()

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

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

# ===== ROI =====
def get_cam1_roi(frame, bbox):
    x1,y1,x2,y2 = bbox
    h = y2 - y1

    y2_new = min(frame.shape[0], y2 + int(1.5*h))

    return frame[y1:y2_new, x1:x2]


def get_cam2_roi(frame, bbox):
    x1,y1,x2,y2 = bbox
    h = y2 - y1

    return frame[y1:y1 + int(0.5*h), x1:x2]

# ===== SIFT =====
def extract_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return des

def match_sift(des1, des2):
    if des1 is None or des2 is None:
        return 0

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)

# ===== MAIN =====
def main():
    global frame1, frame2

    threading.Thread(target=cam_reader, args=(RTSP_CAM1,1), daemon=True).start()
    threading.Thread(target=cam_reader, args=(RTSP_CAM2,2), daemon=True).start()

    while True:
        with lock1:
            f1 = frame1.copy() if frame1 is not None else None
        with lock2:
            f2 = frame2.copy() if frame2 is not None else None

        if f1 is None or f2 is None:
            continue

        # ===== CAM1 =====
        faces = face_model.face_detect(frame=f1)

        cam1_data = []
        if faces is not None:
            for face in faces:
                bbox = face["bbox"]
                identity = face.get("id", "unknown")

                x1,y1,x2,y2 = bbox
                cv2.rectangle(f1,(x1,y1),(x2,y2),(0,255,0),2)

                roi1 = get_cam1_roi(f1, bbox)
                des1 = extract_sift(roi1)

                cam1_data.append((identity, des1, bbox))

        # ===== CAM2 =====
        cam2_boxes = person_detect.person_detect(f2)
        print(cam2_boxes)

        for box in cam2_boxes:
            x1,y1,x2,y2 = box[0]
            cv2.rectangle(f2,(x1,y1),(x2,y2),(255,0,0),2)

        # ===== MATCH =====
        for box in cam2_boxes:
            roi2 = get_cam2_roi(f2, box)
            des2 = extract_sift(roi2)

            best_id = None
            best_score = 0

            for identity, des1, _ in cam1_data:
                score = match_sift(des1, des2)

                if score > best_score:
                    best_score = score
                    best_id = identity

            x1,y1,x2,y2 = box

            if best_score > 10:
                label = f"{best_id}:{best_score}"
                cv2.putText(f2, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            else:
                cv2.putText(f2, "unknown", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        cv2.imshow("Cam1", f1)
        cv2.imshow("Cam2", f2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
