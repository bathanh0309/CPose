from setup import *

import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from insightface.app import FaceAnalysis

# LOAD DATABASE
db = load_db()

# LOAD MODELS
# ===================== INIT INSIGHTFACE =====================
app = FaceAnalysis(
    name="buffalo_s",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    root="./models"
)
app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace models:", app.models.keys())

# ===================== INIT YOLOv8 =====================
yolo = YOLO("./models/yolov8n.pt")
PERSON_CLASS_ID = 0

# ===================== VIDEO =====================
video_path = "./video"
target = "Huy"
view = "Huy"
in_path = f"{video_path}/{target}/{view}.mp4"

# OUTPUT PROCESSED VIDEO
out_path = f"{video_path}/{target}/OUT_{view}.mp4"
OUT_FPS = 10
OUT_SIZE = (960,640)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, OUT_FPS, OUT_SIZE)

cap = cv2.VideoCapture(in_path)
# cap = cv2.VideoCapture(rtsp_imou, cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)

# ===================== VOTING MEMORY =====================
MAX_VOTE = 5  # 5–10 lần
person_votes = {}  # key -> deque
gender_votes = {}
age_votes = {}

# ===================== FPS =====================
fps = 0.0
frame_count = 0
fps_timer = time.time()

# ===================== LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ===================== YOLO PERSON DETECT =====================
    results = yolo(frame, conf=0.4, iou=0.5, verbose=False)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) != PERSON_CLASS_ID:
            continue

        px1, py1, px2, py2 = map(int, box)

        # clamp
        px1, py1 = max(px1, 0), max(py1, 0)
        px2, py2 = min(px2, w), min(py2, h)

        person_roi = frame[py1:py2, px1:px2]
        if person_roi.size == 0:
            continue

        # ===================== FACE DETECT INSIDE PERSON =====================
        faces = app.get(person_roi)

        # key bbox (giảm rung)
        key = (px1 // 20, py1 // 20, px2 // 20, py2 // 20)

        if key not in person_votes:
            person_votes[key] = deque(maxlen=MAX_VOTE)

        if key not in gender_votes:
            gender_votes[key] = deque(maxlen=MAX_VOTE)
        
        if key not in age_votes:
            age_votes[key] = deque(maxlen=MAX_VOTE)

        for face in faces:
            # EMBEDDING
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            pid, sim = match_face(emb, db)

            if pid is not None and sim >= TH_LOW_CONF:
                person_votes[key].append(pid)
            
            # GENDER VOTE
            gender_label = "Male" if face.gender == 1 else "Female"
            gender_votes[key].append(gender_label)

            # AGE VOTE
            age_group = age_to_group(face.age)
            age_votes[key].append(age_group)

        # ===================== FINAL VOTE =====================
        final_id = get_majority(person_votes[key])
        final_gender = get_majority(gender_votes[key])
        final_age = get_majority(age_votes[key])

        if final_id:
            label = f"{final_id}"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        info = []
        if final_gender is not None:
            info.append(str(final_gender))
        if final_age is not None:
            info.append(str(final_age))
        
        if len(info) > 0:
            info_text = " | ".join(info)


            # ===================== DRAW PERSON BOX =====================
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(
                frame, label, (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            # ===================== DRAW FACE INFO (OPTIONAL) =====================
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                fx1 += px1
                fx2 += px1
                fy1 += py1
                fy2 += py1

                age = face.age
                gender = "Male" if face.gender == 1 else "Female"
                age_group = age_to_group(age)

                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                cv2.putText(
                    frame, info_text,
                    (fx1, fy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1
                )

    # ===================== FPS =====================
    frame_count += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps = frame_count / (now - fps_timer)
        fps_timer = now
        frame_count = 0

    cv2.putText(
        frame, f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 255, 0), 2
    )

    frame = cv2.resize(frame, (960, 640))

    # WRITER FRAME INTO OUTPUT VIDEO
    writer.write(frame)

    cv2.imshow("YOLOv8 Person + Face Recognition Vote", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== CLEAN =====================
cap.release()
writer.release()
cv2.destroyAllWindows()
