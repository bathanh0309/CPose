import cv2
import os
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
PERSON_ID = input("Input user register name: ")
SAVE_DIR = f"./db/{PERSON_ID}"
BLUR_THRESHOLD = 50
CAPTURE_PER_POSE = 4
CAPTURE_COOLDOWN = 1

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# INIT MODEL
# =========================
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

# =========================
# UTILS
# =========================
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def draw_guidance_box(frame):
    h, w, _ = frame.shape
    bw, bh = 300, 300
    x1 = (w - bw) // 2
    y1 = (h - bh) // 2
    x2 = x1 + bw
    y2 = y1 + bh
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return x1, y1, x2, y2

def face_inside(face_bbox, box):
    fx1, fy1, fx2, fy2 = face_bbox
    bx1, by1, bx2, by2 = box
    return fx1 > bx1 and fy1 > by1 and fx2 < bx2 and fy2 < by2

# =========================
# USER META INPUT
# =========================
print("=== FACE REGISTRATION ===")
password = input("Set fallback password: ")
password_hash = hash_password(password)

# =========================
# POSE TARGETS
# =========================
POSES = {
    "front": lambda y, p: abs(y) < 10 and abs(p) < 10,
    "right": lambda y, p: p < -15,
    "left":  lambda y, p: p > 15,
}

pose_count = {k: 0 for k in POSES}
embeddings = []

last_capture = 0

print("[INFO] Look at the camera and follow instructions")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    box = draw_guidance_box(frame)
    faces = app.get(frame)

    if len(faces) == 1:
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        if face_inside((x1, y1, x2, y2), box):
            face_img = frame[y1:y2, x1:x2]
            blur = blur_score(face_img)

            yaw, pitch, _ = face.pose

            if blur > BLUR_THRESHOLD:
                for pose, cond in POSES.items():
                    if pose_count[pose] < CAPTURE_PER_POSE and cond(yaw, pitch):
                        if time.time() - last_capture > CAPTURE_COOLDOWN:
                            embeddings.append(face.embedding)
                            pose_count[pose] += 1
                            last_capture = time.time()
                            print(f"[CAPTURE] {pose} ({pose_count[pose]}/{CAPTURE_PER_POSE})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"Blur:{int(blur)}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # =========================
    # UI POSE STATUS
    # =========================
    y_ui = 30
    for pose, cnt in pose_count.items():
        color = (0,255,0) if cnt >= CAPTURE_PER_POSE else (0,0,255)
        cv2.putText(frame, f"{pose}: {cnt}/{CAPTURE_PER_POSE}",
                    (20, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_ui += 30

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if all(cnt >= CAPTURE_PER_POSE for cnt in pose_count.values()):
        break

# =========================
# =========================
cap.release()
cv2.destroyAllWindows()

if len(embeddings) >= 10:
    for i, emb in enumerate(embeddings):
        np.save(f"{SAVE_DIR}/emb_{i:02d}.npy", emb)

    meta = {
        "person_id": PERSON_ID,
        "created_at": datetime.now().isoformat(),
        "num_embeddings": len(embeddings),
        "poses": pose_count,
        "password_hash": password_hash
    }

    with open(f"{SAVE_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[SUCCESS] Face registered successfully")
else:
    print("[FAIL] Not enough data")


# =====================================================

import cv2
import os
import json
import time
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
VIDEO_PATH = "./video/APhu/inside.mp4"   # video hoặc RTSP
PERSON_ID = "APhu"
SAVE_DIR = f"./db/{PERSON_ID}"

MAX_EMB = 12                # số embedding muốn lưu
BLUR_THRESHOLD = 30         # ngưỡng mờ
CAPTURE_INTERVAL = 1     # giãn cách giữa các lần capture (s)

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# INIT MODEL
# =========================
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# UTILS
# =========================
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# =========================
# REGISTER LOOP
# =========================
embeddings = []
last_capture = 0

print("[INFO] Start registering face from video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    if len(faces) == 1 and len(embeddings) < MAX_EMB:
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            blur = blur_score(face_img)

            if blur > BLUR_THRESHOLD:
                if time.time() - last_capture > CAPTURE_INTERVAL:
                    emb = face.embedding
                    emb = emb / np.linalg.norm(emb)  # normalize
                    embeddings.append(emb)
                    last_capture = time.time()

                    print(f"[CAPTURE] {len(embeddings)}/{MAX_EMB}  blur={int(blur)}")

        # DRAW
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"Captured: {len(embeddings)}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Register From Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if len(embeddings) >= MAX_EMB:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE DATA
# =========================
if len(embeddings) >= 5:
    for i, emb in enumerate(embeddings):
        np.save(f"{SAVE_DIR}/emb_{i:02d}.npy", emb)

    meta = {
        "person_id": PERSON_ID,
        "created_at": datetime.now().isoformat(),
        "num_embeddings": len(embeddings),
        "source": VIDEO_PATH
    }

    with open(f"{SAVE_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[SUCCESS] Face features registered")
else:
    print("[FAIL] Not enough good face samples")
