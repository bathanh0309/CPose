from body_detect import PersonDetector
from face_detect import FaceRecognition, age_to_group
from tracking import Tracker

from setting import *

import time
import cv2

# init model
person_detector = PersonDetector()
person_tracker = Tracker()
face_detection = FaceRecognition(data_path="D:\\ModA_FaceReg\\data\\face")
face_detection.load_face_database()

# start video processing
cap = cv2.VideoCapture(VIDEO_PATH)

prev = time.time()
fps_list = deque(maxlen=10)
fps = 0.0
fps_inst = 0.0

# voting
track_votes = defaultdict(list)
track_final = {}
track_attr = {}
unknown_count = 0
voted = []

FRAME_IDX = False
x1, y1, x2, y2 = 0, 0, 0, 0 # init values
color =(0,0,0)
label  = "..."

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if FRAME_IDX:
        # Processing logic
        # Person detection
        detections = person_detector.person_detect(frame)
        tracks = person_tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id 
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            person_roi = frame[y1:y2, x1:x2]    

            label = "Detecting..."
            color = COLOR_DETECTING

            # face record (not vote yet)
            if tid not in track_final:
                faces, emb = face_detection.face_detect(person_roi)
                
                if emb is not None:
                    pid, sim = face_detection.match_face(embedding=emb)
                    print(f"[VOTED] Track_ID {tid} sim={sim:.3f}", end='')
                    
                    if pid is not None and sim >= TH_SIM and pid not in voted:
                        track_votes[tid].append(pid)
                        print(f" - {pid}")
                    else:
                        track_votes[tid].append("UNKNOWN")
                        print(" - UNKNOWN")                            
                # Voting
                if len(track_votes[tid]) >= VOTE_LEN:
                    final = Counter(track_votes[tid]).most_common(1)[0][0]
                    
                    if final == "UNKNOWN":
                        unknown_count += 1
                        final_id = f"UNKNOWN_{unknown_count:01d}"
                        cv2.imwrite(
                            f"{UNKNOWN_DIR}/{final}.jpg",
                            person_roi)
                    else:
                        final_id = final
                        
                    track_final[tid] = final_id
                    voted.append(final_id)
                    track_votes.pop(tid, None)
                    
                    if faces and len(faces) == 1:
                        gender = "Male" if faces[0].gender == 1 else "Female"
                        age = int(faces[0].age)
                    else:
                        gender, age = "?", "?"
                        
                    track_attr[tid] = {
                        "gender": gender,
                        "age": age_to_group(age) # add age_to_group to face_detection.py
                    }
                    
                    print(f"[INFO] Track {tid} -> {final_id} ({gender}, {age_to_group(age)})")

            # compile box info
            if tid in track_final:
                info = track_attr.get(tid, {})
                label = f"{track_final[tid]} | {info.get('gender', '?')} | {info.get('age','?')}"
                color = COLOR_FINAL
                                        
        # calculate fps 
        curr = time.time()
        fps_inst = 1 / max(curr - prev, 1e-4)
        prev = curr

    # display fps and bbox
    fps_list.append(fps_inst)
    fps = sum(fps_list) / len(fps_list)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)

    FRAME_IDX = not FRAME_IDX

    cv2.imshow("Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()