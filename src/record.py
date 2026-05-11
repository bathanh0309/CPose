import cv2
import time
import os
import threading
from datetime import datetime
from src.setting import *

# ===== CONFIG =====
SAVE_DIR = DATA_DIR / "records"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_FPS = 15


# ===== LẤY INDEX =====
def get_next_index():
    files = os.listdir(SAVE_DIR)
    indices = []

    for f in files:
        if f.startswith("cam1_") and f.endswith(".mp4"):
            try:
                idx = int(f.split("_")[1].split(".")[0])
                indices.append(idx)
            except:
                pass

    return max(indices, default=0) + 1


# ===== CAMERA THREAD =====
class CameraStream:
    def __init__(self, rtsp_url, name="cam"):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.name = name
        self.frame = None
        self.ret = False
        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.cap.grab()  # bỏ frame cũ
                ret, frame = self.cap.read()

                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ===== VIDEO WRITER =====
def create_writer(frame, name, idx):
    h, w = frame.shape[:2]

    filename = os.path.join(SAVE_DIR, f"{name}_{idx}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (w, h))

    print(f"[INFO] Recording cam2 -> {filename}")
    return writer


# ===== MAIN =====
def main():
    cam1 = CameraStream(RTSP_CAM1, "cam1")
    cam2 = CameraStream(RTSP_CAM2, "cam2")

    time.sleep(2)  # đợi buffer ổn định

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("❌ Cannot read initial frames")
        return

    idx = get_next_index()
    writer1 = create_writer(frame1, "cam1", idx)
    writer2 = create_writer(frame2, "cam2", idx)

    print("🎥 Press 'q' to stop")

    while True:
        start = time.time()

        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            print("⚠️ Frame error")
            break

        # ===== timestamp =====
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        cv2.putText(frame1, f"Cam1 {now_str}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame2, f"Cam2 {now_str}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # ===== chỉ lưu cam2 =====
        writer1.write(frame1)
        writer2.write(frame2)

        # ===== preview =====
        cv2.imshow("Cam1", frame1)
        cv2.imshow("Cam2", frame2)

        # ===== sync fps =====
        elapsed = time.time() - start
        delay = max(1.0 / TARGET_FPS - elapsed, 0)

        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break

    # ===== release =====
    cam1.release()
    cam2.release()
    writer1.release()
    writer2.release()
    cv2.destroyAllWindows()

    print("✅ Done")


if __name__ == "__main__":
    main()