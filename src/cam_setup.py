import cv2
import numpy as np
from src.setting import *

points_cam1 = []
points_cam2 = []

PTS = int(input("Number of points >= 8: "))
SCALE = 1.2

def get_frame(rtsp):
    cap = cv2.VideoCapture(rtsp)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read RTSP")
    return frame

def undistort(frame, K, dist):
    return cv2.undistort(frame, K, dist)

def click_cam1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_cam1) < PTS:
        x_real = int(x / SCALE)
        y_real = int(y / SCALE)
        points_cam1.append([x_real, y_real])
        print(f"[Cam1] {len(points_cam1)}:", (x_real, y_real))

def click_cam2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_cam2) < PTS:
        x_real = int(x / SCALE)
        y_real = int(y / SCALE)
        points_cam2.append([x_real, y_real])
        print(f"[Cam2] {len(points_cam2)}:", (x_real, y_real))

def main():
    global points_cam1, points_cam2

    f1 = get_frame(RTSP_CAM1)
    f2 = get_frame(RTSP_CAM2)

    cv2.namedWindow("Cam1")
    cv2.namedWindow("Cam2")

    cv2.setMouseCallback("Cam1", click_cam1)
    cv2.setMouseCallback("Cam2", click_cam2)

    print("👉 Click corresponding points, press 's' to solve")

    while True:
        t1 = cv2.resize(f1, None, fx=SCALE, fy=SCALE)
        t2 = cv2.resize(f2, None, fx=SCALE, fy=SCALE)

        for i,p in enumerate(points_cam1):
            x, y = int(p[0] * SCALE), int(p[1] * SCALE)
            cv2.circle(t1, (x,y), 5, (0,0,255), -1)
            cv2.putText(t1, str(i+1), (x,y), 0, 0.6, (0,0,255), 2)

        for i,p in enumerate(points_cam2):
            x, y = int(p[0] * SCALE), int(p[1] * SCALE)
            cv2.circle(t2, (x,y), 5, (255,0,0), -1)
            cv2.putText(t2, str(i+1), (x,y), 0, 0.6, (255,0,0), 2)

        cv2.imshow("Cam1", t1)
        cv2.imshow("Cam2", t2)

        key = cv2.waitKey(1)

        if key == ord('r'):
            points_cam1.clear()
            points_cam2.clear()
            print("Reset")

        if key == ord('s'):
            pts1 = np.array(points_cam1, dtype=np.float32)
            pts2 = np.array(points_cam2, dtype=np.float32)

            H, mask = cv2.findHomography(
                pts1, pts2,
                cv2.RANSAC,
                ransacReprojThreshold=2.0
            )

            print("H:\n", H)

            np.save("homography.npy", H)
            print("✅ Saved homography.npy")
            break

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()