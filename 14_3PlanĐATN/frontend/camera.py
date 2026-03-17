"""
camera.py — CameraStream class và danh sách cameras global
"""

import cv2
import sys
import time
import threading
import logging
from detection import get_model, log_detection, _yolo_available

DETECT_EVERY_N = 5    # detect mỗi N frame
CONF_THRESHOLD = 0.40
LOG_INTERVAL   = 10   # giây — tránh spam log


class CameraStream:
    def __init__(self, cam_id: int):
        self.cam_id           = cam_id
        self.cam_label        = f"CAM-0{cam_id + 1}"
        self.source           = None
        self.status           = "offline"
        self.info             = "—"
        self.person_count     = 0
        self.detection_enabled = True
        self._frame           = None
        self._lock            = threading.Lock()
        self._thread          = None
        self._stop_evt        = threading.Event()
        self.fps_real         = 0.0
        self.resolution       = ""
        self.stream_id        = 0

    # ── Frame access ──────────────────────────────────────────────────────────
    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame

    def _set_frame(self, data: bytes):
        with self._lock:
            self._frame = data

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self, source):
        self._stop_evt.set()          # signal thread cũ dừng
        self.stream_id    += 1
        self.source        = source
        self.status        = "connecting"
        self.info          = "—"
        self.person_count  = 0
        self._frame        = None
        self._stop_evt     = threading.Event()
        self._thread       = threading.Thread(
            target=self._capture_loop,
            args=(self.stream_id, self._stop_evt),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self.stream_id   += 1
        self._thread      = None
        self.status       = "offline"
        self.info         = "—"
        self.person_count = 0
        self._frame       = None

    # ── Capture + Detection loop ───────────────────────────────────────────────
    def _capture_loop(self, my_stream_id: int, stop_evt: threading.Event):
        cap        = None
        frame_cnt  = 0
        t_fps      = time.time()
        t_last_log = 0.0
        last_count = -1
        MAX_DIM    = 1280
        model      = get_model()

        while not stop_evt.is_set():
            try:
                # ── Open ─────────────────────────────────────────────────────
                if cap is None or not cap.isOpened():
                    source = self.source

                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                    if not cap.isOpened():
                        self.status = "error"
                        cap.release(); cap = None
                        stop_evt.wait(2)
                        continue

                    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    src = "RTSP"
                    self.resolution = f"{w}×{h}"
                    self._update_info(src, fps)

                # ── Read ─────────────────────────────────────────────────────
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.status = "error"
                    cap.release(); cap = None
                    stop_evt.wait(1)
                    continue

                self.status = "live"
                frame_cnt  += 1

                # ── Resize ───────────────────────────────────────────────────
                h_f, w_f = frame.shape[:2]
                if max(h_f, w_f) > MAX_DIM:
                    scale = MAX_DIM / max(h_f, w_f)
                    frame = cv2.resize(
                        frame,
                        (int(w_f * scale), int(h_f * scale)),
                        interpolation=cv2.INTER_AREA
                    )

                # ── Detection ────────────────────────────────────────────────
                annotated     = frame.copy()
                current_count = self.person_count   # giữ count cũ nếu skip frame

                if (model is not None
                        and self.detection_enabled
                        and frame_cnt % DETECT_EVERY_N == 0):

                    results = model(frame, classes=[0], conf=CONF_THRESHOLD,
                                    verbose=False)[0]
                    boxes   = results.boxes
                    crops   = []

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)

                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            crops.append(frame[y1:y2, x1:x2].copy())

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (63, 185, 80), 2)
                        cv2.putText(annotated, f"person {conf:.2f}",
                                    (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (63, 185, 80), 1, cv2.LINE_AA)

                    current_count     = len(boxes)
                    self.person_count = current_count

                    # Overlay count
                    txt = f"PERSONS: {current_count}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (8, 8), (tw + 16, th + 16), (22, 27, 34), -1)
                    cv2.putText(annotated, txt, (12, th + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (63, 185, 80), 2, cv2.LINE_AA)

                    # Log
                    now = time.time()
                    should_log = (
                        (current_count != last_count and current_count > 0)
                        or (current_count > 0 and now - t_last_log >= LOG_INTERVAL)
                    )
                    if should_log and crops:
                        log_detection(self.cam_id, self.cam_label, current_count, crops)
                        t_last_log = now
                    last_count = current_count

                    src = "RTSP"
                    self._update_info(src, self.fps_real, person_count=current_count)

                # ── Encode ───────────────────────────────────────────────────
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self._set_frame(buf.tobytes())

                # ── FPS ──────────────────────────────────────────────────────
                if frame_cnt % 30 == 0:
                    elapsed       = time.time() - t_fps
                    self.fps_real = round(30.0 / elapsed, 1) if elapsed > 0 else 0
                    t_fps         = time.time()
                    src           = "RTSP"
                    self._update_info(src, self.fps_real, person_count=self.person_count)

            except Exception as e:
                logging.exception(f"[CAM-{self.cam_id}] error: {e}")
                self.status = "error"
                if cap:
                    cap.release(); cap = None
                stop_evt.wait(2)

        if cap:
            cap.release()

    def _update_info(self, src_type: str, fps, person_count: int = None):
        pc         = self.person_count if person_count is None else person_count
        detect_str = f" | 👤 {pc}" if self.detection_enabled and _yolo_available else ""
        self.info  = f"{src_type} • {self.resolution} • {fps:.0f}fps{detect_str}"


# ─── Global list ─────────────────────────────────────────────────────────────
cameras: list[CameraStream] = [CameraStream(i) for i in range(4)]