from pathlib import Path
import argparse
import time

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent

VIDEO_PATH = ROOT / "data" / "input" / "cam2_2026-01-29_16-26-40.mp4"

TEST_MODELS = [
    {
        "name": "YOLO11 Detection OpenVINO",
        "path": MODELS_DIR / "yolo11n_openvino_model",
        "mode": "predict",
        "task": "detect",
    },
    {
        "name": "YOLO11 Pose OpenVINO",
        "path": MODELS_DIR / "yolo11n-pose_openvino_model",
        "mode": "track",
        "task": "pose",
    },
    {
        "name": "Tracking OpenVINO",
        "path": MODELS_DIR / "tracking_openvino_model",
        "mode": "track",
        "task": "detect",
    },
]


def print_openvino_devices() -> list[str]:
    try:
        try:
            from openvino.runtime import Core
        except ModuleNotFoundError:
            from openvino import Core

        ie = Core()
        devices = list(ie.available_devices)
        print(f"[OpenVINO] devices: {devices}")
        if "GPU" in devices:
            try:
                print(f"[OpenVINO] GPU: {ie.get_property('GPU', 'FULL_DEVICE_NAME')}")
            except Exception as exc:
                print(f"[OpenVINO] GPU name unavailable: {exc}")
        return devices
    except Exception as exc:
        print(f"[OpenVINO] unavailable: {type(exc).__name__}: {exc}")
        return []


def run_inference(model, mode: str, frame, device: str):
    yolo_device = device if device.lower().startswith("intel:") else f"intel:{device.lower()}"
    common = {
        "imgsz": 640,
        "conf": 0.60,
        "iou": 0.5,
        "classes": [0],
        "device": yolo_device,
        "verbose": False,
    }
    if mode == "track":
        return model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            **common,
        )
    return model.predict(frame, **common)


def test_model(model_info: dict, max_frames: int = 200, device: str = "GPU", show: bool = True):
    model_path = model_info["path"]
    name = model_info["name"]
    mode = model_info["mode"]
    task = model_info["task"]

    if not model_path.exists():
        print(f"[SKIP] {name}: not found {model_path}")
        return

    print()
    print("=" * 80)
    print(f"[TEST] {name}")
    print(f"[MODEL] {model_path}")
    yolo_device = device if device.lower().startswith("intel:") else f"intel:{device.lower()}"
    print(f"[DEVICE] OpenVINO={device} | Ultralytics={yolo_device}")
    print("=" * 80)

    model = YOLO(str(model_path), task=task)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    ok, warmup_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame from video: {VIDEO_PATH}")

    print("[WARMUP] Running 5 warm-up inferences...")
    for _ in range(5):
        run_inference(model, mode, warmup_frame, device=device)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    t0 = time.monotonic()

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        results = run_inference(model, mode, frame, device=device)

        annotated = results[0].plot() if show else None

        frame_count += 1
        elapsed = time.monotonic() - t0
        fps = frame_count / max(elapsed, 1e-6)

        if show:
            cv2.putText(
                annotated,
                f"{name} | {device} | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    if show:
        cv2.destroyWindow(name)

    elapsed = time.monotonic() - t0
    fps = frame_count / max(elapsed, 1e-6)

    print(f"[RESULT] Frames: {frame_count}")
    print(f"[RESULT] Time: {elapsed:.2f}s")
    print(f"[RESULT] FPS: {fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenVINO YOLO models with explicit device selection.")
    parser.add_argument("--device", default="GPU", help="OpenVINO device, usually GPU for Iris Xe or CPU as fallback.")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--no-show", action="store_true", help="Disable OpenCV display for pure benchmark timing.")
    args = parser.parse_args()

    print(f"[VIDEO] {VIDEO_PATH}")
    devices = print_openvino_devices()
    device = args.device.replace("intel:", "").upper()
    if device not in devices:
        raise RuntimeError(
            f"Requested OpenVINO device '{device}' is not available. "
            f"Available devices: {devices}"
        )

    for model_info in TEST_MODELS:
        test_model(model_info, max_frames=args.max_frames, device=device, show=not args.no_show)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
