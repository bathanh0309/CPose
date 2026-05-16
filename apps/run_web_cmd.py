import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


HOST = "127.0.0.1"
PORT = "8000"
URL = f"http://{HOST}:{PORT}"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.openvino import available_openvino_devices, select_openvino_device, to_ultralytics_openvino_device


def detect_openvino() -> tuple[list[str], str | None]:
    try:
        devices = available_openvino_devices()
        # Prefer Intel Iris Xe iGPU through OpenVINO when it is available.
        # Web runtime still falls back to CPU on known OpenVINO GPU failures.
        device = select_openvino_device(devices, preferred="GPU.0", fallback="CPU")
        return devices, device
    except Exception as exc:
        print(f"[OpenVINO] unavailable: {type(exc).__name__}: {exc}")
        return [], None


def build_server_env() -> dict[str, str]:
    env = os.environ.copy()
    devices, device = detect_openvino()
    forced_device = env.get("CPOSE_OPENVINO_DEVICE")
    ultralytics_device = forced_device or to_ultralytics_openvino_device(device)

    env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;15000000|max_delay;500000"
    env["CPOSE_OPENVINO_ENABLED"] = "1" if device else "0"
    if ultralytics_device:
        env["CPOSE_OPENVINO_DEVICE"] = ultralytics_device

    print(f"[Python] {sys.executable}")
    print(f"[OpenVINO] devices: {devices or 'none'}")
    if device:
        print(f"[OpenVINO] CPose will prefer device: {ultralytics_device}")
    else:
        print("[OpenVINO] CPose will fall back to the configured PyTorch/CPU runtime.")
    print(f"[OpenCV] OPENCV_FFMPEG_CAPTURE_OPTIONS={env['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")
    return env


def stop_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    env = build_server_env()
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        HOST,
        "--port",
        PORT,
        "--reload",
    ]

    process = subprocess.Popen(command, cwd=str(ROOT), env=env)
    time.sleep(2)
    webbrowser.open(URL)

    print()
    print(f"CPose web is running at {URL}")
    print("Press Ctrl+Z in this cmd window to stop.")
    print()

    try:
        import msvcrt

        while process.poll() is None:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\x1a":
                    print("\nStopping CPose web backend...")
                    stop_process_tree(process)
                    return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping CPose web backend...")
        stop_process_tree(process)
        return 0
    finally:
        stop_process_tree(process)

    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
