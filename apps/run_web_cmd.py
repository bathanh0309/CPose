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


def build_server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;15000000|max_delay;500000"
    env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"[Python] {sys.executable}")
    print("[Runtime] CPU only")
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
