import os
import subprocess
import sys
import time
import webbrowser


HOST = "127.0.0.1"
PORT = "8000"
URL = f"http://{HOST}:{PORT}"


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

    process = subprocess.Popen(command, cwd=os.getcwd())
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
