"""
CPose — main.py
Application entry point.
Run: python main.py
"""
import logging
import webbrowser
import threading

from app import create_app, socketio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

app = create_app()


def _open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    print("=" * 60)
    print("  CPose — Camera-Based Person Detection & Annotation")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    threading.Thread(target=_open_browser, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
