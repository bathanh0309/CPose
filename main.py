"""
CPose - main.py
Main entry point. Orchestrates startup and runs the server.
"""

import logging
import threading
import webbrowser

from app import create_app, get_config, socketio

# Setup console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("[Main]")

# Create app instance using the factory
flask_app = create_app()


def _open_browser():
    """Wait for server startup and open the dashboard."""
    import time

    time.sleep(2.0)
    webbrowser.open(get_config().project.dashboard_url)


if __name__ == "__main__":
    config = get_config()
    dashboard_url = config.project.dashboard_url

    print("=" * 60)
    print("  CPose - AI-Powered Person Recognition & Tracking")
    print(f"  Dashboard: {dashboard_url}")
    print("=" * 60)

    threading.Thread(target=_open_browser, daemon=True).start()

    socketio.run(
        flask_app,
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug,
    )
