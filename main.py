"""
CPose — main.py
Main Entry Point. Orchestrates startup and runs the server.
"""
import logging
import webbrowser
import threading
import os

from app import create_app, socketio, get_config

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
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    print("=" * 60)
    print("  CPose — AI-Powered Person Recognition & Tracking")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    
    # Auto-open browser in a separate thread
    threading.Thread(target=_open_browser, daemon=True).start()
    
    # Get server settings from config
    config = get_config()
    
    # Start the SocketIO server
    socketio.run(
        flask_app, 
        host=config.server.host, 
        port=config.server.port, 
        debug=config.server.debug
    )
