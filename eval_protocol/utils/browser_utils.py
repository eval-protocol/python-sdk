"""
Browser utilities for auto-opening evaluation results in the local UI.
"""

import json
import os
import threading
import time
import webbrowser
from pathlib import Path
from typing import Tuple, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _get_pid_file_path() -> Path:
    """Get the path to the logs server PID file."""
    from eval_protocol.directory_utils import find_eval_protocol_dir

    return Path(find_eval_protocol_dir()) / "logs_server.pid"


def write_pid_file(pid: int, port: int) -> None:
    """
    Write the server PID and port to a file for external processes to check.

    Args:
        pid: The process ID of the logs server
        port: The port the server is running on
    """
    try:
        pid_file = _get_pid_file_path()

        data = {"pid": pid, "port": port}

        with open(pid_file, "w") as f:
            json.dump(data, f)

        # Use print instead of logger to avoid circular imports
        print(f"Wrote PID file: {pid_file} with PID {pid} and port {port}")
    except Exception as e:
        print(f"Warning: Failed to write PID file: {e}")


def is_logs_server_running() -> Tuple[bool, Optional[int]]:
    """
    Check if the logs server is running by reading the PID file and verifying the process.

    Returns:
        Tuple of (is_running, port) where:
        - is_running: True if server is running, False otherwise
        - port: The port the server is running on, or None if not running
    """
    if not PSUTIL_AVAILABLE:
        return False, None

    pid_file = _get_pid_file_path()
    if not pid_file.exists():
        return False, None

    try:
        with open(pid_file, "r") as f:
            data = json.load(f)
            pid = data.get("pid")
            port = data.get("port")
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False, None

    if pid is None:
        return False, None

    try:
        # Check if the process is still running
        process = psutil.Process(pid)
        if not process.is_running():
            return False, None

        # Optionally verify it's listening on the expected port
        if port is not None:
            try:
                connections = process.net_connections()
                for conn in connections:
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        return True, port
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                # If we can't check connections, assume it's running if process exists
                pass

        return True, port
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False, None


def open_browser_tab(url: str, delay: float = 0.5) -> None:
    """
    Open a URL in a new browser tab with an optional delay.

    Args:
        url: The URL to open
        delay: Delay in seconds before opening browser (default: 0.5)
    """

    def _open():
        time.sleep(delay)  # Give the server time to start
        webbrowser.open_new_tab(url)

    thread = threading.Thread(target=_open)
    thread.daemon = True
    thread.start()
