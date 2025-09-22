"""
Utility functions for checking server status and generating UI URLs.
"""

import socket
import urllib.parse
from typing import List, Dict, Any


def is_server_running(host: str = "localhost", port: int = 8000) -> bool:
    """
    Check if a server is running on the specified host and port.

    Args:
            host: The host to check (default: "localhost")
            port: The port to check (default: 8000)

    Returns:
            True if server is running, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def generate_invocation_filter_url(invocation_id: str, base_url: str = "http://localhost:8000") -> str:
    """
    Generate a URL for viewing results filtered by invocation_id.

    Args:
            invocation_id: The invocation ID to filter results by
            base_url: The base URL for the UI (default: "http://localhost:8000")

    Returns:
            URL-encoded URL with filter configuration
    """
    filter_config = [
        {
            "logic": "AND",
            "filters": [
                {
                    "field": "$.execution_metadata.invocation_id",
                    "operator": "equals",
                    "value": invocation_id,
                    "type": "text",
                }
            ],
        }
    ]

    # URL encode the filter config
    filter_config_json = str(filter_config).replace("'", '"')
    encoded_filter = urllib.parse.quote(filter_config_json)

    return f"{base_url}/pivot?filterConfig={encoded_filter}"


def show_results_url(invocation_id: str) -> None:
    """
    Show a URL for viewing evaluation results filtered by invocation_id.

    If the server is not running, prints a message to run "ep logs" to start the local UI.
    If the server is running, prints a URL to view results filtered by invocation_id.

    Args:
            invocation_id: The invocation ID to filter results by
    """
    if is_server_running():
        url = generate_invocation_filter_url(invocation_id)
        print(f"View your evaluation results: {url}")
    else:
        url = generate_invocation_filter_url(invocation_id)
        print(f"Start the local UI with 'ep logs', then visit: {url}")
