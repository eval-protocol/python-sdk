"""
Utility functions for showing evaluation results URLs and checking server status.
"""

import socket
import urllib.parse

from eval_protocol.pytest.store_results_url import store_local_ui_url


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
                    "operator": "==",
                    "value": invocation_id,
                    "type": "text",
                }
            ],
        }
    ]

    # URL encode the filter config
    filter_config_json = str(filter_config).replace("'", '"')
    encoded_filter = urllib.parse.quote(filter_config_json)

    return f"{base_url}?filterConfig={encoded_filter}"


def store_local_ui_results_url(invocation_id: str) -> None:
    """
    Store URLs for viewing evaluation results filtered by invocation_id in pytest stash.

    Args:
                    invocation_id: The invocation ID to filter results by
    """
    pivot_url = generate_invocation_filter_url(invocation_id, "http://localhost:8000/pivot")
    table_url = generate_invocation_filter_url(invocation_id, "http://localhost:8000/table")

    # Store URLs in pytest stash for later printing in pytest_sessionfinish
    store_local_ui_url(invocation_id, pivot_url, table_url)
