"""
Tests for eval_protocol.utils.show_results_url module.
"""

import socket
from unittest.mock import patch, MagicMock
import pytest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from eval_protocol.utils.show_results_url import (
    is_server_running,
    generate_invocation_filter_url,
    store_local_ui_results_url,
)


class TestIsServerRunning:
    """Test the is_server_running function."""

    @patch("socket.socket")
    def test_server_running(self, mock_socket):
        """Test when server is running."""
        # Mock successful connection
        mock_socket_instance = MagicMock()
        mock_socket_instance.connect_ex.return_value = 0
        mock_socket.return_value.__enter__.return_value = mock_socket_instance

        result = is_server_running("localhost", 8000)
        assert result is True
        mock_socket_instance.connect_ex.assert_called_once_with(("localhost", 8000))

    @patch("socket.socket")
    def test_server_not_running(self, mock_socket):
        """Test when server is not running."""
        # Mock failed connection
        mock_socket_instance = MagicMock()
        mock_socket_instance.connect_ex.return_value = 1
        mock_socket.return_value.__enter__.return_value = mock_socket_instance

        result = is_server_running("localhost", 8000)
        assert result is False

    @patch("socket.socket")
    def test_connection_exception(self, mock_socket):
        """Test when connection raises an exception."""
        # Mock connection exception
        mock_socket.side_effect = Exception("Connection failed")

        result = is_server_running("localhost", 8000)
        assert result is False

    def test_default_parameters(self):
        """Test with default parameters."""
        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket_instance.connect_ex.return_value = 0
            mock_socket.return_value.__enter__.return_value = mock_socket_instance

            result = is_server_running()
            assert result is True
            mock_socket_instance.connect_ex.assert_called_once_with(("localhost", 8000))


class TestGenerateInvocationFilterUrl:
    """Test the generate_invocation_filter_url function."""

    def test_basic_url_generation(self):
        """Test basic URL generation with default base URL."""
        invocation_id = "test-123"
        result = generate_invocation_filter_url(invocation_id)

        assert "http://localhost:8000" in result
        assert "filterConfig=" in result
        assert invocation_id in result

    def test_custom_base_url(self):
        """Test URL generation with custom base URL."""
        invocation_id = "test-456"
        base_url = "http://example.com/pivot"
        result = generate_invocation_filter_url(invocation_id, base_url)

        assert base_url in result
        assert "filterConfig=" in result
        assert invocation_id in result

    def test_url_encoding(self):
        """Test that special characters are properly URL encoded."""
        invocation_id = "test with spaces & symbols"
        result = generate_invocation_filter_url(invocation_id)

        # Should be URL encoded
        assert "%20" in result  # spaces
        assert "%26" in result  # ampersand

    def test_filter_config_structure(self):
        """Test that the filter config has the correct structure."""
        invocation_id = "test-789"
        result = generate_invocation_filter_url(invocation_id)

        # Decode the URL to check the filter config
        from urllib.parse import unquote, parse_qs

        parsed_url = parse_qs(result.split("?")[1])
        filter_config_str = unquote(parsed_url["filterConfig"][0])

        # Should contain the expected filter structure
        assert invocation_id in filter_config_str
        assert "execution_metadata.invocation_id" in filter_config_str
        assert "==" in filter_config_str  # operator
        assert "text" in filter_config_str  # type

    def test_pivot_and_table_urls(self):
        """Test URL generation for both pivot and table views."""
        invocation_id = "test-pivot-table"

        pivot_url = generate_invocation_filter_url(invocation_id, "http://localhost:8000/pivot")
        table_url = generate_invocation_filter_url(invocation_id, "http://localhost:8000/table")

        assert "pivot" in pivot_url
        assert "table" in table_url
        assert invocation_id in pivot_url
        assert invocation_id in table_url
        # Both should have the same filter config
        assert pivot_url.split("?")[1] == table_url.split("?")[1]


class TestStoreLocalUiResultsUrl:
    """Test the store_local_ui_results_url function."""

    @patch("eval_protocol.utils.show_results_url.store_local_ui_url")
    def test_stores_urls_correctly(self, mock_store):
        """Test that URLs are stored correctly."""
        invocation_id = "test-invocation"

        store_local_ui_results_url(invocation_id)

        # Should call store_local_ui_url once with correct parameters
        mock_store.assert_called_once()
        call_args = mock_store.call_args[0]

        assert call_args[0] == invocation_id  # invocation_id
        assert "pivot" in call_args[1]  # pivot_url
        assert "table" in call_args[2]  # table_url

    @patch("eval_protocol.utils.show_results_url.store_local_ui_url")
    def test_invocation_id_in_urls(self, mock_store):
        """Test that invocation_id appears in both URLs."""
        invocation_id = "unique-test-id-123"

        store_local_ui_results_url(invocation_id)

        call_args = mock_store.call_args[0]
        pivot_url = call_args[1]
        table_url = call_args[2]

        assert invocation_id in pivot_url
        assert invocation_id in table_url

    @patch("eval_protocol.utils.show_results_url.store_local_ui_url")
    def test_different_invocation_ids(self, mock_store):
        """Test that different invocation IDs produce different URLs."""
        # Test with first invocation ID
        store_local_ui_results_url("id-1")
        call_1 = mock_store.call_args[0]
        mock_store.reset_mock()

        # Test with second invocation ID
        store_local_ui_results_url("id-2")
        call_2 = mock_store.call_args[0]

        # URLs should be different
        assert call_1[1] != call_2[1]  # Pivot URLs different
        assert call_1[2] != call_2[2]  # Table URLs different
        assert "id-1" in call_1[1]
        assert "id-2" in call_2[1]


class TestIntegration:
    """Integration tests for the module."""

    @patch("eval_protocol.utils.show_results_url.store_local_ui_url")
    def test_full_workflow_stores_urls(self, mock_store):
        """Test the full workflow stores URLs correctly."""
        invocation_id = "integration-test"

        store_local_ui_results_url(invocation_id)

        # Verify store_local_ui_url was called
        mock_store.assert_called_once()
        call_args = mock_store.call_args[0]

        assert call_args[0] == invocation_id
        assert "pivot" in call_args[1]
        assert "table" in call_args[2]
        assert "integration-test" in call_args[1]
        assert "integration-test" in call_args[2]


class TestBrowserUtilities:
    """Test browser utility functions."""

    def test_get_pid_file_path(self):
        """Test PID file path generation."""
        from eval_protocol.utils.browser_utils import _get_pid_file_path
        from eval_protocol.directory_utils import find_eval_protocol_dir
        from pathlib import Path

        pid_file = _get_pid_file_path()
        expected = Path(find_eval_protocol_dir()) / "logs_server.pid"
        assert pid_file == expected

    def test_is_logs_server_running_no_pid_file(self, tmp_path, monkeypatch):
        """Test server detection when PID file doesn't exist."""
        from eval_protocol.utils.browser_utils import is_logs_server_running

        # Mock the PID file path to a non-existent file
        monkeypatch.setattr(
            "eval_protocol.utils.browser_utils._get_pid_file_path", lambda: tmp_path / "nonexistent.pid"
        )

        is_running, port = is_logs_server_running()
        assert not is_running
        assert port is None

    def test_is_logs_server_running_invalid_pid_file(self, tmp_path, monkeypatch):
        """Test server detection with invalid PID file content."""
        from eval_protocol.utils.browser_utils import is_logs_server_running

        # Create invalid PID file
        pid_file = tmp_path / "invalid.pid"
        pid_file.write_text("invalid json")
        monkeypatch.setattr("eval_protocol.utils.browser_utils._get_pid_file_path", lambda: pid_file)

        is_running, port = is_logs_server_running()
        assert not is_running
        assert port is None

    def test_is_logs_server_running_missing_pid_key(self, tmp_path, monkeypatch):
        """Test server detection with PID file missing required keys."""
        from eval_protocol.utils.browser_utils import is_logs_server_running
        import json

        # Create PID file with missing pid key
        pid_file = tmp_path / "missing_pid.pid"
        pid_file.write_text(json.dumps({"port": 8000}))
        monkeypatch.setattr("eval_protocol.utils.browser_utils._get_pid_file_path", lambda: pid_file)

        is_running, port = is_logs_server_running()
        assert not is_running
        assert port is None

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_is_logs_server_running_nonexistent_process(self, tmp_path, monkeypatch):
        """Test server detection with PID file pointing to non-existent process."""
        from eval_protocol.utils.browser_utils import is_logs_server_running
        import json

        # Create PID file with non-existent PID
        pid_file = tmp_path / "nonexistent_process.pid"
        pid_file.write_text(json.dumps({"pid": 999999, "port": 8000}))
        monkeypatch.setattr("eval_protocol.utils.browser_utils._get_pid_file_path", lambda: pid_file)

        is_running, port = is_logs_server_running()
        assert not is_running
        assert port is None

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_is_logs_server_running_current_process(self, tmp_path, monkeypatch):
        """Test server detection with PID file pointing to current process."""
        from eval_protocol.utils.browser_utils import is_logs_server_running
        import json
        import os

        # Create PID file with current process PID
        pid_file = tmp_path / "current_process.pid"
        pid_file.write_text(json.dumps({"pid": os.getpid(), "port": 8000}))
        monkeypatch.setattr("eval_protocol.utils.browser_utils._get_pid_file_path", lambda: pid_file)

        is_running, port = is_logs_server_running()
        assert is_running
        assert port == 8000

    def test_open_browser_tab(self, monkeypatch):
        """Test browser tab opening."""
        from eval_protocol.utils.browser_utils import open_browser_tab

        opened_urls = []

        def mock_open_new_tab(url):
            opened_urls.append(url)

        monkeypatch.setattr("webbrowser.open_new_tab", mock_open_new_tab)

        # Test with delay
        open_browser_tab("http://example.com", delay=0.01)

        # Wait a bit for the thread to execute
        import time

        time.sleep(0.02)

        assert len(opened_urls) == 1
        assert opened_urls[0] == "http://example.com"


class TestLogsServerPidFile:
    """Test logs server PID file functionality."""

    def test_write_pid_file(self, tmp_path, monkeypatch):
        """Test PID file writing."""
        from eval_protocol.utils.browser_utils import write_pid_file
        import json

        # Mock the find_eval_protocol_dir function
        monkeypatch.setattr("eval_protocol.directory_utils.find_eval_protocol_dir", lambda: str(tmp_path))

        # Test writing PID file
        write_pid_file(12345, 8000)

        # Check that PID file was created
        pid_file = tmp_path / "logs_server.pid"
        assert pid_file.exists()

        # Check content
        with open(pid_file, "r") as f:
            data = json.load(f)
            assert "pid" in data
            assert "port" in data
            assert data["port"] == 8000
            assert data["pid"] == 12345
