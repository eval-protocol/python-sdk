"""
Tests for eval_protocol.utils.show_results_url module.
"""

import socket
from unittest.mock import patch, MagicMock
import pytest

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
