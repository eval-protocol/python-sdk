"""Tests for FireworksAPIClient user-agent header functionality."""

import re
from unittest.mock import MagicMock, patch

import pytest

from eval_protocol.common_utils import get_user_agent
from eval_protocol.fireworks_api_client import FireworksAPIClient


class TestFireworksAPIClientUserAgent:
    """Test that FireworksAPIClient correctly sets the User-Agent header."""

    def test_get_user_agent_format(self):
        """Test that get_user_agent returns the expected format."""
        user_agent = get_user_agent()
        # Should match format: eval-protocol/{version}
        # Version can be actual version or "unknown"
        assert user_agent.startswith("eval-protocol/")
        assert len(user_agent) > len("eval-protocol/")

    def test_get_user_agent_fallback_logic(self):
        """Test that get_user_agent has fallback logic for when version can't be imported.

        This test verifies the code structure, since actually triggering an import
        failure during the import statement is difficult to test reliably.
        The important behavior (User-Agent header being set) is verified in other tests.
        """
        # Verify the function exists and can be called normally
        user_agent = get_user_agent()
        # The function should always return a valid user agent string
        assert isinstance(user_agent, str)
        assert user_agent.startswith("eval-protocol/")

        # The actual fallback ("eval-protocol/unknown") happens when the import
        # fails, which is hard to simulate without patching at a very low level.
        # The try/except block in the implementation handles this gracefully.

    def test_get_headers_includes_user_agent(self):
        """Test that _get_headers includes the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")
        headers = client._get_headers()

        assert "User-Agent" in headers
        assert headers["User-Agent"] == get_user_agent()

    def test_get_request_includes_user_agent(self):
        """Test that GET requests include the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            client.get("test/path")

            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["Authorization"] == "Bearer test_key"

    def test_post_request_includes_user_agent(self):
        """Test that POST requests include the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.post("test/path", json={"key": "value"})

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["Authorization"] == "Bearer test_key"
            assert headers["Content-Type"] == "application/json"

    def test_post_with_files_excludes_content_type(self):
        """Test that POST requests with files exclude Content-Type header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.post("test/path", files={"file": MagicMock()})

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            # Content-Type should not be set when files are present
            assert "Content-Type" not in headers

    def test_put_request_includes_user_agent(self):
        """Test that PUT requests include the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "put", return_value=mock_response) as mock_put:
            client.put("test/path", json={"key": "value"})

            mock_put.assert_called_once()
            call_kwargs = mock_put.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["Authorization"] == "Bearer test_key"

    def test_patch_request_includes_user_agent(self):
        """Test that PATCH requests include the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "patch", return_value=mock_response) as mock_patch:
            client.patch("test/path", json={"key": "value"})

            mock_patch.assert_called_once()
            call_kwargs = mock_patch.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["Authorization"] == "Bearer test_key"

    def test_delete_request_includes_user_agent(self):
        """Test that DELETE requests include the User-Agent header."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "delete", return_value=mock_response) as mock_delete:
            client.delete("test/path")

            mock_delete.assert_called_once()
            call_kwargs = mock_delete.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["Authorization"] == "Bearer test_key"
            # DELETE requests shouldn't have Content-Type
            assert "Content-Type" not in headers

    def test_additional_headers_merged(self):
        """Test that additional headers passed to requests are merged with User-Agent."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            client.get("test/path", headers={"X-Custom-Header": "custom-value"})

            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            assert headers["X-Custom-Header"] == "custom-value"

    def test_user_agent_consistent_across_methods(self):
        """Test that User-Agent is consistent across all HTTP methods."""
        client = FireworksAPIClient(api_key="test_key", api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        expected_user_agent = get_user_agent()

        # Test all methods
        methods = [
            ("get", lambda: client.get("test/path")),
            ("post", lambda: client.post("test/path", json={})),
            ("put", lambda: client.put("test/path", json={})),
            ("patch", lambda: client.patch("test/path", json={})),
            ("delete", lambda: client.delete("test/path")),
        ]

        for method_name, method_call in methods:
            with patch.object(client._session, method_name, return_value=mock_response) as mock_method:
                method_call()

                call_kwargs = mock_method.call_args[1]
                headers = call_kwargs["headers"]

                assert "User-Agent" in headers, f"{method_name} should include User-Agent"
                assert headers["User-Agent"] == expected_user_agent, (
                    f"{method_name} User-Agent should match expected value"
                )

    def test_user_agent_without_api_key(self):
        """Test that User-Agent is still included even without API key."""
        client = FireworksAPIClient(api_key=None, api_base="https://api.fireworks.ai")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            client.get("test/path")

            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            headers = call_kwargs["headers"]

            assert "User-Agent" in headers
            assert headers["User-Agent"] == get_user_agent()
            # Authorization should not be present
            assert "Authorization" not in headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
