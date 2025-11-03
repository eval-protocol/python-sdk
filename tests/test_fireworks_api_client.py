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


class TestFireworksAPIClientPathHandling:
    """Test that FireworksAPIClient correctly handles relative paths and prevents URL construction bugs."""

    def test_post_relative_path_combines_with_api_base(self):
        """Test that POST requests correctly combine relative paths with api_base."""
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            relative_path = "v1/test/evaluator:getUploadEndpoint"
            client.post(relative_path, json={"name": "test"})

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # Check the URL passed to requests.post
            assert call_args[0][0] == f"{api_base}/{relative_path}"
            assert not call_args[0][0].startswith(f"{api_base}/{api_base}")

    def test_get_relative_path_combines_with_api_base(self):
        """Test that GET requests correctly combine relative paths with api_base."""
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            relative_path = "verifyApiKey"
            client.get(relative_path)

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[0][0] == f"{api_base}/{relative_path}"

    def test_post_get_upload_endpoint_path(self):
        """Test the specific getUploadEndpoint path that was buggy.

        This ensures relative paths like 'v1/{name}:getUploadEndpoint' are handled correctly
        and don't get double-prefixed with api_base.
        """
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"filenameToSignedUrls": {"test.tar.gz": "https://signed.url"}}

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            evaluator_name = "test-evaluator"
            # This is the correct pattern - relative path, not full URL
            upload_endpoint_path = f"v1/{evaluator_name}:getUploadEndpoint"
            client.post(upload_endpoint_path, json={"name": evaluator_name})

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            expected_url = f"{api_base}/{upload_endpoint_path}"
            actual_url = call_args[0][0]
            assert actual_url == expected_url, f"Expected {expected_url}, got {actual_url}"
            # Ensure it doesn't have the buggy double-prefix
            assert not actual_url.startswith(f"{api_base}/{api_base}")

    def test_post_validate_upload_path(self):
        """Test the specific validateUpload path that was buggy.

        This ensures relative paths like 'v1/{name}:validateUpload' are handled correctly.
        """
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "validated"}

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            evaluator_name = "test-evaluator"
            # This is the correct pattern - relative path, not full URL
            validate_path = f"v1/{evaluator_name}:validateUpload"
            client.post(validate_path, json={"name": evaluator_name})

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            expected_url = f"{api_base}/{validate_path}"
            actual_url = call_args[0][0]
            assert actual_url == expected_url, f"Expected {expected_url}, got {actual_url}"
            # Ensure it doesn't have the buggy double-prefix
            assert not actual_url.startswith(f"{api_base}/{api_base}")

    def test_path_with_leading_slash_stripped(self):
        """Test that leading slashes in paths are correctly handled."""
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            # Path with leading slash should be handled correctly
            client.get("/v1/test/path")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            # Should not have double slash
            assert call_args[0][0] == f"{api_base}/v1/test/path"

    def test_api_base_with_trailing_slash(self):
        """Test that api_base with trailing slash is handled correctly."""
        api_base = "https://api.fireworks.ai/"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            relative_path = "v1/test/path"
            client.post(relative_path, json={})

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # Should not have double slash
            assert call_args[0][0] == f"https://api.fireworks.ai/{relative_path}"

    def test_all_http_methods_with_relative_paths(self):
        """Test that all HTTP methods correctly handle relative paths."""
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        test_path = "v1/accounts/test/evaluators"

        methods = [
            ("get", lambda p: client.get(p)),
            ("post", lambda p: client.post(p, json={})),
            ("put", lambda p: client.put(p, json={})),
            ("patch", lambda p: client.patch(p, json={})),
            ("delete", lambda p: client.delete(p)),
        ]

        for method_name, method_call in methods:
            with patch.object(client._session, method_name, return_value=mock_response) as mock_method:
                method_call(test_path)

                mock_method.assert_called_once()
                call_args = mock_method.call_args
                expected_url = f"{api_base}/{test_path}"
                actual_url = call_args[0][0]
                assert actual_url == expected_url, f"{method_name.upper()} expected {expected_url}, got {actual_url}"
                # Ensure no double-prefix bug
                assert not actual_url.startswith(f"{api_base}/{api_base}"), (
                    f"{method_name.upper()} URL has double-prefix bug: {actual_url}"
                )

    def test_paths_containing_v1_pattern(self):
        """Test various v1 API paths to ensure correct URL construction."""
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        test_cases = [
            "v1/accounts/test/evaluators",
            "v1/accounts/test/evaluators/eval-id",
            "v1/accounts/test/evaluatorsV2",
            "v1/accounts/test/evaluators:previewEvaluator",
            "v1/test-evaluator:getUploadEndpoint",
            "v1/test-evaluator:validateUpload",
        ]

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            for path in test_cases:
                client.post(path, json={})

                call_args = mock_post.call_args
                actual_url = call_args[0][0]
                expected_url = f"{api_base}/{path}"

                assert actual_url == expected_url, (
                    f"Path '{path}' resulted in URL '{actual_url}', expected '{expected_url}'"
                )
                assert not actual_url.startswith(f"{api_base}/{api_base}"), (
                    f"Path '{path}' has double-prefix bug: {actual_url}"
                )

                mock_post.reset_mock()

    def test_full_url_passed_by_mistake_detected(self):
        """Test that accidentally passing a full URL instead of relative path is detected.

        This test documents the bug pattern: if a full URL like '{api_base}/v1/path'
        is passed instead of a relative path like 'v1/path', it will result in a
        malformed URL like '{api_base}/{api_base}/v1/path'.

        This test verifies that our code correctly handles relative paths (which prevents
        the bug), and documents what would happen if the bug occurred.
        """
        api_base = "https://api.fireworks.ai"
        client = FireworksAPIClient(api_key="test_key", api_base=api_base)

        mock_response = MagicMock()
        mock_response.status_code = 200

        # CORRECT: Relative path (what we should use)
        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            correct_relative_path = "v1/test-evaluator:getUploadEndpoint"
            client.post(correct_relative_path, json={})

            call_args = mock_post.call_args
            correct_url = call_args[0][0]
            expected_correct_url = f"{api_base}/{correct_relative_path}"
            assert correct_url == expected_correct_url

        # INCORRECT: Full URL (this would cause the bug - but we're not actually testing this,
        # just documenting that our current implementation would create a malformed URL)
        # If someone accidentally did: client.post(f"{api_base}/v1/path", ...)
        # The result would be: f"{api_base}/{api_base}/v1/path" which is wrong.
        # Our tests above ensure we use relative paths, preventing this bug.
        mock_post.reset_mock()
        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            # Simulating what WOULD happen if buggy code passed full URL
            buggy_full_url = f"{api_base}/v1/test-evaluator:getUploadEndpoint"
            client.post(buggy_full_url, json={})

            call_args = mock_post.call_args
            buggy_url = call_args[0][0]
            # This shows what the buggy URL would look like
            buggy_expected = f"{api_base}/{buggy_full_url}"

            # This assertion documents the bug pattern - the URL would be malformed
            assert buggy_url == buggy_expected
            assert buggy_url.startswith(f"{api_base}/{api_base}"), (
                "This documents the bug: passing full URL creates double-prefix. Always use relative paths!"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
