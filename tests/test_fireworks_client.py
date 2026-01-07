"""Tests for the consolidated Fireworks client factory."""

import os
from unittest.mock import patch

import pytest

from eval_protocol.fireworks_client import (
    create_fireworks_client,
    get_fireworks_extra_headers,
)


class TestGetFireworksExtraHeaders:
    """Tests for get_fireworks_extra_headers function."""

    def test_returns_none_when_env_var_not_set(self):
        """Should return None when FIREWORKS_EXTRA_HEADERS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("FIREWORKS_EXTRA_HEADERS", None)
            result = get_fireworks_extra_headers()
            assert result is None

    def test_returns_none_for_empty_string(self):
        """Should return None when FIREWORKS_EXTRA_HEADERS is empty."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": ""}):
            result = get_fireworks_extra_headers()
            assert result is None

    def test_returns_none_for_whitespace_only(self):
        """Should return None when FIREWORKS_EXTRA_HEADERS is whitespace only."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": "   "}):
            result = get_fireworks_extra_headers()
            assert result is None

    def test_parses_valid_json_object(self):
        """Should parse valid JSON object into dict."""
        headers = '{"X-Custom": "value", "X-Another": "test"}'
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": headers}):
            result = get_fireworks_extra_headers()
            assert result == {"X-Custom": "value", "X-Another": "test"}

    def test_returns_none_for_invalid_json(self):
        """Should return None and log warning for invalid JSON."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": "not json"}):
            result = get_fireworks_extra_headers()
            assert result is None

    def test_returns_none_for_json_array(self):
        """Should return None when JSON is an array instead of object."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": '["item1", "item2"]'}):
            result = get_fireworks_extra_headers()
            assert result is None

    def test_returns_none_for_json_string(self):
        """Should return None when JSON is a string instead of object."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": '"just a string"'}):
            result = get_fireworks_extra_headers()
            assert result is None

    def test_returns_none_for_non_string_values(self):
        """Should return None when JSON object has non-string values."""
        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": '{"key": 123}'}):
            result = get_fireworks_extra_headers()
            assert result is None


class TestCreateFireworksClient:
    """Tests for create_fireworks_client function."""

    def test_creates_client_with_explicit_api_key(self):
        """Should create client with explicitly provided API key."""
        client = create_fireworks_client(api_key="test-api-key")
        assert client.api_key == "test-api-key"

    def test_creates_client_with_explicit_base_url(self):
        """Should create client with explicitly provided base URL."""
        client = create_fireworks_client(
            api_key="test-api-key",
            base_url="https://custom.api.example.com",
        )
        assert str(client.base_url).rstrip("/") == "https://custom.api.example.com"

    def test_creates_client_with_explicit_account_id(self):
        """Should create client with explicitly provided account ID."""
        client = create_fireworks_client(
            api_key="test-api-key",
            account_id="test-account-123",
        )
        assert client.account_id == "test-account-123"

    def test_creates_client_with_explicit_extra_headers(self):
        """Should create client with explicitly provided extra headers."""
        extra_headers = {"X-Custom-Header": "test-value"}
        client = create_fireworks_client(
            api_key="test-api-key",
            extra_headers=extra_headers,
        )
        assert "X-Custom-Header" in client._custom_headers
        assert client._custom_headers["X-Custom-Header"] == "test-value"

    def test_merges_env_and_explicit_extra_headers(self):
        """Should merge env var headers with explicit headers, explicit taking precedence."""
        env_headers = '{"X-Env-Header": "env-value", "X-Override": "env"}'
        explicit_headers = {"X-Explicit-Header": "explicit-value", "X-Override": "explicit"}

        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": env_headers}):
            client = create_fireworks_client(
                api_key="test-api-key",
                extra_headers=explicit_headers,
            )
            # Both headers should be present
            assert client._custom_headers["X-Env-Header"] == "env-value"
            assert client._custom_headers["X-Explicit-Header"] == "explicit-value"
            # Explicit should override env
            assert client._custom_headers["X-Override"] == "explicit"

    def test_uses_env_extra_headers_when_no_explicit(self):
        """Should use env var extra headers when no explicit headers provided."""
        env_headers = '{"X-Env-Header": "env-value"}'

        with patch.dict(os.environ, {"FIREWORKS_EXTRA_HEADERS": env_headers}):
            client = create_fireworks_client(api_key="test-api-key")
            assert client._custom_headers["X-Env-Header"] == "env-value"

    def test_resolves_api_key_from_env(self):
        """Should resolve API key from environment when not explicitly provided."""
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "env-api-key"}):
            client = create_fireworks_client()
            assert client.api_key == "env-api-key"

    def test_resolves_base_url_from_env(self):
        """Should resolve base URL from environment when not explicitly provided."""
        with patch.dict(
            os.environ,
            {
                "FIREWORKS_API_KEY": "test-key",
                "FIREWORKS_API_BASE": "https://env.api.example.com",
            },
        ):
            client = create_fireworks_client()
            assert str(client.base_url).rstrip("/") == "https://env.api.example.com"
