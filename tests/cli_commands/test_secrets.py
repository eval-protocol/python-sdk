"""Tests for eval_protocol.cli_commands.secrets module."""

import os
from pathlib import Path

import pytest

from eval_protocol.cli_commands.secrets import load_secrets_from_env_file, mask_secret_value


class TestLoadSecretsFromEnvFile:
    """Tests for load_secrets_from_env_file function."""

    def test_basic_key_value(self, tmp_path: Path):
        """Test basic KEY=value parsing."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=myvalue123\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_KEY": "myvalue123"}

    def test_end_of_line_comment(self, tmp_path: Path):
        """Test that end-of-line comments are properly stripped.

        This was a bug where manual parsing included the comment in the value.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=test_dummy_value_abc123 # this is a comment\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_API_KEY": "test_dummy_value_abc123"}
        # Ensure the comment is NOT included
        assert "# this" not in result["MY_API_KEY"]
        assert "comment" not in result["MY_API_KEY"]

    def test_end_of_line_comment_no_space(self, tmp_path: Path):
        """Test end-of-line comment without space before #."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value#this is a comment\n")

        result = load_secrets_from_env_file(str(env_file))

        # python-dotenv treats # without space as part of value
        # unless the value is quoted. This is the expected behavior.
        assert result == {"KEY": "value#this is a comment"}

    def test_quoted_value_with_hash(self, tmp_path: Path):
        """Test that quoted values preserve # character."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="value#with#hashes"\n')

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"KEY": "value#with#hashes"}

    def test_double_quoted_values(self, tmp_path: Path):
        """Test that double-quoted values are properly unquoted."""
        env_file = tmp_path / ".env"
        env_file.write_text('MY_KEY="value with spaces"\n')

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_KEY": "value with spaces"}

    def test_single_quoted_values(self, tmp_path: Path):
        """Test that single-quoted values are properly unquoted."""
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY='value with spaces'\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_KEY": "value with spaces"}

    def test_comment_lines_ignored(self, tmp_path: Path):
        """Test that full-line comments are ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text("# This is a comment\nMY_KEY=myvalue\n# Another comment\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_KEY": "myvalue"}

    def test_empty_lines_ignored(self, tmp_path: Path):
        """Test that empty lines are ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\n\n\nKEY2=value2\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_multiple_keys(self, tmp_path: Path):
        """Test parsing multiple key-value pairs."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\nKEY3=value3\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}

    def test_file_not_found(self, tmp_path: Path):
        """Test that non-existent file returns empty dict."""
        env_file = tmp_path / "nonexistent.env"

        result = load_secrets_from_env_file(str(env_file))

        assert result == {}

    def test_empty_file(self, tmp_path: Path):
        """Test that empty file returns empty dict."""
        env_file = tmp_path / ".env"
        env_file.write_text("")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {}

    def test_value_with_equals_sign(self, tmp_path: Path):
        """Test that values containing = are properly parsed."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value=with=equals\n")

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"KEY": "value=with=equals"}

    def test_quoted_value_with_comment(self, tmp_path: Path):
        """Test quoted value followed by a comment."""
        env_file = tmp_path / ".env"
        env_file.write_text('MY_KEY="myvalue123" # this is a comment\n')

        result = load_secrets_from_env_file(str(env_file))

        assert result == {"MY_KEY": "myvalue123"}

    def test_complex_env_file(self, tmp_path: Path):
        """Test a complex .env file with various formats."""
        env_content = """# Configuration file
# Last updated: 2024-01-15

# API Keys
SERVICE_A_KEY=dummy_key_aaa # Service A key
SERVICE_B_KEY="dummy_key_bbb" # Service B key

# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME='mydb'

# Feature flags
ENABLE_FEATURE=true # enable new feature

# Empty values are skipped
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        result = load_secrets_from_env_file(str(env_file))

        assert result["SERVICE_A_KEY"] == "dummy_key_aaa"
        assert result["SERVICE_B_KEY"] == "dummy_key_bbb"
        assert result["DB_HOST"] == "localhost"
        assert result["DB_PORT"] == "5432"
        assert result["DB_NAME"] == "mydb"
        assert result["ENABLE_FEATURE"] == "true"
        # Ensure no comments leaked into values
        assert "# Service" not in result.get("SERVICE_A_KEY", "")
        assert "# Service" not in result.get("SERVICE_B_KEY", "")

    def test_export_prefix_handled(self, tmp_path: Path):
        """Test that 'export' prefix is handled (if supported by dotenv)."""
        env_file = tmp_path / ".env"
        env_file.write_text("export MY_KEY=myvalue123\n")

        result = load_secrets_from_env_file(str(env_file))

        # python-dotenv handles 'export' prefix
        assert result == {"MY_KEY": "myvalue123"}


class TestMaskSecretValue:
    """Tests for mask_secret_value function."""

    def test_normal_length_value(self):
        """Test masking a normal length secret."""
        result = mask_secret_value("abcdefghijklmnopqrstu")
        assert result == "abcdef***rstu"
        assert len(result) < len("abcdefghijklmnopqrstu")

    def test_short_value(self):
        """Test masking a very short secret."""
        result = mask_secret_value("abc")
        assert result == "a***c"

    def test_empty_value(self):
        """Test masking an empty value."""
        result = mask_secret_value("")
        assert result == "<empty>"

    def test_none_value(self):
        """Test masking None (edge case)."""
        result = mask_secret_value(None)
        assert result == "<empty>"

    def test_exact_boundary_length(self):
        """Test masking value at exactly prefix+suffix length (10 chars)."""
        # prefix_len=6, suffix_len=4, so <= 10 chars uses short format
        result = mask_secret_value("1234567890")
        assert result == "1***0"

    def test_just_over_boundary(self):
        """Test masking value just over the boundary."""
        # 11 chars should use the full format
        result = mask_secret_value("12345678901")
        assert result == "123456***8901"
