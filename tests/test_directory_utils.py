import os
import tempfile
from unittest.mock import patch
import pytest

from eval_protocol.directory_utils import find_eval_protocol_dir, find_eval_protocol_datasets_dir


class TestDirectoryUtils:
    """Test directory utility functions."""

    def test_find_eval_protocol_dir_uses_home_folder(self):
        """Test that find_eval_protocol_dir always maps to home folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                result = find_eval_protocol_dir()
                expected = os.path.expanduser("~/.eval_protocol")
                assert result == expected
                assert result.endswith(".eval_protocol")
                assert os.path.exists(result)

    def test_find_eval_protocol_dir_creates_directory(self):
        """Test that find_eval_protocol_dir creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                # Ensure the directory doesn't exist initially
                eval_protocol_dir = os.path.join(temp_dir, ".eval_protocol")
                if os.path.exists(eval_protocol_dir):
                    os.rmdir(eval_protocol_dir)

                # Call the function
                result = find_eval_protocol_dir()

                # Verify the directory was created
                assert result == eval_protocol_dir
                assert os.path.exists(result)
                assert os.path.isdir(result)

    def test_find_eval_protocol_dir_handles_tilde_expansion(self):
        """Test that find_eval_protocol_dir properly handles tilde expansion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                result = find_eval_protocol_dir()
                expected = os.path.expanduser("~/.eval_protocol")
                assert result == expected
                assert result.startswith(temp_dir)

    def test_find_eval_protocol_datasets_dir_uses_home_folder(self):
        """Test that find_eval_protocol_datasets_dir also uses home folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                result = find_eval_protocol_datasets_dir()
                expected = os.path.expanduser("~/.eval_protocol/datasets")
                assert result == expected
                assert result.endswith(".eval_protocol/datasets")
                assert os.path.exists(result)
                assert os.path.isdir(result)

    def test_find_eval_protocol_datasets_dir_creates_directory(self):
        """Test that find_eval_protocol_datasets_dir creates the datasets directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                # Ensure the directories don't exist initially
                eval_protocol_dir = os.path.join(temp_dir, ".eval_protocol")
                datasets_dir = os.path.join(eval_protocol_dir, "datasets")
                if os.path.exists(datasets_dir):
                    os.rmdir(datasets_dir)
                if os.path.exists(eval_protocol_dir):
                    os.rmdir(eval_protocol_dir)

                # Call the function
                result = find_eval_protocol_datasets_dir()

                # Verify both directories were created
                assert result == datasets_dir
                assert os.path.exists(result)
                assert os.path.isdir(result)
                assert os.path.exists(eval_protocol_dir)
                assert os.path.isdir(eval_protocol_dir)

    def test_find_eval_protocol_dir_consistency(self):
        """Test that multiple calls to find_eval_protocol_dir return the same path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                result1 = find_eval_protocol_dir()
                result2 = find_eval_protocol_dir()
                assert result1 == result2

    def test_find_eval_protocol_datasets_dir_consistency(self):
        """Test that multiple calls to find_eval_protocol_datasets_dir return the same path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOME": temp_dir}):
                result1 = find_eval_protocol_datasets_dir()
                result2 = find_eval_protocol_datasets_dir()
                assert result1 == result2
