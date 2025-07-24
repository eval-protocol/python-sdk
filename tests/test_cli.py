import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from eval_protocol.cli import deploy_command, main, parse_args, preview_command


class TestCLI:
    """Tests for the CLI functionality."""

    def test_parse_args(self):
        """Test the argument parser."""
        # Test preview command
        # Note: This test is less comprehensive than tests/test_cli_args.py
        # It doesn't check for --remote-url here.
        args, _ = parse_args(  # Unpack tuple
            ["preview", "--samples", "test.jsonl", "--metrics-folders", "m=p"]
        )  # Added metrics folders to pass new check
        assert args.command == "preview"
        assert args.samples == "test.jsonl"
        assert args.max_samples == 5  # default value

        # Test deploy command
        args, _ = parse_args(["deploy", "--id", "test-eval", "--metrics-folders", "test=./test"])  # Unpack tuple
        assert args.command == "deploy"
        assert args.id == "test-eval"
        assert args.metrics_folders == ["test=./test"]
        assert not args.force  # default value

    @patch("eval_protocol.cli_commands.preview.check_environment", return_value=True)
    @patch("eval_protocol.cli_commands.preview.preview_evaluation")
    def test_preview_command(self, mock_preview_eval, mock_preview_check_env):
        """Test the preview command (local mode)."""
        mock_preview_result = MagicMock()
        mock_preview_result.display = MagicMock()
        mock_preview_eval.return_value = mock_preview_result

        args = argparse.Namespace()
        args.metrics_folders = ["test=./test"]
        args.samples = "test.jsonl"
        args.max_samples = 5
        args.huggingface_dataset = None
        args.huggingface_split = "train"
        args.huggingface_prompt_key = "prompt"
        args.huggingface_response_key = "response"
        args.huggingface_key_map = None
        args.remote_url = None  # Added for compatibility with updated preview_command

        with patch("eval_protocol.cli_commands.preview.Path.exists", return_value=True):
            result = preview_command(args)

            assert result == 0
            mock_preview_check_env.assert_called_once()
            mock_preview_eval.assert_called_once_with(
                metric_folders=["test=./test"],
                sample_file="test.jsonl",
                max_samples=5,
                huggingface_dataset=None,
                huggingface_split="train",
                huggingface_prompt_key="prompt",
                huggingface_response_key="response",
                huggingface_message_key_map=None,
            )
            mock_preview_result.display.assert_called_once()

    @patch("eval_protocol.cli_commands.deploy.check_environment", return_value=True)
    @patch("eval_protocol.cli_commands.deploy.create_evaluation")
    def test_deploy_command(self, mock_create_eval, mock_deploy_check_env):
        """Test the deploy command (local mode)."""
        mock_create_eval.return_value = {"name": "test-evaluator"}

        args = argparse.Namespace()
        args.metrics_folders = ["test=./test"]
        args.id = "test-eval"
        args.display_name = "Test Evaluator"
        args.description = "Test description"
        args.force = True
        args.huggingface_dataset = None
        args.huggingface_split = "train"
        args.huggingface_prompt_key = "prompt"
        args.huggingface_response_key = "response"
        args.huggingface_key_map = None
        args.remote_url = None

        # Add attributes accessed by deploy_command, with defaults for non-GCP target
        args.target = "fireworks"  # Explicitly set for this local mode test
        args.function_ref = None
        args.gcp_project = None
        args.gcp_region = None
        args.gcp_ar_repo = None
        args.service_account = None
        args.entry_point = "reward_function"  # Default from parser
        args.runtime = "python311"  # Default from parser
        args.gcp_auth_mode = None  # Default from parser

        # For local deploy, metrics_folders is required. This is checked inside deploy_command.
        # The test_parse_args in test_cli_args.py covers parser-level requirement changes.

        result = deploy_command(args)

        assert result == 0
        mock_deploy_check_env.assert_called_once()
        mock_create_eval.assert_called_once_with(
            evaluator_id="test-eval",
            metric_folders=["test=./test"],
            display_name="Test Evaluator",
            description="Test description",
            force=True,
            huggingface_dataset=None,
            huggingface_split="train",
            huggingface_message_key_map=None,  # This is derived from args.huggingface_key_map
            huggingface_prompt_key="prompt",
            huggingface_response_key="response",
            # remote_url=None removed as it relies on default
        )

    @patch("eval_protocol.cli_commands.deploy.check_environment", return_value=False)
    @patch("eval_protocol.cli_commands.preview.check_environment", return_value=False)
    def test_command_environment_check(self, mock_preview_check_env, mock_deploy_check_env):
        """Test that commands check the environment and fail if check_environment returns False."""
        preview_args = argparse.Namespace()
        # For preview_command to proceed to check_environment, it needs either remote_url or metrics_folders,
        # and also sample sources.
        preview_args.metrics_folders = ["test=./test"]
        preview_args.samples = "test.jsonl"
        preview_args.max_samples = 1
        preview_args.huggingface_dataset = None
        preview_args.huggingface_split = "train"
        preview_args.huggingface_prompt_key = "prompt"
        preview_args.huggingface_response_key = "response"
        preview_args.huggingface_key_map = None
        preview_args.remote_url = None  # Added for compatibility

        deploy_args = argparse.Namespace()
        deploy_args.id = "test-eval"
        # For deploy_command to proceed to check_environment, it needs id.
        # If not remote_url, it also needs metrics_folders.
        deploy_args.metrics_folders = ["test=./test"]
        deploy_args.display_name = None
        deploy_args.description = None
        deploy_args.force = False
        deploy_args.huggingface_dataset = None
        deploy_args.huggingface_split = "train"
        deploy_args.huggingface_prompt_key = "prompt"
        deploy_args.huggingface_response_key = "response"
        deploy_args.huggingface_key_map = None
        deploy_args.remote_url = None
        deploy_args.target = "fireworks"  # Ensure target is set
        deploy_args.function_ref = None
        deploy_args.gcp_project = None
        deploy_args.gcp_region = None
        deploy_args.gcp_ar_repo = None
        deploy_args.service_account = None
        deploy_args.entry_point = "reward_function"
        deploy_args.runtime = "python311"
        deploy_args.gcp_auth_mode = None

        # Mock Path.exists for preview_args if it uses samples file
        with patch("eval_protocol.cli_commands.preview.Path.exists", return_value=True):
            preview_result = preview_command(preview_args)

        deploy_result = deploy_command(deploy_args)

        assert preview_result == 1
        assert deploy_result == 1
        mock_preview_check_env.assert_called_once()
        mock_deploy_check_env.assert_called_once()
