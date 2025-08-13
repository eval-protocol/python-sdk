import argparse

import pytest

# Module to be tested
from eval_protocol.cli import parse_args


class TestCliArgParsing:
    # --- Tests for 'preview' command ---
    def test_preview_with_remote_url_and_samples(self):
        args_list = [
            "preview",
            "--remote-url",
            "http://example.com/eval",
            "--samples",
            "dummy.jsonl",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "preview"
        assert parsed.remote_url == "http://example.com/eval"
        assert parsed.samples == "dummy.jsonl"
        assert parsed.metrics_folders is None  # Should be None if not provided

    def test_preview_with_remote_url_and_hf_dataset(self):
        args_list = [
            "preview",
            "--remote-url",
            "http://example.com/eval",
            "--hf",
            "dataset_name",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "preview"
        assert parsed.remote_url == "http://example.com/eval"
        assert parsed.huggingface_dataset == "dataset_name"

    def test_preview_with_remote_url_and_metrics_folders(self):
        """Metrics folders should be accepted by argparse but logic in command might ignore/warn."""
        args_list = [
            "preview",
            "--remote-url",
            "http://example.com/eval",
            "--metrics-folders",
            "mf=path",
            "--samples",
            "s.jsonl",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "preview"
        assert parsed.remote_url == "http://example.com/eval"
        assert parsed.metrics_folders == ["mf=path"]

    def test_preview_without_remote_url_requires_metrics_folders_or_command_logic_handles(
        self,
    ):
        """Argparse allows no metrics_folders, command logic should enforce if needed."""
        args_list = [
            "preview",
            "--samples",
            "dummy.jsonl",
        ]  # No --remote-url, no --metrics-folders
        parsed, _ = parse_args(args_list)
        assert parsed.command == "preview"
        assert parsed.remote_url is None
        assert parsed.metrics_folders is None
        # The command logic in preview.py now checks:
        # if not args.remote_url and not args.metrics_folders: error

    def test_preview_traditional_with_metrics_folders(self):
        args_list = [
            "preview",
            "--metrics-folders",
            "mf=path",
            "--samples",
            "dummy.jsonl",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "preview"
        assert parsed.metrics_folders == ["mf=path"]
        assert parsed.remote_url is None

    # --- Tests for 'deploy' command ---
    def test_deploy_with_remote_url(self):
        args_list = [
            "deploy",
            "--id",
            "my-eval",
            "--remote-url",
            "http://example.com/deploy-eval",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "deploy"
        assert parsed.id == "my-eval"
        assert parsed.remote_url == "http://example.com/deploy-eval"
        assert parsed.metrics_folders is None  # Not required, should be None if not given

    def test_deploy_with_remote_url_and_metrics_folders(self):
        """Metrics folders should be accepted by argparse but logic in command might ignore/warn."""
        args_list = [
            "deploy",
            "--id",
            "my-eval",
            "--remote-url",
            "http://example.com/eval",
            "--metrics-folders",
            "mf=path",
        ]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "deploy"
        assert parsed.id == "my-eval"
        assert parsed.remote_url == "http://example.com/eval"
        assert parsed.metrics_folders == ["mf=path"]

    def test_deploy_traditional_without_remote_url(self):
        args_list = ["deploy", "--id", "my-eval", "--metrics-folders", "mf=path"]
        parsed, _ = parse_args(args_list)
        assert parsed.command == "deploy"
        assert parsed.id == "my-eval"
        assert parsed.metrics_folders == ["mf=path"]
        assert parsed.remote_url is None

    def test_deploy_traditional_metrics_folders_still_optional_at_parser_level(self):
        """
        --metrics-folders is required=False at parser level.
        The command logic in deploy.py enforces it if --remote-url is not present.
        """
        args_list = [
            "deploy",
            "--id",
            "my-eval",
        ]  # No --metrics-folders, no --remote-url
        # This should parse fine, but deploy_command will raise error.
        parsed, _ = parse_args(args_list)
        assert parsed.command == "deploy"
        assert parsed.id == "my-eval"
        assert parsed.metrics_folders is None
        assert parsed.remote_url is None

    def test_deploy_id_is_required(self):
        with pytest.raises(SystemExit):  # argparse exits on missing required arg
            parse_args(["deploy"])  # Missing --id

    # General verbose flag
    def test_verbose_flag(self):
        # Global flags like -v or --verbose should typically come before the subcommand
        parsed_verbose_short, _ = parse_args(["-v", "preview", "--samples", "s.jsonl", "--metrics-folders", "m=p"])
        assert parsed_verbose_short.verbose is True

        parsed_verbose_long, _ = parse_args(
            ["--verbose", "preview", "--samples", "s.jsonl", "--metrics-folders", "m=p"]
        )
        assert parsed_verbose_long.verbose is True

        parsed_not_verbose, _ = parse_args(["preview", "--samples", "s.jsonl", "--metrics-folders", "m=p"])
        assert parsed_not_verbose.verbose is False
