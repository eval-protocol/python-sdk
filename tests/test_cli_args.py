import argparse
import subprocess
import sys
from typing import TypedDict

import pytest

# Module to be tested
from eval_protocol.cli import parse_args
from eval_protocol.cli_commands.utils import add_args_from_callable_signature


def test_unknown_flag_fails_fast(capsys):
    with pytest.raises(SystemExit) as e:
        parse_args(["create", "rft", "--definitely-not-a-real-flag"])
    assert e.value.code == 2
    out = capsys.readouterr()
    # argparse writes errors to stderr
    assert "unrecognized arguments" in out.err
    assert "--definitely-not-a-real-flag" in out.err


def test_create_rft_help_does_not_error():
    """Smoke test: `python -m eval_protocol create rft --help` should exit cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "eval_protocol", "create", "rft", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "create rft" in combined
    assert "--dry-run" in combined


def test_verbose_flag():
    """Test verbose flag with upload command."""
    parsed_verbose_short, _ = parse_args(["-v", "upload", "--path", "."])
    assert parsed_verbose_short.verbose is True

    parsed_verbose_long, _ = parse_args(["--verbose", "upload", "--path", "."])
    assert parsed_verbose_long.verbose is True

    parsed_not_verbose, _ = parse_args(["upload", "--path", "."])
    assert parsed_not_verbose.verbose is False


def test_add_args_skips_duplicate_nested_shorthand_flags():
    class AwsS3Config(TypedDict):
        credentials_secret: str

    class AzureBlobStorageConfig(TypedDict):
        credentials_secret: str

    def create(
        *,
        aws_s3_config: AwsS3Config | None = None,
        azure_blob_storage_config: AzureBlobStorageConfig | None = None,
    ) -> None:
        return None

    parser = argparse.ArgumentParser()
    add_args_from_callable_signature(parser, create)

    parsed = parser.parse_args(
        [
            "--credentials-secret",
            "aws-short",
            "--azure-blob-storage-config-credentials-secret",
            "azure-prefixed",
        ]
    )

    assert parsed.aws_s3_config_credentials_secret == "aws-short"
    assert parsed.azure_blob_storage_config_credentials_secret == "azure-prefixed"
    assert "--credentials-secret" in parser._option_string_actions


def test_add_args_preserves_top_level_canonical_flag_over_nested_shorthand():
    class InferenceParameters(TypedDict):
        extra_body: str

    def create(
        *,
        inference_parameters: InferenceParameters | None = None,
        extra_body: str | None = None,
    ) -> None:
        return None

    parser = argparse.ArgumentParser()
    add_args_from_callable_signature(parser, create)

    parsed = parser.parse_args(
        [
            "--extra-body",
            "top-level",
            "--inference-parameters-extra-body",
            "nested",
        ]
    )

    assert parsed.extra_body == "top-level"
    assert parsed.inference_parameters_extra_body == "nested"
