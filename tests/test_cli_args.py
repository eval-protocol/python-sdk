import subprocess
import sys

import pytest

# Module to be tested
from eval_protocol.cli import parse_args


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
