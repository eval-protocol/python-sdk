"""
Pytest plugin for Eval Protocol developer ergonomics.

Adds a discoverable CLI flag `--ep-max-rows` to control how many rows
evaluation_test processes. This sets the environment variable
`EP_MAX_DATASET_ROWS` so the core decorator can apply it uniformly to
both URL datasets and in-memory input_messages.

Usage:
  - CLI: pytest --ep-max-rows=2  # or --ep-max-rows=all for no limit
  - Defaults: If not provided, no override is applied (tests use the
    max_dataset_rows value set in the decorator).
"""

import os
from typing import Optional

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("eval-protocol")
    group.addoption(
        "--ep-max-rows",
        action="store",
        default=None,
        help=(
            "Limit number of dataset rows processed by evaluation_test. "
            "Pass an integer (e.g., 2, 50) or 'all' for no limit."
        ),
    )
    group.addoption(
        "--ep-print-summary",
        action="store_true",
        default=False,
        help=(
            "Print a concise summary line (suite/model/effort/agg score) at the end of each evaluation_test."
        ),
    )
    group.addoption(
        "--ep-summary-json",
        action="store",
        default=None,
        help=(
            "Write a JSON summary artifact at the given path (e.g., ./outputs/aime_low.json)."
        ),
    )


def _normalize_max_rows(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip().lower()
    if s == "all":
        return "None"
    # Validate int; if invalid, ignore and return None (no override)
    try:
        int(s)
        return s
    except ValueError:
        return None


def pytest_configure(config: pytest.Config) -> None:
    cli_val = config.getoption("--ep-max-rows")
    norm = _normalize_max_rows(cli_val)
    if norm is not None:
        os.environ["EP_MAX_DATASET_ROWS"] = norm

    if config.getoption("--ep-print-summary"):
        os.environ["EP_PRINT_SUMMARY"] = "1"

    summary_json_path = config.getoption("--ep-summary-json")
    if summary_json_path:
        os.environ["EP_SUMMARY_JSON"] = summary_json_path


