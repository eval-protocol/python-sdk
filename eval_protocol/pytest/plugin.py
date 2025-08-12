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
import logging
from typing import Optional


def pytest_addoption(parser) -> None:
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
    group.addoption(
        "--ep-input-param",
        action="append",
        default=None,
        help=(
            "Override rollout input parameters. Can be used multiple times. "
            "Format: key=value or JSON via @path.json. Examples: "
            "--ep-input-param temperature=0 --ep-input-param @params.json"
        ),
    )
    group.addoption(
        "--ep-reasoning-effort",
        action="store",
        default=None,
        help=(
            "Set reasoning.effort for providers that support it (e.g., Fireworks) via LiteLLM extra_body. "
            "Values: low|medium|high"
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


def pytest_configure(config) -> None:
    # Quiet LiteLLM INFO spam early in pytest session unless user set a level
    try:
        if os.environ.get("LITELLM_LOG") is None:
            os.environ["LITELLM_LOG"] = "ERROR"
        _llog = logging.getLogger("LiteLLM")
        _llog.setLevel(logging.CRITICAL)
        _llog.propagate = False
        for _h in list(_llog.handlers):
            _llog.removeHandler(_h)
    except Exception:
        pass

    cli_val = config.getoption("--ep-max-rows")
    norm = _normalize_max_rows(cli_val)
    if norm is not None:
        os.environ["EP_MAX_DATASET_ROWS"] = norm

    if config.getoption("--ep-print-summary"):
        os.environ["EP_PRINT_SUMMARY"] = "1"

    summary_json_path = config.getoption("--ep-summary-json")
    if summary_json_path:
        os.environ["EP_SUMMARY_JSON"] = summary_json_path

    # Allow ad-hoc overrides of input params via CLI flags
    try:
        import json as _json
        import pathlib as _pathlib
        merged: dict = {}
        input_params_opts = config.getoption("--ep-input-param")
        if input_params_opts:
            for opt in input_params_opts:
                if opt is None:
                    continue
                opt = str(opt)
                if opt.startswith("@"):  # load JSON file
                    p = _pathlib.Path(opt[1:])
                    if p.is_file():
                        with open(p, "r", encoding="utf-8") as f:
                            obj = _json.load(f)
                            if isinstance(obj, dict):
                                merged.update(obj)
                elif "=" in opt:
                    k, v = opt.split("=", 1)
                    # Try parse JSON values, fallback to string
                    try:
                        merged[k] = _json.loads(v)
                    except Exception:
                        merged[k] = v
        reasoning_effort = config.getoption("--ep-reasoning-effort")
        if reasoning_effort:
            # Always place under extra_body to avoid LiteLLM rejecting top-level params
            eb = merged.setdefault("extra_body", {})
            eb["reasoning_effort"] = str(reasoning_effort)
        if merged:
            os.environ["EP_INPUT_PARAMS_JSON"] = _json.dumps(merged)
    except Exception:
        # best effort, do not crash pytest session
        pass


