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

import logging
import os
from typing import Optional
import json
import pathlib


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
        "--ep-num-runs",
        action="store",
        default=None,
        help=("Override the number of runs for evaluation_test. Pass an integer (e.g., 1, 5, 10)."),
    )
    group.addoption(
        "--ep-max-concurrent-rollouts",
        action="store",
        default=None,
        help=("Override the maximum number of concurrent rollouts. Pass an integer (e.g., 8, 50, 100)."),
    )
    group.addoption(
        "--ep-print-summary",
        action="store_true",
        default=False,
        help=("Print a concise summary line (suite/model/effort/agg score) at the end of each evaluation_test."),
    )
    group.addoption(
        "--ep-summary-json",
        action="store",
        default=None,
        help=("Write a JSON summary artifact at the given path (e.g., ./outputs/aime_low.json)."),
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
            "Values: low|medium|high|none"
        ),
    )
    group.addoption(
        "--ep-max-retry",
        action="store",
        default=None,
        help=("Failed rollouts (with rollout_status.code indicating error) will be retried up to this many times."),
    )
    group.addoption(
        "--ep-fail-on-max-retry",
        action="store",
        default="true",
        choices=["true", "false"],
        help=(
            "Whether to fail the entire rollout when permanent failures occur after max retries. "
            "Default: true (fail on permanent failures). Set to 'false' to continue with remaining rollouts."
        ),
    )
    group.addoption(
        "--ep-success-threshold",
        action="store",
        default=None,
        help=("Override the success threshold for evaluation_test. Pass a float between 0.0 and 1.0 (e.g., 0.8)."),
    )
    group.addoption(
        "--ep-se-threshold",
        action="store",
        default=None,
        help=(
            "Override the standard error threshold for evaluation_test. "
            "Pass a float >= 0.0 (e.g., 0.05). If only this is set, success threshold defaults to 0.0."
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


def _normalize_number(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip()
    # Validate int; if invalid, ignore and return None (no override)
    try:
        num = int(s)
        if num <= 0:
            return None  # num_runs must be positive
        return str(num)
    except ValueError:
        return None


def _normalize_success_threshold(val: Optional[str]) -> Optional[float]:
    """Normalize success threshold value as float between 0.0 and 1.0."""
    if val is None:
        return None

    try:
        threshold_float = float(val.strip())
        if 0.0 <= threshold_float <= 1.0:
            return threshold_float
        else:
            return None  # threshold must be between 0 and 1
    except ValueError:
        return None


def _normalize_se_threshold(val: Optional[str]) -> Optional[float]:
    """Normalize standard error threshold value as float >= 0.0."""
    if val is None:
        return None

    try:
        threshold_float = float(val.strip())
        if threshold_float >= 0.0:
            return threshold_float
        else:
            return None  # standard error must be >= 0
    except ValueError:
        return None


def _build_passed_threshold_env(success: Optional[float], se: Optional[float]) -> Optional[str]:
    """Build the EP_PASSED_THRESHOLD environment variable value from the two separate thresholds."""
    if success is None and se is None:
        return None

    if se is None:
        return str(success)
    else:
        success_val = success if success is not None else 0.0
        threshold_dict = {"success": success_val, "standard_error": se}
        return json.dumps(threshold_dict)


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

    num_runs_val = config.getoption("--ep-num-runs")
    norm_runs = _normalize_number(num_runs_val)
    if norm_runs is not None:
        os.environ["EP_NUM_RUNS"] = norm_runs

    max_concurrent_val = config.getoption("--ep-max-concurrent-rollouts")
    norm_concurrent = _normalize_number(max_concurrent_val)
    if norm_concurrent is not None:
        os.environ["EP_MAX_CONCURRENT_ROLLOUTS"] = norm_concurrent

    if config.getoption("--ep-print-summary"):
        os.environ["EP_PRINT_SUMMARY"] = "1"

    summary_json_path = config.getoption("--ep-summary-json")
    if summary_json_path:
        os.environ["EP_SUMMARY_JSON"] = summary_json_path

    max_retry = config.getoption("--ep-max-retry")
    norm_max_retry = _normalize_number(max_retry)
    if norm_max_retry is not None:
        os.environ["EP_MAX_RETRY"] = norm_max_retry

    fail_on_max_retry = config.getoption("--ep-fail-on-max-retry")
    if fail_on_max_retry is not None:
        os.environ["EP_FAIL_ON_MAX_RETRY"] = fail_on_max_retry

    success_threshold_val = config.getoption("--ep-success-threshold")
    se_threshold_val = config.getoption("--ep-se-threshold")
    norm_success = _normalize_success_threshold(success_threshold_val)
    norm_se = _normalize_se_threshold(se_threshold_val)
    threshold_env = _build_passed_threshold_env(norm_success, norm_se)
    if threshold_env is not None:
        os.environ["EP_PASSED_THRESHOLD"] = threshold_env

    # Allow ad-hoc overrides of input params via CLI flags
    try:
        merged: dict = {}
        input_params_opts = config.getoption("--ep-input-param")
        if input_params_opts:
            for opt in input_params_opts:
                if opt is None:
                    continue
                opt = str(opt)
                if opt.startswith("@"):  # load JSON file
                    p = pathlib.Path(opt[1:])
                    if p.is_file():
                        with open(p, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                            if isinstance(obj, dict):
                                merged.update(obj)
                elif "=" in opt:
                    k, v = opt.split("=", 1)
                    # Try parse JSON values, fallback to string
                    try:
                        merged[k] = json.loads(v)
                    except Exception:
                        merged[k] = v
        reasoning_effort = config.getoption("--ep-reasoning-effort")
        if reasoning_effort:
            # Always place under extra_body to avoid LiteLLM rejecting top-level params
            eb = merged.setdefault("extra_body", {})
            # Convert "none" string to None value for API compatibility
            eb["reasoning_effort"] = None if reasoning_effort.lower() == "none" else str(reasoning_effort)
        if merged:
            os.environ["EP_INPUT_PARAMS_JSON"] = json.dumps(merged)
    except Exception:
        # best effort, do not crash pytest session
        pass
