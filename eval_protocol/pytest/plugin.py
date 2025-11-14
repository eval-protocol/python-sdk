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
import sys
from pytest import StashKey
import pytest


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
    # deprecate this later
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
    group.addoption(
        "--ep-no-upload",
        action="store_true",
        default=False,
        help=(
            "Disable saving and uploading of detailed experiment JSON files to Fireworks. "
            "Default: false (experiment JSONs are saved and uploaded by default)."
        ),
    )
    group.addoption(
        "--ep-jsonl-path",
        default=None,
        help=("Load input from a jsonl file that is already in EvaluationRow or openai CHAT format"),
    )
    group.addoption(
        "--ep-completion-params",
        default=[],
        action="append",
        help=("Overwrite completion params with json. Can be used multiple times. "),
    )
    group.addoption(
        "--ep-remote-rollout-processor-base-url",
        default=None,
        help=("If set, use this base URL for remote rollout processing. Example: http://localhost:8000"),
    )
    group.addoption(
        "--ep-no-op-rollout-processor",
        action="store_true",
        default=False,
        help=(
            "Override the rollout processor to use NoOpRolloutProcessor, which passes input dataset through unchanged."
        ),
    )
    group.addoption(
        "--ep-output-dir",
        default=None,
        help=("If set, save evaluation results to this directory in jsonl format."),
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

    if config.getoption("--ep-output-dir"):
        # set this to save eval results to the target dir in jsonl format
        os.environ["EP_OUTPUT_DIR"] = config.getoption("--ep-output-dir")

    if config.getoption("--ep-no-op-rollout-processor"):
        os.environ["EP_USE_NO_OP_ROLLOUT_PROCESSOR"] = "1"

    if config.getoption("--ep-no-upload"):
        os.environ["EP_NO_UPLOAD"] = "1"

    if config.getoption("--ep-jsonl-path"):
        os.environ["EP_JSONL_PATH"] = config.getoption("--ep-jsonl-path")

    if config.getoption("--ep-completion-params"):
        # redump to json to make sure they are legit
        os.environ["EP_COMPLETION_PARAMS"] = json.dumps(
            [json.loads(s) for s in config.getoption("--ep-completion-params") or []]
        )

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


def _print_experiment_links(session):
    """Print all collected Fireworks experiment links from pytest stash."""
    try:
        # Late import to avoid circulars; if missing key, skip printing
        EXPERIMENT_LINKS_STASH_KEY = None
        try:
            from .store_experiment_link import EXPERIMENT_LINKS_STASH_KEY as _KEY  # type: ignore

            EXPERIMENT_LINKS_STASH_KEY = _KEY
        except Exception:
            EXPERIMENT_LINKS_STASH_KEY = None

        # Get links from pytest stash
        links = []
        if EXPERIMENT_LINKS_STASH_KEY is not None and EXPERIMENT_LINKS_STASH_KEY in session.stash:
            links = session.stash[EXPERIMENT_LINKS_STASH_KEY]

        # Only print when there is at least one successful link.
        # Suppress the entire section if all links are failures (noise).
        if any(link.get("status") == "success" for link in links):
            print("\n" + "=" * 80, file=sys.__stderr__)
            print("🔥 FIREWORKS EXPERIMENT LINKS", file=sys.__stderr__)
            print("=" * 80, file=sys.__stderr__)

            for link in links:
                print(f"Experiment {link['experiment_id']}: {link['job_link']}", file=sys.__stderr__)

            print("=" * 80, file=sys.__stderr__)
            return True
        return False
    except Exception:
        return False


def _print_local_ui_results_urls(session):
    """Print all collected evaluation results URLs from pytest stash."""
    try:
        # Late import to avoid circulars; if missing key, skip printing
        RESULTS_URLS_STASH_KEY = None
        try:
            from .store_results_url import RESULTS_URLS_STASH_KEY as _URL_KEY  # type: ignore

            RESULTS_URLS_STASH_KEY = _URL_KEY
        except Exception:
            RESULTS_URLS_STASH_KEY = None

        # Get URLs from pytest stash
        urls_dict = {}
        if RESULTS_URLS_STASH_KEY is not None and RESULTS_URLS_STASH_KEY in session.stash:
            urls_dict = session.stash[RESULTS_URLS_STASH_KEY]

        if urls_dict:
            print("\n" + "=" * 80, file=sys.__stderr__)
            print("📊 LOCAL UI EVALUATION RESULTS", file=sys.__stderr__)
            print("=" * 80, file=sys.__stderr__)

            for url_data in urls_dict.values():
                print(f"📊 Invocation {url_data['invocation_id']}:", file=sys.__stderr__)
                print(f"  📊 Aggregate scores: {url_data['pivot_url']}", file=sys.__stderr__)
                print(f"  📋 Trajectories: {url_data['table_url']}", file=sys.__stderr__)

            print("=" * 80, file=sys.__stderr__)
            return True
        return False
    except Exception:
        return False


def pytest_sessionfinish(session, exitstatus):
    """Print all collected Fireworks experiment links and evaluation results URLs from pytest stash."""
    try:
        # Print experiment links and results URLs separately
        links_printed = _print_experiment_links(session)
        urls_printed = _print_local_ui_results_urls(session)

        # Flush stderr if anything was printed
        if links_printed or urls_printed:
            err_stream = getattr(sys, "__stderr__", None)
            if err_stream is not None:
                try:
                    err_stream.flush()  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass
