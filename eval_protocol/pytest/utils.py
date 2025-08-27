import asyncio
import inspect
import os
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvalMetadata, EvaluationRow, Status
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import (
    CompletionParams,
    DatasetPathParam,
    EvaluationInputParam,
    InputMessagesParam,
    RolloutProcessorConfig,
)
from eval_protocol.pytest.exception_config import ExceptionHandlerConfig, get_default_exception_handler_config

import logging
import json


def execute_function(func: Callable, **kwargs) -> Any:
    """
    Execute a function with proper async handling.

    This is a pure function that handles both async and non-async function execution
    with proper event loop management for async functions.

    Args:
        func: The function to execute
        **kwargs: Arguments to pass to the function

    Returns:
        The result of the function execution
    """
    is_async = asyncio.iscoroutinefunction(func)
    if is_async:
        # Handle async functions with proper event loop management
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is already running, create a task and wait for it
                task = loop.create_task(func(**kwargs))
                # Use asyncio.wait to avoid run_until_complete on running loop
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, func(**kwargs))
                    results = future.result()
            elif not loop.is_closed():
                # Use existing loop that's not running
                task = loop.create_task(func(**kwargs))
                results = loop.run_until_complete(task)
            else:
                # Loop is closed, create a new one
                results = asyncio.run(func(**kwargs))
        except RuntimeError:
            # No event loop or other issues, create a new one
            results = asyncio.run(func(**kwargs))
    else:
        results = func(**kwargs)
    return results


AggregationMethod = Literal["mean", "max", "min"]


def aggregate(scores: List[float], method: AggregationMethod) -> float:
    if not scores:
        return 0.0
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    if method == "min":
        return min(scores)
    raise ValueError(f"Unknown aggregation method: {method}")


def create_dynamically_parameterized_wrapper(test_func, wrapper_body, test_param_names):
    """
    Creates a wrapper function with dynamic parameters for pytest parameterization.

    This function takes a test function and creates a wrapper that:
    1. Preserves the original function's metadata using functools.wraps
    2. Creates a new function signature with the specified parameter names that maps to pytest.mark.parametrize decorator
    3. Returns a callable that can be used with pytest.mark.parametrize

    The function signature is dynamically created to match the parameter names expected by
    pytest.mark.parametrize, ensuring that pytest can properly map the test parameters
    to the function arguments.

    Args:
        test_func: The original test function to wrap
        wrapper_body: The function body that contains the actual test logic
        test_param_names: List of parameter names for the dynamic signature

    Returns:
        A wrapper function with the specified parameter signature that calls wrapper_body
    """
    from functools import wraps

    @wraps(test_func)
    async def wrapper(**kwargs):
        return await wrapper_body(**kwargs)

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in test_param_names]
    wrapper.__signature__ = inspect.Signature(parameters)

    return wrapper


def log_eval_status_and_rows(
    eval_metadata: Optional[EvalMetadata],
    rows: Optional[List[EvaluationRow]] | None,
    status: Status,
    passed: bool,
    logger: DatasetLogger,
) -> None:
    """Update eval status and emit rows to the given logger.

    If no rows are provided, emits a minimal placeholder row so downstream
    consumers still observe a terminal status.
    """
    if eval_metadata is None:
        return

    eval_metadata.status = status
    eval_metadata.passed = passed

    rows_to_log: List[EvaluationRow] = rows or []
    if not rows_to_log:
        error_row = EvaluationRow(messages=[], eval_metadata=eval_metadata, evaluation_result=None)
        logger.log(error_row)
    else:
        for r in rows_to_log:
            if r.eval_metadata is not None:
                r.eval_metadata.status = status
            logger.log(r)


def parse_ep_max_rows(default_value: Optional[int]) -> Optional[int]:
    """Read EP_MAX_DATASET_ROWS env override as int or None.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_MAX_DATASET_ROWS")
    if raw is None:
        return default_value
    # plugin.py stores "None" as string for the "all" case
    return None if raw.lower() == "none" else int(raw)


def parse_ep_num_runs(default_value: int) -> int:
    """Read EP_NUM_RUNS env override as int.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_NUM_RUNS")
    return int(raw) if raw is not None else default_value


def parse_ep_max_concurrent_rollouts(default_value: int) -> int:
    """Read EP_MAX_CONCURRENT_ROLLOUTS env override as int.

    Assumes the environment variable was already validated by plugin.py.
    """
    raw = os.getenv("EP_MAX_CONCURRENT_ROLLOUTS")
    return int(raw) if raw is not None else default_value


def parse_ep_completion_params(completion_params: List[CompletionParams]) -> List[CompletionParams]:
    """Apply EP_INPUT_PARAMS_JSON overrides to completion_params.

    Reads the environment variable set by plugin.py and applies deep merge to each completion param.
    """
    try:
        _env_override = os.getenv("EP_INPUT_PARAMS_JSON")
        if _env_override:
            override_obj = json.loads(_env_override)
            if isinstance(override_obj, dict):
                # Apply override to each completion_params item
                return [deep_update_dict(dict(cp), override_obj) for cp in completion_params]
    except Exception:
        pass
    return completion_params


def parse_ep_passed_threshold(default_value: Optional[Union[float, dict]]) -> Optional[Union[float, dict]]:
    """Read EP_PASSED_THRESHOLD env override as float or dict.

    Assumes the environment variable was already validated by plugin.py.
    Supports both float values (e.g., "0.8") and JSON dict format (e.g., '{"success":0.8}').
    """
    raw = os.getenv("EP_PASSED_THRESHOLD")
    if raw is None:
        return default_value

    try:
        return float(raw)
    except ValueError:
        pass

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"EP_PASSED_THRESHOLD env var exists but can't be parsed: {raw}") from e


def deep_update_dict(base: dict, override: dict) -> dict:
    """Recursively update nested dictionaries in-place and return base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def generate_parameter_combinations(
    input_dataset: Optional[List[DatasetPathParam]],
    completion_params: List[CompletionParams],
    input_messages: Optional[List[InputMessagesParam]],
    input_rows: Optional[List[EvaluationRow]],
    evaluation_test_kwargs: Optional[List[EvaluationInputParam]],
    max_dataset_rows: Optional[int],
    combine_datasets: bool,
) -> List[tuple]:
    """
    Generate all combinations of parameters for pytest parameterization.

    Args:
        input_dataset: Dataset paths to use
        completion_params: Completion parameters to test
        input_messages: Input messages to use
        input_rows: Pre-constructed EvaluationRow objects to use
        evaluation_test_kwargs: Additional kwargs for evaluation tests
        max_dataset_rows: Maximum number of dataset rows to process
        combine_datasets: Whether to combine multiple datasets into one test

    Returns:
        List of parameter tuples for pytest.mark.parametrize
    """
    combinations = []

    # Handle optional parameters with defaults
    # Optionally combine multiple dataset paths into one logical dataset,
    # or parameterize to run one dataset per test invocation.
    if input_dataset is not None:
        if combine_datasets:
            datasets: List[Optional[List[DatasetPathParam]]] = [input_dataset]  # type: ignore
        else:
            # Fan out: one dataset path per parameterization
            if isinstance(input_dataset, list):  # type: ignore
                datasets = [[p] for p in input_dataset]  # type: ignore
            else:
                datasets = [[input_dataset]]  # type: ignore
    else:
        datasets = [None]

    cps: List[Optional[CompletionParams]] = completion_params if completion_params is not None else [None]  # type: ignore

    # Apply EP_MAX_DATASET_ROWS to input_messages, but do NOT parameterize over
    # each row. Instead, pass the entire sliced list through in a single test run
    # so summaries aggregate all rows together (AIME-style behavior).
    if input_messages is not None and isinstance(input_messages, list):
        effective_max_rows = parse_ep_max_rows(max_dataset_rows)
        if effective_max_rows is not None:
            sliced_messages = input_messages[:effective_max_rows]  # type: ignore
        else:
            sliced_messages = input_messages  # type: ignore
        # Wrap as a single parameter payload
        messages = [sliced_messages]  # type: ignore
    else:
        messages = [None]  # type: ignore

    # Handle input_rows - similar to input_messages, apply max_dataset_rows if specified
    if input_rows is not None and isinstance(input_rows, list):
        effective_max_rows = parse_ep_max_rows(max_dataset_rows)
        if effective_max_rows is not None:
            sliced_rows = input_rows[:effective_max_rows]  # type: ignore
        else:
            sliced_rows = input_rows  # type: ignore
        # Wrap as a single parameter payload
        rows = [sliced_rows]  # type: ignore
    else:
        rows = [None]  # type: ignore

    kwargs: List[Optional[EvaluationInputParam]] = (
        evaluation_test_kwargs if evaluation_test_kwargs is not None else [None]
    )  # type: ignore

    # Generate all combinations
    for ds in datasets:
        for cp in cps:
            for im in messages:
                for ir in rows:
                    for etk in kwargs:
                        # if no dataset, no messages, and no rows, raise an error
                        if ds is None and im is None and ir is None:
                            raise ValueError(
                                "No dataset, messages, or rows provided. Please provide at least one of input_dataset, input_messages, or input_rows."
                            )
                        combinations.append((ds, cp, im, ir, etk))

    return combinations


async def rollout_processor_with_retry(
    rollout_processor: RolloutProcessor,
    fresh_dataset: List[EvaluationRow],
    config: RolloutProcessorConfig,
):
    """
    Wrapper around rollout_processor that handles retry logic using the Python backoff library.

    Provides configurable exception handling with automatic retry for specific exception types:
    - Retryable exceptions (e.g., ConnectionError, TimeoutError) are automatically retried with backoff
    - Fail-fast exceptions (e.g., ValueError, TypeError) are not retried and return immediately
    - Unknown exceptions can be configured to either re-raise or return as failed rows

    The backoff behavior (exponential/constant, delays, max attempts) is fully configurable
    through the ExceptionHandlerConfig in the RolloutProcessorConfig.

    Yields results as they complete, allowing for concurrent processing while handling
    retries transparently in the background.
    """

    # Use provided exception handler config or fall back to default
    # Environment variable overrides are automatically applied in __post_init__
    exception_config = config.exception_handler_config or get_default_exception_handler_config()

    try:
        # Create initial batch of tasks (preserves indexing for mock processors)
        try:
            base_tasks = rollout_processor(fresh_dataset, config)
        except Exception as e:
            print(f"❌ Rollout processor failed to initialize: {e}")
            raise e

        # Create a single backoff-decorated retry function that can be reused
        @exception_config.get_backoff_decorator()
        async def execute_row_with_backoff_retry(row: EvaluationRow):
            """Execute rollout for a single row with backoff retry."""
            retry_config = replace(config, kwargs={**(config.kwargs or {}), "start_server": False})
            retry_tasks = rollout_processor([row], retry_config)
            return await retry_tasks[0]

        async def execute_row_with_backoff(task: asyncio.Task, row: EvaluationRow) -> EvaluationRow:
            """Execute a single row task with backoff retry."""

            try:
                # Try original task first
                result = await task
                result.rollout_status = Status.rollout_finished()
                return result
            except Exception as e:
                # NOTE: we perform these checks because we don't put the backoff decorator on initial batch call. we don't want to retry whole batch if anything fails.
                # Check if this exception should be retried
                is_retryable = any(isinstance(e, exc_type) for exc_type in exception_config.retryable_exceptions)
                giveup_func = exception_config.backoff_config.giveup_func
                should_giveup = giveup_func and giveup_func(e)

                if is_retryable and not should_giveup:
                    # Use shared backoff function for retryable exceptions
                    try:
                        result = await execute_row_with_backoff_retry(row)
                        result.rollout_status = Status.rollout_finished()
                        return result
                    except Exception as retry_error:
                        # Backoff gave up
                        logging.error(
                            f"❌ Rollout failed, (retried {exception_config.backoff_config.max_tries} times): {repr(retry_error)}"
                        )
                        row.rollout_status = Status.rollout_error(str(retry_error))
                        return row
                else:
                    # Non-retryable exception - fail immediately
                    logging.error(f"❌ Rollout failed (non-retryable error encountered): {repr(e)}")
                    row.rollout_status = Status.rollout_error(repr(e))
                    return row

        async def execute_row_with_backoff_and_log(task: asyncio.Task, row: EvaluationRow) -> EvaluationRow:
            """Execute a single row task with backoff retry and logging."""
            result = await execute_row_with_backoff(task, row)
            # Log the row after execution completes (success or failure)
            config.logger.log(result)
            return result

        # Process all tasks concurrently with backoff retry
        retry_tasks = [
            asyncio.create_task(execute_row_with_backoff_and_log(task, fresh_dataset[i]))
            for i, task in enumerate(base_tasks)
        ]

        # Yield results as they complete
        for task in asyncio.as_completed(retry_tasks):
            result = await task
            yield result

    finally:
        rollout_processor.cleanup()


def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filenames by replacing special characters with dashes."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return safe[:120]


def extract_effort_tag(params: dict) -> Optional[str]:
    """
    Extract effort tag from completion parameters for use in file naming.

    Args:
        params: Completion parameters dictionary

    Returns:
        Effort tag string if found, None otherwise
    """
    try:
        if not isinstance(params, dict):
            return None
        # Common locations
        if "extra_body" in params and isinstance(params["extra_body"], dict):
            eb = params["extra_body"]
            if isinstance(eb.get("reasoning"), dict) and "effort" in eb["reasoning"]:
                return str(eb["reasoning"]["effort"]).lower()
            if "reasoning_effort" in eb:
                return str(eb["reasoning_effort"]).lower()
        if "reasoning" in params and isinstance(params["reasoning"], dict) and "effort" in params["reasoning"]:
            return str(params["reasoning"]["effort"]).lower()
    except Exception:
        return None
    return None
