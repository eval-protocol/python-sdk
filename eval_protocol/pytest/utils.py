import asyncio
import inspect
import os
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvalMetadata, EvaluationRow, RolloutStatus
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import (
    CompletionParams,
    DatasetPathParam,
    EvaluationInputParam,
    InputMessagesParam,
    RolloutProcessorConfig,
)


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
    status: Literal["finished", "error"],
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
    """Read EP_MAX_DATASET_ROWS env override as int or None."""
    raw = os.getenv("EP_MAX_DATASET_ROWS")
    if raw is None:
        return default_value
    s = raw.strip().lower()
    if s == "none":
        return None
    try:
        return int(s)
    except ValueError:
        return default_value


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

    kwargs: List[Optional[EvaluationInputParam]] = (
        evaluation_test_kwargs if evaluation_test_kwargs is not None else [None]
    )  # type: ignore

    # Generate all combinations
    for ds in datasets:
        for cp in cps:
            for im in messages:
                for etk in kwargs:
                    # if no dataset and no messages, raise an error
                    if ds is None and im is None:
                        raise ValueError(
                            "No dataset or messages provided. Please provide at least one of input_dataset or input_messages."
                        )
                    combinations.append((ds, cp, im, etk))

    return combinations


async def rollout_processor_with_retry(
    rollout_processor: RolloutProcessor,
    fresh_dataset: List[EvaluationRow],
    config: RolloutProcessorConfig,
    max_retry: int,
):
    """
    Wrapper around rollout_processor that handles retry logic internally.
    Uses async queue pattern to yield results immediately as they become available.
    Yields both successful and failed results, leaving it up to the user to handle them in test_func.
    """

    try:
        queue: asyncio.Queue[EvaluationRow] = asyncio.Queue()
        retry_counts = {r.execution_metadata.rollout_id: 0 for r in fresh_dataset}
        failed_permanently = []

        async def retry_handler(failed_row: EvaluationRow):
            rollout_id = failed_row.execution_metadata.rollout_id
            current_attempts = retry_counts.get(rollout_id, 0)

            if current_attempts >= max_retry:
                assert (
                    failed_row.rollout_status and failed_row.rollout_status.status == RolloutStatus.Status.ERROR
                ), f"Rollout {failed_row.execution_metadata.rollout_id} did not fail with error status"
                failed_permanently.append(failed_row)
                await queue.put(failed_row)  # put failed row on queue
                return

            retry_counts[rollout_id] = current_attempts + 1

            # add kwargs start_server=False to config so we don't start new MCP server
            retry_config = replace(config, kwargs={**(config.kwargs or {}), "start_server": False})

            retry_tasks = rollout_processor([failed_row], retry_config)

            try:
                retry_result = await retry_tasks[0]
                retry_result.rollout_status.status = RolloutStatus.Status.FINISHED
                await queue.put(retry_result)
            except Exception as e:
                failed_row.rollout_status.status = RolloutStatus.Status.ERROR
                failed_row.rollout_status.termination_reason = str(e)
                asyncio.create_task(retry_handler(failed_row))  # retry failed, spawn another retry

        async def initial_processor():
            """Process initial batch and spawn retries for failures"""
            # catch any task creation errors and raise them immediately, i.e. port already in use
            try:
                base_tasks = rollout_processor(fresh_dataset, config)
            except Exception as e:
                print(f"❌ Rollout processor failed to initialize: {e}")
                raise e

            pending = set(base_tasks)

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    task_index = base_tasks.index(task)

                    try:
                        result = await task
                        result.rollout_status.status = RolloutStatus.Status.FINISHED
                        await queue.put(result)
                    except Exception as e:
                        failed_row = fresh_dataset[task_index]
                        failed_row.rollout_status.status = RolloutStatus.Status.ERROR
                        failed_row.rollout_status.termination_reason = str(e)
                        asyncio.create_task(retry_handler(failed_row))  # rollout errored, spawn retry task

        processor_task = asyncio.create_task(initial_processor())

        # yield results as they become available
        completed_count = 0
        total_expected = len(fresh_dataset)

        while completed_count < total_expected:
            finished_row = await queue.get()

            # only permanent failure rows are put on the queue, so we can check for them here
            if finished_row.rollout_status and finished_row.rollout_status.status == RolloutStatus.Status.ERROR:
                if max_retry > 0 and os.getenv("EP_FAIL_ON_MAX_RETRY", "true") != "false":
                    raise RuntimeError(
                        f"Rollout {finished_row.execution_metadata.rollout_id} failed after {max_retry} retries. Errors: {finished_row.rollout_status.termination_reason}"
                    )

            completed_count += 1
            yield finished_row

        await processor_task  # explicitly wait for task completion and catch any exceptions

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
