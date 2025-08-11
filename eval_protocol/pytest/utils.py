import asyncio
import inspect
from typing import Any, Callable, List, Literal, Optional

from eval_protocol.dataset_logger.dataset_logger import DatasetLogger
from eval_protocol.models import EvalMetadata, EvaluationRow


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
    def wrapper(**kwargs):
        return wrapper_body(**kwargs)

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
