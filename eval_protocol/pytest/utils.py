import asyncio
import inspect
from typing import Any, Callable, List

from ..models import EvaluateResult, EvaluationRow


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
            if not loop.is_closed():
                # Use existing loop
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


def evaluate(
    rows: List[EvaluationRow], reward_fn: Callable[..., EvaluateResult], **kwargs: Any
) -> List[EvaluationRow]:
    """Apply a reward function to each row and attach the result."""
    evaluated: List[EvaluationRow] = []
    for row in rows:
        result = reward_fn(messages=row.messages, ground_truth=row.ground_truth, **kwargs)
        row.evaluation_result = result
        evaluated.append(row)
    return evaluated


def aggregate(scores: List[float], method: str) -> float:
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
