import asyncio
from collections.abc import Callable
import functools

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import EvaluationTestMode, TestFunction


def create_dual_mode_wrapper(  # pyright: ignore[reportUnknownParameterType]
    test_func: TestFunction,
    mode: EvaluationTestMode,
    max_concurrent_rollouts: int,
    max_concurrent_evaluations: int,
    pytest_wrapper: Callable[[], None],
):
    """
    Creates a wrapper that supports both pytest parameterized execution and direct function calls.

    This wrapper enables the decorated evaluation test function to be used in two ways:
    1. As a pytest test (via pytest.mark.parametrize) with full parameterization
    2. As a direct function call with EvaluationRow data for programmatic use

    The wrapper automatically detects the calling pattern and routes to the appropriate
    execution path, ensuring consistent behavior regardless of how the function is invoked.

    Returns:
        A callable that can handle both pytest test execution and direct function calls
    """

    # Check if the test function is async
    is_async = asyncio.iscoroutinefunction(test_func)

    async def call_test_func(**call_kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Helper to call test_func with proper async/sync handling"""
        if is_async:
            return await test_func(**call_kwargs)  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues, reportCallIssue]
        else:
            return test_func(**call_kwargs)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]

    async def dual_mode_wrapper(*args, **kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        # Check if this is a direct call with the expected signature
        if mode == "pointwise":
            # For pointwise mode, check if called with a single row argument
            if len(args) == 1 and isinstance(args[0], EvaluationRow) and not kwargs:  # pyright: ignore[reportUnknownArgumentType]
                return await call_test_func(row=args[0])  # pyright: ignore[reportUnknownVariableType]
        else:
            # For batch mode, check if called with rows argument
            if (
                len(args) == 1  # pyright: ignore[reportUnknownArgumentType]
                and isinstance(args[0], list)
                and all(isinstance(r, EvaluationRow) for r in args[0])  # pyright: ignore[reportUnknownVariableType]
                and not kwargs
            ):
                return await call_test_func(rows=args[0])  # pyright: ignore[reportUnknownVariableType]
            # Also check if called with keyword argument 'rows'
            if (
                len(args) == 0  # pyright: ignore[reportUnknownArgumentType]
                and "rows" in kwargs
                and isinstance(kwargs["rows"], list)
                and all(isinstance(r, EvaluationRow) for r in kwargs["rows"])  # pyright: ignore[reportUnknownVariableType]
            ):
                return await call_test_func(**kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        # If not a direct call, use the pytest wrapper
        return await pytest_wrapper(*args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]

    dual_mode_wrapper._origin_func = test_func  # pyright: ignore[reportFunctionMemberAccess]
    dual_mode_wrapper._metainfo = {  # pyright: ignore[reportFunctionMemberAccess]
        "mode": mode,
        "max_rollout_concurrency": max_concurrent_rollouts,
        "max_evaluation_concurrency": max_concurrent_evaluations,
    }

    # Copy all attributes from the pytest wrapper to our dual mode wrapper

    functools.update_wrapper(dual_mode_wrapper, pytest_wrapper)  # pyright: ignore[reportUnknownArgumentType]

    return dual_mode_wrapper  # pyright: ignore[reportUnknownVariableType]
