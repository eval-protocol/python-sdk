import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast
from eval_protocol.models import EvaluationRow, EvaluateResult, Status
from eval_protocol.pytest.types import Dataset, EvaluationInputParam, TestFunction


async def execute_pytest(
    test_func: TestFunction,
    processed_row: EvaluationRow | None = None,
    processed_dataset: Dataset | None = None,
    evaluation_test_kwargs: EvaluationInputParam | None = None,
) -> EvaluationRow | Dataset:
    """
    Generic function that handles both sync and async test functions.
    """
    if evaluation_test_kwargs is not None:
        if "row" in evaluation_test_kwargs:
            raise ValueError("'row' is a reserved parameter for the evaluation function")
        if "rows" in evaluation_test_kwargs:
            raise ValueError("'rows' is a reserved parameter for the evaluation function")
    else:
        evaluation_test_kwargs = {}

    # Handle both sync and async test functions
    if asyncio.iscoroutinefunction(test_func):
        if processed_row is not None:
            test_func = cast(Callable[[EvaluationRow], Awaitable[EvaluationRow]], test_func)
            return await test_func(processed_row, **evaluation_test_kwargs)
        if processed_dataset is not None:
            test_func = cast(Callable[[list[EvaluationRow]], Awaitable[list[EvaluationRow]]], test_func)
            return await test_func(processed_dataset, **evaluation_test_kwargs)
        test_func = cast(Callable[[], Awaitable[EvaluationRow]], test_func)
        return await test_func(**evaluation_test_kwargs)
    else:
        if processed_row is not None:
            test_func = cast(Callable[[EvaluationRow], EvaluationRow], test_func)
            return test_func(processed_row, **evaluation_test_kwargs)
        if processed_dataset is not None:
            test_func = cast(Callable[[Dataset], Dataset], test_func)
            return test_func(processed_dataset, **evaluation_test_kwargs)
        test_func = cast(Callable[[], EvaluationRow], test_func)
        return test_func(**evaluation_test_kwargs)


async def execute_pytest_with_exception_handling(
    test_func: TestFunction,
    evaluation_test_kwargs: dict[str, Any],
    processed_row: EvaluationRow | None = None,
    processed_dataset: list[EvaluationRow] | None = None,
) -> EvaluationRow | list[EvaluationRow]:
    """Helper function to execute pytest with consistent exception handling.

    Args:
        test_func: The test function to execute
        evaluation_test_kwargs: Kwargs for the evaluation function
        processed_row: Single row for pointwise evaluation (mutually exclusive with processed_dataset)
        processed_dataset: Dataset for groupwise/all evaluation (mutually exclusive with processed_row)

    Returns:
        The result of execute_pytest, or the input data with error results on exception
    """
    try:
        if processed_row is not None:
            return await execute_pytest(
                test_func,
                processed_row=processed_row,
                evaluation_test_kwargs=evaluation_test_kwargs,
            )
        else:
            return await execute_pytest(
                test_func,
                processed_dataset=processed_dataset,
                evaluation_test_kwargs=evaluation_test_kwargs,
            )
    except Exception as e:
        if os.getenv("EP_RAISE_EVAL_EXCEPTIONS", "true").strip() == "false":
            # Handle single row case
            if processed_row is not None:
                result = processed_row
                result.evaluation_result = EvaluateResult(
                    score=0.0,
                    is_score_valid=False,
                    reason=f"Error during evaluation: {type(e).__name__}: {e}",
                )
                if result.eval_metadata is not None:
                    result.eval_metadata.status = Status.error(
                        f"Error during evaluation: {type(e).__name__}: {e}",
                    )
                return result
            # Handle list of rows case
            elif processed_dataset is not None:
                results = processed_dataset
                for row in results:
                    row.evaluation_result = EvaluateResult(
                        score=0.0,
                        is_score_valid=False,
                        reason=f"Error during evaluation: {type(e).__name__}: {e}",
                    )
                    if row.eval_metadata is not None:
                        row.eval_metadata.status = Status.error(
                            f"Error during evaluation: {type(e).__name__}: {e}",
                        )
                return results
            else:
                # This should never happen since one of processed_row/processed_dataset must be provided
                raise ValueError("Neither processed_row nor processed_dataset was provided")
        # Default: raise exceptions unless explicitly disabled
        else:
            raise
