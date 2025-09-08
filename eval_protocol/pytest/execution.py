import asyncio
from collections.abc import Awaitable, Callable
from typing import cast
from eval_protocol.models import EvaluationRow
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
