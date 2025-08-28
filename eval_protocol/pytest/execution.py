import asyncio
from collections.abc import Awaitable
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.types import EvaluationInputParam, TestFunction


async def execute_pytest(
    test_func: TestFunction,
    processed_row: EvaluationRow | None = None,
    processed_dataset: list[EvaluationRow] | None = None,
    evaluation_test_kwargs: EvaluationInputParam | None = None,
) -> EvaluationRow | list[EvaluationRow]:
    if evaluation_test_kwargs is not None:
        if "row" in evaluation_test_kwargs:
            raise ValueError("'row' is a reserved parameter for the evaluation function")
        if "rows" in evaluation_test_kwargs:
            raise ValueError("'rows' is a reserved parameter for the evaluation function")

    # Handle both sync and async test functions
    if asyncio.iscoroutinefunction(test_func):
        if processed_row is not None:
            return await test_func(processed_row)
        if processed_dataset is not None:
            return await test_func(processed_dataset)
        return await test_func()
    else:
        if processed_row is not None:
            row = test_func(processed_row)
        if processed_dataset is not None:
            return test_func(processed_dataset)
        return test_func()
