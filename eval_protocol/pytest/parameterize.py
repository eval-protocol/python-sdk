import inspect
from typing import TypedDict
from collections.abc import Callable, Sequence, Iterable, Awaitable

from _pytest.mark import ParameterSet

from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.generate_parameter_combinations import CombinationTuple
from eval_protocol.pytest.types import DatasetPathParam, EvaluationInputParam, InputMessagesParam, TestFunction


class PytestParametrizeArgs(TypedDict):
    argnames: Sequence[str]
    argvalues: Iterable[ParameterSet | Sequence[object] | object]


def pytest_parametrize(
    combinations: list[CombinationTuple],
    input_dataset: list[DatasetPathParam] | None,
    completion_params: Sequence[CompletionParams | None] | None,
    input_messages: Sequence[InputMessagesParam | None] | None,
    input_rows: Sequence[list[EvaluationRow]] | None,
    evaluation_test_kwargs: Sequence[EvaluationInputParam | None] | None,
) -> PytestParametrizeArgs:
    """
    This function dynamically generates pytest.mark.parametrize arguments for a given
    set of combinations. This is the magic that allows developers to pass in their
    inputs in a single decorator and generate all combinations of experiments
    without having to create their own fixtures and confirming to eval-protocol's
    API.
    """

    # Create parameter tuples for pytest.mark.parametrize
    argnames: list[str] = []
    if input_dataset is not None:
        argnames.append("dataset_path")
    if completion_params is not None:
        argnames.append("completion_params")
    if input_messages is not None:
        argnames.append("input_messages")
    if input_rows is not None:
        argnames.append("input_rows")
    if evaluation_test_kwargs is not None:
        argnames.append("evaluation_test_kwargs")

    argvalues: list[ParameterSet | Sequence[object] | object] = []
    param_tuples: list[tuple[object, ...]] = []
    for combo in combinations:
        dataset, cp, messages, rows, etk = combo
        param_tuple: list[object] = []
        if input_dataset is not None:
            param_tuple.append(dataset)
        if completion_params is not None:
            param_tuple.append(cp)
        if input_messages is not None:
            param_tuple.append(messages)
        if input_rows is not None:
            param_tuple.append(rows)
        if evaluation_test_kwargs is not None:
            param_tuple.append(etk)
        # do validation that the length of argnames is the same as the length of param_tuple
        if len(argnames) != len(param_tuple):
            raise ValueError(
                f"The length of argnames ({len(argnames)}) is not the same as the length of param_tuple ({len(param_tuple)})"
            )
        param_tuples.append(tuple(param_tuple))

    return PytestParametrizeArgs(argnames=argnames, argvalues=argvalues)


def create_dynamically_parameterized_wrapper(
    test_func: TestFunction, wrapper_body: Callable[..., Awaitable[None]], test_param_names: Sequence[str]
) -> Callable[..., None]:
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

    @wraps(test_func)  # pyright: ignore[reportArgumentType]
    async def wrapper(**kwargs) -> None:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return await wrapper_body(**kwargs)

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in test_param_names]
    wrapper.__signature__ = inspect.Signature(parameters)  # pyright: ignore[reportAttributeAccessIssue]

    return wrapper  # pyright: ignore[reportUnknownVariableType, reportReturnType]
