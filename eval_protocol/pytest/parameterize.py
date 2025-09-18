import inspect
from typing import TypedDict, Protocol
from collections.abc import Callable, Sequence, Iterable, Awaitable

from _pytest.mark import ParameterSet

from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.generate_parameter_combinations import CombinationTuple
from eval_protocol.pytest.types import DatasetPathParam, EvaluationInputParam, InputMessagesParam, TestFunction


class PytestMarkParametrizeKwargs(TypedDict):
    argnames: Sequence[str]
    argvalues: Iterable[ParameterSet | Sequence[object] | object]
    ids: Iterable[str] | None


class ParametrizeArgs(TypedDict):
    """
    This contains all the necessary information to properly hijack the test
    function's signature and dynamically inject usage of
    pytest.mark.parametrize. The two will differ when a user manually provides
    the pytest.mark.parametrize decorator instead of passing completion_params
    on their own.
    """

    # for create_dynamically_parameterized_wrapper
    sig_parameters: Sequence[str]

    # for pytest.mark.parametrize
    pytest_parametrize_kwargs: PytestMarkParametrizeKwargs


class ParameterIdGenerator(Protocol):
    """Protocol for generating pytest parameter IDs from parameter combinations."""

    def generate_id(self, combo: CombinationTuple) -> str | None:
        """Generate an ID for a parameter combination.

        Args:
            combo: The parameter combination tuple

        Returns:
            A string ID for the combination, or None to use default pytest ID
        """
        ...


class DefaultParameterIdGenerator(ParameterIdGenerator):
    """Default ID generator that creates meaningful IDs from parameter combinations."""

    def __init__(self, max_length: int = 200):
        """Initialize the ID generator with configuration options.

        Args:
            max_length: Maximum length of generated IDs
        """
        self.max_length = max_length

    def generate_id(self, combo: CombinationTuple) -> str | None:
        """Generate an ID for a parameter combination."""
        dataset, completion_params, messages, rows, evaluation_test_kwargs = combo

        if completion_params:
            id = self.generate_id_from_dict(completion_params, self.max_length)
            if id:
                return id
        else:
            if rows:
                return f"rows(len={len(rows)})"
            elif messages:
                return f"messages(len={len(messages)})"
            elif dataset:
                return f"dataset(len={len(dataset)})"
        return None

    @staticmethod
    def generate_id_from_dict(d: dict[str, object], max_length: int = 200) -> str | None:
        # Get all string, numeric, and boolean values from completion_params, sorted by key
        str_values = []
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, (str, int, float, bool)):
                str_values.append(str(value))

        if str_values:
            id_str = ":".join(str_values)

            # Truncate if too long
            if len(id_str) > max_length:
                id_str = id_str[: max_length - 3] + "..."

            return id_str
        return None


def pytest_parametrize(
    combinations: list[CombinationTuple],
    input_dataset: Sequence[DatasetPathParam] | None,
    completion_params: Sequence[CompletionParams | None] | None,
    completion_params_provided: bool,
    input_messages: Sequence[list[InputMessagesParam] | None] | None,
    input_rows: Sequence[list[EvaluationRow]] | None,
    evaluation_test_kwargs: Sequence[EvaluationInputParam | None] | None,
    id_generator: ParameterIdGenerator | None = None,
) -> ParametrizeArgs:
    """
    This function dynamically generates pytest.mark.parametrize arguments for a given
    set of combinations. This is the magic that allows developers to pass in their
    inputs in a single decorator and generate all combinations of experiments
    without having to create their own fixtures and confirming to eval-protocol's
    API.
    """

    # Create parameter tuples for pytest.mark.parametrize
    argnames: list[str] = []
    sig_parameters: list[str] = []
    if input_dataset is not None:
        argnames.append("dataset_path")
        sig_parameters.append("dataset_path")
    if completion_params is not None:
        if completion_params_provided:
            argnames.append("completion_params")
        sig_parameters.append("completion_params")
    if input_messages is not None:
        argnames.append("input_messages")
        sig_parameters.append("input_messages")
    if input_rows is not None:
        argnames.append("input_rows")
        sig_parameters.append("input_rows")
    if evaluation_test_kwargs is not None:
        argnames.append("evaluation_test_kwargs")
        sig_parameters.append("evaluation_test_kwargs")

    # Use default ID generator if none provided
    if id_generator is None:
        id_generator = DefaultParameterIdGenerator()

    argvalues: list[ParameterSet | Sequence[object] | object] = []
    ids: list[str] = []

    for combo in combinations:
        dataset, cp, messages, rows, etk = combo
        param_tuple: list[object] = []

        # Build parameter tuple based on what's provided
        if input_dataset is not None:
            param_tuple.append(dataset)
        if completion_params_provided:
            param_tuple.append(cp)
        if input_messages is not None:
            param_tuple.append(messages)
        if input_rows is not None:
            param_tuple.append(rows)
        if evaluation_test_kwargs is not None:
            param_tuple.append(etk)

        # Validate parameter tuple length
        if len(argnames) != len(param_tuple):
            raise ValueError(
                f"The length of argnames ({len(argnames)}) is not the same as the length of param_tuple ({len(param_tuple)})"
            )

        argvalues.append(tuple(param_tuple))

        # Generate ID for this combination
        combo_id = id_generator.generate_id(combo)
        if combo_id is not None:
            ids.append(combo_id)

    # Return None for ids if no IDs were generated (let pytest use defaults)
    return ParametrizeArgs(
        pytest_parametrize_kwargs=PytestMarkParametrizeKwargs(
            argnames=argnames, argvalues=argvalues, ids=ids if ids else None
        ),
        sig_parameters=sig_parameters,
    )


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
