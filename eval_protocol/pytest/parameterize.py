import ast
import inspect
from typing import TypedDict, Protocol
from collections.abc import Callable, Sequence, Iterable, Awaitable

from _pytest.mark import ParameterSet

from eval_protocol.data_loader.models import EvaluationDataLoader
from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.generate_parameter_combinations import CombinationTuple
from eval_protocol.pytest.types import DatasetPathParam, EvaluationInputParam, InputMessagesParam, TestFunction


def _has_pytest_parametrize_with_completion_params(test_func: TestFunction) -> bool:
    """
    Check if a test function has a pytest.mark.parametrize decorator with argnames="completion_params".

    This function uses inspect.getsource and ast to parse the function's source code and look for
    pytest.mark.parametrize decorators that include "completion_params" in their argnames.

    Args:
        test_func: The test function to analyze

    Returns:
        True if the function has a pytest.mark.parametrize decorator with "completion_params" in argnames,
        False otherwise

    Raises:
        OSError: If the source code cannot be retrieved (e.g., function is defined in interactive mode)
        SyntaxError: If the source code cannot be parsed as valid Python
    """
    try:
        source = inspect.getsource(test_func)
    except OSError:
        # Function source cannot be retrieved (e.g., defined in interactive mode)
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Source code cannot be parsed
        return False

    # Walk through the AST to find pytest.mark.parametrize decorators
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Check decorators on this function
            for decorator in node.decorator_list:
                if _is_pytest_parametrize_with_completion_params(decorator):
                    return True

    return False


def _is_pytest_parametrize_with_completion_params(decorator: ast.expr) -> bool:
    """
    Check if a decorator is pytest.mark.parametrize with "completion_params" in argnames.

    Args:
        decorator: AST node representing a decorator

    Returns:
        True if this is a pytest.mark.parametrize decorator with "completion_params" in argnames
    """
    # Look for pytest.mark.parametrize pattern
    if isinstance(decorator, ast.Call):
        # Check if it's pytest.mark.parametrize
        if isinstance(decorator.func, ast.Attribute):
            if (
                isinstance(decorator.func.value, ast.Attribute)
                and isinstance(decorator.func.value.value, ast.Name)
                and decorator.func.value.value.id == "pytest"
                and decorator.func.value.attr == "mark"
                and decorator.func.attr == "parametrize"
            ):
                # Validate argvalues if present
                _validate_parametrize_argvalues(decorator)

                # Check positional arguments first (argnames is typically the first positional arg)
                if len(decorator.args) > 0:
                    argnames_arg = decorator.args[0]
                    if _check_argnames_for_completion_params(argnames_arg):
                        return True

                # Check keyword arguments for argnames
                for keyword in decorator.keywords:
                    if keyword.arg == "argnames":
                        if _check_argnames_for_completion_params(keyword.value):
                            return True

    return False


def _ast_dict_to_string(dict_node: ast.Dict) -> str:
    """
    Convert an AST Dict node to its string representation.

    Args:
        dict_node: AST node representing a dictionary

    Returns:
        String representation of the dictionary
    """
    if not dict_node.keys:
        return "{}"

    pairs = []
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is not None:
            key_str = _ast_node_to_string(key)
            value_str = _ast_node_to_string(value)
            pairs.append(f"{key_str}: {value_str}")

    return "{" + ", ".join(pairs) + "}"


def _ast_node_to_string(node: ast.expr) -> str:
    """
    Convert an AST node to its string representation.

    Args:
        node: AST node to convert

    Returns:
        String representation of the node
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return repr(node.value)
        else:
            return str(node.value)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Dict):
        return _ast_dict_to_string(node)
    elif isinstance(node, ast.List):
        elements = [_ast_node_to_string(elt) for elt in node.elts]
        return "[" + ", ".join(elements) + "]"
    elif isinstance(node, ast.Tuple):
        elements = [_ast_node_to_string(elt) for elt in node.elts]
        return "(" + ", ".join(elements) + ")"
    else:
        # For complex expressions, return a simplified representation
        return "<complex expression>"


def _validate_parametrize_argvalues(decorator: ast.Call) -> None:
    """
    Validate that pytest.mark.parametrize argvalues is a list/tuple, not a dict.

    Args:
        decorator: AST node representing the pytest.mark.parametrize decorator call

    Raises:
        ValueError: If argvalues is a dict instead of a list/tuple
    """
    # Check positional arguments (argvalues is typically the second positional arg)
    if len(decorator.args) > 1:
        argvalues_arg = decorator.args[1]
        if isinstance(argvalues_arg, ast.Dict):
            dict_repr = _ast_dict_to_string(argvalues_arg)
            raise ValueError(
                f"For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict. "
                f"Use [{dict_repr}] instead of {dict_repr}."
            )

    # Check keyword arguments for argvalues
    for keyword in decorator.keywords:
        if keyword.arg == "argvalues":
            if isinstance(keyword.value, ast.Dict):
                dict_repr = _ast_dict_to_string(keyword.value)
                raise ValueError(
                    f"For evaluation_test with completion_params, pytest.mark.parametrize argvalues must be a list or tuple, not a dict. "
                    f"Use [{dict_repr}] instead of {dict_repr}."
                )


def _check_argnames_for_completion_params(argnames_node: ast.expr) -> bool:
    """
    Check if an argnames AST node contains "completion_params".

    Args:
        argnames_node: AST node representing the argnames value

    Returns:
        True if argnames contains "completion_params"
    """
    if isinstance(argnames_node, ast.Constant):
        # Single string case: argnames="completion_params"
        if argnames_node.value == "completion_params":
            return True
    elif isinstance(argnames_node, ast.List):
        # List case: argnames=["completion_params", ...]
        for elt in argnames_node.elts:
            if isinstance(elt, ast.Constant) and elt.value == "completion_params":
                return True
    elif isinstance(argnames_node, ast.Tuple):
        # Tuple case: argnames=("completion_params", ...)
        for elt in argnames_node.elts:
            if isinstance(elt, ast.Constant) and elt.value == "completion_params":
                return True

    return False


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
        dataset, completion_params, messages, rows, evaluation_test_kwargs, data_loaders = combo

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
    test_func: TestFunction | None,
    input_dataset: Sequence[DatasetPathParam] | None,
    completion_params: Sequence[CompletionParams | None] | None,
    completion_params_provided: bool,
    input_messages: Sequence[list[InputMessagesParam] | None] | None,
    input_rows: Sequence[list[EvaluationRow]] | None,
    data_loaders: Sequence[EvaluationDataLoader] | EvaluationDataLoader | None,
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

    if test_func is not None:
        has_pytest_parametrize = _has_pytest_parametrize_with_completion_params(test_func)
    else:
        has_pytest_parametrize = False

    # Create parameter tuples for pytest.mark.parametrize
    argnames: list[str] = []
    sig_parameters: list[str] = []
    if input_dataset is not None:
        argnames.append("dataset_path")
        sig_parameters.append("dataset_path")
    if completion_params is not None:
        """
        manually adding completion_params as a pytest.mark.parametrize decorator
        automatically adds it to the function signature so we only need to add
        it if we provided completion_params using the evaluation_test decorator.
        """
        if completion_params_provided and not has_pytest_parametrize:
            argnames.append("completion_params")
        if has_pytest_parametrize or completion_params_provided:
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
    if data_loaders is not None:
        argnames.append("data_loaders")
        sig_parameters.append("data_loaders")

    # Use default ID generator if none provided
    if id_generator is None:
        id_generator = DefaultParameterIdGenerator()

    argvalues: list[ParameterSet | Sequence[object] | object] = []
    ids: list[str] = []

    for combo in combinations:
        dataset, cp, messages, rows, etk, dl = combo
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
        if data_loaders is not None:
            param_tuple.append(dl)

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
