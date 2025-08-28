from typing import TypedDict
from collections.abc import Sequence, Iterable

from _pytest.mark import ParameterSet

from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.types import DatasetPathParam, EvaluationInputParam, InputMessagesParam
from eval_protocol.pytest.utils import CombinationTuple


class PytestParametrizeArgs(TypedDict):
    argnames: str | Sequence[str]
    argvalues: Iterable[ParameterSet | Sequence[object] | object]


def pytest_parametrize(
    combinations: list[CombinationTuple],
    input_dataset: list[DatasetPathParam] | None,
    completion_params: list[CompletionParams | None] | None,
    input_messages: list[InputMessagesParam | None] | None,
    input_rows: list[EvaluationRow] | None,
    evaluation_test_kwargs: list[EvaluationInputParam | None] | None,
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
