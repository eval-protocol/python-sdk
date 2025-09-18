from typing import TypedDict
from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.types import Dataset, DatasetPathParam, EvaluationInputParam, InputMessagesParam
from eval_protocol.pytest.utils import parse_ep_max_rows
from collections.abc import Sequence


InputDatasetKwarg = list[DatasetPathParam] | None
"""
Either a single dataset path or a list of dataset paths depending on if
combine_datasets is True or False. If True, then you would expect to see a list
of dataset paths. If False, then you would expect to see a list with a single
dataset path.
"""

CompletionParamsKwarg = CompletionParams | None
"""
Either a single completion params object or None.
"""

InputMessagesKwarg = list[InputMessagesParam] | None
InputRowsKwarg = Dataset | None
EvaluationTestKwargs = EvaluationInputParam | None

CombinationTuple = tuple[
    InputDatasetKwarg,
    CompletionParamsKwarg,
    InputMessagesKwarg,
    InputRowsKwarg,
    EvaluationTestKwargs,
]


class ParameterizedTestKwargs(TypedDict, total=False):
    """
    These are the type of parameters that can be passed to the generated pytest
    function. Every experiment is a unique combination of these parameters.
    """

    dataset_path: InputDatasetKwarg
    completion_params: CompletionParamsKwarg
    input_messages: InputMessagesKwarg
    input_rows: InputRowsKwarg
    evaluation_test_kwargs: EvaluationTestKwargs


def generate_parameter_combinations(
    input_dataset: Sequence[DatasetPathParam] | None,
    completion_params: Sequence[CompletionParams | None],
    input_messages: Sequence[list[InputMessagesParam] | None] | None,
    input_rows: Sequence[list[EvaluationRow] | None] | None,
    evaluation_test_kwargs: Sequence[EvaluationInputParam | None] | None,
    max_dataset_rows: int | None,
    combine_datasets: bool,
) -> list[CombinationTuple]:
    """
    Generate all combinations of parameters for pytest parameterization.

    Args:
        input_dataset: Dataset paths to use
        completion_params: Completion parameters to test
        input_messages: Input messages to use
        input_rows: Pre-constructed EvaluationRow objects to use
        evaluation_test_kwargs: Additional kwargs for evaluation tests
        max_dataset_rows: Maximum number of dataset rows to process
        combine_datasets: Whether to combine multiple datasets into one test

    Returns:
        List of parameter tuples for pytest.mark.parametrize
    """
    # Optionally combine multiple dataset paths into one logical dataset,
    # or parameterize to run one dataset per test invocation.
    datasets: Sequence[list[DatasetPathParam] | None] = [None]
    if input_dataset is not None:
        if combine_datasets:
            datasets = [list(input_dataset)]
        else:
            # Fan out: one dataset path per parameterization
            datasets = [[p] for p in input_dataset]

    cps: Sequence[CompletionParams | None] = completion_params

    # Apply EP_MAX_DATASET_ROWS to input_messages, but do NOT parameterize over
    # each row. Instead, pass the entire sliced list through in a single test run
    # so summaries aggregate all rows together (AIME-style behavior).
    messages: Sequence[list[InputMessagesParam] | None] = [None]
    if input_messages is not None:
        effective_max_rows = parse_ep_max_rows(max_dataset_rows)
        if effective_max_rows is not None:
            sliced_messages: Sequence[list[InputMessagesParam] | None] = [
                dataset_messages[:effective_max_rows]
                for dataset_messages in input_messages
                if dataset_messages is not None
            ]
        else:
            sliced_messages = input_messages
        # Wrap as a single parameter payload
        messages = sliced_messages

    # Handle input_rows - similar to input_messages, apply max_dataset_rows if specified
    if input_rows is not None:
        effective_max_rows = parse_ep_max_rows(max_dataset_rows)
        if effective_max_rows is not None:
            input_rows = [row[:effective_max_rows] for row in input_rows if row is not None]
    else:
        input_rows = [None]

    if evaluation_test_kwargs is None:
        evaluation_test_kwargs = [None]

    combinations: list[CombinationTuple] = []

    # Generate all combinations
    for ds in datasets:
        for cp in cps:
            for im in messages:
                for ir in input_rows:
                    for etk in evaluation_test_kwargs:
                        # if no dataset, no messages, and no rows, raise an error
                        if ds is None and im is None and ir is None:
                            raise ValueError(
                                "No dataset, messages, or rows provided. Please provide at least one of input_dataset, input_messages, or input_rows."
                            )
                        combinations.append((ds, cp, im, ir, etk))

    return combinations
