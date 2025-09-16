"""Utilities for preparing datasets for evaluation tests."""

from collections.abc import Callable
from typing import Any

from eval_protocol.human_id import generate_id, num_combinations
from eval_protocol.models import EvaluationRow
from eval_protocol.pytest.generate_parameter_combinations import ParameterizedTestKwargs
from eval_protocol.pytest.types import Dataset

from ..common_utils import load_jsonl


def load_and_prepare_rows(
    kwargs: ParameterizedTestKwargs,
    *,
    dataset_adapter: Callable[[list[dict[str, Any]]], Dataset],
    preprocess_fn: Callable[[list[EvaluationRow]], list[EvaluationRow]] | None,
    max_dataset_rows: int | None,
) -> list[EvaluationRow]:
    """Load and preprocess evaluation rows based on parameterized pytest kwargs.

    This helper consolidates the logic that loads input data from various sources
    (dataset paths, raw messages, or pre-built :class:`EvaluationRow` objects),
    applies optional preprocessing, and ensures each row has a stable
    ``row_id``. The behavior mirrors the original inline implementation inside
    :func:`eval_protocol.pytest.evaluation_test.evaluation_test`.
    """

    data: list[EvaluationRow] = []

    if kwargs.get("dataset_path") is not None:
        ds_arg = kwargs["dataset_path"]
        data_jsonl: list[dict[str, Any]] = []
        for path in ds_arg or []:
            data_jsonl.extend(load_jsonl(path))
        if max_dataset_rows is not None:
            data_jsonl = data_jsonl[:max_dataset_rows]
        data = dataset_adapter(data_jsonl)
    elif kwargs.get("input_messages") is not None:
        input_messages = kwargs["input_messages"] or []
        data = [EvaluationRow(messages=dataset_messages) for dataset_messages in input_messages]
    elif kwargs.get("input_rows") is not None:
        input_rows = kwargs["input_rows"] or []
        data = [row.model_copy(deep=True) for row in input_rows]
    else:
        raise ValueError("No input dataset, input messages, or input rows provided")

    if preprocess_fn:
        data = preprocess_fn(data)

    for row in data:
        if row.input_metadata.row_id is None:
            index = hash(row)
            max_index = num_combinations() - 1
            index = abs(index) % (max_index + 1)
            row.input_metadata.row_id = generate_id(seed=0, index=index)

    return data
