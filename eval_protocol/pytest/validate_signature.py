from collections.abc import Sequence
from inspect import Signature
from typing import get_origin, get_args

from eval_protocol.models import CompletionParams, EvaluationRow
from eval_protocol.pytest.types import EvaluationTestMode


def _is_list_of_evaluation_row(annotation) -> bool:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    """Check if annotation is list[EvaluationRow] or equivalent."""
    origin = get_origin(annotation)  # pyright: ignore[reportUnknownArgumentType, reportAny]
    if origin is not list:
        return False

    args = get_args(annotation)
    if len(args) != 1:
        return False

    # Check if the single argument is EvaluationRow or equivalent
    arg = args[0]  # pyright: ignore[reportAny]
    return arg is EvaluationRow or str(arg) == str(EvaluationRow)  # pyright: ignore[reportAny]


def validate_signature(
    signature: Signature, mode: EvaluationTestMode, completion_params: Sequence[CompletionParams | None] | None
) -> None:
    # For pointwise/groupwise mode, we expect a different signature
    # we expect single row to be passed in as the original row
    if mode == "pointwise":
        # Pointwise mode: function should accept messages and other row-level params
        if "row" not in signature.parameters:
            raise ValueError("In pointwise mode, your eval function must have a parameter named 'row'")

        # validate that "Row" is of type EvaluationRow
        if signature.parameters["row"].annotation is not EvaluationRow:  # pyright: ignore[reportAny]
            raise ValueError("In pointwise mode, the 'row' parameter must be of type EvaluationRow")

        # validate that the function has a return type of EvaluationRow
        if signature.return_annotation is not EvaluationRow:  # pyright: ignore[reportAny]
            raise ValueError("In pointwise mode, your eval function must return an EvaluationRow instance")

        # additional check for groupwise evaluation
    elif mode == "groupwise":
        if "rows" not in signature.parameters:
            raise ValueError("In groupwise mode, your eval function must have a parameter named 'rows'")

        # validate that "Rows" is of type List[EvaluationRow]
        if not _is_list_of_evaluation_row(signature.parameters["rows"].annotation):  # pyright: ignore[reportAny]
            raise ValueError(
                f"In groupwise mode, the 'rows' parameter must be of type List[EvaluationRow]. Got {str(signature.parameters['rows'].annotation)} instead"  # pyright: ignore[reportAny]
            )

        # validate that the function has a return type of List[EvaluationRow]
        if not _is_list_of_evaluation_row(signature.return_annotation):  # pyright: ignore[reportAny]
            raise ValueError("In groupwise mode, your eval function must return a list of EvaluationRow instances")
        if completion_params is not None and len(completion_params) < 2:
            raise ValueError("In groupwise mode, you must provide at least 2 completion parameters")
    else:
        # all mode: function should accept input_dataset and model
        if "rows" not in signature.parameters:
            raise ValueError("In all mode, your eval function must have a parameter named 'rows'")

        # validate that "Rows" is of type List[EvaluationRow]
        if not _is_list_of_evaluation_row(signature.parameters["rows"].annotation):  # pyright: ignore[reportAny]
            raise ValueError(
                f"In all mode, the 'rows' parameter must be of type list[EvaluationRow]. Got {str(signature.parameters['rows'].annotation)} instead"  # pyright: ignore[reportAny]
            )

        # validate that the function has a return type of List[EvaluationRow]
        if not _is_list_of_evaluation_row(signature.return_annotation):  # pyright: ignore[reportAny]
            raise ValueError("In all mode, your eval function must return a list of EvaluationRow instances")
